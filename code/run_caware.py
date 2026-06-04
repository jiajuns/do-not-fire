"""Does the method absorbing the 2025 correlated-source theory help? Correlation-AWARE staggering
(separate spatially-near = more-correlated robots in phase) vs plain EVEN staggering vs SYNC, under
CLUSTERED spawns (robots seeded close -> high initial correlation, where correlation-aware
scheduling should have the most room). N=8, interval 12, broadcast model. Key comparison: does
corr-aware staggering beat even staggering when information is correlated? Group setcomm_caware."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_caware"
SEEDS=list(range(int(os.environ.get("CA_SEEDS","8")))); IV=int(os.environ.get("CA_IV","12")); N=8; NUM=20
WORKERS=int(os.environ.get("CA_WORKERS","8")); SPAWN=os.environ.get("CA_SPAWN","clustered")
# (method, comm-mode, stagger_pattern)
CONDS=[("Sync","periodic",""),("StagEven","staggered","even"),("StagCorr","staggered","corraware")]
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv"); return os.path.exists(p) and os.path.getsize(p)>100
def one(j):
    m,mode,pat,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]=SPAWN
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    cmd=[PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),"--num-maps",str(NUM),
         "--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","caware.log"),"a").write(m+"\n")
def main():
    jobs=[]
    for nm,mode,pat in CONDS:
        m=f"{nm}_{IV}"
        for s in SEEDS:
            if not ok(m,s): jobs.append((m,mode,pat,s))
    t0=time.time(); log(f"caware: sync/even/corraware x {len(SEEDS)} seeds @ iv{IV}, spawn={SPAWN} = {len(jobs)} jobs")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","CAWARE_DONE"),"w").close(); log(f"CAWARE DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["cond"]=d.method_name.str.extract(r"^(Sync|StagEven|StagCorr)_"); d=d.dropna(subset=["cond"]); rows=[]
        def pair(A,B,lab):
            X=d[d.cond==A].set_index(["map_id","seed"]); Y=d[d.cond==B].set_index(["map_id","seed"])
            idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(comparison=lab,metric=met.split("_")[0],a_mean=round(float(a.mean()),2),b_mean=round(float(c.mean()),2),
                    pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),5)))
        pair("StagEven","Sync","even_vs_sync"); pair("StagCorr","Sync","corr_vs_sync"); pair("StagCorr","StagEven","corr_vs_even")
        import pandas as pd2; o=pd2.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_caware_paired.csv"),index=False)
        log("correlation-aware vs even staggering (clustered spawn; corr_vs_even neg=corr-aware better):\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log(f"an failed {e}\n{traceback.format_exc()[-300:]}")
if __name__=="__main__": sys.exit(main())
