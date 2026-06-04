"""Joint submodular de-synchronisation vs the separable schedulers, in the regime where coupling
bites: k>=2 broadcasts/step (N/T>=2) + CLUSTERED (high-overlap) team, where refreshing two nearby
stale robots is redundant. Compares SYNC vs even STAGGER vs CWD (per-arm Whittle index) vs SUBMOD
(greedy joint max-coverage). Key: submod_vs_even (does coupling-awareness beat uniform staggering?)
and submod_vs_cwd (does joint beat per-arm?). Group setcomm_submod."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_submod"
SEEDS=list(range(int(os.environ.get("SM_SEEDS","10")))); IV=int(os.environ.get("SM_IV","8")); N=int(os.environ.get("SM_N","16")); NUM=20
WORKERS=int(os.environ.get("SM_WORKERS","10")); SPAWN=os.environ.get("SM_SPAWN","clustered")
CONDS=[("Sync","periodic",""),("StagEven","staggered","even"),("CWD","cwd",""),("Submod","submod","")]
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv"); return os.path.exists(p) and os.path.getsize(p)>100
def one(j):
    m,mode,pat,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"
    if SPAWN!="default": env["BLMRE_SPAWN_CLUSTER"]=SPAWN
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    cmd=[PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),"--num-maps",str(NUM),
         "--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","submod.log"),"a").write(m+"\n")
def main():
    jobs=[(f"{nm}_{IV}",mode,pat,s) for nm,mode,pat in CONDS for s in SEEDS if not ok(f"{nm}_{IV}",s)]
    t0=time.time(); log(f"submod: sync/even/cwd/submod x {len(SEEDS)} seeds @ N={N} T={IV} spawn={SPAWN} = {len(jobs)} jobs")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","SUBMOD_DONE"),"w").close(); log(f"SUBMOD DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","total_communication_volume","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["cond"]=d.method_name.str.extract(r"^(Sync|StagEven|CWD|Submod)_"); d=d.dropna(subset=["cond"]); rows=[]
        def pair(A,B,lab):
            X=d[d.cond==A].set_index(["map_id","seed"]); Y=d[d.cond==B].set_index(["map_id","seed"])
            idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(comparison=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),5)))
        pair("StagEven","Sync","even_vs_sync"); pair("Submod","Sync","submod_vs_sync")
        pair("Submod","StagEven","submod_vs_even"); pair("Submod","CWD","submod_vs_cwd")
        import pandas as pd2; o=pd2.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_submod_paired.csv"),index=False)
        vol=d.groupby("cond")["total_communication_volume"].mean().round(0)
        log("joint submodular vs even/cwd (submod_vs_even, submod_vs_cwd are KEY; neg=submod better):\n"+o.to_string(index=False)+"\n-- comm volume --\n"+vol.to_string())
    except Exception as e:
        import traceback; log(f"an failed {e}\n{traceback.format_exc()[-300:]}")
if __name__=="__main__": sys.exit(main())
