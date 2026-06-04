"""Strengthened method: contention-weighted de-synchronisation (CWD), a Max-Weight / Whittle-index
broadcast policy that refreshes the robots that are both stale and currently contending. Compares
SYNC (periodic) vs even STAGGER (the homogeneous-optimal special case, Prop 3) vs CWD, under
DISPERSED spawns (heterogeneous, dynamic contention -> where adaptivity should help) and DEFAULT
random spawns. Key comparison: cwd_vs_even (does the Whittle-weighted scheduler beat uniform even
staggering?). N=8, interval 12. Group setcomm_cwd."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_cwd"
SEEDS=list(range(int(os.environ.get("CW_SEEDS","8")))); IV=int(os.environ.get("CW_IV","12")); N=8; NUM=20
WORKERS=int(os.environ.get("CW_WORKERS","8"))
SPAWNS=os.environ.get("CW_SPAWNS","dispersed,default").split(",")
# (method, comm-mode, stagger_pattern)
CONDS=[("Sync","periodic",""),("StagEven","staggered","even"),("CWD","cwd","")]
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv"); return os.path.exists(p) and os.path.getsize(p)>100
def one(j):
    m,mode,pat,sp,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"
    if sp!="default": env["BLMRE_SPAWN_CLUSTER"]=sp
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    cmd=[PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),"--num-maps",str(NUM),
         "--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","cwd.log"),"a").write(m+"\n")
def main():
    jobs=[]
    for sp in SPAWNS:
        for nm,mode,pat in CONDS:
            m=f"{sp}_{nm}_{IV}"
            for s in SEEDS:
                if not ok(m,s): jobs.append((m,mode,pat,sp,s))
    t0=time.time(); log(f"cwd: spawns {SPAWNS} x sync/even/cwd x {len(SEEDS)} seeds @ iv{IV} = {len(jobs)} jobs")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","CWD_DONE"),"w").close(); log(f"CWD DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["sp"]=d.method_name.str.extract(r"^(dispersed|default)_"); d["cond"]=d.method_name.str.extract(r"_(Sync|StagEven|CWD)_")
        d=d.dropna(subset=["sp","cond"]); rows=[]
        for sp in SPAWNS:
            ds=d[d.sp==sp]
            def pair(A,B,lab):
                X=ds[ds.cond==A].set_index(["map_id","seed"]); Y=ds[ds.cond==B].set_index(["map_id","seed"])
                idx=X.index.intersection(Y.index)
                if len(idx)<8: return
                for met in ["total_travel_distance","overlap_ratio"]:
                    a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                    try: _,p=stats.wilcoxon(a,c)
                    except: p=float("nan")
                    rows.append(dict(spawn=sp,comparison=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),5)))
            pair("StagEven","Sync","even_vs_sync"); pair("CWD","Sync","cwd_vs_sync"); pair("CWD","StagEven","cwd_vs_even")
        import pandas as pd2; o=pd2.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_cwd_paired.csv"),index=False)
        log("CWD (Whittle-weighted) vs even staggering (cwd_vs_even neg=CWD better):\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log(f"an failed {e}\n{traceback.format_exc()[-300:]}")
if __name__=="__main__": sys.exit(main())
