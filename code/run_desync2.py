"""Deployability test: in a realistic decentralised setting the no-coordinator DEFAULT is synchronised
comm (robots boot together / share one schedule, can't compute even offsets without global N+clock).
Can DESYNC self-organise AWAY from a synchronised cold-start to even spacing and RECOVER the de-sync
benefit, using only local overhearing? Conditions: Sync (periodic = realistic default, worst) |
StagIdeal (centralised even-staggering, no drift = unreachable oracle) | DESYNCsync (DESYNC from a
~synchronised cold start + small clock drift to break symmetry) | DESYNCeven (DESYNC from coordinated
even start = best-case ref). N in {8,16} (smaller N converges in fewer O(N^2) rounds), T=8 clustered,
16 seeds. WIN = desync_vs_sync significantly NEGATIVE (recovers benefit) with desync_vs_ideal small.
Group setcomm_desync2."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_desync2"; NUM=20; IV=8; SPAWN="clustered"
SEEDS=list(range(int(os.environ.get("DS2_SEEDS","16")))); WORKERS=int(os.environ.get("DS2_WORKERS","5"))
NS=[8,16]
# (cond, comm_mode, stagger_pattern, drift, desync_init)
CONDS=[("Sync","periodic","",0.0,""),("StagIdeal","staggered","even",0.0,""),
       ("DESYNCsync","desync","",0.05,"sync"),("DESYNCeven","desync","",0.05,"even")]
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv")
    if not os.path.exists(p): return False
    try: return sum(1 for _ in open(p))>=NUM+1
    except: return False
def jobs_spec():
    return [(f"{c}_N{N}",mode,pat,d,init,N) for (c,mode,pat,d,init) in CONDS for N in NS]
def one(j):
    m,mode,pat,d,init,N,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]=SPAWN
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    env["BLMRE_CLOCK_DRIFT"]=str(d)
    if init: env["BLMRE_DESYNC_INIT"]=init
    else: env.pop("BLMRE_DESYNC_INIT",None)
    cmd=["timeout","1800",PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),
         "--num-maps",str(NUM),"--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","desync2.log"),"a").write(m+"\n")
def main():
    spec=jobs_spec(); jobs=[(m,mode,pat,d,init,N,s) for (m,mode,pat,d,init,N) in spec for s in SEEDS if not ok(m,s)]
    t0=time.time(); log(f"[{time.strftime('%H:%M')}] desync2: {len(jobs)} jobs ({len(spec)} configs x {len(SEEDS)} seeds) NS={NS} T={IV} workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","DESYNC2_DONE"),"w").close(); log(f"[{time.strftime('%H:%M')}] DESYNC2 DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv") if "coverage" not in f],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","total_communication_volume","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        rows=[]
        def pick(name): return d[d["method_name"]==name]
        def pair(A,B,tags,lab):
            X=pick(A).set_index(["map_id","seed"]); Y=pick(B).set_index(["map_id","seed"])
            X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(**tags,cmp=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),4)))
        for N in NS:
            pair(f"DESYNCsync_N{N}",f"Sync_N{N}",dict(N=N),"desync_vs_sync")     # THE deployability win
            pair(f"DESYNCsync_N{N}",f"StagIdeal_N{N}",dict(N=N),"desync_vs_ideal")
            pair(f"Sync_N{N}",f"StagIdeal_N{N}",dict(N=N),"sync_vs_ideal")        # full gap reference
            pair(f"DESYNCeven_N{N}",f"StagIdeal_N{N}",dict(N=N),"desyncEven_vs_ideal")
        o=pd.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_desync2_paired.csv"),index=False)
        log("=== DESYNC2 paired (neg=first better). WIN = desync_vs_sync significantly NEGATIVE & desync_vs_ideal small ===\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log("an failed: "+str(e)+"\n"+traceback.format_exc()[-400:])
if __name__=="__main__": sys.exit(main())
