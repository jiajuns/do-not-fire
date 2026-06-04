"""Correlation-structure experiment (tests Theorem 1 of AoI-scheduling-of-correlated-sources, 2025,
in the multi-robot exploration setting). De-synchronisation should help MORE when robots' information
is more correlated. We vary initial spawn correlation via BLMRE_SPAWN_CLUSTER: CLUSTERED starts
(robots seeded close together -> highly overlapping/correlated early views) vs DISPERSED starts
(farthest-point spread -> low correlation), and compare staggered vs synchronised periodic at a
matched budget (N=8, interval 12). Prediction: the de-sync benefit (stag vs sync) is larger under
clustered spawns. Group setcomm_corr; paired Wilcoxon stag vs sync within each spawn condition."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_corr"
SEEDS=list(range(int(os.environ.get("CO_SEEDS","8")))); IV=int(os.environ.get("CO_IV","12")); N=8; NUM=20
WORKERS=int(os.environ.get("CO_WORKERS","8"))
SPAWNS=["clustered","dispersed"]
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv"); return os.path.exists(p) and os.path.getsize(p)>100
def one(j):
    m,mode,sp,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]=sp
    cmd=[PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),"--num-maps",str(NUM),
         "--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","corr.log"),"a").write(m+"\n")
def main():
    jobs=[]
    for sp in SPAWNS:
        for mode,tag in [("periodic","Sync"),("staggered","Stag")]:
            m=f"{sp}_{tag}_{IV}"
            for s in SEEDS:
                if not ok(m,s): jobs.append((m,mode,sp,s))
    t0=time.time(); log(f"corr: clustered/dispersed x sync/stag x {len(SEEDS)} seeds @ iv{IV} = {len(jobs)} jobs")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","CORR_DONE"),"w").close(); log(f"CORR DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["sp"]=d.method_name.str.extract(r"^(clustered|dispersed)_"); d["tag"]=d.method_name.str.extract(r"_(Sync|Stag)_")
        d=d.dropna(subset=["sp","tag"]); rows=[]
        for sp in SPAWNS:
            S=d[(d.sp==sp)&(d.tag=="Sync")].set_index(["map_id","seed"]); G=d[(d.sp==sp)&(d.tag=="Stag")].set_index(["map_id","seed"])
            idx=S.index.intersection(G.index)
            if len(idx)<8: rows.append(dict(spawn=sp,n=len(idx),note="few")); continue
            for met in ["total_travel_distance","overlap_ratio"]:
                a=G.loc[idx,met].values; c=S.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(spawn=sp,metric=met.split("_")[0],sync=round(float(c.mean()),2),stag=round(float(a.mean()),2),
                    desync_pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),5)))
        import pandas as pd2; o=pd2.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_corr_paired.csv"),index=False)
        log("de-sync benefit by initial correlation (expect clustered > dispersed):\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log(f"an failed {e}\n{traceback.format_exc()[-300:]}")
if __name__=="__main__": sys.exit(main())
