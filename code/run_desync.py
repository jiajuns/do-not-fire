"""Decentralised de-synchronisation under clock drift: does self-organising DESYNC maintain the
even-staggering benefit where a fixed centralised schedule (StagFix) degrades under drift?
Baselines: Sync (periodic, floor) | StagIdeal (centralised even-staggering, NO drift = oracle upper
bound) | StagFix (centralised even offsets + per-robot clock drift, no re-adaptation) | DESYNC
(self-organising midpoint rule + same drift). Sweep drift sigma in {0,0.05,0.10,0.20}, N=16 T=8
clustered. PREDICT: as drift grows, StagFix degrades toward Sync while DESYNC tracks StagIdeal, so
DESYNC << StagFix (the positive result) and DESYNC ~= StagIdeal. Group setcomm_desync."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_desync"; NUM=20; N=16; IV=8; SPAWN="clustered"
SEEDS=list(range(int(os.environ.get("DS_SEEDS","12")))); WORKERS=int(os.environ.get("DS_WORKERS","5"))
DRIFTS=[0.0,0.05,0.10,0.20]
def dtag(d): return f"d{int(round(d*100)):02d}"
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv")
    if not os.path.exists(p): return False
    try: return sum(1 for _ in open(p))>=NUM+1
    except: return False
def jobs_spec():
    # (method_name, comm_mode, stagger_pattern, drift)
    js=[("Sync","periodic","",0.0),("StagIdeal","staggered","even",0.0)]  # drift-independent refs
    for d in DRIFTS:
        js.append((f"StagFix_{dtag(d)}","stagfix","",d))
        js.append((f"DESYNC_{dtag(d)}","desync","",d))
    return js
def one(j):
    m,mode,pat,d,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]=SPAWN
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    env["BLMRE_CLOCK_DRIFT"]=str(d)
    cmd=["timeout","1800",PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),
         "--num-maps",str(NUM),"--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","desync.log"),"a").write(m+"\n")
def main():
    spec=jobs_spec(); jobs=[(m,mode,pat,d,s) for (m,mode,pat,d) in spec for s in SEEDS if not ok(m,s)]
    t0=time.time(); log(f"[{time.strftime('%H:%M')}] desync: {len(jobs)} jobs ({len(spec)} configs x {len(SEEDS)} seeds) N={N} T={IV} drifts={DRIFTS} workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","DESYNC_DONE"),"w").close(); log(f"[{time.strftime('%H:%M')}] DESYNC DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv") if "coverage" not in f],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","total_communication_volume","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["m"]=d["method_name"]; rows=[]
        def pick(name): return d[d["m"]==name]
        def pair(A,B,tags,lab):
            X=pick(A).set_index(["map_id","seed"]); Y=pick(B).set_index(["map_id","seed"])
            X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(**tags,cmp=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),4)))
        pair("Sync","StagIdeal",dict(drift="-"),"sync_vs_ideal")  # Pillar-A reference gap
        for dd in [0.05,0.10,0.20]:
            t=dtag(dd)
            pair(f"DESYNC_{t}",f"StagFix_{t}",dict(drift=dd),"desync_vs_stagfix")  # THE win
            pair(f"StagFix_{t}","StagIdeal",dict(drift=dd),"stagfix_vs_ideal")      # StagFix degrades?
            pair(f"DESYNC_{t}","StagIdeal",dict(drift=dd),"desync_vs_ideal")        # DESYNC tracks oracle?
        o=pd.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_desync_paired.csv"),index=False)
        log("=== DESYNC paired (neg=first better). WIN = desync_vs_stagfix negative & growing with drift; stagfix_vs_ideal positive & growing; desync_vs_ideal ~0 ===\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log("an failed: "+str(e)+"\n"+traceback.format_exc()[-400:])
if __name__=="__main__": sys.exit(main())
