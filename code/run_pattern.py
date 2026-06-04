"""Empirical validation of Proposition 3 (evenly-spaced staggering is bound-optimal). At N=8,
interval 12, broadcast model, we compare three matched-budget periodic schedules that differ ONLY
in broadcast phase pattern: synchronised (all phase 0), HALF-split staggered (two phase clusters at
0 and T/2, a suboptimal de-sync), and EVEN staggered (phi_k=round(kT/N), the bound-optimal schedule).
Proposition 3 predicts the de-sync benefit ordering sync (worst) < half < even (best). Group
setcomm_pattern; paired Wilcoxon vs the synchronised control."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_pattern"
SEEDS=list(range(int(os.environ.get("PT_SEEDS","8")))); IV=int(os.environ.get("PT_IV","12")); N=8; NUM=20
WORKERS=int(os.environ.get("PT_WORKERS","8"))
# (method_name, comm-mode, stagger_pattern)
CONDS=[("Sync","periodic",""),("StagHalf","staggered","half"),("StagEven","staggered","even")]
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv"); return os.path.exists(p) and os.path.getsize(p)>100
def one(j):
    m,mode,pat,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    cmd=[PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),"--num-maps",str(NUM),
         "--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","pattern.log"),"a").write(m+"\n")
def main():
    jobs=[]
    for nm,mode,pat in CONDS:
        m=f"{nm}_{IV}"
        for s in SEEDS:
            if not ok(m,s): jobs.append((m,mode,pat,s))
    t0=time.time(); log(f"pattern: sync/half/even x {len(SEEDS)} seeds @ iv{IV} = {len(jobs)} jobs")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","PATTERN_DONE"),"w").close(); log(f"PATTERN DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["cond"]=d.method_name.str.extract(r"^(Sync|StagHalf|StagEven)_"); d=d.dropna(subset=["cond"]); rows=[]
        S=d[d.cond=="Sync"].set_index(["map_id","seed"])
        for cond in ["StagHalf","StagEven"]:
            G=d[d.cond==cond].set_index(["map_id","seed"]); idx=S.index.intersection(G.index)
            if len(idx)<8: continue
            for met in ["total_travel_distance","overlap_ratio"]:
                a=G.loc[idx,met].values; c=S.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(vs_sync=cond,metric=met.split("_")[0],sync=round(float(c.mean()),2),this=round(float(a.mean()),2),
                    desync_pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),5)))
        import pandas as pd2; o=pd2.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_pattern_paired.csv"),index=False)
        log("Prop 3 check: de-sync benefit by phase pattern (neg=better; expect even>half>sync):\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log(f"an failed {e}\n{traceback.format_exc()[-300:]}")
if __name__=="__main__": sys.exit(main())
