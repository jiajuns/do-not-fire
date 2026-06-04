"""OW-Desync on the FROZEN EXPLORER (range-limited interference channel). Conditions: Sync (periodic),
StagEven (uniform even-staggering), OWDesync (overlap-weighted conflict-free activation). All under the
SAME range-limited channel (send_msg delivers only within R_COMM); only the schedule differs. Sweep
R_COMM {0=full-mesh control, 15, 30, 60} x heterogeneous sensing (10,30) x 12 seeds, N=16 T=8 clustered.
Primary = does-no-harm on coverage/travel/overlap (peak-AoI win is proven standalone). Group setcomm_owtb."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_owtb"; NUM=20; N=16; IV=8; HET="10,30"
SEEDS=list(range(int(os.environ.get("OT_SEEDS","12")))); WORKERS=int(os.environ.get("OT_WORKERS","12"))
RCOMMS=[0,15,30,60]
CONDS=[("Sync","periodic",""),("StagEven","staggered","even"),("OWDesync","owdesync","")]
def mname(c,R): return f"{c}_R{R}"
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv")
    if not os.path.exists(p): return False
    try: return sum(1 for _ in open(p))>=NUM+1
    except: return False
def one(j):
    c,mode,pat,R,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]="clustered"
    env["BLMRE_HETERO_SRANGE"]=HET; env["BLMRE_R_COMM"]=str(R)
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    m=mname(c,R)
    cmd=["timeout","1800",PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),
         "--num-maps",str(NUM),"--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","owtb.log"),"a").write(m+"\n")
def main():
    jobs=[(c,mode,pat,R,s) for (c,mode,pat) in CONDS for R in RCOMMS for s in SEEDS if not ok(mname(c,R),s)]
    t0=time.time(); log(f"[{time.strftime('%H:%M')}] owtb: {len(jobs)} jobs (3 conds x {len(RCOMMS)} R_COMM x {len(SEEDS)} seeds) N={N} het={HET} workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","OWTB_DONE"),"w").close(); log(f"[{time.strftime('%H:%M')}] OWTB DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv") if "coverage" not in f],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","final_coverage_ratio","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        ext=d["method_name"].str.extract(r"^(Sync|StagEven|OWDesync)_R(\d+)")
        d["cond"]=ext[0]; d["R"]=pd.to_numeric(ext[1],errors="coerce"); d=d.dropna(subset=["cond"]); rows=[]
        def pair(sub,A,B,R,lab):
            X=sub[sub["cond"]==A].set_index(["map_id","seed"]); Y=sub[sub["cond"]==B].set_index(["map_id","seed"])
            X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio","final_coverage_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(R=int(R),cmp=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),4)))
        for R in RCOMMS:
            sub=d[d["R"]==R]
            pair(sub,"OWDesync","StagEven",R,"ow_vs_even"); pair(sub,"OWDesync","Sync",R,"ow_vs_sync"); pair(sub,"StagEven","Sync",R,"even_vs_sync")
        o=pd.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_owtb_paired.csv"),index=False)
        log("=== OWTB paired (travel/overlap neg=better; coverage pos=better; ow_vs_even = does-no-harm check) ===\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log("an failed: "+str(e)+"\n"+traceback.format_exc()[-400:])
if __name__=="__main__": sys.exit(main())
