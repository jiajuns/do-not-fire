"""~10h EXTREME-heterogeneity x high-loss sweep: does pushing value asymmetry far past 3x (so the
Tripathi-Modiano k>1 condition is firmly met) + high packet loss finally make the value/loss-aware
Whittle (vlw) beat even-staggering on travel/overlap? Either finds a winning corner (the positive
result) or makes the near-optimality null bulletproof. conds {StagEven,CWD,VLW} x het {10,40 (16x area),
5,40 (64x)} x loss {0.5,0.7} x N {16,20}, T=8, clustered, seeds 0-11. 30-min cap, resume-safe,
auto-aggregate + paired Wilcoxon. Group setcomm_xhet."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_xhet"; NUM=20; IV=8; SPAWN="clustered"
SEEDS=list(range(int(os.environ.get("XH_SEEDS","12")))); WORKERS=int(os.environ.get("XH_WORKERS","5"))
HETS=[("h1040","10,40"),("h540","5,40")]            # 16x and 64x sensor-area asymmetry
LOSSES=[("L50","0.5"),("L70","0.7")]                # high packet loss
NS=[16,20]
CONDS=[("StagEven","staggered","even"),("CWD","cwd",""),("VLW","vlw","")]
def mname(ht,lt,N,cond): return f"{ht}_{lt}_N{N}_{cond}"
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv")
    if not os.path.exists(p): return False
    try: return sum(1 for _ in open(p))>=NUM+1
    except: return False
def build():
    jobs=[]
    for ht,hv in HETS:
        for lt,lv in LOSSES:
            for N in NS:
                for cond,mode,pat in CONDS:
                    for s in SEEDS:
                        m=mname(ht,lt,N,cond)
                        if not ok(m,s): jobs.append((ht,hv,lt,lv,N,cond,mode,pat,s))
    return jobs
def one(j):
    ht,hv,lt,lv,N,cond,mode,pat,s=j; env=dict(os.environ)
    env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]=SPAWN; env["BLMRE_HETERO_SRANGE"]=hv
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    m=mname(ht,lt,N,cond)
    cmd=["timeout","1800",PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),
         "--num-maps",str(NUM),"--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),
         "--interval",str(IV),"--comm-drop",lv]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","xhet.log"),"a").write(m+"\n")
def main():
    jobs=build(); t0=time.time()
    log(f"[{time.strftime('%H:%M')}] xhet: {len(jobs)} jobs (conds3 x het{len(HETS)} x loss{len(LOSSES)} x N{len(NS)} x {len(SEEDS)} seeds) workers={WORKERS}")
    done=[0]
    def wrap(j):
        r=one(j); done[0]+=1
        if done[0]%20==0: log(f"[{time.strftime('%H:%M')}] {done[0]}/{len(jobs)} ({int((time.time()-t0)/60)}min)")
        return r
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(wrap,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","XHET_DONE"),"w").close(); log(f"[{time.strftime('%H:%M')}] XHET DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv") if "coverage" not in f],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","total_communication_volume","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        ext=d["method_name"].str.extract(r"^(h1040|h540)_(L50|L70)_N(\d+)_(StagEven|CWD|VLW)")
        d["het"]=ext[0]; d["loss"]=ext[1]; d["N"]=pd.to_numeric(ext[2],errors="coerce"); d["cond"]=ext[3]
        d=d.dropna(subset=["cond"]); rows=[]
        def pair(sub,A,B,tags,lab):
            X=sub[sub["cond"]==A].set_index(["map_id","seed"]); Y=sub[sub["cond"]==B].set_index(["map_id","seed"])
            X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(**tags,cmp=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),4)))
        for ht,_ in HETS:
            for lt,_ in LOSSES:
                for N in NS:
                    sub=d[(d["het"]==ht)&(d["loss"]==lt)&(d["N"]==N)]
                    pair(sub,"VLW","StagEven",dict(het=ht,loss=lt,N=N),"vlw_vs_even")
                    pair(sub,"VLW","CWD",dict(het=ht,loss=lt,N=N),"vlw_vs_cwd")
        o=pd.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_xhet_paired.csv"),index=False)
        log("=== XHET paired (neg pct = vlw better; looking for a SIGNIFICANT vlw_vs_even win) ===\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log("an failed: "+str(e)+"\n"+traceback.format_exc()[-400:])
if __name__=="__main__": sys.exit(main())
