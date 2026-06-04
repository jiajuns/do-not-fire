"""Value- and loss-aware Whittle (vlw) vs even-staggering, across the 2x2 regime the theory predicts:
heterogeneous value (mixed sensor ranges) x lossy channel. Tripathi-Modiano (TNET 2024): with
heterogeneous cost the Whittle-optimal cycle is asymmetric (round-robin strictly suboptimal) once
the value asymmetry is large; Ayik-Ceran-Uysal (2023): under loss the closed-form age/AoII Whittle
beats round-robin. PREDICT: VLW beats StagEven significantly in HET (more under LOSS); ties in
HOM+reliable (graceful degradation = sanity check that Pillar-A baseline is preserved). CWD
(value-BLIND Whittle) is the control: shows VALUE is the active ingredient. Group setcomm_vlw."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_vlw"; NUM=20; N=16; IV=8
SEEDS=list(range(int(os.environ.get("VLW_SEEDS","12")))); WORKERS=int(os.environ.get("VLW_WORKERS","14"))
DROP=os.environ.get("VLW_DROP","0.5"); HET=os.environ.get("VLW_HET","10,30"); SPAWN=os.environ.get("VLW_SPAWN","clustered")
CONDS=[("StagEven","staggered","even"),("CWD","cwd",""),("VLW","vlw","")]
# regime: (tag, hetero?, loss?)
REGIMES=[("HOMrel",False,False),("HOMloss",False,True),("HETrel",True,False),("HETloss",True,True)]
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv")
    if not os.path.exists(p): return False
    try: return sum(1 for _ in open(p))>=NUM+1
    except: return False
def one(j):
    reg,het,loss,nm,mode,pat,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]=SPAWN
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    if het: env["BLMRE_HETERO_SRANGE"]=HET
    else: env.pop("BLMRE_HETERO_SRANGE",None)
    m=f"{reg}_{nm}_{IV}"
    cmd=["timeout","1800",PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),
         "--num-maps",str(NUM),"--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    if loss: cmd+=["--comm-drop",DROP]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","vlw.log"),"a").write(m+"\n")
def main():
    jobs=[]
    for reg,het,loss in REGIMES:
        for nm,mode,pat in CONDS:
            for s in SEEDS:
                if not ok(f"{reg}_{nm}_{IV}",s): jobs.append((reg,het,loss,nm,mode,pat,s))
    t0=time.time(); log(f"[{time.strftime('%H:%M')}] vlw: {len(jobs)} jobs (4 regimes x3 conds x{len(SEEDS)} seeds) N={N} T={IV} drop={DROP} het={HET} spawn={SPAWN} workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","VLW_DONE"),"w").close(); log(f"[{time.strftime('%H:%M')}] VLW DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","total_communication_volume","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        ext=d["method_name"].str.extract(r"^(HOMrel|HOMloss|HETrel|HETloss)_(StagEven|CWD|VLW)_")
        d["reg"]=ext[0]; d["cond"]=ext[1]; d=d.dropna(subset=["cond"]); rows=[]
        def pair(sub,A,B,reg,lab):
            X=sub[sub["cond"]==A].set_index(["map_id","seed"]); Y=sub[sub["cond"]==B].set_index(["map_id","seed"])
            X=X[~X.index.duplicated()]; Y=Y[~Y.index.duplicated()]; idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(regime=reg,cmp=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),4)))
        for reg,_,_ in REGIMES:
            sub=d[d["reg"]==reg]
            pair(sub,"VLW","StagEven",reg,"vlw_vs_even"); pair(sub,"CWD","StagEven",reg,"cwd_vs_even"); pair(sub,"VLW","CWD",reg,"vlw_vs_cwd")
        o=pd.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_vlw_paired.csv"),index=False)
        vol=d.groupby(["reg","cond"])["total_communication_volume"].mean().round(0)
        log("=== VLW paired (neg pct = vlw better; expect win in HET, esp HETloss; tie in HOMrel) ===\n"+o.to_string(index=False)+"\n-- comm volume (reg,cond) --\n"+vol.to_string())
    except Exception as e:
        import traceback; log("an failed: "+str(e)+"\n"+traceback.format_exc()[-400:])
if __name__=="__main__": sys.exit(main())
