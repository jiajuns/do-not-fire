"""Packet-loss robustness of de-synchronisation: does the free de-sync benefit survive a lossy
channel? Synchronised vs staggered periodic (N=8, interval 12, broadcast model) under per-message
drop probabilities {0,0.1,0.2,0.3}. Group setcomm_ploss; paired Wilcoxon stag vs sync per drop."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_ploss"
DROPS=[float(x) for x in os.environ.get("PL_DROPS","0,0.1,0.2,0.3").split(",")]
SEEDS=list(range(int(os.environ.get("PL_SEEDS","6")))); IV=int(os.environ.get("PL_IV","12")); N=8; NUM=20
WORKERS=int(os.environ.get("PL_WORKERS","8")); BUD=int(os.environ.get("PL_BUDGET_MIN","180"))*60
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv"); return os.path.exists(p) and os.path.getsize(p)>100
def one(j):
    m,mode,dr,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"
    cmd=[PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),"--num-maps",str(NUM),
         "--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV),"--comm-drop",str(dr)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","ploss.log"),"a").write(m+"\n")
def main():
    jobs=[]
    for dr in DROPS:
        dt=str(dr).replace(".","p")
        for mode,tag in [("periodic","Sync"),("staggered","Stag")]:
            for s in SEEDS:
                m=f"d{dt}_{tag}_{IV}"
                if not ok(m,s): jobs.append((m,mode,dr,s))
    t0=time.time(); log(f"ploss: drops {DROPS} x sync/stag x {len(SEEDS)} seeds = {len(jobs)} jobs")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","PLOSS_DONE"),"w").close(); log(f"PLOSS DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        d["dr"]=d.method_name.str.extract(r"^d([0-9p]+)_")[0].str.replace("p",".",regex=False).astype(float)
        d["tag"]=d.method_name.str.extract(r"_(Sync|Stag)_"); d=d.dropna(subset=["dr","tag"]); rows=[]
        for dr in DROPS:
            S=d[(d.dr==dr)&(d.tag=="Sync")].set_index(["map_id","seed"]); G=d[(d.dr==dr)&(d.tag=="Stag")].set_index(["map_id","seed"])
            idx=S.index.intersection(G.index)
            if len(idx)<8: rows.append(dict(drop=dr,n=len(idx),note="few")); continue
            for met in ["total_travel_distance","overlap_ratio"]:
                a=G.loc[idx,met].values; c=S.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(drop=dr,metric=met.split("_")[0],sync=round(float(c.mean()),2),stag=round(float(a.mean()),2),
                    desync_pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),5)))
        import pandas as pd2; o=pd2.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_ploss_paired.csv"),index=False)
        log("de-sync under packet loss (neg=de-sync better):\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log(f"an failed {e}\n{traceback.format_exc()[-300:]}")
if __name__=="__main__": sys.exit(main())
