"""Scaled-up gradient + team-size study for the joint-submodular scheduler.
CORE: {StagEven, CWD, Submod} x {clustered, default, dispersed} x 16 seeds @ N=16 T=8
      -> does submod's overlap advantage over even-staggering grow MONOTONICALLY with
         inter-robot spatial overlap (clustered>default>dispersed)? Firms up the marginal
         p=0.015 single point into a structural gradient. Also submod_vs_cwd per overlap level.
SCALE: {StagEven, Submod} x {clustered, dispersed} x {N=12, N=20} x 10 seeds @ T=8
      -> does the coupling advantage scale with team size?
Each job capped at 30 min wall (timeout) so a pathological map drops that (map,seed) cell
symmetrically instead of hanging for hours. Group setcomm_grad."""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); GROUP="setcomm_grad"; NUM=20; IV=8
WORKERS=int(os.environ.get("GR_WORKERS","12")); CAP=os.environ.get("GR_CAP","1800")
CORE_SEEDS=list(range(int(os.environ.get("GR_CORE_SEEDS","16"))))
SCALE_SEEDS=list(range(int(os.environ.get("GR_SCALE_SEEDS","10"))))
# (cond, mode, pattern)
CORE_CONDS=[("StagEven","staggered","even"),("CWD","cwd",""),("Submod","submod","")]
SCALE_CONDS=[("StagEven","staggered","even"),("Submod","submod","")]
CORE_SPAWNS=["clustered","default","dispersed"]; SCALE_SPAWNS=["clustered","dispersed"]
def mname(spawn,nm,N): return f"{spawn}_{nm}_N{N}_{IV}"
def ok(m,s):
    p=os.path.join(HERE,"results","raw",GROUP,f"{GROUP}_{m}_seed{s}.csv"); return os.path.exists(p) and sum(1 for _ in open(p))>=NUM+1
def build():
    jobs=[]
    for nm,mode,pat in CORE_CONDS:
        for sp in CORE_SPAWNS:
            for s in CORE_SEEDS: jobs.append((mname(sp,nm,16),mode,pat,sp,16,s))
    for nm,mode,pat in SCALE_CONDS:
        for sp in SCALE_SPAWNS:
            for N in (12,20):
                for s in SCALE_SEEDS: jobs.append((mname(sp,nm,N),mode,pat,sp,N,s))
    return [j for j in jobs if not ok(j[0],j[5])]
def one(j):
    m,mode,pat,sp,N,s=j; env=dict(os.environ); env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]=sp
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    cmd=["timeout",CAP,PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),
         "--num-maps",str(NUM),"--group",GROUP,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(IV)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","gradient.log"),"a").write(m+"\n")
def main():
    jobs=build(); t0=time.time()
    log(f"gradient: {len(jobs)} jobs ({len(CORE_SEEDS)} core seeds x3 conds x3 spawns @N16 + {len(SCALE_SEEDS)} scale seeds x2 conds x2 spawns x{{N12,N20}}), workers={WORKERS} cap={CAP}s")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(one,jobs))
    subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{GROUP}","--group",GROUP,"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","GRAD_DONE"),"w").close(); log(f"GRAD DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import re, pandas as pd; from scipy import stats
        d=pd.concat([pd.read_csv(f) for f in glob.glob(f"results/raw/{GROUP}/*.csv")],ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","total_communication_volume","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        ext=d.method_name.str.extract(r"^(clustered|default|dispersed)_(StagEven|CWD|Submod)_N(\d+)_")
        d["spawn"]=ext[0]; d["cond"]=ext[1]; d["N"]=pd.to_numeric(ext[2],errors="coerce"); d=d.dropna(subset=["cond"]); rows=[]
        def pair(sub,A,B,N,sp,lab):
            X=sub[sub.cond==A].set_index(["map_id","seed"]); Y=sub[sub.cond==B].set_index(["map_id","seed"])
            idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(N=int(N),spawn=sp,comparison=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),5)))
        for sp in CORE_SPAWNS:
            sub=d[(d.N==16)&(d.spawn==sp)]; pair(sub,"Submod","StagEven",16,sp,"submod_vs_even"); pair(sub,"Submod","CWD",16,sp,"submod_vs_cwd")
        for N in (12,20):
            for sp in SCALE_SPAWNS:
                sub=d[(d.N==N)&(d.spawn==sp)]; pair(sub,"Submod","StagEven",N,sp,"submod_vs_even")
        o=pd.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_grad_paired.csv"),index=False)
        vol=d.groupby(["N","spawn","cond"])["total_communication_volume"].mean().round(0)
        log("GRADIENT (submod_vs_even overlap should grow more negative clustered->dispersed; neg=submod better):\n"+o.to_string(index=False)+"\n-- comm volume (N,spawn,cond) --\n"+vol.to_string())
    except Exception as e:
        import traceback; log(f"an failed {e}\n{traceback.format_exc()[-400:]}")
if __name__=="__main__": sys.exit(main())
