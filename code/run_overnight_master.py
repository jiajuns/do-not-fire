"""~10h overnight master: maps WHERE and HOW MUCH the joint-submodular scheduler beats
even-staggering, across the broadcast-rate k=N/T and the team-overlap axes. One 16-worker
pool, every job wall-capped at 30 min (pathological maps drop that (map,seed) cell instead of
hanging), resumes any already-collected CSV by row-count, auto-aggregates + paired Wilcoxon at
the end. Uniform method name = {spawn}_{cond}_N{N}_{T} so it resumes run_gradient output too.

Studies (each its own results group):
  grad    : {even,cwd,submod} x {clustered,default,dispersed} x N16 x T8 x 16 seeds  -> overlap gradient
  ksweep  : {even,submod}      x clustered x N16 x T{4,6,8,12,16} x 24 seeds          -> gap vs k=16/T
  nsweep  : {even,submod}      x clustered x N{4,8,12,16,20,24} x T8 x 20 seeds        -> gap vs N (k=N/8)
  power   : {even,cwd,submod}  x clustered x N16 x T8 x 64 seeds                       -> tight p on the key pair
"""
import os, sys, time, glob, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); PY=sys.executable
MAPS=os.path.join(ROOT,"maps_test"); NUM=20
WORKERS=int(os.environ.get("OM_WORKERS","16")); CAP=os.environ.get("OM_CAP","1800")
MODE={"StagEven":("staggered","even"),"CWD":("cwd",""),"Submod":("submod",""),"Sync":("periodic","")}
def S(n): return list(range(n))
STUDIES=[
 dict(group="setcomm_grad",  conds=["StagEven","CWD","Submod"], spawns=["clustered","default","dispersed"], Ns=[16], Ts=[8], seeds=S(16)),
 dict(group="setcomm_ksweep",conds=["StagEven","Submod"],        spawns=["clustered"], Ns=[16], Ts=[4,6,8,12,16], seeds=S(24)),
 dict(group="setcomm_nsweep",conds=["StagEven","Submod"],        spawns=["clustered"], Ns=[4,8,12,16,20,24], Ts=[8], seeds=S(20)),
 dict(group="setcomm_power", conds=["StagEven","CWD","Submod"],  spawns=["clustered"], Ns=[16], Ts=[8], seeds=S(64)),
]
def name(sp,c,N,T): return f"{sp}_{c}_N{N}_{T}"
def ok(group,m,s):
    p=os.path.join(HERE,"results","raw",group,f"{group}_{m}_seed{s}.csv")
    if not os.path.exists(p): return False
    try: return sum(1 for _ in open(p))>=NUM+1
    except: return False
def build():
    jobs=[]
    for st in STUDIES:
        for c in st["conds"]:
            mode,pat=MODE[c]
            for sp in st["spawns"]:
                for N in st["Ns"]:
                    for T in st["Ts"]:
                        for s in st["seeds"]:
                            m=name(sp,c,N,T)
                            if not ok(st["group"],m,s): jobs.append((st["group"],m,mode,pat,sp,N,T,s))
    return jobs
def one(j):
    group,m,mode,pat,sp,N,T,s=j; env=dict(os.environ)
    env["BLMRE_GATE_POSITIONS"]="1"; env["BLMRE_SPAWN_CLUSTER"]=sp
    if pat: env["BLMRE_STAGGER_PATTERN"]=pat
    cmd=["timeout",CAP,PY,"run_eval_one.py","--method-name",m,"--comm-mode",mode,"--seed",str(s),
         "--num-maps",str(NUM),"--group",group,"--map-dir",MAPS,"--device","cpu","--n-agents",str(N),"--interval",str(T)]
    return subprocess.run(cmd,cwd=HERE,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
def log(m):
    print(m,flush=True); os.makedirs(os.path.join(HERE,"results","overnight"),exist_ok=True)
    open(os.path.join(HERE,"results","overnight","master.log"),"a").write(m+"\n")
def main():
    jobs=build(); t0=time.time()
    log(f"[{time.strftime('%H:%M')}] master: {len(jobs)} jobs queued, workers={WORKERS}, cap={CAP}s")
    done=[0]
    def wrap(j):
        r=one(j); done[0]+=1
        if done[0]%25==0: log(f"[{time.strftime('%H:%M')}] {done[0]}/{len(jobs)} done ({int((time.time()-t0)/60)}min)")
        return r
    with ThreadPoolExecutor(max_workers=WORKERS) as ex: list(ex.map(wrap,jobs))
    for st in STUDIES:
        subprocess.run([PY,"scripts/aggregate_results.py","--input_dir",f"results/raw/{st['group']}","--group",st["group"],"--output_dir","results/summary"])
    _an(); open(os.path.join(HERE,"results","overnight","MASTER_DONE"),"w").close()
    log(f"[{time.strftime('%H:%M')}] MASTER DONE {int((time.time()-t0)/60)}min")
def _an():
    try:
        import re, pandas as pd; from scipy import stats
        groups=["setcomm_grad","setcomm_ksweep","setcomm_nsweep","setcomm_power"]
        frames=[]
        for g in groups:
            for f in glob.glob(f"results/raw/{g}/*.csv"):
                df=pd.read_csv(f); df["grp"]=g; frames.append(df)
        d=pd.concat(frames,ignore_index=True)
        for c in ["total_travel_distance","overlap_ratio","total_communication_volume","map_id","seed"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        ext=d.method_name.str.extract(r"^(clustered|default|dispersed)_(StagEven|CWD|Submod)_N(\d+)_(\d+)$")
        d["spawn"]=ext[0]; d["cond"]=ext[1]; d["N"]=pd.to_numeric(ext[2],errors="coerce"); d["T"]=pd.to_numeric(ext[3],errors="coerce")
        d=d.dropna(subset=["cond"]); rows=[]
        def pair(sub,A,B,tags,lab):
            X=sub[sub.cond==A].set_index(["map_id","seed"]); Y=sub[sub.cond==B].set_index(["map_id","seed"])
            idx=X.index.intersection(Y.index)
            if len(idx)<8: return
            for met in ["total_travel_distance","overlap_ratio"]:
                a=X.loc[idx,met].values; c=Y.loc[idx,met].values
                try: _,p=stats.wilcoxon(a,c)
                except: p=float("nan")
                rows.append(dict(**tags,comparison=lab,metric=met.split("_")[0],pct=round(float(100*(a.mean()-c.mean())/c.mean()),2),n=len(idx),p=round(float(p),5)))
        # ksweep: gap vs k=16/T (group has both even & submod, clustered N16)
        for T in [4,6,8,12,16]:
            sub=d[(d["grp"]=="setcomm_ksweep")&(d.N==16)&(d.T==T)&(d.spawn=="clustered")]
            pair(sub,"Submod","StagEven",dict(study="ksweep",k=round(16/T,2),N=16,T=T,spawn="clustered"),"submod_vs_even")
        # nsweep: gap vs N (k=N/8), clustered T8
        for N in [4,8,12,16,20,24]:
            sub=d[(d["grp"]=="setcomm_nsweep")&(d.N==N)&(d.T==8)&(d.spawn=="clustered")]
            pair(sub,"Submod","StagEven",dict(study="nsweep",k=round(N/8,2),N=N,T=8,spawn="clustered"),"submod_vs_even")
        # grad: overlap gradient (N16 T8), submod vs even + vs cwd
        for sp in ["clustered","default","dispersed"]:
            sub=d[(d["grp"]=="setcomm_grad")&(d.N==16)&(d.T==8)&(d.spawn==sp)]
            pair(sub,"Submod","StagEven",dict(study="grad",k=2.0,N=16,T=8,spawn=sp),"submod_vs_even")
            pair(sub,"Submod","CWD",dict(study="grad",k=2.0,N=16,T=8,spawn=sp),"submod_vs_cwd")
        # power: tight p (clustered N16 T8, up to 64 seeds)
        sub=d[(d["grp"]=="setcomm_power")&(d.N==16)&(d.T==8)&(d.spawn=="clustered")]
        pair(sub,"Submod","StagEven",dict(study="power",k=2.0,N=16,T=8,spawn="clustered"),"submod_vs_even")
        pair(sub,"Submod","CWD",dict(study="power",k=2.0,N=16,T=8,spawn="clustered"),"submod_vs_cwd")
        o=pd.DataFrame(rows); o.to_csv(os.path.join(HERE,"results","summary","setcomm_master_paired.csv"),index=False)
        log("=== MASTER paired (neg pct = submod better; overlap is the coupling-sensitive metric) ===\n"+o.to_string(index=False))
    except Exception as e:
        import traceback; log("an failed: "+str(e)+"\n"+traceback.format_exc()[-500:])
if __name__=="__main__": sys.exit(main())
