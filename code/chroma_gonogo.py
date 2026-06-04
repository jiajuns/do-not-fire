"""CHROMA-DeSync GO/NO-GO v2: validate BOTH the centralized core (age-MWIS, prior-art-ish) AND the
genuinely-novel DECENTRALIZED piece (clique-restricted DESYNC: each robot fires on its own phase, and
nudges its phase toward the midpoint of its CONFLICTING neighbours only -> conflict cliques self-
organize to collision-free even spacing, no central solver, no global N). Metric: network peak-AoI at
matched volume on a range-limited interference graph. Gates: full-mesh ties; homogeneous ties; clustered
reuse -> CHROMA wins. cdesync (decentralized) must track mwis (centralized) to earn the novelty."""
import os, numpy as np
from scipy import stats

def greedy_color(conf):
    N=len(conf); color=-np.ones(N,int)
    for i in np.argsort(-conf.sum(1)):
        used={color[j] for j in range(N) if conf[i,j] and color[j]>=0}
        c=0
        while c in used: c+=1
        color[i]=c
    return color

def topk_is(conf,w,K):
    chosen=[]; avail=np.ones(len(w),bool)
    for i in np.argsort(-w):
        if not avail[i]: continue
        chosen.append(i); avail[i]=False; avail[conf[i]]=False
        if len(chosen)>=K: break
    return chosen

def run(N,R,layout,loss,sched,T,rng):
    if layout=="clustered":
        nc=max(2,N//5); ctr=rng.uniform(0,1,(nc,2))
        pos=np.clip(ctr[rng.integers(0,nc,N)]+rng.normal(0,0.06,(N,2)),0,1)
    else:
        pos=rng.uniform(0,1,(N,2))
    D=np.sqrt(((pos[:,None,:]-pos[None,:,:])**2).sum(-1))
    comm=(D<R)&~np.eye(N,dtype=bool); conf=(D<R)&~np.eye(N,dtype=bool)
    li,lr=np.where(comm)
    if len(li)==0: return None
    aoi=np.ones(len(li))
    color=greedy_color(conf); chi=int(color.max()+1); K=max(1,round(N/max(1,chi)))
    a=np.ones(N)
    P=float(max(2,chi)); theta=rng.uniform(0,P,N); last=-np.ones(N)   # cdesync state
    peak=[]; mean=[]; vol=0
    for t in range(T):
        if sched=="rr": S=[i for i in range(N) if color[i]==(t%chi)]
        elif sched=="mwis": S=topk_is(conf,a.copy(),K)
        elif sched=="cdesync":
            theta+=1.0; S=[i for i in range(N) if theta[i]>=P]
            for i in S:
                theta[i]-=P
                nb=np.where(conf[i]&(last>=0))[0]
                if len(nb)>=1:
                    ph=((last[nb]-t)%P)            # conflicting neighbours' phase positions in [0,P)
                    ph=np.sort(ph)
                    if len(ph)>=2:
                        prevp=ph[ph<=0.0]; nextp=ph[ph>0.0]
                        lo=ph.max()-P; hi=ph.min()  # wrap nearest-before / nearest-after around 0
                        mid=(lo+hi)/2.0
                        theta[i]=float(np.clip(theta[i]-0.5*mid,-P,P))
                last[i]=t
        else: S=[]
        Sset=set(S); vol+=len(S)
        for k,(i,r) in enumerate(zip(li,lr)):
            if i in Sset:
                collide=any((j!=i) and comm[j,r] for j in S)
                aoi[k]=1.0 if (not collide and rng.random()>=loss) else aoi[k]+1.0
            else: aoi[k]+=1.0
        for i in range(N):
            a[i]=aoi[li==i].sum() if comm[i].any() else 1.0
        peak.append(aoi.max()); mean.append(aoi.mean())
    return dict(peak=float(np.mean(peak)),mean=float(np.mean(mean)),chi=chi,vol=vol)

def main():
    SEEDS=range(int(os.environ.get("CH_SEEDS","80"))); T=160
    cfgs=[("FULLMESH(G1)",2.0,"clustered",0.0),("HOMO_unif(G2)",0.45,"uniform",0.0),
          ("REUSE_clustered",0.30,"clustered",0.0),("REUSE_clustered_loss",0.30,"clustered",0.3),
          ("HIREUSE_clustered",0.22,"clustered",0.0)]
    for N in (16,24):
        print(f"\n===== N={N}: peak-AoI vs reuse-aware-RR (neg=better). mwis=centralized core, cdesync=DECENTRALIZED novel =====")
        print(f"{'regime':22s} {'chi':>3s} {'rrPeak':>7s} {'mwis d%':>8s} {'p':>6s} {'cdesync d%':>10s} {'p':>6s} {'cdMean%':>7s}")
        for name,R,lay,loss in cfgs:
            rp=[];mp=[];cp=[];cm=[];rm=[]
            for s in SEEDS:
                rr=run(N,R,lay,loss,"rr",T,np.random.default_rng(7000+s))
                mw=run(N,R,lay,loss,"mwis",T,np.random.default_rng(7000+s))
                cd=run(N,R,lay,loss,"cdesync",T,np.random.default_rng(7000+s))
                if rr and mw and cd:
                    rp.append(rr["peak"]);mp.append(mw["peak"]);cp.append(cd["peak"]);cm.append(cd["mean"]);rm.append(rr["mean"])
            rp,mp,cp,cm,rm=map(np.array,(rp,mp,cp,cm,rm))
            def pct(x): return 100*(x.mean()-rp.mean())/rp.mean()
            def pv(x):
                try: return stats.wilcoxon(x,rp)[1]
                except: return float("nan")
            print(f"{name:22s} {rr['chi']:>3d} {rp.mean():>7.1f} {pct(mp):>8.1f} {pv(mp):>6.3f} {pct(cp):>10.1f} {pv(cp):>6.3f} {100*(cm.mean()-rm.mean())/rm.mean():>7.1f}")
    print("\nGO if cdesync (decentralized) dPeak NEGATIVE & p<0.05 in clustered reuse, ~0 at fullmesh/homo -> the NOVEL piece earns its keep.")
if __name__=="__main__": main()
