"""OW-Desync (corrected realization): practical scheduler = each slot transmit the MAX-WEIGHT
independent set with weight w_i * AoI_i (conflict-free, realizable, no f* targeting). Convex program
gives the optimality CERTIFICATE A_p* (lower bound). Weights w_i = value_i * coupling_i (value from
heterogeneous sensing; coupling = overlap kernel) -> genuinely heterogeneous & exploration-grounded.
Gate: symmetric weights / full-mesh -> OW-Desync ties RR (even-staggering special case); heterogeneous
-> OW-Desync realized beats RR on weighted peak-AoI and approaches the bound."""
import os, numpy as np
from scipy.optimize import minimize

def bron_kerbosch(adj):
    N=len(adj); cl=[]
    def bk(R,P,X):
        if not P and not X: cl.append(R); return
        for v in list(P):
            Nv=set(np.where(adj[v])[0]); bk(R|{v},P&Nv,X&Nv); P=P-{v}; X=X|{v}
    bk(set(),set(range(N)),set()); return [list(c) for c in cl]

def setup(N,R,layout,wmode,rng,sigma=0.12):
    if layout=="clustered":
        nc=max(2,N//5); ctr=rng.uniform(0,1,(nc,2)); pos=np.clip(ctr[rng.integers(0,nc,N)]+rng.normal(0,0.06,(N,2)),0,1)
    else: pos=rng.uniform(0,1,(N,2))
    D=np.sqrt(((pos[:,None,:]-pos[None,:,:])**2).sum(-1)); conf=(D<R)&~np.eye(N,dtype=bool)
    K=np.exp(-D**2/(2*sigma**2)); np.fill_diagonal(K,0.0); coupling=1.0+K.sum(1)
    if wmode=="symm": val=np.ones(N)
    elif wmode=="het2": val=np.where(rng.random(N)<0.5,1.0,9.0)        # 2-level sensor value (e.g. 10 vs 30 range -> area ~9x)
    else: val=np.clip(rng.lognormal(0,1.0,N),0.2,None)                # continuous heterogeneous value
    return conf,val*coupling

def color(conf):
    N=len(conf); c=-np.ones(N,int)
    for i in np.argsort(-conf.sum(1)):
        u={c[j] for j in range(N) if conf[i,j] and c[j]>=0}; k=0
        while k in u: k+=1
        c[i]=k
    return c

def mwis(conf,wt,K=None):
    chosen=[]; avail=np.ones(len(wt),bool)
    for i in np.argsort(-wt):
        if not avail[i]: continue
        chosen.append(i); avail[i]=False; avail[conf[i]]=False
        if K and len(chosen)>=K: break
    return chosen

def realized_wpaoi(conf,w,sched,col,chi,T,rng):
    N=len(w); aoi=np.ones(N); peak=np.ones(N)
    for t in range(1,T+1):
        if sched=="rr": S=[i for i in range(N) if col[i]==(t%chi)]
        else: S=mwis(conf, w*aoi)                       # OW-Desync: max-weight IS by w_i*AoI_i
        for i in range(N):
            aoi[i]=1.0 if i in S else aoi[i]+1.0; peak[i]=max(peak[i],aoi[i])
    return float(np.sum(w*peak))

def bound(conf,w,B,N):
    eps=1e-3; cl=bron_kerbosch(conf) or [[i] for i in range(N)]; f0=np.full(N,min(1.0,B/N))
    obj=lambda f: float(np.sum(w/np.maximum(f,eps))); jac=lambda f:-w/np.maximum(f,eps)**2
    cons=[{"type":"ineq","fun":(lambda f,C=C:1.0-np.sum(f[C]))} for C in cl]+[{"type":"ineq","fun":lambda f:B-np.sum(f)}]
    r=minimize(obj,f0,jac=jac,bounds=[(eps,1.0)]*N,constraints=cons,method="SLSQP",options={"maxiter":400,"ftol":1e-9})
    return float(np.sum(w/np.clip(r.x,eps,1.0)))

def main():
    SEEDS=range(int(os.environ.get("OW_SEEDS","40"))); T=200
    cfgs=[("FULLMESH symmW",2.0,"clustered","symm"),("clustered symmW",0.30,"clustered","symm"),
          ("clustered het2(9x)",0.30,"clustered","het2"),("clustered hetLN",0.30,"clustered","hetln"),
          ("hireuse hetLN",0.22,"clustered","hetln")]
    for N in (16,24):
        print(f"\n===== N={N}: realized weighted peak-AoI, OW-Desync vs RR (neg%=OW better); bound=certificate =====")
        print(f"{'regime':22s} {'chi':>3s} {'OW/RR':>7s} {'d%':>6s} {'OW/bound':>9s}")
        for name,R,lay,wm in cfgs:
            ow=[];rr=[];owb=[]
            for s in SEEDS:
                rng=np.random.default_rng(11000+s); conf,w=setup(N,R,lay,wm,rng)
                col=color(conf); chi=int(col.max()+1); B=N/max(1,chi)
                ro=realized_wpaoi(conf,w,"ow",col,chi,T,rng); rrv=realized_wpaoi(conf,w,"rr",col,chi,T,rng)
                ow.append(ro); rr.append(rrv); owb.append(ro/max(bound(conf,w,B,N),1e-9))
            ow,rr,owb=map(np.array,(ow,rr,owb)); ratio=ow.mean()/rr.mean()
            print(f"{name:22s} {chi:>3d} {ratio:>7.3f} {100*(ratio-1):>6.1f} {owb.mean():>9.2f}")
    print("\nGATE: symmW/FULLMESH OW/RR ~1.0 (even-staggering special case); heterogeneous-W OW/RR <1 (OW realized beats RR); OW/bound ~1 = near the certificate.")
if __name__=="__main__": main()
