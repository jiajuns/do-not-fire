"""OW-Desync step 1+2: (1) overlap-GROUNDED weights w_i = sum_j kappa(d_ij) (robot i's overlap-coupling
from the staleness->overlap bound; high in dense/clustered regions) instead of synthetic weights; and
(2) REALIZE the convex-optimal stationary frequencies f* as an actual deficit-max-weight-IS schedule and
confirm the realized weighted peak-AoI achieves the convex lower bound and beats round-robin. Ties when
symmetric / full-mesh (even-staggering = special case)."""
import os, numpy as np
from scipy.optimize import minimize

def bron_kerbosch(adj):
    N=len(adj); cl=[]
    def bk(R,P,X):
        if not P and not X: cl.append(R); return
        for v in list(P):
            Nv=set(np.where(adj[v])[0]); bk(R|{v},P&Nv,X&Nv); P=P-{v}; X=X|{v}
    bk(set(),set(range(N)),set()); return [list(c) for c in cl]

def setup(N,R,layout,rng,sigma=0.12):
    if layout=="clustered":
        nc=max(2,N//5); ctr=rng.uniform(0,1,(nc,2)); pos=np.clip(ctr[rng.integers(0,nc,N)]+rng.normal(0,0.06,(N,2)),0,1)
    else: pos=rng.uniform(0,1,(N,2))
    D=np.sqrt(((pos[:,None,:]-pos[None,:,:])**2).sum(-1)); conf=(D<R)&~np.eye(N,dtype=bool)
    K=np.exp(-D**2/(2*sigma**2)); np.fill_diagonal(K,0.0); coupling=K.sum(1)   # overlap-coupling weight
    return conf,coupling

def color(conf):
    N=len(conf); c=-np.ones(N,int)
    for i in np.argsort(-conf.sum(1)):
        u={c[j] for j in range(N) if conf[i,j] and c[j]>=0}; k=0
        while k in u: k+=1
        c[i]=k
    return c

def solve_ow(w,cliques,B,N):
    eps=1e-3; f0=np.full(N,min(1.0,B/N))
    obj=lambda f: float(np.sum(w/np.maximum(f,eps))); jac=lambda f: -w/np.maximum(f,eps)**2
    cons=[{"type":"ineq","fun":(lambda f,C=C:1.0-np.sum(f[C]))} for C in cliques]+[{"type":"ineq","fun":lambda f:B-np.sum(f)}]
    r=minimize(obj,f0,jac=jac,bounds=[(eps,1.0)]*N,constraints=cons,method="SLSQP",options={"maxiter":400,"ftol":1e-9})
    f=np.clip(r.x,eps,1.0); return f,float(np.sum(w/f))

def realize(conf,w,f,T,rng):
    # deficit-driven max-weight independent set per slot, targeting frequencies f
    N=len(w); sent=np.zeros(N); aoi=np.ones(N); peak=np.ones(N)
    for t in range(1,T+1):
        deficit=f*t-sent; order=np.argsort(-deficit)        # who is most behind their target rate
        avail=np.ones(N,bool); S=[]
        for i in order:
            if deficit[i]<=0 or not avail[i]: continue
            S.append(i); avail[i]=False; avail[conf[i]]=False
        for i in range(N):
            if i in S: aoi[i]=1.0; sent[i]+=1
            else: aoi[i]+=1.0
            peak[i]=max(peak[i],aoi[i])
    return float(np.sum(w*peak))   # realized weighted peak-AoI (sum form, matches Talak Ap)

def main():
    SEEDS=range(int(os.environ.get("OW_SEEDS","40"))); T=200
    cfgs=[("FULLMESH",2.0,"clustered"),("uniform layout",0.30,"uniform"),("clustered (overlap-het)",0.30,"clustered"),("clustered hireuse",0.22,"clustered")]
    for N in (16,24):
        print(f"\n===== N={N}: overlap-grounded weights w_i=sum_j kappa(d_ij). ratio = OW/RR (<1 = OW better) =====")
        print(f"{'regime':24s} {'chi':>3s} {'bound ratio':>11s} {'realized ratio':>14s} {'real~bound':>10s}")
        for name,R,lay in cfgs:
            br=[];rr=[]
            for s in SEEDS:
                rng=np.random.default_rng(9000+s); conf,w=setup(N,R,lay,rng)
                col=color(conf); chi=int(col.max()+1); B=N/max(1,chi)
                cliques=bron_kerbosch(conf) or [[i] for i in range(N)]
                f_ow,ap_ow=solve_ow(w,cliques,B,N); ap_rr=float(np.sum(w/(1.0/chi)))
                br.append(ap_ow/ap_rr)
                f_rr=np.full(N,1.0/chi)
                re_ow=realize(conf,w,f_ow,T,rng); re_rr=realize(conf,w,f_rr,T,rng)
                rr.append(re_ow/max(re_rr,1e-9))
            br=np.array(br); rr=np.array(rr)
            print(f"{name:24s} {chi:>3d} {br.mean():>11.3f} {rr.mean():>14.3f} {'OK' if abs(br.mean()-rr.mean())<0.12 else 'gap':>10s}")
    print("\nGATE: FULLMESH/uniform ratio~1.0 (even-staggering special case); clustered (overlap-heterogeneous) ratio<1; realized~bound confirms the optimum is achievable by a real schedule.")
if __name__=="__main__": main()
