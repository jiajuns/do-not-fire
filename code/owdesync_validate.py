"""OW-Desync validation suite: (A) win vs weight-heterogeneity gradient, (B) win vs reuse/R_comm
gradient, (C) budget effect. Two measures per cell: CLOSED-FORM bound ratio A_p(OW)/A_p(RR) (the
theorem, exact), and a DETERMINISTIC deficit-realized time-average weighted network peak-AoI ratio
(fair, low-variance; avoids the i.i.d.-sampling peak-variance artifact). Deterministic deficit
scheduler realizes f* with bounded inter-service gaps. Gates: symmetric/full-mesh -> ratio ~1.0
(even-staggering special case); heterogeneous -> ratio<1 and monotone in heterogeneity."""
import os, numpy as np
from scipy.optimize import minimize

def setup(N,R,layout,wcv,rng,sigma=0.12):
    if layout=="clustered":
        nc=max(2,N//5); ctr=rng.uniform(0,1,(nc,2)); pos=np.clip(ctr[rng.integers(0,nc,N)]+rng.normal(0,0.06,(N,2)),0,1)
    else: pos=rng.uniform(0,1,(N,2))
    D=np.sqrt(((pos[:,None,:]-pos[None,:,:])**2).sum(-1)); conf=(D<R)&~np.eye(N,dtype=bool)
    K=np.exp(-D**2/(2*sigma**2)); np.fill_diagonal(K,0.0); coupling=1.0+K.sum(1)
    val=np.ones(N) if wcv<=0 else np.clip(rng.lognormal(0,wcv,N),0.1,None)
    return conf, val*coupling

def color(conf):
    N=len(conf); c=-np.ones(N,int)
    for i in np.argsort(-conf.sum(1)):
        u={c[j] for j in range(N) if conf[i,j] and c[j]>=0}; k=0
        while k in u: k+=1
        c[i]=k
    return c

def solve_f(w,edges,B,N):
    eps=1e-3; f0=np.full(N,min(1.0,B/N))
    obj=lambda f: float(np.sum(w/np.maximum(f,eps))); jac=lambda f:-w/np.maximum(f,eps)**2
    cons=[{"type":"ineq","fun":(lambda f,e=e:1.0-f[e[0]]-f[e[1]])} for e in edges]+[{"type":"ineq","fun":lambda f:B-np.sum(f)}]
    r=minimize(obj,f0,jac=jac,bounds=[(eps,1.0)]*N,constraints=cons,method="SLSQP",options={"maxiter":100,"ftol":1e-9})
    return np.clip(r.x,eps,1.0)

def realize_det(conf,w,target_f,col,chi,T,policy):
    N=len(w); sent=np.zeros(N); aoi=np.ones(N); wp=0.0
    for t in range(1,T+1):
        if policy=="rr": S=[i for i in range(N) if col[i]==(t%chi)]
        else:                                  # deterministic deficit scheduler realizing target_f
            deficit=target_f*t-sent; avail=np.ones(N,bool); S=[]
            for i in np.argsort(-deficit):
                if deficit[i]<=0 or not avail[i]: continue
                S.append(i); avail[i]=False; avail[conf[i]]=False
        for i in range(N):
            aoi[i]=1.0 if i in S else aoi[i]+1.0
            if i in S: sent[i]+=1
        wp+=float(np.max(w*aoi))
    return wp/T

def cell(N,R,layout,wcv,Bfac,seeds):
    bnd=[];rel=[]
    for s in seeds:
        rng=np.random.default_rng(21000+s); conf,w=setup(N,R,layout,wcv,rng)
        col=color(conf); chi=int(col.max()+1); B=Bfac*N/max(1,chi)
        edges=[(i,j) for i in range(N) for j in range(i+1,N) if conf[i,j]]
        f=solve_f(w,edges,B,N)
        ap_ow=float(np.sum(w/f)); ap_rr=float(np.sum(w/(1.0/chi))); bnd.append(ap_ow/ap_rr)
        ro=realize_det(conf,w,f,col,chi,300,"ow"); rr=realize_det(conf,w,None,col,chi,300,"rr"); rel.append(ro/max(rr,1e-9))
    return np.mean(bnd),np.mean(rel)

def main():
    N=int(os.environ.get("OW_N","16")); seeds=range(int(os.environ.get("OW_SEEDS","30")))
    print(f"N={N}, {len(list(seeds))} seeds. ratio<1 = OW-Desync better (boundRatio=theorem, relRatio=deterministic realized)")
    print("\n== (A) win vs weight-heterogeneity (clustered R=0.30, B=1x) ==")
    print(f"{'wcv':>5s} {'boundRatio':>11s} {'relRatio':>9s}")
    for wcv in [0.0,0.5,1.0,1.5,2.0]:
        b,r=cell(N,0.30,"clustered",wcv,1.0,seeds); print(f"{wcv:>5.1f} {b:>11.3f} {r:>9.3f}")
    print("\n== (B) win vs reuse/R_comm (clustered, wcv=1.5, B=1x) ==")
    print(f"{'R':>5s} {'boundRatio':>11s} {'relRatio':>9s}")
    for R in [2.0,0.45,0.35,0.30,0.25,0.22]:
        b,r=cell(N,R,"clustered",1.5,1.0,seeds); print(f"{R:>5.2f} {b:>11.3f} {r:>9.3f}")
    print("\n== (C) budget effect (clustered R=0.30, wcv=1.5) ==")
    print(f"{'Bfac':>5s} {'boundRatio':>11s} {'relRatio':>9s}")
    for bf in [0.5,1.0,1.5,2.0]:
        b,r=cell(N,0.30,"clustered",1.5,bf,seeds); print(f"{bf:>5.1f} {b:>11.3f} {r:>9.3f}")
    print("\nGATE: wcv=0 ~1.0 (even-staggering special case); ratio decreases monotonically with heterogeneity; full-mesh(R=2) ~1.0.")
if __name__=="__main__": main()
