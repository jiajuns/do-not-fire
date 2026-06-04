"""OW-Desync paper-blocker fixes + re-validation: (1) realized scheduler now MODELS Bernoulli(gamma)
packet loss (AoI resets only on successful delivery) -> fair under reliability heterogeneity; (2) add
truly-symmetric weights w=1 (no coupling) to confirm the EXACT tie with round-robin (even-staggering is
the special case); matched budget only (Bfac=1). Reuses the master's formulation."""
import os, numpy as np
from scipy.optimize import minimize

def setup(N,R,wcv,gcv,wmodel,rng,sigma=0.12):
    nc=max(2,N//5); ctr=rng.uniform(0,1,(nc,2)); pos=np.clip(ctr[rng.integers(0,nc,N)]+rng.normal(0,0.06,(N,2)),0,1)
    D=np.sqrt(((pos[:,None,:]-pos[None,:,:])**2).sum(-1)); conf=(D<R)&~np.eye(N,dtype=bool)
    K=np.exp(-D**2/(2*sigma**2)); np.fill_diagonal(K,0.0); coupling=1.0+K.sum(1)
    val=np.ones(N) if wcv<=0 else np.clip(rng.lognormal(0,wcv,N),0.1,None)
    w={"symm":np.ones(N),"coupling":coupling,"value":val,"valcoup":val*coupling}[wmodel]
    gamma=np.ones(N) if gcv<=0 else np.clip(1.0-np.abs(rng.normal(0,gcv,N)),0.2,1.0)
    return conf,w,gamma

def color(conf):
    N=len(conf); c=-np.ones(N,int)
    for i in np.argsort(-conf.sum(1)):
        u={c[j] for j in range(N) if conf[i,j] and c[j]>=0}; k=0
        while k in u: k+=1
        c[i]=k
    return c

def solve_f(w,gamma,edges,B,N):
    eps=1e-3; f0=np.full(N,min(1.0,B/N))
    obj=lambda f: float(np.sum(w/(gamma*np.maximum(f,eps)))); jac=lambda f:-w/(gamma*np.maximum(f,eps)**2)
    cons=[{"type":"ineq","fun":(lambda f,e=e:1.0-f[e[0]]-f[e[1]])} for e in edges]+[{"type":"ineq","fun":lambda f:B-np.sum(f)}]
    r=minimize(obj,f0,jac=jac,bounds=[(eps,1.0)]*N,constraints=cons,method="SLSQP",options={"maxiter":120,"ftol":1e-9})
    return np.clip(r.x,eps,1.0)

def realize(conf,w,gamma,tf,col,chi,T,policy,rng):
    N=len(w); sent=np.zeros(N); aoi=np.ones(N); s=0.0
    for t in range(1,T+1):
        if policy=="rr": S=[i for i in range(N) if col[i]==(t%chi)]
        else:
            d=tf*t-sent; avail=np.ones(N,bool); S=[]
            for i in np.argsort(-d):
                if d[i]<=0 or not avail[i]: continue
                S.append(i); avail[i]=False; avail[conf[i]]=False
        for i in range(N):
            if i in S:
                sent[i]+=1
                aoi[i]=1.0 if rng.random()<gamma[i] else aoi[i]+1.0   # Bernoulli(gamma) delivery
            else: aoi[i]+=1.0
        s+=float(np.max(w*aoi))
    return s/T

def cell(N,R,wcv,gcv,wmodel,seeds):
    br=[];rr=[]
    for sd in range(seeds):
        rng=np.random.default_rng(41000+sd); conf,w,gamma=setup(N,R,wcv,gcv,wmodel,rng)
        col=color(conf); chi=int(col.max()+1); B=N/max(1,chi)
        edges=[(i,j) for i in range(N) for j in range(i+1,N) if conf[i,j]]
        f=solve_f(w,gamma,edges,B,N)
        br.append(float(np.sum(w/(gamma*f)))/float(np.sum(w/(gamma*(1.0/chi)))))
        ro=realize(conf,w,gamma,f,col,chi,300,"ow",rng); rv=realize(conf,w,gamma,None,col,chi,300,"rr",rng)
        rr.append(ro/max(rv,1e-9))
    return np.mean(br),np.mean(rr)

def main():
    seeds=int(os.environ.get("OW_SEEDS","60"))
    print("=== FIX1: exact tie with truly-symmetric weights w=1 (R=0.30, gcv=0) ===")
    for N in (16,24):
        b,r=cell(N,0.30,0.0,0.0,"symm",seeds); print(f"  N={N} symm: boundRatio={b:.3f} relRatio={r:.3f}  (expect ~1.000)")
    print("=== FIX2: reliability heterogeneity WITH loss modelled in realization (value weights) ===")
    for gcv in (0.0,0.3,0.6):
        for wcv in (0.0,1.5):
            b,r=cell(16,0.30,wcv,gcv,"value",seeds); print(f"  N=16 gcv={gcv} wcv={wcv}: boundRatio={b:.3f} relRatio={r:.3f}")
    print("=== heterogeneity gradient (value weights, gcv=0), confirm clean monotone ===")
    for wcv in (0.0,0.5,1.0,1.5,2.0):
        b,r=cell(16,0.30,wcv,0.0,"value",seeds); print(f"  wcv={wcv}: boundRatio={b:.3f} relRatio={r:.3f}")
if __name__=="__main__": main()
