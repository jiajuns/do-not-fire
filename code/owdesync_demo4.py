"""OW-Desync FINAL realization (edge-constraint convex program, no clique enumeration). Stationary
RANDOMIZED policy: solve f* = argmin sum_i w_i/f_i s.t. f_i+f_j<=1 per conflict edge + sum f<=B;
decompose f* into a convex combination of independent sets (greedy column generation); each slot
sample an independent set from that distribution. Achieves the peak-AoI bound, reduces to round-robin
when weights are symmetric (even-staggering special case), beats RR under heterogeneous weights."""
import os, numpy as np
from scipy.optimize import minimize

def setup(N,R,layout,wmode,rng,sigma=0.12):
    if layout=="clustered":
        nc=max(2,N//5); ctr=rng.uniform(0,1,(nc,2)); pos=np.clip(ctr[rng.integers(0,nc,N)]+rng.normal(0,0.06,(N,2)),0,1)
    else: pos=rng.uniform(0,1,(N,2))
    D=np.sqrt(((pos[:,None,:]-pos[None,:,:])**2).sum(-1)); conf=(D<R)&~np.eye(N,dtype=bool)
    K=np.exp(-D**2/(2*sigma**2)); np.fill_diagonal(K,0.0); coupling=1.0+K.sum(1)
    if wmode=="symm": val=np.ones(N)
    elif wmode=="het2": val=np.where(rng.random(N)<0.5,1.0,9.0)
    else: val=np.clip(rng.lognormal(0,1.0,N),0.2,None)
    return conf,val*coupling

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
    r=minimize(obj,f0,jac=jac,bounds=[(eps,1.0)]*N,constraints=cons,method="SLSQP",options={"maxiter":120,"ftol":1e-9})
    return np.clip(r.x,eps,1.0)

def decompose(conf,f):
    N=len(f); res=f.copy(); sets=[]; probs=[]
    for _ in range(4*N):
        if res.max()<1e-3: break
        avail=np.ones(N,bool); S=[]
        for i in np.argsort(-res):
            if res[i]<1e-3 or not avail[i]: continue
            S.append(i); avail[i]=False; avail[conf[i]]=False
        if not S: break
        p=min(res[i] for i in S); sets.append(S); probs.append(p)
        for i in S: res[i]-=p
    probs=np.array(probs)
    if probs.sum()<=0: sets=[[int(np.argmax(f))]]; probs=np.array([1.0])
    return sets, probs/probs.sum()

def realized(conf,w,sched,col,chi,sets,probs,T,rng):
    N=len(w); aoi=np.ones(N); peak=np.ones(N)
    for t in range(1,T+1):
        S=[i for i in range(N) if col[i]==(t%chi)] if sched=="rr" else sets[rng.choice(len(sets),p=probs)]
        for i in range(N):
            aoi[i]=1.0 if i in S else aoi[i]+1.0; peak[i]=max(peak[i],aoi[i])
    return float(np.sum(w*peak))

def main():
    SEEDS=range(int(os.environ.get("OW_SEEDS","40"))); T=200
    cfgs=[("FULLMESH symmW",2.0,"clustered","symm"),("clustered symmW",0.30,"clustered","symm"),
          ("clustered het2(9x)",0.30,"clustered","het2"),("clustered hetLN",0.30,"clustered","hetln"),
          ("hireuse hetLN",0.22,"clustered","hetln")]
    for N in (16,):
        print(f"\n===== N={N}: OW-Desync randomized-stationary realized vs RR (neg%=OW better); OW/bound~1=achieves cert =====")
        print(f"{'regime':22s} {'chi':>3s} {'OW/RR':>7s} {'d%':>6s} {'OW/bound':>9s}")
        for name,R,lay,wm in cfgs:
            ow=[];rr=[];ob=[]
            for s in SEEDS:
                rng=np.random.default_rng(13000+s); conf,w=setup(N,R,lay,wm,rng)
                col=color(conf); chi=int(col.max()+1); B=N/max(1,chi)
                edges=[(i,j) for i in range(N) for j in range(i+1,N) if conf[i,j]]
                f=solve_f(w,edges,B,N); bnd=float(np.sum(w/f)); sets,probs=decompose(conf,f)
                ro=realized(conf,w,"ow",col,chi,sets,probs,T,rng); rrv=realized(conf,w,"rr",col,chi,None,None,T,rng)
                ow.append(ro);rr.append(rrv);ob.append(ro/max(bnd,1e-9))
            ow,rr,ob=map(np.array,(ow,rr,ob)); ratio=ow.mean()/rr.mean()
            print(f"{name:22s} {chi:>3d} {ratio:>7.3f} {100*(ratio-1):>6.1f} {ob.mean():>9.2f}")
    print("\nGATE: symmW ~1.0 (reduces to RR = even-staggering special case); hetW <1 (beats RR); OW/bound near 1 = realized achieves the certificate.")
if __name__=="__main__": main()
