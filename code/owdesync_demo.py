"""OW-Desync: validate the algorithm+guarantee. Generalize Talak-Modiano peak-AoI optimality
(Ap = sum_i w_i/(gamma_i f_i) over activation frequencies on the conflict graph) with OUR
heterogeneous overlap-weights w_i and an explicit per-slot budget B. The OW-Desync schedule = the
convex-program-optimal stationary activation frequencies f*. Baseline = reuse-aware round-robin
(uniform f_i = 1/chi). At MATCHED total budget B = N/chi:
  - symmetric weights w_i=1  -> f* uniform = round-robin  => TIE (even-staggering is the special case)
  - heterogeneous weights    -> f* non-uniform           => OW-Desync provably LOWER weighted peak-AoI
  - full-mesh (chi=N)        -> one tx/slot               => TIE
Reports ratio A_p(OW)/A_p(RR) (<1 = OW better). Pure convex computation (no AoI sim)."""
import os, numpy as np
from scipy.optimize import minimize
from scipy import stats

def bron_kerbosch(adj):
    N=len(adj); cliques=[]
    def bk(R,P,X):
        if not P and not X: cliques.append(R); return
        piv=max(P|X, key=lambda u:len(P & set(np.where(adj[u])[0]))) if (P|X) else None
        for v in list(P - (set(np.where(adj[piv])[0]) if piv is not None else set())):
            Nv=set(np.where(adj[v])[0])
            bk(R|{v}, P & Nv, X & Nv); P=P-{v}; X=X|{v}
    bk(set(),set(range(N)),set()); return [list(c) for c in cliques]

def conflict_graph(N,R,layout,rng):
    if layout=="clustered":
        nc=max(2,N//5); ctr=rng.uniform(0,1,(nc,2))
        pos=np.clip(ctr[rng.integers(0,nc,N)]+rng.normal(0,0.06,(N,2)),0,1)
    else: pos=rng.uniform(0,1,(N,2))
    D=np.sqrt(((pos[:,None,:]-pos[None,:,:])**2).sum(-1))
    conf=(D<R)&~np.eye(N,dtype=bool); return conf

def greedy_color(conf):
    N=len(conf); col=-np.ones(N,int)
    for i in np.argsort(-conf.sum(1)):
        used={col[j] for j in range(N) if conf[i,j] and col[j]>=0}
        c=0
        while c in used: c+=1
        col[i]=c
    return col

def solve_ow(w,gamma,cliques,B,N):
    # min sum w_i/(gamma_i f_i) s.t. sum_{i in C} f_i <= 1 for each clique, sum f_i <= B, eps<=f_i<=1
    eps=1e-3
    f0=np.full(N,min(1.0,B/N))
    obj=lambda f: float(np.sum(w/(gamma*np.maximum(f,eps))))
    jac=lambda f: -w/(gamma*np.maximum(f,eps)**2)
    cons=[{"type":"ineq","fun":(lambda f,C=C: 1.0-np.sum(f[C]))} for C in cliques]
    cons.append({"type":"ineq","fun":lambda f: B-np.sum(f)})
    res=minimize(obj,f0,jac=jac,bounds=[(eps,1.0)]*N,constraints=cons,
                 method="SLSQP",options={"maxiter":300,"ftol":1e-9})
    f=np.clip(res.x,eps,1.0); return f, float(np.sum(w/(gamma*f)))

def main():
    SEEDS=range(int(os.environ.get("OW_SEEDS","40")))
    cfgs=[("FULLMESH symm",2.0,"clustered",0.0),
          ("HETgraph symmW",0.30,"clustered",0.0),
          ("HETw mild (cv1)",0.30,"clustered",1.0),
          ("HETw strong (cv2)",0.30,"clustered",2.0),
          ("HETw strong+hireuse",0.22,"clustered",2.0)]
    for N in (16,24):
        print(f"\n===== N={N}: weighted peak-AoI ratio A_p(OW-Desync)/A_p(RR), <1 = OW provably better =====")
        print(f"{'regime':22s} {'chi':>3s} {'ratio':>7s} {'min':>6s} {'max':>6s} {'frac<0.99':>9s}")
        for name,R,lay,wcv in cfgs:
            ratios=[]
            for s in SEEDS:
                rng=np.random.default_rng(5000+s)
                conf=conflict_graph(N,R,lay,rng)
                col=greedy_color(conf); chi=int(col.max()+1)
                w=np.ones(N) if wcv==0 else np.clip(rng.lognormal(0,wcv,N),0.1,None)  # heterogeneous overlap-weights
                gamma=np.ones(N)
                B=N/max(1,chi)                      # matched budget to round-robin total frequency
                cliques=bron_kerbosch(conf) or [[i] for i in range(N)]
                f_ow,ap_ow=solve_ow(w,gamma,cliques,B,N)
                ap_rr=float(np.sum(w/(gamma*(1.0/chi))))   # round-robin uniform freq 1/chi
                ratios.append(ap_ow/ap_rr)
            ratios=np.array(ratios)
            print(f"{name:22s} {chi:>3d} {ratios.mean():>7.3f} {ratios.min():>6.3f} {ratios.max():>6.3f} {np.mean(ratios<0.99):>9.2f}")
    print("\nGATE: FULLMESH & symmW ratio ~1.00 (even-staggering = special case); HETw ratio <1 = OW-Desync provably lower weighted peak-AoI.")
if __name__=="__main__": main()
