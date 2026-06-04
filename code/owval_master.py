"""OW-Desync comprehensive overnight validation (standalone scheduling-theory sim, parallel).
Primary metric = CLOSED-FORM bound ratio A_p(OW)/A_p(RR) (the theorem, exact). Secondary = deterministic
deficit-realized time-average weighted network peak-AoI ratio. Sweeps: heterogeneity x reuse x budget x
channel-reliability x baselines x weight-models x N-scaling, 200 seeds/cell, Wilcoxon p. Robust: each
cell independent; results appended to CSV; DONE marker at end."""
import os, sys, time, itertools, numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy import stats
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(HERE,"results","overnight"); os.makedirs(OUT,exist_ok=True)
CSV=os.path.join(OUT,"owval_results.csv")

def setup(N,R,layout,wcv,gcv,wmodel,rng,sigma=0.12):
    if layout=="clustered":
        nc=max(2,N//5); ctr=rng.uniform(0,1,(nc,2)); pos=np.clip(ctr[rng.integers(0,nc,N)]+rng.normal(0,0.06,(N,2)),0,1)
    else: pos=rng.uniform(0,1,(N,2))
    D=np.sqrt(((pos[:,None,:]-pos[None,:,:])**2).sum(-1)); conf=(D<R)&~np.eye(N,dtype=bool)
    K=np.exp(-D**2/(2*sigma**2)); np.fill_diagonal(K,0.0); coupling=1.0+K.sum(1)
    val=np.ones(N) if wcv<=0 else np.clip(rng.lognormal(0,wcv,N),0.1,None)
    if wmodel=="coupling": w=coupling
    elif wmodel=="value": w=val
    else: w=val*coupling
    gamma=np.ones(N) if gcv<=0 else np.clip(1.0-np.abs(rng.normal(0,gcv,N)),0.2,1.0)  # heterogeneous reliability
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

def realize_det(conf,w,tf,col,chi,T,policy):
    N=len(w); sent=np.zeros(N); aoi=np.ones(N); s=0.0
    for t in range(1,T+1):
        if policy=="rr": S=[i for i in range(N) if col[i]==(t%chi)]
        elif policy=="mwis":
            wt=w*aoi; avail=np.ones(N,bool); S=[]
            for i in np.argsort(-wt):
                if not avail[i]: continue
                S.append(i); avail[i]=False; avail[conf[i]]=False
        else:
            d=tf*t-sent; avail=np.ones(N,bool); S=[]
            for i in np.argsort(-d):
                if d[i]<=0 or not avail[i]: continue
                S.append(i); avail[i]=False; avail[conf[i]]=False
        for i in range(N):
            aoi[i]=1.0 if i in S else aoi[i]+1.0
            if i in S: sent[i]+=1
        s+=float(np.max(w*aoi))
    return s/T

def run_cell(args):
    sid,cfg=args
    N,R,layout,wcv,Bfac,gcv,wmodel,seeds=cfg
    br=[];rr=[];mw=[]
    for sd in range(seeds):
        rng=np.random.default_rng(31000+sd)
        conf,w,gamma=setup(N,R,layout,wcv,gcv,wmodel,rng)
        col=color(conf); chi=int(col.max()+1); B=Bfac*N/max(1,chi)
        edges=[(i,j) for i in range(N) for j in range(i+1,N) if conf[i,j]]
        f=solve_f(w,gamma,edges,B,N)
        ap_ow=float(np.sum(w/(gamma*f))); ap_rr=float(np.sum(w/(gamma*(1.0/chi)))); br.append(ap_ow/ap_rr)
        ro=realize_det(conf,w,f,col,chi,300,"ow"); rv=realize_det(conf,w,None,col,chi,300,"rr")
        mwv=realize_det(conf,w,None,col,chi,300,"mwis")
        rr.append(ro/max(rv,1e-9)); mw.append(mwv/max(rv,1e-9))
    br=np.array(br); rr=np.array(rr); mw=np.array(mw)
    try: _,p=stats.wilcoxon(br,np.ones_like(br))
    except: p=float("nan")
    return dict(sid=sid,N=N,R=R,layout=layout,wcv=wcv,Bfac=Bfac,gcv=gcv,wmodel=wmodel,seeds=seeds,
                boundRatio=round(float(br.mean()),4),boundP=round(float(p),5),
                relRatio=round(float(rr.mean()),4),mwisRatio=round(float(mw.mean()),4),
                fracWin=round(float(np.mean(br<0.99)),3))

def build():
    cells=[]; S=500
    # (A) heterogeneity gradient x N
    for N in [8,12,16,24,32,40]:
        for wcv in [0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0]:
            cells.append((N,0.30,"clustered",wcv,1.0,0.0,"valcoup",S))
    # (B) reuse gradient
    for N in [16,24,32]:
        for R in [2.0,0.5,0.4,0.35,0.30,0.25,0.22]:
            for wcv in [0.0,1.5]: cells.append((N,R,"clustered",wcv,1.0,0.0,"valcoup",S))
    # (C) budget
    for N in [16,24]:
        for bf in [0.5,1.0,1.5,2.0,3.0]: cells.append((N,0.30,"clustered",1.5,bf,0.0,"valcoup",S))
    # (D) reliability heterogeneity
    for N in [16,24]:
        for gcv in [0.0,0.3,0.6]:
            for wcv in [0.0,1.5]: cells.append((N,0.30,"clustered",wcv,1.0,gcv,"valcoup",S))
    # (E) weight models
    for N in [16,24]:
        for wm in ["coupling","value","valcoup"]:
            for wcv in [0.5,1.5]: cells.append((N,0.30,"clustered",wcv,1.0,0.0,wm,S))
    # (F) large-N scaling (heavier solves)
    for N in [8,16,24,32,40,48]:
        cells.append((N,0.30,"clustered",1.5,1.0,0.0,"valcoup",max(120,S//2)))
    # uniform-layout control
    for N in [16,24]:
        for wcv in [0.0,1.5]: cells.append((N,0.30,"uniform",wcv,1.0,0.0,"valcoup",S))
    return list(enumerate(cells))

def main():
    cells=build(); t0=time.time()
    open(CSV,"w").write("sid,N,R,layout,wcv,Bfac,gcv,wmodel,seeds,boundRatio,boundP,relRatio,mwisRatio,fracWin\n")
    print(f"[{time.strftime('%H:%M')}] owval: {len(cells)} cells, 14 workers",flush=True)
    with Pool(int(os.environ.get("OW_WORKERS","14"))) as pool:
        for i,r in enumerate(pool.imap_unordered(run_cell,cells)):
            line=",".join(str(r[k]) for k in ["sid","N","R","layout","wcv","Bfac","gcv","wmodel","seeds","boundRatio","boundP","relRatio","mwisRatio","fracWin"])
            open(CSV,"a").write(line+"\n")
            if (i+1)%10==0: print(f"[{time.strftime('%H:%M')}] {i+1}/{len(cells)} ({int((time.time()-t0)/60)}min)",flush=True)
    open(os.path.join(OUT,"OWVAL_DONE"),"w").close()
    print(f"[{time.strftime('%H:%M')}] OWVAL DONE {int((time.time()-t0)/60)}min, {len(cells)} cells -> {CSV}",flush=True)
if __name__=="__main__": main()
