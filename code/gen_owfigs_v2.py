import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
try: plt.style.use("seaborn-v0_8-whitegrid")
except Exception: pass
plt.rcParams.update({"font.size":12,"axes.titlesize":11.5,"axes.labelsize":11,"figure.dpi":200,
  "axes.edgecolor":"0.45","axes.linewidth":0.9,"legend.framealpha":0.9,"savefig.bbox":"tight"})
OUT="results/figures"; os.makedirs(OUT,exist_ok=True)
TEAL="#1b9e77"; RED="#d95f02"; BLUE="#3b6fb6"; PURP="#7570b3"; rng=np.random.default_rng(7)

def schematic():
    P=rng.uniform(0.05,0.95,(9,2)); R=0.30; D=np.sqrt(((P[:,None]-P[None])**2).sum(-1))
    val=rng.uniform(0.5,1.7,9)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4.4))
    for ax,title,mode in zip(axes,["Synchronised / full broadcast","OW-Desync activation"],["sync","ow"]):
        ax.set_xlim(-0.1,1.1); ax.set_ylim(-0.1,1.1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
        for i in range(9):
            for j in range(i+1,9):
                if D[i,j]<2*R: ax.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]],color="0.82",lw=0.8,zorder=1)
        if mode=="sync": fire=set(range(9))
        else:
            avail=np.ones(9,bool); fire=set()
            for i in np.argsort(-val):
                if not avail[i]: continue
                fire.add(int(i)); avail[i]=False; avail[D[i]<2*R]=False
        for i in range(9):
            on=i in fire; col=(TEAL if (on and mode=="ow") else (RED if on else "0.78"))
            ax.add_patch(Circle(P[i],0.020+0.028*val[i]/1.7,color=col,ec="k",lw=0.6,zorder=3))
            if on: ax.add_patch(Circle(P[i],R,fill=False,ls="--",ec=col,lw=1.1,alpha=0.55,zorder=2))
    axes[0].text(0.5,-0.07,"every robot fires → collisions, team stales together",ha="center",fontsize=8.5,transform=axes[0].transAxes,color=RED)
    axes[1].text(0.5,-0.07,"conflict-free, high-value independent set (spatial reuse)",ha="center",fontsize=8.5,transform=axes[1].transAxes,color=TEAL)
    fig.suptitle("Range-limited interference channel: OW-Desync fires a conflict-free, value-weighted subset",y=1.01,fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_method_schematic.png"); plt.close(fig)

def sawtooth():
    N=8;T=8;t=np.arange(0,2*T+1)
    sync=np.array([N*(tt%T) for tt in t],float)
    phi=np.round(np.arange(N)*T/N).astype(int); stag=np.array([sum((tt-p)%T for p in phi) for tt in t],float)
    fig,ax=plt.subplots(figsize=(6.5,4.1))
    ax.fill_between(t,stag,sync,where=sync>=stag,color=RED,alpha=0.10,label="staleness peak avoided")
    ax.plot(t,sync,color=RED,lw=2.2,marker="o",ms=4,label="synchronised (peaks align)")
    ax.plot(t,stag,color=TEAL,lw=2.2,marker="s",ms=4,label="even-staggered (peak spread)")
    ax.set_xlabel("time step"); ax.set_ylabel(r"team belief staleness  $\sum_i \tau_i$")
    ax.set_title("De-synchronisation lowers the team-staleness peak (Prop. 3 mechanism)")
    ax.legend(fontsize=9,loc="upper right"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_staleness_sawtooth.png"); plt.close(fig)

def forest():
    rows=[]
    def grab(path,met="total"):
        try: return pd.read_csv(path)
        except Exception: return None
    xh=grab("results/summary/setcomm_xhet_paired.csv")
    if xh is not None:
        for _,r in xh[(xh.cmp=="vlw_vs_even")&(xh.metric=="total")].iterrows():
            rows.append((f"value-Whittle  (het {r.het},{r.loss},N{int(r.N)})",float(r.pct),float(r.p),"fancy"))
    ot=grab("results/summary/setcomm_owtb_paired.csv")
    if ot is not None:
        for _,r in ot[(ot.cmp=="ow_vs_even")&(ot.metric=="total")].iterrows():
            rows.append((f"OW-Desync  (testbed R={int(r.R)})",float(r.pct),float(r.p),"fancy"))
        for _,r in ot[(ot.cmp=="even_vs_sync")&(ot.metric=="total")].iterrows():
            rows.append((f"de-sync vs sync (Pillar A, R={int(r.R)})",float(r.pct),float(r.p),"pillar"))
    if not rows: return
    rows=rows[:16]; y=np.arange(len(rows))[::-1]
    fig,ax=plt.subplots(figsize=(7.4,0.42*len(rows)+1.2))
    for yi,(lab,pct,p,kind) in zip(y,rows):
        col=BLUE if kind=="pillar" else ("0.45")
        ax.plot([0,pct],[yi,yi],color=col,lw=1,alpha=0.5)
        ax.scatter([pct],[yi],s=46,color=col,edgecolor="k",lw=0.5,zorder=3,marker=("D" if kind=="pillar" else "o"))
    ax.axvline(0,color="k",lw=1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows],fontsize=8)
    ax.set_xlabel("travel change vs even-staggering (%)")
    ax.set_title("On the exploration objective, no sophisticated scheduler beats even-staggering;\nonly synchronised-vs-de-synchronised (Pillar A, blue) moves it")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_nearopt_forest.png"); plt.close(fig)

def robust():
    d=pd.read_csv("results/overnight/owval_results.csv")
    b=d[(d.layout=="clustered")&(d.wmodel=="valcoup")]
    fig,axs=plt.subplots(2,2,figsize=(9,6.6))
    # (a) reuse R
    s=b[(b.Bfac==1.0)&(b.gcv==0.0)&(b.wcv==1.5)].groupby("R")["boundRatio"].mean().sort_index()
    axs[0,0].plot(s.index,s.values,marker="o",color=TEAL,lw=2); axs[0,0].axhline(1,ls="--",c="0.5")
    axs[0,0].set_xlabel("interference range $R$"); axs[0,0].set_title("(a) vs spatial reuse")
    # (b) budget
    s=b[(b.R==0.30)&(b.gcv==0.0)&(b.wcv==1.5)].groupby("Bfac")["boundRatio"].mean().sort_index()
    axs[0,1].plot(s.index,s.values,marker="s",color=PURP,lw=2); axs[0,1].axhline(1,ls="--",c="0.5")
    axs[0,1].set_xlabel("budget factor"); axs[0,1].set_title("(b) vs per-slot budget")
    # (c) N
    s=b[(b.R==0.30)&(b.Bfac==1.0)&(b.gcv==0.0)&(b.wcv==1.5)].groupby("N")["boundRatio"].mean().sort_index()
    axs[1,0].plot(s.index,s.values,marker="^",color=BLUE,lw=2); axs[1,0].axhline(1,ls="--",c="0.5")
    axs[1,0].set_xlabel("team size $N$"); axs[1,0].set_title("(c) vs team size")
    # (d) reliability gcv
    s=b[(b.R==0.30)&(b.Bfac==1.0)&(b.wcv==1.5)].groupby("gcv")["boundRatio"].mean().sort_index()
    axs[1,1].plot(s.index,s.values,marker="D",color=RED,lw=2); axs[1,1].axhline(1,ls="--",c="0.5")
    axs[1,1].set_xlabel("channel-reliability heterogeneity"); axs[1,1].set_title("(d) vs reliability spread")
    for ax in axs.flat: ax.set_ylabel(r"$A_p$(OW)/$A_p$(RR)"); ax.set_ylim(0.45,1.05)
    fig.suptitle("OW-Desync peak-AoI gain is robust across reuse, budget, team size, and channel reliability",y=1.0,fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_owdesync_robust.png"); plt.close(fig)

for fn in (schematic,sawtooth,forest,robust):
    try: fn(); print("ok",fn.__name__)
    except Exception as e:
        import traceback; print("FAIL",fn.__name__,e); traceback.print_exc()
print("figs:",[f for f in os.listdir(OUT) if f.startswith("fig_") and f.endswith(".png")])
