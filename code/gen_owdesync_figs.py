import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":160,"font.family":"DejaVu Sans"})
OUT="results/figures"; os.makedirs(OUT,exist_ok=True)

# ---- Fig 1: heterogeneity gradient (ratio vs CV, per N) ----
d=pd.read_csv("results/overnight/owval_results.csv")
base=d[(d.layout=="clustered")&(d.R==0.30)&(d.Bfac==1.0)&(d.gcv==0.0)&(d.wmodel=="valcoup")]
fig,ax=plt.subplots(figsize=(6.2,4.2))
cmap=plt.cm.viridis(np.linspace(0,0.85,base.N.nunique()))
for c,N in zip(cmap,sorted(base.N.unique())):
    s=base[base.N==N].sort_values("wcv")
    ax.plot(s.wcv,s.boundRatio,marker="o",ms=4,color=c,label=f"N={int(N)}")
ax.axhline(1.0,ls="--",c="0.3",lw=1.2); ax.text(0.04,1.012,"even-staggering = round-robin (special case)",fontsize=9,color="0.3")
ax.fill_between([-0.1,3.1],0.4,1.0,color="tab:green",alpha=0.05)
ax.set_xlim(-0.05,3.05); ax.set_ylim(0.42,1.06)
ax.set_xlabel("per-robot weight heterogeneity (coefficient of variation)")
ax.set_ylabel(r"$A_p(\mathrm{OW\text{-}Desync})\,/\,A_p(\mathrm{round\text{-}robin})$")
ax.set_title("OW-Desync lowers weighted peak-AoI under heterogeneity\n(ties round-robin when symmetric)",fontsize=11)
ax.legend(ncol=3,fontsize=8,loc="lower left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_owdesync_gradient.png"); plt.close(fig)

# ---- Fig 2: heatmap N x CV ----
piv=base.pivot_table("boundRatio","N","wcv")
fig,ax=plt.subplots(figsize=(7.2,3.6))
im=ax.imshow(piv.values,aspect="auto",cmap="RdYlGn_r",vmin=0.45,vmax=1.0,origin="lower")
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:g}" for c in piv.columns])
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([int(i) for i in piv.index])
ax.set_xlabel("weight heterogeneity (CV)"); ax.set_ylabel("team size $N$")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v=piv.values[i,j]
        if not np.isnan(v): ax.text(j,i,f"{v:.2f}",ha="center",va="center",color=("w" if v<0.72 else "k"),fontsize=7.5)
fig.colorbar(im,label=r"$A_p$(OW)/$A_p$(RR)",fraction=0.046,pad=0.02)
ax.set_title("OW-Desync peak-AoI gain across team size and heterogeneity",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_owdesync_heatmap.png"); plt.close(fig)

# ---- Fig 3: on-testbed does-no-harm + Pillar A ----
t=pd.read_csv("results/summary/setcomm_owtb_paired.csv"); tt=t[t.metric=="total"]
Rs=[r for r in [0,30,60] if r in set(tt.R.unique())]
cmps=[("ow_vs_even","OW-Desync vs even (does-no-harm)","tab:gray"),
      ("even_vs_sync","even vs sync (Pillar A)","tab:blue"),
      ("ow_vs_sync","OW-Desync vs sync","tab:green")]
fig,ax=plt.subplots(figsize=(6.6,4.2)); w=0.26; x=np.arange(len(Rs))
for k,(cmp,lab,col) in enumerate(cmps):
    vals=[float(tt[(tt.R==R)&(tt.cmp==cmp)].pct.mean()) for R in Rs]
    ps=[float(tt[(tt.R==R)&(tt.cmp==cmp)].p.mean()) for R in Rs]
    bars=ax.bar(x+(k-1)*w,vals,w,label=lab,color=col)
    for b,p in zip(bars,ps):
        if p<0.05: ax.text(b.get_x()+b.get_width()/2,b.get_height()+(0.1 if b.get_height()>=0 else -0.3),"*",ha="center",fontsize=12)
ax.axhline(0,c="k",lw=0.8); ax.set_xticks(x); ax.set_xticklabels(["full-mesh" if R==0 else f"$R_c$={int(R)}" for R in Rs])
ax.set_ylabel("travel change vs comparison baseline (%)")
ax.set_title("On the frozen explorer (range-limited channel):\nOW-Desync does no harm; de-synchronisation beats synchronisation",fontsize=10.5)
ax.legend(fontsize=8.5,loc="lower left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_owtb.png"); plt.close(fig)
print("OK figs:", os.listdir(OUT))
