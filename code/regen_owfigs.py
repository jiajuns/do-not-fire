"""Regenerate the OW-Desync + new mechanism figures in the unified house style, from local data.
Covers: fig_owdesync_gradient, fig_owdesync_heatmap, fig_owtb (OW-Desync results) and the four new
figures fig_method_schematic, fig_staleness_sawtooth, fig_nearopt_forest, fig_owdesync_robust.
Reads jeit/_owdata/*.csv (pulled from pro6000). Run locally: `py regen_owfigs.py`.
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import house_style as hs
hs.apply()
P = hs.PAL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "_owdata")
OUT = os.path.join(HERE, "results", "figures")
PAPER = os.path.join(ROOT, "paper", "mdpi_template", "figures")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)


def save(fig, name):
    fig.tight_layout()
    for d in [OUT, PAPER]:
        fig.savefig(os.path.join(d, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- gradient
def gradient():
    d = pd.read_csv(os.path.join(DATA, "owval_results.csv"))
    base = d[(d.layout == "clustered") & (d.R == 0.30) & (d.Bfac == 1.0) & (d.gcv == 0.0) & (d.wmodel == "valcoup")]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.grid(axis="both", alpha=0.4)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, base.N.nunique()))
    for c, N in zip(cmap, sorted(base.N.unique())):
        s = base[base.N == N].sort_values("wcv")
        ax.plot(s.wcv, s.boundRatio, marker="o", ms=5, color=c, label=f"N={int(N)}", markeredgecolor="white", markeredgewidth=0.7)
    ax.axhline(1.0, ls="--", c=P["ink"], lw=1.3)
    ax.text(0.05, 1.012, "even-staggering = round-robin (symmetric special case)", fontsize=8.5, color=P["ink"])
    ax.fill_between([-0.1, 3.1], 0.4, 1.0, color=P["ow"], alpha=0.06)
    ax.set_xlim(-0.05, 3.05); ax.set_ylim(0.42, 1.06)
    ax.set_xlabel("per-robot weight heterogeneity (coefficient of variation)")
    ax.set_ylabel(r"$A_p(\mathrm{OW\text{-}Desync})\,/\,A_p(\mathrm{round\text{-}robin})$")
    ax.set_title("OW-Desync lowers weighted peak-AoI under heterogeneity\n(ties round-robin when symmetric)")
    ax.legend(ncol=3, fontsize=8, loc="lower left")
    save(fig, "fig_owdesync_gradient")


# ---------------------------------------------------------------- heatmap
def heatmap():
    d = pd.read_csv(os.path.join(DATA, "owval_results.csv"))
    base = d[(d.layout == "clustered") & (d.R == 0.30) & (d.Bfac == 1.0) & (d.gcv == 0.0) & (d.wmodel == "valcoup")]
    piv = base.pivot_table("boundRatio", "N", "wcv")
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.grid(False)
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn_r", vmin=0.45, vmax=1.0, origin="lower")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([int(i) for i in piv.index])
    ax.set_xlabel("weight heterogeneity (CV)"); ax.set_ylabel("team size $N$")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=("w" if v < 0.72 else "#222"), fontsize=7.5)
    fig.colorbar(im, label=r"$A_p$(OW)/$A_p$(RR)", fraction=0.046, pad=0.02)
    ax.set_title("OW-Desync peak-AoI gain across team size and heterogeneity")
    save(fig, "fig_owdesync_heatmap")


# ---------------------------------------------------------------- owtb
def owtb():
    t = pd.read_csv(os.path.join(DATA, "setcomm_owtb_paired.csv"))
    cmps = [("ow_vs_even", "OW-Desync vs.\\ even (does no harm)", P["gray"]),
            ("even_vs_sync", "even vs.\\ sync (Pillar A)", P["desync"]),
            ("ow_vs_sync", "OW-Desync vs.\\ sync", P["ow"])]
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.2)); w = 0.26
    for mi, (met, title) in enumerate([("total", "(a) total travel"), ("overlap", "(b) sensing overlap")]):
        tt = t[t.metric == met]
        Rs = [r for r in [0, 30, 60] if r in set(tt.R.unique())]; x = np.arange(len(Rs))
        for k, (cmp, lab, col) in enumerate(cmps):
            vals = [float(tt[(tt.R == R) & (tt.cmp == cmp)].pct.mean()) for R in Rs]
            ps = [float(tt[(tt.R == R) & (tt.cmp == cmp)].p.mean()) for R in Rs]
            bars = ax[mi].bar(x + (k - 1) * w, vals, w, label=(lab.replace("\\", "") if mi == 0 else None),
                              color=col, edgecolor="white", linewidth=0.7)
            for b, p in zip(bars, ps):
                if p < 0.05:
                    ax[mi].text(b.get_x() + b.get_width() / 2,
                                b.get_height() + (0.12 if b.get_height() >= 0 else -0.45), "*", ha="center", fontsize=12)
        ax[mi].axhline(0, c=P["ink"], lw=1.0); ax[mi].set_xticks(x)
        ax[mi].set_xticklabels(["full-mesh" if R == 0 else f"$R_c$={int(R)}" for R in Rs])
        ax[mi].set_title(title)
    ax[0].set_ylabel("change vs. comparison baseline (%)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=9, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.suptitle("On the frozen explorer (range-limited channel): OW-Desync does no harm; "
                 "de-synchronisation beats synchronisation ($\\ast\\,p<0.05$)", y=1.02, fontsize=10.5)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    for d in [OUT, PAPER]:
        fig.savefig(os.path.join(d, "fig_owtb.png"), dpi=300)
    plt.close(fig); print("saved fig_owtb")
    return


# ---------------------------------------------------------------- schematic
def schematic():
    Pp = rng.uniform(0.05, 0.95, (9, 2)); R = 0.30
    D = np.sqrt(((Pp[:, None] - Pp[None]) ** 2).sum(-1)); val = rng.uniform(0.5, 1.7, 9)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5))
    for ax, title, mode in zip(axes, ["Synchronised / full broadcast", "OW-Desync activation"], ["sync", "ow"]):
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.12, 1.1); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_title(title)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(9):
            for j in range(i + 1, 9):
                if D[i, j] < 2 * R:
                    ax.plot([Pp[i, 0], Pp[j, 0]], [Pp[i, 1], Pp[j, 1]], color="#C7CCD4", lw=1.0, zorder=1)
        if mode == "sync":
            fire = set(range(9))
        else:
            avail = np.ones(9, bool); fire = set()
            for i in np.argsort(-val):
                if not avail[i]:
                    continue
                fire.add(int(i)); avail[i] = False; avail[D[i] < 2 * R] = False
        for i in range(9):
            on = i in fire
            col = (P["ow"] if (on and mode == "ow") else (P["sync"] if on else "#B9C0CA"))
            ax.add_patch(Circle(Pp[i], 0.020 + 0.030 * val[i] / 1.7, color=col, ec="white", lw=0.9, zorder=3))
            if on:
                ax.add_patch(Circle(Pp[i], R, fill=False, ls="--", ec=col, lw=1.2, alpha=0.55, zorder=2))
    axes[0].text(0.5, -0.09, "every robot fires → collisions, team stales together", ha="center",
                 fontsize=9, transform=axes[0].transAxes, color=P["sync"])
    axes[1].text(0.5, -0.09, "conflict-free, high-value independent set (spatial reuse)", ha="center",
                 fontsize=9, transform=axes[1].transAxes, color=P["ow"])
    fig.suptitle("Range-limited interference channel: OW-Desync fires a conflict-free, value-weighted subset",
                 y=1.02, fontsize=11)
    save(fig, "fig_method_schematic")


# ---------------------------------------------------------------- sawtooth
def sawtooth():
    N = 8; T = 8; t = np.arange(0, 2 * T + 1)
    sync = np.array([N * (tt % T) for tt in t], float)
    phi = np.round(np.arange(N) * T / N).astype(int)
    stag = np.array([sum((tt - p) % T for p in phi) for tt in t], float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.fill_between(t, stag, sync, where=sync >= stag, color=P["sync"], alpha=0.10, label="staleness peak avoided")
    ax.plot(t, sync, color=P["sync"], marker="o", ms=4, label="synchronised (peaks align)")
    ax.plot(t, stag, color=P["desync"], marker="s", ms=4, label="even-staggered (peak spread)")
    ax.set_xlabel("time step"); ax.set_ylabel(r"team belief staleness  $\sum_i \tau_i$")
    ax.set_title("De-synchronisation lowers the team-staleness peak (Prop. 3 mechanism)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    save(fig, "fig_staleness_sawtooth")


# ---------------------------------------------------------------- forest
def forest():
    rows = []
    try:
        xh = pd.read_csv(os.path.join(DATA, "setcomm_xhet_paired.csv"))
        for _, r in xh[(xh.cmp == "vlw_vs_even") & (xh.metric == "total")].iterrows():
            rows.append((f"value-Whittle  (het {r.het},{r.loss},N{int(r.N)})", float(r.pct), float(r.p), "fancy"))
    except Exception as e:
        print("xhet skip", e)
    try:
        ot = pd.read_csv(os.path.join(DATA, "setcomm_owtb_paired.csv"))
        for _, r in ot[(ot.cmp == "ow_vs_even") & (ot.metric == "total")].iterrows():
            rows.append((f"OW-Desync  (testbed R={int(r.R)})", float(r.pct), float(r.p), "fancy"))
        for _, r in ot[(ot.cmp == "even_vs_sync") & (ot.metric == "total")].iterrows():
            rows.append((f"de-sync vs sync (Pillar A, R={int(r.R)})", float(r.pct), float(r.p), "pillar"))
    except Exception as e:
        print("owtb skip", e)
    if not rows:
        return
    rows = rows[:16]; y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(rows) + 1.3))
    ax.grid(axis="x", alpha=0.4); ax.set_axisbelow(True)
    for yi, (lab, pct, p, kind) in zip(y, rows):
        col = P["desync"] if kind == "pillar" else P["gray"]
        ax.plot([0, pct], [yi, yi], color=col, lw=1.2, alpha=0.5)
        ax.scatter([pct], [yi], s=55, color=col, edgecolor="white", lw=0.8, zorder=3,
                   marker=("D" if kind == "pillar" else "o"))
    ax.axvline(0, color=P["ink"], lw=1.1)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("travel change vs even-staggering (%)")
    ax.set_title("On the exploration objective, no sophisticated scheduler beats even-staggering;\nonly synchronised-vs-de-synchronised (Pillar A, blue) moves it")
    save(fig, "fig_nearopt_forest")


# ---------------------------------------------------------------- robust
def robust():
    d = pd.read_csv(os.path.join(DATA, "owval_results.csv"))
    b = d[(d.layout == "clustered") & (d.wmodel == "valcoup")]
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.6))
    specs = [
        (axs[0], "R", b[(b.Bfac == 1.0) & (b.gcv == 0.0) & (b.wcv == 1.5) & (b.R <= 0.5)], "o", P["ow"], "interference range $R$", "(a) vs spatial reuse"),
        (axs[1], "N", b[(b.R == 0.30) & (b.Bfac == 1.0) & (b.gcv == 0.0) & (b.wcv == 1.5)], "^", P["desync"], "team size $N$", "(b) vs team size"),
        (axs[2], "gcv", b[(b.R == 0.30) & (b.Bfac == 1.0) & (b.wcv == 1.5)], "D", P["sync"], "channel-reliability heterogeneity", "(c) vs reliability spread"),
    ]
    for i, (ax, key, sub, mk, col, xl, tt) in enumerate(specs):
        s = sub.groupby(key)["boundRatio"].mean().sort_index()
        ax.plot(s.index, s.values, marker=mk, ms=7, color=col, markeredgecolor="white", markeredgewidth=0.9)
        ax.fill_between(s.index, s.values, 1.0, color=col, alpha=0.08)
        ax.axhline(1, ls="--", c=P["gray"], lw=1.2)
        ax.set_xlabel(xl); ax.set_title(tt); ax.set_ylim(0.45, 1.05)
        if i == 0:
            ax.set_ylabel(r"$A_p$(OW-Desync)$/A_p$(round-robin)")
    fig.suptitle("OW-Desync peak-AoI gain is robust across spatial reuse, team size, and channel reliability",
                 y=1.03, fontsize=11)
    save(fig, "fig_owdesync_robust")


if __name__ == "__main__":
    for f in [gradient, heatmap, owtb, schematic, sawtooth, forest, robust]:
        try:
            f()
        except Exception as e:
            import traceback
            print(f.__name__, "FAILED:", e, traceback.format_exc()[-400:])
    print("DONE; figures in", PAPER)
