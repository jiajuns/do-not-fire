"""Rebuild fig_multiN from the CLEAN per-N decisive control (run_syncN), replacing the earlier
ANCOVA-on-genN version whose linear log-comm covariate could not fully remove the saturated-regime
confound. Here the comparison is phase-only: staggered vs synchronised periodic at the SAME interval
(identical per-robot rate AND total communication volume), at interval 12, N in {4,6,8,10,12}, n=120
paired (map x seed) per N. Shows travel and overlap reduction per N with paired-Wilcoxon significance.
Reads _pro6000_pull/results/summary/setcomm_syncN_paired.csv; writes code/results/figures + paper.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRED = os.path.join(ROOT, "_pro6000_pull", "results", "summary", "setcomm_syncN_paired.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures")
PAPER = os.path.join(ROOT, "paper", "mdpi_template", "figures")
os.makedirs(OUT, exist_ok=True)

NATURE = ["#3B4CC0", "#9B5FB5"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#2b2b2b",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def star(p):
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    d = pd.read_csv(PAIRED)
    d = d[d["metric"].isin(["total_travel_distance", "overlap_ratio"])]
    Ns = sorted(d["N"].unique())
    trav = {int(r.N): (r.stag_minus_sync_pct, r.p) for r in d[d.metric == "total_travel_distance"].itertuples()}
    ovl = {int(r.N): (r.stag_minus_sync_pct, r.p) for r in d[d.metric == "overlap_ratio"].itertuples()}

    print("per-N de-sync effect (interval 12, neg=de-sync better):")
    for N in Ns:
        N = int(N)
        print(f"  N={N:2d}  travel {trav[N][0]:+6.1f}% (p={trav[N][1]:.4f})  overlap {ovl[N][0]:+6.1f}% (p={ovl[N][1]:.4f})")

    Ns = [int(n) for n in Ns]
    x = np.arange(len(Ns)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4))
    b1 = ax.bar(x - w / 2, [trav[n][0] for n in Ns], w, label="travel", color=NATURE[0])
    b2 = ax.bar(x + w / 2, [ovl[n][0] for n in Ns], w, label="sensing overlap", color=NATURE[1])
    ax.axhline(0, color="k", lw=0.9)
    for bars, dct in [(b1, trav), (b2, ovl)]:
        for bar, n in zip(bars, Ns):
            e, p = dct[n]
            ax.text(bar.get_x() + bar.get_width() / 2, e - 0.5, star(p), ha="center", va="top", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([f"$N={n}$" for n in Ns])
    ax.set_ylabel("reduction vs. synchronised (%)\n(negative = de-sync better)")
    ax.set_title("De-synchronisation advantage across team size\n(phase-only control, interval 12)")
    ax.legend(loc="lower right")
    lo = min(min(trav[n][0] for n in Ns), min(ovl[n][0] for n in Ns))
    ax.set_ylim(lo - 3.0, 2.0)
    fig.tight_layout()
    for dd in [OUT, PAPER]:
        fig.savefig(os.path.join(dd, "fig_multiN.png"), dpi=300)
        fig.savefig(os.path.join(dd, "fig_multiN.pdf"))
    plt.close(fig)
    print("saved fig_multiN -> code/results/figures + paper/figures")


if __name__ == "__main__":
    main()
