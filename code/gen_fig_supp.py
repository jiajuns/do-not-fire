"""Breadth + correlation figures for the pure-A paper.
 fig_breadth : de-synchronisation travel benefit holds across packet loss and across sensor range.
 fig_corr    : de-synchronisation benefit grows with information correlation (clustered vs dispersed
               spawns), the multi-robot test of the 2025 correlated-source AoI theorem.
Reads _pro6000_pull/results/summary/setcomm_{ploss,srange,corr}_paired.csv."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM = os.path.join(ROOT, "_pro6000_pull", "results", "summary")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures")
PAPER = os.path.join(ROOT, "paper", "mdpi_template", "figures")
os.makedirs(OUT, exist_ok=True)
BLUE, PURP, CORAL = "#3B4CC0", "#9B5FB5", "#C45B5B"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.linewidth": 0.8,
    "axes.edgecolor": "#2b2b2b", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def star(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")


def save(fig, name):
    fig.tight_layout()
    for d in [OUT, PAPER]:
        fig.savefig(os.path.join(d, name + ".png"), dpi=300)
        fig.savefig(os.path.join(d, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


def fig_breadth():
    pl = pd.read_csv(os.path.join(SUM, "setcomm_ploss_paired.csv"))
    sr = pd.read_csv(os.path.join(SUM, "setcomm_srange_paired.csv"))
    plt_t = pl[pl.metric == "total"].sort_values("drop")
    sr_t = sr[sr.metric == "total"].sort_values("srange")
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
    # packet loss
    x = (plt_t["drop"] * 100).values
    y = plt_t["desync_pct"].values
    ax[0].plot(x, y, "-o", color=BLUE, ms=6)
    for xi, yi, pi in zip(x, y, plt_t["p"].values):
        ax[0].annotate(star(pi), (xi, yi), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7.5)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("packet-loss probability (%)")
    ax[0].set_ylabel("de-sync travel effect (%)\n(negative = de-sync better)")
    ax[0].set_title("Robust to packet loss")
    ax[0].set_ylim(min(y) - 3, 2)
    # sensor range
    col = sr_t.columns[sr_t.columns.str.contains("pct")][0]
    x2 = sr_t["srange"].values
    y2 = sr_t[col].values
    ax[1].plot(x2, y2, "-o", color=PURP, ms=6)
    for xi, yi, pi in zip(x2, y2, sr_t["p"].values):
        ax[1].annotate(star(pi), (xi, yi), textcoords="offset points", xytext=(0, -11), ha="center", fontsize=7.5)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("sensor range (m)")
    ax[1].set_title("Holds across sensor range")
    ax[1].set_ylim(min(y2) - 3, 2)
    save(fig, "fig_breadth")


def fig_corr():
    c = pd.read_csv(os.path.join(SUM, "setcomm_corr_paired.csv"))
    ct = c[c.metric == "total"]
    order = ["dispersed", "clustered"]
    labels = ["dispersed\n(low correlation)", "clustered\n(high correlation)"]
    vals = [float(ct[ct.spawn == s]["desync_pct"].iloc[0]) for s in order]
    ps = [float(ct[ct.spawn == s]["p"].iloc[0]) for s in order]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    bars = ax.bar(range(2), vals, color=[BLUE, CORAL], width=0.55)
    ax.axhline(0, color="k", lw=0.9)
    for b, v, p in zip(bars, vals, ps):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.6, f"{v:.1f}%\n{star(p)}", ha="center", va="top", fontsize=8)
    ax.set_xticks(range(2)); ax.set_xticklabels(labels)
    ax.set_ylabel("de-sync travel effect (%)\n(negative = de-sync better)")
    ax.set_title("De-synchronisation benefit grows with information correlation")
    ax.set_ylim(min(vals) - 3, 1)
    save(fig, "fig_corr")


if __name__ == "__main__":
    fig_breadth()
    fig_corr()
