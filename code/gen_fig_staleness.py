"""Theory<->experiment figure (fig_staleness): de-synchronisation decorrelates staleness at
constant mean. For a period-T broadcast schedule, per-link staleness is a sawtooth; the
staleness->overlap bound (Proposition 2) is driven by the PEAK of the team staleness sum
S(t)=sum_i ((t-phi_i) mod T), not its mean. Synchronised broadcasting (phi_i=0) makes all links
peak together; even staggering (phi_i=round(iT/N)) spreads them. The time-MEAN of S(t) is identical
(5.5N at T=12) for both, but staggering roughly HALVES the peak and collapses its variability --
exactly what lowers the overlap envelope. Left: peak vs mean team staleness, sync vs stagger, across
N. Writes code/results/figures + paper/figures.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures")
PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "mdpi_template", "figures")
os.makedirs(OUT, exist_ok=True)
NATURE = ["#3B4CC0", "#9B5FB5", "#4C8DBE"]
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.linewidth": 0.8,
    "axes.edgecolor": "#2b2b2b", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

T = 12
Ns = [4, 6, 8, 10, 12]
peak_sync, peak_stag, mean_both = [], [], []
for N in Ns:
    t = np.arange(T)
    Ss = np.array([(int(tt) % T) * N for tt in t], dtype=float)
    phi = [round(i * T / N) for i in range(N)]
    Sg = np.array([sum((int(tt) - p) % T for p in phi) for tt in t], dtype=float)
    peak_sync.append(Ss.max()); peak_stag.append(Sg.max()); mean_both.append(Ss.mean())

x = np.arange(len(Ns)); w = 0.38
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.bar(x - w / 2, peak_sync, w, label="synchronised (peak)", color=NATURE[0])
ax.bar(x + w / 2, peak_stag, w, label="staggered (peak)", color=NATURE[1])
ax.plot(x, mean_both, "o--", color="#444", lw=1.2, ms=4, label="mean (identical)")
for i in range(len(Ns)):
    red = 100 * (peak_stag[i] - peak_sync[i]) / peak_sync[i]
    ax.text(x[i], peak_stag[i] - 3, f"{red:.0f}%", ha="center", va="top", fontsize=7.5, color=NATURE[1])
ax.set_xticks(x); ax.set_xticklabels([f"$N={n}$" for n in Ns])
ax.set_ylabel("team staleness $\\sum_i \\tau_i$  (steps)")
ax.set_title("De-synchronisation collapses the staleness peak at constant mean\n(broadcast period $T=12$)")
ax.legend(loc="upper left")
fig.tight_layout()
for d in [OUT, PAPER]:
    fig.savefig(os.path.join(d, "fig_staleness.png"), dpi=300)
    fig.savefig(os.path.join(d, "fig_staleness.pdf"))
plt.close(fig)
print("peak reduction by N:", {Ns[i]: round(100 * (peak_stag[i] - peak_sync[i]) / peak_sync[i], 1) for i in range(len(Ns))})
print("saved fig_staleness -> code/results/figures + paper/figures")
