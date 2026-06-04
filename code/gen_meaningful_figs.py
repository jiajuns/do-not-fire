"""Three additional meaningful figures for the paper, unified house style, self-contained.
  fig_desync_converge : decentralised DESYNC reaches even staggering from a synchronised cold start,
                        with no coordinator (validates the decentralised claim).
  fig_ow_allocation   : OW-Desync gives more airtime to high-value robots (water-filling f* vs round
                        robin), the mechanism behind the weighted peak-AoI gain.
  fig_mechanism_control: the two causal controls behind the de-synchronisation effect (dispersion
                        unchanged vs overlap down; belief-coupling ablation).
Run locally: `py gen_meaningful_figs.py`.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import house_style as hs
hs.apply()
P = hs.PAL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "figures")
PAPER = os.path.join(ROOT, "paper", "mdpi_template", "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    for d in [OUT, PAPER]:
        fig.savefig(os.path.join(d, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------- 1. decentralised convergence
def desync_converge():
    N, alpha, rounds = 8, 0.45, 45
    rng = np.random.default_rng(1)
    phi = (rng.uniform(0, 0.03, N)) % 1.0           # synchronised cold start
    hist = [phi.copy()]
    coh = [abs(np.mean(np.exp(2j * np.pi * phi)))]   # Kuramoto coherence (1 = synchronised)
    for _ in range(rounds):
        order = np.argsort(phi); s = phi[order]; new = phi.copy()
        for k in range(N):
            i = order[k]
            prev = s[k - 1] if k > 0 else s[N - 1] - 1.0
            nxt = s[k + 1] if k < N - 1 else s[0] + 1.0
            new[i] = ((1 - alpha) * phi[i] + alpha * (prev + nxt) / 2.0) % 1.0
        phi = new
        hist.append(phi.copy()); coh.append(abs(np.mean(np.exp(2j * np.pi * phi))))
    hist = np.array(hist); coh = np.array(coh); rr = np.arange(len(coh))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, N))
    snaps = [0, 4, 12, rounds]
    fig = plt.figure(figsize=(9.8, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.15], hspace=0.5, wspace=0.35)
    th = np.linspace(0, 2 * np.pi, 200)
    for c, rd in enumerate(snaps):
        ax = fig.add_subplot(gs[0, c]); ax.set_aspect("equal"); ax.axis("off"); ax.grid(False)
        ax.plot(np.cos(th), np.sin(th), color="#C7CCD4", lw=1.2)
        ph = hist[rd]
        ax.scatter(np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph), s=90, c=cmap, edgecolor="white",
                   linewidth=1.0, zorder=3)
        ax.set_title(f"round {rd}", fontsize=10)
    fig.text(0.5, 0.955, "(a) broadcast phases on the cycle: from clustered (synchronised) to evenly spaced",
             ha="center", fontsize=10.5, color=P["ink"])
    axb = fig.add_subplot(gs[1, :])
    axb.plot(rr, coh, color=P["desync"], lw=2.4, marker="o", ms=4)
    axb.axhline(0, ls="--", color=P["gray"], lw=1.1)
    axb.annotate("synchronised cold start", (0, coh[0]), xytext=(14, -6),
                 textcoords="offset points", fontsize=9, color=P["sync"])
    axb.annotate("even-staggered (peak-AoI optimal)", (rr[-1], coh[-1]), xytext=(-46, 30),
                 textcoords="offset points", fontsize=9, color=P["desync"],
                 arrowprops=dict(arrowstyle="->", color=P["gray"], lw=0.9))
    axb.set_xlabel("decentralised update round (local overhearing only)")
    axb.set_ylabel("synchronisation coherence")
    axb.set_title("(b) coherence collapses to the even-staggered optimum with no coordinator")
    axb.set_ylim(-0.05, 1.03)
    fig.suptitle("Decentralised de-synchronisation reaches even staggering from a synchronised start "
                 "($N=8$)", y=1.02, fontsize=11.5)
    save(fig, "fig_desync_converge")


# ---------------------------------------------------------- 2. OW-Desync allocation
def ow_allocation():
    # representative heterogeneous single-medium instance; water-filling closed form f_i ~ sqrt(w_i/gamma_i)
    w = np.array([0.40, 0.55, 0.70, 0.90, 1.15, 1.50, 2.00, 2.55])
    gam = np.ones_like(w)
    N = len(w)
    f_ow = np.sqrt(w / gam); f_ow = f_ow / f_ow.sum()          # total airtime 1 reallocated
    f_rr = np.ones(N) / N                                       # round-robin: uniform
    Ap_ow = np.sum(w / (gam * f_ow)); Ap_rr = np.sum(w / (gam * f_rr))
    ratio = Ap_ow / Ap_rr
    x = np.arange(N); bw = 0.4
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.bar(x - bw / 2, f_rr, bw, label="round-robin (uniform)", color=P["gray"], edgecolor="white", linewidth=0.8)
    ax.bar(x + bw / 2, f_ow, bw, label="OW-Desync $f^\\star$", color=P["ow"], edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{wi:.2f}" for wi in w])
    ax.set_xlabel("per-robot value weight $w_i$ (low $\\to$ high)")
    ax.set_ylabel("long-run activation frequency $f_i$")
    ax.set_title("OW-Desync reallocates airtime to high-value robots\n"
                 f"(weighted peak-AoI $A_p$: {ratio:.2f}$\\times$ round-robin, $-{100*(1-ratio):.0f}\\%$)")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(x, w, "--o", color=P["accent"], lw=1.6, ms=5, label="weight $w_i$")
    ax2.set_ylabel("value weight $w_i$", color=P["accent"]); ax2.tick_params(axis="y", colors=P["accent"])
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.5)
    save(fig, "fig_ow_allocation")


# ---------------------------------------------------------- 3. causal mechanism controls
def mechanism_control():
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.9))
    # (a) dispersion unchanged vs overlap down
    labs = ["team spatial\ndispersion", "redundant sensing\noverlap"]
    vals = [-0.9, -14.5]; ps = [0.57, 0.0005]
    cols = [P["gray"], P["desync"]]
    b = ax[0].bar(range(2), vals, color=cols, width=0.55, edgecolor="white", linewidth=0.8)
    ax[0].axhline(0, color=P["ink"], lw=1.0)
    for bar, v, p in zip(b, vals, ps):
        star = "$p=0.57$ (n.s.)" if p > 0.05 else "$p<10^{-3}$"
        ax[0].text(bar.get_x() + bar.get_width() / 2, v + (0.6 if v >= 0 else -0.6),
                   f"{v:+.1f}%\n{star}", ha="center", va=("bottom" if v >= 0 else "top"), fontsize=8.5)
    ax[0].set_xticks(range(2)); ax[0].set_xticklabels(labs)
    ax[0].set_ylabel("change, de-sync vs sync (%)")
    ax[0].set_title("(a) mechanism is decorrelation, not declustering")
    ax[0].set_ylim(-19, 5)
    # (b) belief-coupling ablation
    labs2 = ["belief-coupling\noff", "belief-coupling\non"]
    vals2 = [0.0, -2.7]; ps2 = [1.0, 0.03]
    b2 = ax[1].bar(range(2), vals2, color=[P["gray"], P["desync"]], width=0.55, edgecolor="white", linewidth=0.8)
    ax[1].axhline(0, color=P["ink"], lw=1.0)
    for bar, v, p in zip(b2, vals2, ps2):
        star = "n.s." if p > 0.05 else "$*$"
        ax[1].text(bar.get_x() + bar.get_width() / 2, v - 0.12,
                   f"{v:.1f}%\n{star}", ha="center", va="top", fontsize=8.5)
    ax[1].set_xticks(range(2)); ax[1].set_xticklabels(labs2)
    ax[1].set_ylabel("de-sync travel effect (%)")
    ax[1].set_title("(b) effect is tied to the shared teammate belief")
    ax[1].set_ylim(-3.6, 1.0)
    fig.suptitle("De-synchronisation works by decorrelating stale beliefs across robots",
                 y=1.03, fontsize=11)
    save(fig, "fig_mechanism_control")


if __name__ == "__main__":
    for f in [desync_converge, ow_allocation, mechanism_control]:
        try:
            f()
        except Exception as e:
            import traceback
            print(f.__name__, "FAILED:", e, traceback.format_exc()[-400:])
    print("DONE")
