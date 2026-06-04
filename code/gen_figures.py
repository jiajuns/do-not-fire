"""Generate the consolidated publication figures (MDPI Sensors) from all result groups.
Each figure is wrapped in try/except so missing groups are skipped (e.g. sync/drop2 fill in
later). Outputs results/figures/*.pdf + *.png. Run on the server (has raw data + matplotlib)."""
import os, glob, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results", "figures"); os.makedirs(FIG, exist_ok=True)

# --- Nature-style blue-purple theme (clean, despined, sans-serif, 300 dpi) ---
NATURE_BLUEPURPLE = ["#3B4CC0", "#7A4FA0", "#4C8DBE", "#9B5FB5", "#5C5FB8", "#8E6FBF"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#2b2b2b",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.fontsize": 8, "legend.frameon": False,
    "lines.linewidth": 1.7, "lines.markersize": 5,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.prop_cycle": plt.cycler(color=NATURE_BLUEPURPLE),
})


def load(g):
    fs = glob.glob(os.path.join(HERE, "results", "raw", g, "*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for c in ["total_travel_distance", "overlap_ratio", "final_coverage_ratio",
              "total_communication_volume", "map_id", "seed"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name + ".pdf")); fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig); print("saved", name)


def fig_pareto():
    d = load("setcomm_genN")
    if d is None:
        d = load("setcomm_deep")
    if d is None:
        return
    if d["method_name"].astype(str).str.endswith("_N8").any():
        d = d[d["method_name"].astype(str).str.endswith("_N8")].copy()
        d["method_name"] = d["method_name"].str.replace(r"_N8$", "", regex=True)
    def fam(b):
        if b == "Original" or b.startswith("Periodic"): return "Synchronised (control)"
        if b.startswith("Random"): return "Random (async)"
        if b.startswith("EventTrig"): return "Event-triggered"
        if b.startswith("SETComm"): return "Spiking (ours)"
        return "Ablation"
    famcol = {"Synchronised (control)": "#4C8DBE", "Event-triggered": "#7A4FA0",
              "Random (async)": "#5C5FB8", "Spiking (ours)": "#6A2C91", "Ablation": "#AEB3D4"}
    g = d.groupby("method_name").agg(comm=("total_communication_volume", "mean"),
                                     travel=("total_travel_distance", "mean")).reset_index()
    g["fam"] = g["method_name"].map(fam)
    g["comm"] = g["comm"] / 1e3
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    # faint raw points for context
    ax.scatter(g["comm"], g["travel"], s=16, alpha=0.15,
               color=[famcol.get(f, "0.6") for f in g["fam"]], zorder=1)
    # per-family lower-envelope Pareto front: best travel achievable at each budget
    for f in ["Synchronised (control)", "Event-triggered", "Random (async)", "Spiking (ours)"]:
        sub = g[g.fam == f].sort_values("comm")
        if len(sub) < 2:
            continue
        fx, fy, best = [], [], np.inf
        for xi, yi in zip(sub["comm"].values, sub["travel"].values):
            if yi < best - 1e-9:
                best = yi; fx.append(xi); fy.append(yi)
        emph = f == "Spiking (ours)"
        ax.plot(fx, fy, "-o", color=famcol[f], label=f,
                lw=2.8 if emph else 1.6, ms=5.5 if emph else 4,
                alpha=1.0 if (emph or "control" in f) else 0.85, zorder=3 if emph else 2)
    ax.set_xlabel("communication volume ($10^3$ bytes)")
    ax.set_ylabel("total travel distance (lower is better)")
    ax.set_title("Communication--efficiency Pareto fronts ($N=8$)")
    ax.legend(fontsize=7.5, title="front per scheduler (lower-left is better)")
    save(fig, "fig_pareto")


def fig_paretoN():
    """Pareto fronts (ours vs synchronised control) across team sizes, from setcomm_genN."""
    d = load("setcomm_genN")
    if d is None:
        return
    d = d.dropna(subset=["total_communication_volume", "total_travel_distance"])
    d["N"] = d["method_name"].str.extract(r"_N(\d+)$").astype(float)
    base = d["method_name"].str.replace(r"_N\d+$", "", regex=True)

    def fam(b):
        if b == "Original" or b.startswith("Periodic"): return "Synchronised (control)"
        if b.startswith("Random"): return "Random (async)"
        if b.startswith("EventTrig"): return "Event-triggered"
        if b.startswith("SETComm"): return "Spiking (ours)"
        return "Ablation"
    d["fam"] = base.map(fam)
    famcol = {"Synchronised (control)": "#4C8DBE", "Random (async)": "#5C5FB8",
              "Spiking (ours)": "#6A2C91"}
    Ns = [2, 4, 8, 12]
    fig, axes = plt.subplots(1, len(Ns), figsize=(13.5, 3.4))
    for ax, N in zip(axes, Ns):
        sub = d[d["N"] == N]
        g = sub.groupby("method_name").agg(comm=("total_communication_volume", "mean"),
                                            travel=("total_travel_distance", "mean"),
                                            fam=("fam", "first")).reset_index()
        g["comm"] = g["comm"] / 1e3
        for f in ["Synchronised (control)", "Random (async)", "Spiking (ours)"]:
            s = g[g.fam == f].sort_values("comm")
            if len(s) < 2:
                continue
            fx, fy, best = [], [], np.inf
            for xi, yi in zip(s["comm"].values, s["travel"].values):
                if yi < best - 1e-9:
                    best = yi; fx.append(xi); fy.append(yi)
            emph = f == "Spiking (ours)"
            ax.plot(fx, fy, "-o", color=famcol[f], label=f,
                    lw=2.6 if emph else 1.5, ms=5 if emph else 3.5,
                    alpha=1.0 if (emph or "control" in f) else 0.8, zorder=3 if emph else 2)
        ax.set_title(f"$N={int(N)}$"); ax.set_xlabel("comm ($10^3$ B)")
    axes[0].set_ylabel("total travel (lower is better)")
    axes[-1].legend(fontsize=7, loc="upper right")
    save(fig, "fig_paretoN")


def fig_bsweep():
    d = load("setcomm_targeted")
    if d is None:
        return
    def parse(nm):
        m = re.match(r"T_(voi_lif|voi|err|prox|stale|round_robin|random)(?:_s[\d.]+)?_B(\d+)$", str(nm))
        return (m.group(1), int(m.group(2))) if m else (None, None)
    d["md"], d["B"] = zip(*d["method_name"].map(parse))
    d = d.dropna(subset=["md", "B"])
    d = d[~d["method_name"].str.contains("_s")]  # default sigma only
    show = {"voi": "Greedy VoI (ablation)", "round_robin": "Round-robin", "random": "Random", "voi_lif": "VoI-LIF"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for met, ax, lab in [("total_travel_distance", axes[0], "total travel"),
                         ("overlap_ratio", axes[1], "overlap ratio")]:
        for md, name in show.items():
            sub = d[d["md"] == md].groupby("B")[met].mean()
            ax.plot(sub.index, sub.values, "-o", ms=4, label=name)
        ax.set_xlabel("per-step link budget B"); ax.set_ylabel(lab + " (lower=better)")
    axes[0].set_title("Targeted scheduling vs budget (N=8)"); axes[0].legend(fontsize=8)
    save(fig, "fig_bsweep")


def fig_learned():
    p = os.path.join(HERE, "results", "summary", "setcomm_learned_pivot.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    order = ["Learned", "VoI", "RoundRobin", "Random"]
    lab = {"Learned": "Learned (ours)", "VoI": "Greedy VoI", "RoundRobin": "Round-robin", "Random": "Random"}
    fig, ax = plt.subplots(figsize=(6, 4))
    Bs = [b for b in sorted(d["B"].unique()) if b <= 4]  # bandwidth-limited regime (paper scope)
    w = 0.2
    for i, m in enumerate(order):
        vals = [d[(d["B"] == b) & (d["nm"] == m)]["travel"].mean() for b in Bs]
        ax.bar(np.arange(len(Bs)) + i * w, vals, w, label=lab[m])
    ax.set_xticks(np.arange(len(Bs)) + 1.5 * w); ax.set_xticklabels([f"B={b}" for b in Bs])
    ax.set_ylabel("total travel (held-out maps, lower=better)")
    ax.set_title("Learned VoI scheduler vs baselines (bandwidth-limited regime)")
    ax.legend(fontsize=8); ax.set_ylim(bottom=850)
    save(fig, "fig_learned")


def fig_multiN():
    d = load("setcomm_genN")
    if d is None:
        return
    d["base"] = d["method_name"].str.replace(r"_N\d+$", "", regex=True)
    d["N"] = d["method_name"].str.extract(r"_N(\d+)$").astype(float)
    d["async"] = (~(d["base"].str.startswith("Periodic") | (d["base"] == "Original"))).astype(float)
    d["logc"] = np.log10(d["total_communication_volume"] + 1)
    Ns, eff, sig = [], [], []
    for N in sorted(d["N"].dropna().unique()):
        s = d[d["N"] == N].dropna(subset=["total_travel_distance", "logc"])
        md = pd.get_dummies(s["map_id"].astype(int), prefix="m", drop_first=True).astype(float)
        X = np.column_stack([np.ones(len(s)), s["logc"].values, s["async"].values] + [md[c].values for c in md.columns]).astype(float)
        y = s["total_travel_distance"].values.astype(float)
        beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta; dof = len(y) - rank; s2 = resid @ resid / dof
        se = np.sqrt(max(np.linalg.pinv(X.T @ X)[2, 2] * s2, 0)); t = beta[2] / se if se > 0 else 0
        p = 2 * st.t.sf(abs(t), dof)
        Ns.append(int(N)); eff.append(100 * beta[2] / y.mean()); sig.append(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(n) for n in Ns], eff, color=NATURE_BLUEPURPLE[1])
    for b, p in zip(bars, sig):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), "*" if p < 0.05 else "", ha="center", va="top")
    ax.axhline(0, color="k", lw=0.8); ax.set_xlabel("team size N"); ax.set_ylabel("async travel effect (% vs sync, matched comm)")
    ax.set_title("De-synchronisation advantage across N (* p<0.05)")
    save(fig, "fig_multiN")


def fig_c0():
    d = load("setcomm_c0")
    if d is None:
        return
    d["c0"] = d["method_name"].str.extract(r"_c([\d.]+)_").astype(float)
    d["B"] = d["method_name"].str.extract(r"_B(\d+)$").astype(int)
    fig, ax = plt.subplots(figsize=(6, 4))
    for B in sorted(d["B"].unique()):
        s = d[d["B"] == B].groupby("c0")["total_travel_distance"].mean()
        ax.plot(s.index, s.values, "-o", label=f"B={B}")
    ax.set_xlabel("LIF floor current c0"); ax.set_ylabel("total travel (lower=better)")
    ax.set_title("voi_lif: floor vs performance (smaller c0 -> better, still bounded staleness)")
    ax.legend(fontsize=9)
    save(fig, "fig_c0")


def fig_sync():
    p = os.path.join(HERE, "results", "summary", "setcomm_sync_pivot.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, name in [("Sync", "Synchronised"), ("Stag", "Staggered (de-sync)")]:
        s = d[d["tag"] == tag].sort_values("iv")
        ax.plot(s["iv"], s["travel"], "-o", label=name)
    ax.set_xlabel("period (interval)"); ax.set_ylabel("total travel (lower=better)")
    ax.set_title("Synchronisation isolation: same budget, only phase differs"); ax.legend()
    save(fig, "fig_sync")


def fig_drop():
    p = os.path.join(HERE, "results", "summary", "setcomm_drop2_pivot.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in ["Learned", "VoI", "RoundRobin", "Random"]:
        s = d[d["nm"] == m].sort_values("drop")
        ax.plot(s["drop"], s["travel"], "-o", label=m)
    ax.set_xlabel("packet-loss probability"); ax.set_ylabel("total travel (lower=better)")
    ax.set_title("Robustness to packet loss (N=8, B=4)"); ax.legend(fontsize=9)
    save(fig, "fig_drop")


if __name__ == "__main__":
    for f in [fig_pareto, fig_paretoN, fig_bsweep, fig_learned, fig_multiN, fig_c0, fig_sync, fig_drop]:
        try:
            f()
        except Exception as e:
            import traceback; print(f.__name__, "FAILED", e, traceback.format_exc()[-300:])
    print("FIGURES:", sorted(os.listdir(FIG)))
