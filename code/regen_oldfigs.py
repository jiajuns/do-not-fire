"""Regenerate ALL the paper's existing data figures in the unified house style, from local data.
Covers: fig_sync, fig_multiN, fig_generality, fig_prop2, fig_staleness, fig_breadth, fig_corr.
Run locally with `py regen_oldfigs.py` (data lives in _server_backup + _pro6000_pull). Each figure is
wrapped so one failure does not kill the rest. Writes code/results/figures + paper/figures.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st
import house_style as hs
hs.apply()
P = hs.PAL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PULL = os.path.join(ROOT, "_pro6000_pull", "results")
BK = os.path.join(ROOT, "_server_backup", "code", "results")
OUT = os.path.join(HERE, "results", "figures")
PAPER = os.path.join(ROOT, "paper", "mdpi_template", "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    for d in [OUT, PAPER]:
        fig.savefig(os.path.join(d, name + ".png"), dpi=300)
        fig.savefig(os.path.join(d, name + ".pdf"))
    plt.close(fig)
    print("saved", name)


def star(p):
    return "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")


def load_raw(base, group):
    fs = glob.glob(os.path.join(base, "raw", group, "*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for c in ["total_travel_distance", "overlap_ratio", "final_coverage_ratio",
              "total_communication_volume", "map_id", "seed"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def paired_effect(d, mask_sync, mask_stag, metric="total_travel_distance"):
    S = d[mask_sync].set_index(["map_id", "seed"])
    G = d[mask_stag].set_index(["map_id", "seed"])
    S = S[~S.index.duplicated()]; G = G[~G.index.duplicated()]
    idx = S.index.intersection(G.index)
    a = G.loc[idx, metric].values; c = S.loc[idx, metric].values
    try:
        _, p = st.wilcoxon(a, c)
    except Exception:
        p = float("nan")
    return 100.0 * (a.mean() - c.mean()) / c.mean(), p, len(idx)


# ---------------------------------------------------------------- fig_sync
def fig_sync():
    d = pd.read_csv(os.path.join(BK, "summary", "setcomm_sync_pivot.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.9))
    panels = [("travel", "total travel (lower better)", "(a) travel distance", 1.0),
              ("overlap", "sensing overlap (lower better)", "(b) sensing overlap", 1.0),
              ("comm", "realised volume (kB)", "(c) communication volume", 1e-3)]
    for k, (col, ylab, title, sc) in enumerate(panels):
        ser = {}
        for tag, name, c in [("Sync", "synchronised", P["sync"]), ("Stag", "even-staggered", P["desync"])]:
            s = d[d["tag"] == tag].sort_values("iv")
            ax[k].plot(s["iv"], s[col] * sc, "-o", color=c, ms=6, label=name)
            ser[tag] = s.set_index("iv")[col] * sc
        iv = sorted(set(ser["Sync"].index) & set(ser["Stag"].index))
        sy = ser["Sync"].reindex(iv).values; sg = ser["Stag"].reindex(iv).values
        if col != "comm":
            ax[k].fill_between(iv, sg, sy, where=sy >= sg, color=P["desync"], alpha=0.10)
        else:
            ax[k].fill_between(iv, sy, sg, where=sg >= sy, color=P["accent"], alpha=0.14)
            ax[k].annotate("de-sync transmits\n$\\geq$ bytes, yet wins", xy=(0.97, 0.92),
                           xycoords="axes fraction", ha="right", va="top", fontsize=8.5, color=P["accent"])
        ax[k].set_xlabel("broadcast period $T$"); ax[k].set_title(title); ax[k].set_ylabel(ylab)
    ax[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Synchronisation is the controllable cost: same broadcast rate, only the phase differs",
                 y=1.04, fontsize=12)
    save(fig, "fig_sync")


# ---------------------------------------------------------------- fig_multiN
def fig_multiN():
    # phase-only paired control (Sync vs Stag, interval 12) across team size, from syncN raw
    d = load_raw(PULL, "setcomm_syncN")
    d["N"] = pd.to_numeric(d.method_name.str.extract(r"N(\d+)_")[0], errors="coerce")
    d["tag"] = d.method_name.str.extract(r"_(Sync|Stag)_")[0]
    d["iv"] = pd.to_numeric(d.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")
    d = d[d.iv == 12]
    Ns = sorted(int(n) for n in d["N"].dropna().unique())
    tr, ov, ptr, pov = [], [], [], []
    for N in Ns:
        sub = d[d.N == N]
        S = sub[sub.tag == "Sync"].set_index(["map_id", "seed"])
        G = sub[sub.tag == "Stag"].set_index(["map_id", "seed"])
        S = S[~S.index.duplicated()]; G = G[~G.index.duplicated()]
        idx = S.index.intersection(G.index)
        for met, arrp, arrpp in [("total_travel_distance", tr, ptr), ("overlap_ratio", ov, pov)]:
            a = G.loc[idx, met].values; c = S.loc[idx, met].values
            arrp.append(100 * (a.mean() - c.mean()) / c.mean())
            try:
                _, p = st.wilcoxon(a, c)
            except Exception:
                p = 1.0
            arrpp.append(p)
    x = np.arange(len(Ns)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    b1 = ax.bar(x - w / 2, tr, w, label="total travel", color=P["desync"], edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w / 2, ov, w, label="sensing overlap", color=P["purple"], edgecolor="white", linewidth=0.8)
    for bars, vals, ps in [(b1, tr, ptr), (b2, ov, pov)]:
        for bar, v, p in zip(bars, vals, ps):
            ax.text(bar.get_x() + bar.get_width() / 2, v - 0.25, star(p), ha="center", va="top", fontsize=7.5)
    ax.axhline(0, color=P["ink"], lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("team size $N$")
    ax.set_ylabel("reduction vs synchronised (%)  (negative = de-sync better)")
    ax.set_title("De-synchronisation advantage across team sizes\n(phase-only control, matched bandwidth, interval $12$; $*\\,p<0.05$, $**\\,p<0.01$)")
    lo = min(min(tr), min(ov))
    ax.set_ylim(lo - abs(lo) * 0.22 - 1.5, max(2, max(max(tr), max(ov)) + 1.5))
    ax.legend(loc="lower left")
    save(fig, "fig_multiN")


# ---------------------------------------------------------------- fig_generality
def fig_generality():
    tbb = load_raw(PULL, "setcomm_3bb")
    tbb["bb"] = tbb.method_name.str.extract(r"^(dars|fnear|futil)_")
    tbb["tag"] = tbb.method_name.str.extract(r"_(Sync|Stag)_")
    tbb["iv"] = pd.to_numeric(tbb.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")
    dc = load_raw(PULL, "setcomm_dcpilot")
    dc["bb"] = dc.method_name.str.extract(r"^(fnear|futil)_")
    dc["w"] = dc.method_name.str.extract(r"_dc([0-9p]+)_")[0].str.replace("p", ".", regex=False)
    dc["tag"] = dc.method_name.str.extract(r"_(Sync|Stag)_")
    dc["iv"] = pd.to_numeric(dc.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")
    rng = np.random.default_rng(0)

    def boot_ci(d, mask_sync, mask_stag, metric="total_travel_distance", nb=3000):
        S = d[mask_sync].set_index(["map_id", "seed"]); G = d[mask_stag].set_index(["map_id", "seed"])
        S = S[~S.index.duplicated()]; G = G[~G.index.duplicated()]
        idx = S.index.intersection(G.index)
        a = G.loc[idx, metric].values; c = S.loc[idx, metric].values
        base = c.mean(); pct = 100 * (a.mean() - base) / base; diff = a - c
        bs = np.array([100 * diff[rng.integers(0, len(diff), len(diff))].mean() / base for _ in range(nb)])
        try:
            _, p = st.wilcoxon(a, c)
        except Exception:
            p = float("nan")
        return pct, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), p

    res = {}
    for bb in ["dars", "futil"]:
        res[bb] = boot_ci(tbb, (tbb.bb == bb) & (tbb.iv == 12) & (tbb.tag == "Sync"),
                          (tbb.bb == bb) & (tbb.iv == 12) & (tbb.tag == "Stag"))
    res["fnear"] = boot_ci(dc, (dc.bb == "fnear") & (dc.w == "1.0") & (dc.iv == 12) & (dc.tag == "Sync"),
                           (dc.bb == "fnear") & (dc.w == "1.0") & (dc.iv == 12) & (dc.tag == "Stag"))
    order = ["fnear", "futil", "dars"]
    labels = ["nearest-frontier\n(myopic)", "utility-frontier\n(weak coupling)", "DARS\n(learned, strong)"]
    effs = [res[b][0] for b in order]; ps = [res[b][3] for b in order]
    err = [[res[b][0] - res[b][1] for b in order], [res[b][2] - res[b][0] for b in order]]
    colors = [P["accent"] if e > 0 else P["desync"] for e in effs]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    bars = ax.bar(range(len(order)), effs, color=colors, width=0.62, edgecolor="white", linewidth=0.8,
                  yerr=err, capsize=5, error_kw=dict(ecolor=P["ink"], lw=1.2))
    ax.axhline(0, color=P["ink"], lw=1.0)
    for k, b in enumerate(bars):
        e = effs[k]; hi = res[order[k]][2]; lo = res[order[k]][1]
        if e > 0:
            ytext = hi + 0.6; va = "bottom"
        else:
            ytext = lo - 0.6; va = "top"
        ax.text(b.get_x() + b.get_width() / 2, ytext, f"{e:+.1f}%", ha="center", va=va, fontsize=8.5)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels)
    ax.set_ylabel("de-sync travel effect (%)  (negative = de-sync better)")
    ax.set_title("De-synchronisation pays off in proportion to coordination\n(interval 12, $N=8$; bars are 95% bootstrap CIs)")
    lo_all = min(res[b][1] for b in order); hi_all = max(res[b][2] for b in order)
    ax.set_ylim(lo_all - 4.0, hi_all + 7.5)
    ya = hi_all + 5.0
    ax.annotate("increasing use of shared teammate belief",
                xy=(2.35, ya), xytext=(-0.35, ya),
                ha="left", va="center", fontsize=8, color=P["gray"],
                arrowprops=dict(arrowstyle="->", color=P["gray"], lw=1.0))
    save(fig, "fig_generality")


# ---------------------------------------------------------------- fig_prop2
def fig_prop2():
    N = 8
    def peak_aoi(tag, T):
        if tag == "Sync":
            return N * (T - 1)
        phi = [round(k * T / N) for k in range(N)]
        return max(sum((t - p) % T for p in phi) for t in range(T))
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(BK, "raw", "setcomm_sync", "*.csv"))],
                  ignore_index=True)
    d["overlap_ratio"] = pd.to_numeric(d["overlap_ratio"], errors="coerce")
    d["tag"] = d.method_name.str.replace(r"_\d+$", "", regex=True)
    d["iv"] = pd.to_numeric(d.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")
    rows = []
    for (tag, iv), g in d.groupby(["tag", "iv"]):
        rows.append(dict(tag=tag, iv=int(iv), peak=peak_aoi(tag, int(iv)), ov=float(g.overlap_ratio.mean())))
    r = pd.DataFrame(rows)
    rho, pv = st.spearmanr(r.peak, r.ov)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    z = np.polyfit(r.peak, r.ov, 1); xs = np.linspace(r.peak.min(), r.peak.max(), 50)
    ax.plot(xs, np.polyval(z, xs), "--", color=P["gray"], lw=1.4, zorder=2)
    for tag, c, lab, off in [("Sync", P["sync"], "synchronised", (7, 5)),
                             ("Stag", P["desync"], "staggered (de-sync)", (7, -13))]:
        s = r[r.tag == tag]
        ax.scatter(s.peak, s.ov, c=c, s=70, label=lab, zorder=3, edgecolor="white", linewidth=1.0)
        for _, row in s.iterrows():
            ax.annotate(f"$T={row.iv}$", (row.peak, row.ov), textcoords="offset points",
                        xytext=off, fontsize=7.5, color=c)
    ax.set_xlabel(r"peak team Age-of-Information  $\max_t\sum_i\tau_i$  (steps)")
    ax.set_ylabel("sensing overlap (lower is better)")
    ax.set_title(f"Overlap rises with peak AoI (Prop. 2; Spearman $\\rho={rho:.2f}$, $p<10^{{-3}}$)")
    ax.legend(loc="upper left")
    save(fig, "fig_prop2")


# ---------------------------------------------------------------- fig_staleness
def fig_staleness():
    T = 12; Ns = [4, 6, 8, 10, 12]
    peak_sync, peak_stag, mean_both = [], [], []
    for N in Ns:
        t = np.arange(T)
        Ss = np.array([(int(tt) % T) * N for tt in t], dtype=float)
        phi = [round(i * T / N) for i in range(N)]
        Sg = np.array([sum((int(tt) - p) % T for p in phi) for tt in t], dtype=float)
        peak_sync.append(Ss.max()); peak_stag.append(Sg.max()); mean_both.append(Ss.mean())
    x = np.arange(len(Ns)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(x - w / 2, peak_sync, w, label="synchronised (peak)", color=P["sync"], edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, peak_stag, w, label="even-staggered (peak)", color=P["desync"], edgecolor="white", linewidth=0.8)
    ax.plot(x, mean_both, "o--", color=P["ink"], lw=1.4, ms=5, label="mean (identical)")
    for i in range(len(Ns)):
        red = 100 * (peak_stag[i] - peak_sync[i]) / peak_sync[i]
        ax.text(x[i] + w / 2, peak_stag[i] - 3, f"{red:.0f}%", ha="center", va="top", fontsize=7.5, color="white", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"$N={n}$" for n in Ns])
    ax.set_ylabel(r"team staleness $\sum_i \tau_i$  (steps)")
    ax.set_title("De-synchronisation collapses the staleness peak at constant mean\n(broadcast period $T=12$)")
    ax.legend(loc="upper left")
    save(fig, "fig_staleness")


# ---------------------------------------------------------------- fig_breadth
def fig_breadth():
    pl = pd.read_csv(os.path.join(PULL, "summary", "ploss_full.csv"))
    sr = pd.read_csv(os.path.join(PULL, "summary", "srange_full.csv"))
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.8))
    specs = [
        (ax[0, 0], pl, "total",   P["desync"], "packet-loss probability (%)", "(a) travel vs packet loss",   100),
        (ax[0, 1], pl, "overlap", P["purple"], "packet-loss probability (%)", "(b) overlap vs packet loss",  100),
        (ax[1, 0], sr, "total",   P["desync"], "sensor range (m)",            "(c) travel vs sensor range",   1),
        (ax[1, 1], sr, "overlap", P["purple"], "sensor range (m)",            "(d) overlap vs sensor range",  1),
    ]
    for a, df, met, col, xl, title, xs in specs:
        s = df[df.metric == met].sort_values("key")
        x = s.key.values * xs; y = s.desync_pct.values; ps = s.p.values
        a.plot(x, y, "-o", color=col, ms=6)
        a.fill_between(x, y, 0, where=(y <= 0), color=col, alpha=0.10)
        for xi, yi, pi in zip(x, y, ps):
            a.annotate(star(pi), (xi, yi), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7.5)
        a.axhline(0, color=P["ink"], lw=1.0); a.set_xlabel(xl); a.set_title(title)
        a.set_ylim(min(y) - 3.5, 2.5)
    ax[0, 0].set_ylabel("de-sync effect (%)\n(negative = better)")
    ax[1, 0].set_ylabel("de-sync effect (%)\n(negative = better)")
    fig.suptitle("De-synchronisation benefit is robust across packet loss and sensor range "
                 "($N=8$, interval $12$; $*\\,p<0.05$, $**\\,p<0.01$)", y=1.01, fontsize=11.5)
    save(fig, "fig_breadth")


# ---------------------------------------------------------------- fig_corr
def fig_corr():
    c = pd.read_csv(os.path.join(PULL, "summary", "setcomm_corr_paired.csv"))
    order = ["dispersed", "clustered"]
    labels = ["dispersed\n(low corr.)", "clustered\n(high corr.)"]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.1))
    for k, (met, title) in enumerate([("total", "(a) total travel"), ("overlap", "(b) sensing overlap")]):
        ct = c[c.metric == met]
        vals = [float(ct[ct.spawn == s]["desync_pct"].iloc[0]) for s in order]
        ps = [float(ct[ct.spawn == s]["p"].iloc[0]) for s in order]
        bars = ax[k].bar(range(2), vals, color=[P["gray"], P["desync"]], width=0.55, edgecolor="white", linewidth=0.8)
        ax[k].axhline(0, color=P["ink"], lw=1.0)
        for b, v, p in zip(bars, vals, ps):
            ax[k].text(b.get_x() + b.get_width() / 2, v - 0.5, f"{v:.1f}%\n{star(p)}", ha="center", va="top", fontsize=8.5)
        ax[k].set_xticks(range(2)); ax[k].set_xticklabels(labels); ax[k].set_title(title)
        ax[k].set_ylim(min(vals) - 3.5, 1.5)
    ax[0].set_ylabel("de-sync effect (%)  (negative = better)")
    fig.suptitle("De-synchronisation benefit grows with the team's information correlation", y=1.03, fontsize=11.5)
    save(fig, "fig_corr")


if __name__ == "__main__":
    for f in [fig_sync, fig_multiN, fig_generality, fig_prop2, fig_staleness, fig_breadth, fig_corr]:
        try:
            f()
        except Exception as e:
            import traceback
            print(f.__name__, "FAILED:", e, traceback.format_exc()[-400:])
    print("DONE; figures in", PAPER)
