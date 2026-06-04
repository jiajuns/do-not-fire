"""Cross-backbone de-synchronisation gradient figure (fig_generality).

Shows the de-sync travel effect (staggered vs synchronised periodic, interval 12, N=8) for three
planners ordered by how much they coordinate through the shared teammate belief:
  nearest-frontier (myopic)      -> de-sync HURTS  (+3.9%, p=0.011, clean n=120 from the pilot)
  utility-frontier (weak coord.) -> de-sync weak   (-1.2%, ns)
  DARS (learned, strong coord.)  -> de-sync STRONG (-11.3%, p=0.0002)

dars/futil come from setcomm_3bb; fnear from the deconfliction pilot (full n=120 at iv12, vs the
truncated fnear in 3bb). Reads _pro6000_pull, writes code/results/figures + copies to paper.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as st

PULL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_pro6000_pull", "results", "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "figures")
PAPER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "mdpi_template", "figures")
os.makedirs(OUT, exist_ok=True)

NATURE = ["#3B4CC0", "#7A4FA0", "#4C8DBE", "#9B5FB5", "#5C5FB8", "#8E6FBF"]
HURT = "#C45B5B"  # muted coral for the "de-sync hurts" bar
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.linewidth": 0.8, "axes.edgecolor": "#2b2b2b",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load_group(group):
    fs = glob.glob(os.path.join(PULL, group, "*.csv"))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    for c in ["total_travel_distance", "overlap_ratio", "map_id", "seed"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def paired_effect(d, mask_sync, mask_stag, metric="total_travel_distance"):
    S = d[mask_sync].set_index(["map_id", "seed"])
    G = d[mask_stag].set_index(["map_id", "seed"])
    idx = S.index.intersection(G.index)
    a = G.loc[idx, metric].values  # staggered
    c = S.loc[idx, metric].values  # synchronised
    try:
        _, p = st.wilcoxon(a, c)
    except Exception:
        p = float("nan")
    return 100.0 * (a.mean() - c.mean()) / c.mean(), p, len(idx)


def main():
    tbb = load_group("setcomm_3bb")
    tbb["bb"] = tbb.method_name.str.extract(r"^(dars|fnear|futil)_")
    tbb["tag"] = tbb.method_name.str.extract(r"_(Sync|Stag)_")
    tbb["iv"] = pd.to_numeric(tbb.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")

    dc = load_group("setcomm_dcpilot")
    dc["bb"] = dc.method_name.str.extract(r"^(fnear|futil)_")
    dc["w"] = dc.method_name.str.extract(r"_dc([0-9p]+)_")[0].str.replace("p", ".", regex=False)
    dc["tag"] = dc.method_name.str.extract(r"_(Sync|Stag)_")
    dc["iv"] = pd.to_numeric(dc.method_name.str.extract(r"_(\d+)$")[0], errors="coerce")

    rows = []
    # dars, futil from 3bb at iv12
    for bb in ["dars", "futil"]:
        e, p, n = paired_effect(tbb, (tbb.bb == bb) & (tbb.iv == 12) & (tbb.tag == "Sync"),
                                 (tbb.bb == bb) & (tbb.iv == 12) & (tbb.tag == "Stag"))
        rows.append((bb, e, p, n))
    # fnear from pilot, default deconfliction weight (dc1p0), iv12, full n
    e, p, n = paired_effect(dc, (dc.bb == "fnear") & (dc.w == "1.0") & (dc.iv == 12) & (dc.tag == "Sync"),
                            (dc.bb == "fnear") & (dc.w == "1.0") & (dc.iv == 12) & (dc.tag == "Stag"))
    rows.append(("fnear", e, p, n))

    res = {bb: (e, p, n) for bb, e, p, n in rows}
    print("per-backbone de-sync travel effect (iv12, neg=de-sync better):")
    for bb in ["fnear", "futil", "dars"]:
        e, p, n = res[bb]
        print(f"  {bb:6s}  {e:+6.2f}%   p={p:.4f}   n={n}")

    # ordered by reliance on shared teammate belief: myopic -> weak -> strong
    order = ["fnear", "futil", "dars"]
    labels = ["nearest-frontier\n(myopic)", "utility-frontier\n(weak coupling)", "DARS\n(learned, strong)"]
    effs = [res[b][0] for b in order]
    ps = [res[b][1] for b in order]
    ns = [res[b][2] for b in order]
    colors = [HURT if e > 0 else NATURE[0] for e in effs]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(order)), effs, color=colors, width=0.62)
    ax.axhline(0, color="k", lw=0.9)
    for i, (b, e, p, n) in enumerate(zip(bars, effs, ps, ns)):
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
        va = "bottom" if e > 0 else "top"
        off = 0.4 if e > 0 else -0.4
        ax.text(b.get_x() + b.get_width() / 2, e + off, f"{e:+.1f}%\n{star}", ha="center", va=va, fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("de-sync travel effect (%)\n(negative = de-sync better)")
    ax.set_title("De-synchronisation effect by planner (interval 12, $N=8$)")
    ax.set_ylim(min(effs) - 3.5, max(effs) + 7.0)
    ax.annotate("increasing use of shared teammate belief",
                xy=(2.35, max(effs) + 5.0), xytext=(-0.35, max(effs) + 5.0),
                ha="left", va="center", fontsize=7.5, color="#555",
                arrowprops=dict(arrowstyle="->", color="#999", lw=0.9))
    fig.tight_layout()
    for d in [OUT, PAPER]:
        fig.savefig(os.path.join(d, "fig_generality.png"), dpi=300)
        fig.savefig(os.path.join(d, "fig_generality.pdf"))
    plt.close(fig)
    print("saved fig_generality -> code/results/figures + paper/figures")


if __name__ == "__main__":
    main()
