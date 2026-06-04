#!/usr/bin/env python3
"""Analysis for SET-Comm experiments: per-method summary, communication-vs-coverage
Pareto front, paired significance tests (bootstrap CI + Wilcoxon), and a Pareto plot.

Reads the per-episode raw CSVs written by test_driver.py / experiment_logger.py
(results/raw/<group>/*.csv). Pure pandas/scipy/matplotlib — no simulator needed.

Example:
    python analyze_setcomm.py --group setcomm_full --baseline Periodic_5
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

COV = "final_coverage_ratio"
COMM = "total_communication_volume"
EFF = "total_travel_distance"  # efficiency axis: comm mainly affects travel/overlap, not final coverage

SUMMARY_METRICS = [
    COV, COMM, "communication_count", "total_travel_distance",
    "episode_steps", "overlap_ratio", "mean_broadcast_interval",
]


def load_group(input_dir):
    files = [f for f in glob.glob(os.path.join(input_dir, "*.csv"))]
    if not files:
        raise FileNotFoundError(f"no per-episode CSVs in {input_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    for c in SUMMARY_METRICS + ["seed", "map_id", "success"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize(df):
    rows = []
    for method, g in df.groupby("method_name"):
        row = {"method_name": method, "n": len(g)}
        for m in SUMMARY_METRICS:
            if m in g:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"{COMM}_mean").reset_index(drop=True)


def pareto_front(summary):
    """Mark methods on the comm-vs-travel front (minimize communication AND travel).

    Under position-gating coverage saturates, so the real tradeoff is communication
    vs exploration efficiency (travel distance / overlap). Lower-left is better.
    """
    pts = summary[["method_name", f"{COMM}_mean", f"{EFF}_mean"]].to_numpy()
    on_front = []
    for i, (_, ci, ti) in enumerate(pts):
        dominated = False
        for j, (_, cj, tj) in enumerate(pts):
            if j == i:
                continue
            if cj <= ci and tj <= ti and (cj < ci or tj < ti):
                dominated = True
                break
        on_front.append(not dominated)
    summary = summary.copy()
    summary["on_pareto_front"] = on_front
    return summary


def _paired(df, baseline, target, metric):
    """Pair episodes of two methods on (seed, map_id); return aligned arrays."""
    keys = ["seed", "map_id"]
    b = df[df.method_name == baseline].groupby(keys)[metric].mean()
    t = df[df.method_name == target].groupby(keys)[metric].mean()
    j = pd.concat([b.rename("b"), t.rename("t")], axis=1).dropna()
    return j["b"].to_numpy(), j["t"].to_numpy()


def bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    if len(diff) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_stats(df, baseline, targets, metric):
    try:
        from scipy.stats import wilcoxon
        have_scipy = True
    except Exception:
        have_scipy = False
    rows = []
    for tgt in targets:
        b, t = _paired(df, baseline, tgt, metric)
        diff = t - b
        lo, hi = bootstrap_ci(diff)
        p = np.nan
        if have_scipy and len(diff) > 0 and np.any(diff != 0):
            try:
                p = float(wilcoxon(t, b).pvalue)
            except Exception:
                p = np.nan
        rows.append({
            "metric": metric, "baseline": baseline, "target": tgt, "n_pairs": len(diff),
            "mean_baseline": float(b.mean()) if len(b) else np.nan,
            "mean_target": float(t.mean()) if len(t) else np.nan,
            "mean_diff": float(diff.mean()) if len(diff) else np.nan,
            "ci95_low": lo, "ci95_high": hi, "wilcoxon_p": p,
        })
    return pd.DataFrame(rows)


def plot_pareto(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for _, r in summary.iterrows():
        name = r["method_name"]
        x, y = r[f"{COMM}_mean"], r[f"{EFF}_mean"]
        if name.startswith("SETComm_L"):
            marker, color = "*", "tab:red"
        elif name.startswith("SETComm"):
            marker, color = "o", "tab:blue"
        elif name.startswith("Random"):
            marker, color = "s", "tab:gray"
        else:
            marker, color = "^", "tab:green"  # periodic / Original
        ax.scatter(x, y, marker=marker, color=color, s=60, zorder=3)
        ax.annotate(name, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("total communication volume (bytes)")
    ax.set_ylabel("total travel distance (lower = more efficient)")
    ax.set_title("Communication vs exploration efficiency (lower-left better)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Analyze SET-Comm results.")
    ap.add_argument("--group", default="setcomm_full")
    ap.add_argument("--input-dir", default=None, help="override results/raw/<group>")
    ap.add_argument("--out-dir", default=None, help="override results/summary/<group>")
    ap.add_argument("--baseline", default="Periodic_5")
    args = ap.parse_args()

    input_dir = args.input_dir or os.path.join(PROJECT_ROOT, "results", "raw", args.group)
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, "results", "summary", args.group)
    os.makedirs(out_dir, exist_ok=True)

    df = load_group(input_dir)
    summary = pareto_front(summarize(df))
    summary.to_csv(os.path.join(out_dir, f"{args.group}_setcomm_summary.csv"), index=False)

    methods = [m for m in df.method_name.unique() if m != args.baseline]
    metrics = [COMM, EFF, "overlap_ratio", "episode_steps", COV]
    metrics = [m for m in metrics if m in df.columns]
    stats = pd.concat([paired_stats(df, args.baseline, methods, m) for m in metrics],
                      ignore_index=True)
    stats.to_csv(os.path.join(out_dir, f"{args.group}_paired_stats.csv"), index=False)

    plot_pareto(summary, os.path.join(out_dir, f"{args.group}_pareto"))

    print(summary.to_string(index=False))
    print("\nPaired vs", args.baseline)
    print(stats.to_string(index=False))
    print(f"\n|#Wrote summary + stats + pareto -> {out_dir}")


if __name__ == "__main__":
    main()
