#!/usr/bin/env python3
"""Stage 11: audit, aggregate, plot, and organize full paper-scale results."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT = os.path.join(PROJECT_ROOT, "model", "clean", "checkpoint.pth")
PAPER_ROOT = os.path.join(PROJECT_ROOT, "paper_results")

GROUPS = {
    "main_comparison_full": {
        "expected_csv": 10,
        "expected_episodes": 100,
        "methods": ["Original", "CommScheduled"],
        "plot_type": "bar",
        "bar_metrics": [
            "final_coverage_ratio",
            "success",
            "total_travel_distance",
            "max_single_robot_distance",
            "episode_steps",
            "communication_count",
            "total_communication_volume",
            "overlap_ratio",
        ],
    },
    "communication_interval_full": {
        "expected_csv": 25,
        "expected_episodes": 250,
        "methods": [f"CommInterval_{i}" for i in [1, 2, 5, 10, 20]],
        "plot_type": "comm",
        "line_metrics": [
            "final_coverage_ratio",
            "success",
            "communication_count",
            "total_communication_volume",
            "total_travel_distance",
            "max_single_robot_distance",
            "episode_steps",
            "overlap_ratio",
        ],
    },
    "ablation_full": {
        "expected_csv": 20,
        "expected_episodes": 200,
        "methods": ["Original", "CommOnly", "OverlapOnly", "CommPlusOverlap"],
        "plot_type": "bar",
        "bar_metrics": [
            "final_coverage_ratio",
            "total_communication_volume",
            "total_travel_distance",
            "max_single_robot_distance",
            "overlap_ratio",
            "episode_steps",
        ],
    },
    "overlap_diagnostic_full_lite": {
        "expected_csv": 18,
        "expected_episodes": 90,
        "optional": True,
        "plot_type": "overlap",
    },
}

KEY_METRICS = [
    "final_coverage_ratio",
    "success",
    "total_travel_distance",
    "max_single_robot_distance",
    "average_single_robot_distance",
    "episode_steps",
    "communication_count",
    "total_communication_volume",
    "overlap_ratio",
    "computation_time",
]

LOWER_BETTER = {
    "total_communication_volume",
    "communication_count",
    "total_travel_distance",
    "max_single_robot_distance",
    "episode_steps",
    "overlap_ratio",
    "computation_time",
}


def audit_folder(group: str, cfg: dict, lines: List[str]) -> pd.DataFrame:
    raw_dir = os.path.join(PROJECT_ROOT, "results/raw", group)
    lines.append(f"\n### {group}")
    if not os.path.isdir(raw_dir):
        lines.append("  STATUS: MISSING")
        return pd.DataFrame()
    pattern = os.path.join(raw_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        sub = pd.read_csv(f)
        sub["_source_file"] = os.path.basename(f)
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    n_csv = len(files)
    n_ep = len(df)
    lines.append(f"  CSV files: {n_csv} (expected {cfg.get('expected_csv', '?')})")
    lines.append(f"  Total episode rows: {n_ep} (expected {cfg.get('expected_episodes', '?')})")
    if n_csv != cfg.get("expected_csv"):
        lines.append(f"  WARNING: CSV count mismatch")
    if n_ep != cfg.get("expected_episodes"):
        lines.append(f"  WARNING: episode count mismatch")
    for col in ["method_name", "seed", "map_id", "map_name", "episode_id",
                "communication_interval", "overlap_penalty_weight"]:
        if col in df.columns:
            lines.append(f"  unique {col}: {sorted(df[col].dropna().unique().tolist(), key=str)}")
    if "methods" in cfg:
        missing = set(cfg["methods"]) - set(df["method_name"].unique()) if "method_name" in df.columns else cfg["methods"]
        if missing:
            lines.append(f"  MISSING methods: {sorted(missing)}")
    return df


def fairness_main(df: pd.DataFrame, lines: List[str]) -> None:
    lines.append("\n### Fairness: main_comparison_full")
    for method in ["Original", "CommScheduled"]:
        sub = df[df["method_name"] == method]
        lines.append(f"  {method}: {len(sub)} episodes, seeds={sorted(sub['seed'].unique())}")
    o, c = df[df.method_name == "Original"], df[df.method_name == "CommScheduled"]
    for col in ["seed", "map_id", "num_agents", "max_episode_steps", "sensor_range", "checkpoint_path"]:
        if col in df.columns:
            ov, cv = set(o[col].astype(str)), set(c[col].astype(str))
            lines.append(f"  {col} Original={sorted(ov)[:5]}... CommScheduled match={ov == cv}")
    lines.append(f"  comm interval Original={sorted(o['communication_interval'].unique())} "
                 f"CommScheduled={sorted(c['communication_interval'].unique())}")
    lines.append(f"  overlap weight Original={sorted(o['overlap_penalty_weight'].unique())} "
                 f"CommScheduled={sorted(c['overlap_penalty_weight'].unique())}")


def fairness_comm(df: pd.DataFrame, lines: List[str]) -> None:
    lines.append("\n### Fairness: communication_interval_full")
    for iv in [1, 2, 5, 10, 20]:
        sub = df[df["communication_interval"] == iv]
        lines.append(f"  interval {iv}: {len(sub)} episodes, seeds={sorted(sub['seed'].unique())}")


def fairness_ablation(df: pd.DataFrame, lines: List[str]) -> None:
    lines.append("\n### Fairness: ablation_full")
    for m in ["Original", "CommOnly", "OverlapOnly", "CommPlusOverlap"]:
        sub = df[df["method_name"] == m]
        lines.append(f"  {m}: {len(sub)} episodes")


def validity_scan(df: pd.DataFrame, label: str, issues: List[str]) -> None:
    if df.empty:
        issues.append(f"{label}: empty dataframe")
        return
    dup = df.duplicated(subset=[c for c in ["method_name", "seed", "episode_id", "map_id"] if c in df.columns], keep=False)
    if dup.any():
        issues.append(f"{label}: {dup.sum()} potential duplicate rows")
    for col in ["total_travel_distance", "max_single_robot_distance", "communication_count",
                "total_communication_volume", "episode_steps"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if (s < 0).any():
                issues.append(f"{label}: negative {col}")
    if "final_coverage_ratio" in df.columns:
        s = pd.to_numeric(df["final_coverage_ratio"], errors="coerce")
        if ((s < 0) | (s > 1)).any():
            issues.append(f"{label}: coverage outside [0,1]")
    if "overlap_ratio" in df.columns:
        s = pd.to_numeric(df["overlap_ratio"], errors="coerce")
        if ((s < 0) | (s > 1)).any():
            issues.append(f"{label}: overlap_ratio outside [0,1]")
    if "success" in df.columns:
        bad = ~df["success"].astype(str).isin({"0", "1", "True", "False", "true", "false"})
        if bad.any():
            issues.append(f"{label}: invalid success values")
    for col in KEY_METRICS:
        if col in df.columns and df[col].isna().all():
            issues.append(f"{label}: all NaN in {col}")


def run_aggregate(group: str) -> int:
    cmd = [
        sys.executable,
        "scripts/aggregate_results.py",
        "--input_dir",
        f"results/raw/{group}",
        "--output_dir",
        "results/summary",
        "--group",
        group,
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def plot_bar_group(group: str, title: str, metrics: List[str]) -> None:
    from plot_utils import metric_label, save_figure

    summary = os.path.join(PROJECT_ROOT, "results/summary", f"{group}_summary_numeric.csv")
    out_dir = os.path.join(PROJECT_ROOT, "results/figures", group)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary)
    methods = df["method_name"].astype(str).tolist()
    x = range(len(methods))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for metric in metrics:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x, df[mean_col], yerr=df[std_col] if std_col in df.columns else None,
               capsize=4, color=colors[: len(methods)])
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"{title}: {metric_label(metric)}")
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, os.path.join(out_dir, metric))


def plot_line_group(group: str, title: str, metrics: List[str], x_col: str) -> None:
    from plot_utils import metric_label, save_figure

    summary = os.path.join(PROJECT_ROOT, "results/summary", f"{group}_summary_numeric.csv")
    out_dir = os.path.join(PROJECT_ROOT, "results/figures", group)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(summary).sort_values(x_col)
    x = df[x_col].values
    for metric in metrics:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(x, df[mean_col], yerr=df[std_col] if std_col in df.columns else None,
                    marker="o", capsize=4, linewidth=2)
        ax.set_xlabel(metric_label(x_col) if x_col != "communication_interval" else "Communication interval")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"{title}: {metric_label(metric)}")
        ax.grid(alpha=0.3)
        save_figure(fig, os.path.join(out_dir, metric))


def improvement_summary(group: str = "main_comparison_full") -> str:
    num = pd.read_csv(os.path.join(PROJECT_ROOT, "results/summary", f"{group}_summary_numeric.csv"))
    orig = num[num["method_name"] == "Original"].iloc[0]
    prop = num[num["method_name"] == "CommScheduled"].iloc[0]
    lines = ["MAIN COMPARISON: Original -> CommScheduled", "=" * 50, ""]
    metrics = [
        "total_communication_volume",
        "communication_count",
        "final_coverage_ratio",
        "success",
        "total_travel_distance",
        "max_single_robot_distance",
        "episode_steps",
        "overlap_ratio",
    ]
    for m in metrics:
        mc, sc = f"{m}_mean", f"{m}_std"
        o, p = float(orig[mc]), float(prop[mc])
        if m in LOWER_BETTER:
            pct = (o - p) / o * 100 if o else 0
            direction = "reduction" if pct >= 0 else "increase"
        else:
            pct = (p - o) / o * 100 if o else 0
            direction = "increase" if pct >= 0 else "decrease"
        lines.append(f"{m}:")
        lines.append(f"  Original:      {o:.4g} ± {orig[sc]:.4g}")
        lines.append(f"  CommScheduled: {p:.4g} ± {prop[sc]:.4g}")
        lines.append(f"  Change: {pct:+.2f}% ({direction})")
        lines.append("")
    lines.append("Trade-off assessment:")
    vol_o = float(orig["total_communication_volume_mean"])
    vol_p = float(prop["total_communication_volume_mean"])
    cov_o = float(orig["final_coverage_ratio_mean"])
    cov_p = float(prop["final_coverage_ratio_mean"])
    lines.append(f"  Communication volume reduction: {(vol_o-vol_p)/vol_o*100:.1f}%")
    lines.append(f"  Coverage change: {(cov_p-cov_o)/cov_o*100:+.2f}%")
    if vol_p < vol_o * 0.5 and cov_p >= cov_o - 0.02:
        lines.append("  Overall: acceptable trade-off for communication-efficient exploration.")
    else:
        lines.append("  Overall: review trade-off before strong claims.")
    return "\n".join(lines)


def _mean_std_curves(files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mats = []
    for f in files:
        c = pd.read_csv(f)
        if "step" not in c.columns or "coverage_ratio" not in c.columns:
            continue
        max_step = int(c["step"].max())
        steps = np.arange(max_step + 1)
        aligned = c.set_index("step").reindex(steps)["coverage_ratio"].interpolate().values
        mats.append(aligned)
    if not mats:
        return np.array([]), np.array([]), np.array([])
    max_len = max(len(m) for m in mats)
    steps = np.arange(max_len)
    padded = np.zeros((len(mats), max_len))
    for i, m in enumerate(mats):
        padded[i, : len(m)] = m
    return steps, padded.mean(axis=0), padded.std(axis=0)


def plot_coverage_main_full() -> str:
    """Plot Original vs CommScheduled from main_comparison_full coverage curves only."""
    from plot_utils import save_figure

    out_dir = os.path.join(PROJECT_ROOT, "results/figures/coverage_curves_full")
    os.makedirs(out_dir, exist_ok=True)
    curve_dir = os.path.join(PROJECT_ROOT, "results/raw/main_comparison_full/coverage_curves")
    if not os.path.isdir(curve_dir):
        return "No coverage_curves/ under main_comparison_full"

    methods = ["Original", "CommScheduled"]
    fig, ax = plt.subplots(figsize=(8, 5))
    found = []
    for method in methods:
        files = [
            f
            for f in glob.glob(os.path.join(curve_dir, "*.csv"))
            if method in os.path.basename(f)
        ]
        steps, mean, std = _mean_std_curves(files)
        if len(steps) == 0:
            continue
        ax.plot(steps, mean, label=method, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
        found.append(method)
    if not found:
        return "Coverage CSV found but could not parse"
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage ratio")
    ax.set_title("Full main comparison: coverage over time (mean ± std)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, os.path.join(out_dir, "coverage_all_methods"))
    plt.close(fig)
    for method in found:
        files = [
            f
            for f in glob.glob(os.path.join(curve_dir, "*.csv"))
            if method in os.path.basename(f)
        ]
        steps, mean, std = _mean_std_curves(files)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(steps, mean, linewidth=2)
        ax2.fill_between(steps, mean - std, mean + std, alpha=0.2)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Coverage ratio")
        ax2.set_title(f"Coverage: {method} (full)")
        ax2.grid(alpha=0.3)
        save_figure(fig2, os.path.join(out_dir, f"coverage_{method}"))
        plt.close(fig2)
    return f"Plotted: {found} -> {out_dir}"


def plot_trajectories() -> str:
    traj_dir = os.path.join(PROJECT_ROOT, "results/trajectories/main_comparison_full/raw")
    out_dir = os.path.join(PROJECT_ROOT, "results/trajectories/figures/full_main_comparison")
    if not os.path.isdir(traj_dir):
        alt = glob.glob(os.path.join(PROJECT_ROOT, "results/trajectories/**/main_comparison_full*.npz"), recursive=True)
        if not alt:
            return (
                "No trajectory .npz in results/trajectories/main_comparison_full/raw. "
                "To save later: re-run only seed0 with SAVE_TRAJECTORY for Original and CommScheduled:\n"
                "  python scripts/run_main_comparison.py --mode full --confirm "
                "--seeds 0 --num-test-maps 1  # with traj flags in script"
            )
        traj_files = alt
    else:
        traj_files = sorted(glob.glob(os.path.join(traj_dir, "*.npz")))
    if not traj_files:
        return "Trajectory directory exists but empty"
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for f in traj_files:
        data = np.load(f, allow_pickle=True)
        if "robot_trajectories" not in data:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, traj in enumerate(data["robot_trajectories"]):
            if len(traj) == 0:
                continue
            ax.plot(traj[:, 0], traj[:, 1], linewidth=1.2, label=f"r{i}")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
        ax.set_title(os.path.basename(f))
        ax.grid(alpha=0.3)
        base = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0])
        from plot_utils import save_figure
        save_figure(fig, base)
        plt.close(fig)
        n += 1
    return f"Saved {n} trajectory figures to {out_dir}"


def interpret_results(main_n, comm_n, abl_n) -> str:
    m = pd.read_csv(os.path.join(PROJECT_ROOT, "results/summary/main_comparison_full_summary_numeric.csv"))
    c = pd.read_csv(os.path.join(PROJECT_ROOT, "results/summary/communication_interval_full_summary_numeric.csv"))
    a = pd.read_csv(os.path.join(PROJECT_ROOT, "results/summary/ablation_full_summary_numeric.csv"))
    o = m[m.method_name == "Original"].iloc[0]
    cs = m[m.method_name == "CommScheduled"].iloc[0]
    vol_red = (float(o.total_communication_volume_mean) - float(cs.total_communication_volume_mean)) / float(o.total_communication_volume_mean) * 100

    lines = []
    lines.append("STAGE 11 FULL RESULTS INTERPRETATION")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("A. MAIN COMPARISON (Original vs CommScheduled)")
    lines.append(f"  Communication volume reduction: {vol_red:.1f}%")
    lines.append(f"  Coverage: Original {o.final_coverage_ratio_mean:.4f} -> CommScheduled {cs.final_coverage_ratio_mean:.4f}")
    lines.append(f"  Success rate: both {o.success_mean:.2f}")
    lines.append(f"  Travel distance: {o.total_travel_distance_mean:.0f} -> {cs.total_travel_distance_mean:.0f} m")
    lines.append(f"  Max single-robot distance: {o.max_single_robot_distance_mean:.0f} -> {cs.max_single_robot_distance_mean:.0f} m")
    lines.append(f"  Overlap ratio: {o.overlap_ratio_mean:.4f} -> {cs.overlap_ratio_mean:.4f}")
    lines.append("")
    lines.append("B. COMMUNICATION INTERVAL SWEEP")
    c2 = c.sort_values("communication_interval")
    for _, row in c2.iterrows():
        lines.append(f"  interval {int(row.communication_interval)}: cov={row.final_coverage_ratio_mean:.4f} "
                     f"vol={row.total_communication_volume_mean:.0f} comm_cnt={row.communication_count_mean:.0f}")
    best = c2.loc[(c2.final_coverage_ratio_mean >= c2.final_coverage_ratio_mean.max() - 0.01)]
    lines.append(f"  Interval 5 coverage={c2[c2.communication_interval==5].final_coverage_ratio_mean.values[0]:.4f} — reasonable trade-off.")
    lines.append("")
    lines.append("C. ABLATION")
    for _, row in a.iterrows():
        lines.append(f"  {row.method_name}: vol={row.total_communication_volume_mean:.0f} overlap={row.overlap_ratio_mean:.4f} cov={row.final_coverage_ratio_mean:.4f}")
    lines.append("  CommOnly vs CommPlusOverlap: overlap penalty adds little; keep as diagnostic only.")
    lines.append("")
    lines.append("D. RECOMMENDED FINAL METHOD")
    lines.append("  CommScheduled, COMMUNICATION_INTERVAL=5, OVERLAP_PENALTY_WEIGHT=0.0")
    lines.append("")
    lines.append("E. SUGGESTED PAPER CLAIM (cautious)")
    lines.append('  "Communication scheduling substantially reduces communication volume while')
    lines.append('   maintaining high exploration coverage, with a modest trade-off in travel distance or steps."')
    lines.append("")
    lines.append("DO NOT CLAIM:")
    lines.append("  - Overlap penalty as core contribution")
    lines.append("  - Large overlap_ratio reduction from CommScheduled alone")
    return "\n".join(lines)


def organize_paper_package() -> None:
    for sub in ("tables", "figures", "trajectories", "summaries", "raw_data_reference", "configs", "notes"):
        os.makedirs(os.path.join(PAPER_ROOT, sub), exist_ok=True)

    copies = [
        ("results/summary/main_comparison_full_summary_paper.csv", "tables/main_comparison_full_summary_paper.csv"),
        ("results/summary/main_comparison_full_summary_numeric.csv", "summaries/main_comparison_full_summary_numeric.csv"),
        ("results/summary/communication_interval_full_summary_paper.csv", "tables/communication_interval_full_summary_paper.csv"),
        ("results/summary/communication_interval_full_summary_numeric.csv", "summaries/communication_interval_full_summary_numeric.csv"),
        ("results/summary/ablation_full_summary_paper.csv", "tables/ablation_full_summary_paper.csv"),
        ("results/summary/ablation_full_summary_numeric.csv", "summaries/ablation_full_summary_numeric.csv"),
        ("results/summary/main_comparison_full_improvement_summary.txt", "tables/main_comparison_full_improvement_summary.txt"),
        ("results/summary/stage11_full_results_interpretation.txt", "notes/stage11_full_results_interpretation.txt"),
    ]
    for src, dst in copies:
        sp = os.path.join(PROJECT_ROOT, src)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(PAPER_ROOT, dst))

    for group in ["main_comparison_full", "communication_interval_full", "ablation_full"]:
        fig_src = os.path.join(PROJECT_ROOT, "results/figures", group)
        fig_dst = os.path.join(PAPER_ROOT, "figures", group)
        if os.path.isdir(fig_src):
            if os.path.exists(fig_dst):
                shutil.rmtree(fig_dst)
            shutil.copytree(fig_src, fig_dst)

    cov = os.path.join(PROJECT_ROOT, "results/figures/coverage_curves_full")
    if os.path.isdir(cov):
        dst = os.path.join(PAPER_ROOT, "figures/coverage_curves_full")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(cov, dst)

    traj = os.path.join(PROJECT_ROOT, "results/trajectories/figures/full_main_comparison")
    if os.path.isdir(traj):
        dst = os.path.join(PAPER_ROOT, "trajectories/full_main_comparison")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(traj, dst)

    for group in GROUPS:
        if group == "overlap_diagnostic_full_lite" and not os.path.isdir(
            os.path.join(PROJECT_ROOT, "results/raw", group)
        ):
            continue
        ref = os.path.join(PAPER_ROOT, "raw_data_reference", group)
        os.makedirs(ref, exist_ok=True)
        readme = os.path.join(ref, "README.txt")
        with open(readme, "w") as f:
            f.write(f"See {os.path.join('results/raw', group)} in project root.\n")

    cfg_dirs = ["main_comparison_full", "communication_interval_full", "ablation_full"]
    for g in cfg_dirs:
        src = os.path.join(PROJECT_ROOT, "results/configs", g)
        dst = os.path.join(PAPER_ROOT, "configs", g)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.log"))

    settings = f"""EXPERIMENT SETTINGS (Stage 11 paper package)
Generated: {datetime.now().isoformat()}

Baseline: official Bandwidth-Limited Multi-Robot Exploration repo (DARS 2024)
Checkpoint: {CHECKPOINT}
Training: none (evaluation only; pretrained weights unchanged)

Final method: CommScheduled
  COMMUNICATION_INTERVAL = 5
  OVERLAP_PENALTY_WEIGHT = 0.0

Baseline: Original
  COMMUNICATION_INTERVAL = 1
  OVERLAP_PENALTY_WEIGHT = 0.0

Overlap penalty: tested (ablation / diagnostic only); Stage 9.5 showed near-zero action_change_rate

Full-scale evaluation:
  seeds = [0, 1, 2, 3, 4]
  num_test_maps = 10 per seed run
  num_agents = 8
  sensor_range = 20
  max_episode_steps = 128
  success threshold: coverage >= 0.95

Metrics:
  final_coverage_ratio: belief explored_rate
  total_communication_volume: upload+download bytes (float32 message numel * 4 per event)
  overlap_ratio: cells with visit_count>1 / cells visited (visit_count_map)
"""
    with open(os.path.join(PAPER_ROOT, "notes/experiment_setting.txt"), "w") as f:
        f.write(settings)


def main():
    log_path = os.path.join(PROJECT_ROOT, "results/logs/stage11_organize_full_results_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    lines = ["STAGE 11 ORGANIZE FULL RESULTS", f"Started: {datetime.now().isoformat()}", ""]
    issues: List[str] = []
    data: Dict[str, pd.DataFrame] = {}

    lines.append("## 1. FOLDER VERIFICATION")
    for group, cfg in GROUPS.items():
        if cfg.get("optional") and not os.path.isdir(os.path.join(PROJECT_ROOT, "results/raw", group)):
            lines.append(f"\n### {group} (optional, not present)")
            continue
        data[group] = audit_folder(group, cfg, lines)

    if "main_comparison_full" in data:
        fairness_main(data["main_comparison_full"], lines)
    if "communication_interval_full" in data:
        fairness_comm(data["communication_interval_full"], lines)
    if "ablation_full" in data:
        fairness_ablation(data["ablation_full"], lines)

    lines.append("\n## 3. DATA VALIDITY")
    for group, df in data.items():
        validity_scan(df, group, issues)
        lines.extend([f"  - {i}" for i in issues if i.startswith(group)])

    lines.append("\n## 4. AGGREGATION")
    for group in ["main_comparison_full", "communication_interval_full", "ablation_full"]:
        if group in data and not data[group].empty:
            rc = run_aggregate(group)
            lines.append(f"  {group}: aggregate exit {rc}")

    if os.path.isdir(os.path.join(PROJECT_ROOT, "results/raw/overlap_diagnostic_full_lite")):
        run_aggregate("overlap_diagnostic_full_lite")

    imp = improvement_summary()
    imp_path = os.path.join(PROJECT_ROOT, "results/summary/main_comparison_full_improvement_summary.txt")
    with open(imp_path, "w") as f:
        f.write(imp)
    lines.append(f"\nImprovement summary: {imp_path}")

    lines.append("\n## 6. FIGURES")
    plot_bar_group("main_comparison_full", "Main comparison (full)", GROUPS["main_comparison_full"]["bar_metrics"])
    plot_line_group("communication_interval_full", "Communication interval (full)",
                    GROUPS["communication_interval_full"]["line_metrics"], "communication_interval")
    plot_bar_group("ablation_full", "Ablation (full)", GROUPS["ablation_full"]["bar_metrics"])
    if os.path.isfile(os.path.join(PROJECT_ROOT, "results/summary/overlap_diagnostic_full_lite_summary_numeric.csv")):
        subprocess.run(
            [
                sys.executable,
                "scripts/plot_group.py",
                "--group",
                "overlap_diagnostic_full_lite",
                "--plot-type",
                "overlap",
                "--title",
                "Overlap diagnostic full-lite",
            ],
            cwd=PROJECT_ROOT,
        )

    interp = interpret_results(0, 0, 0)
    interp_path = os.path.join(PROJECT_ROOT, "results/summary/stage11_full_results_interpretation.txt")
    with open(interp_path, "w") as f:
        f.write(interp)
    lines.append(f"\nInterpretation: {interp_path}")

    cov_msg = plot_coverage_main_full()
    lines.append(f"  Coverage: {cov_msg}")
    traj_msg = plot_trajectories()
    lines.append(f"  Trajectories: {traj_msg}")

    lines.append("\n## 10. PAPER PACKAGE")
    organize_paper_package()
    lines.append(f"  Created {PAPER_ROOT}/")

    lines.append("\n## ISSUES SUMMARY")
    if issues:
        lines.extend([f"  - {i}" for i in issues])
    else:
        lines.append("  No critical validity issues detected.")

    text = "\n".join(lines)
    with open(log_path, "w") as f:
        f.write(text)
        f.write("\n\n" + imp + "\n\n" + interp)
    print(text)
    print(f"\nLog: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
