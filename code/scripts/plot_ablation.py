#!/usr/bin/env python3
"""Bar plots for ablation summary."""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import metric_label, save_figure

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS = [
    "final_coverage_ratio",
    "total_travel_distance",
    "max_single_robot_distance",
    "total_communication_volume",
    "overlap_ratio",
    "episode_steps",
]


def main():
    summary = os.path.join(PROJECT_ROOT, "results/summary/ablation_summary_numeric.csv")
    out_dir = os.path.join(PROJECT_ROOT, "results/figures/ablation")
    df = pd.read_csv(summary)
    methods = df["method_name"].astype(str).tolist()
    x = range(len(methods))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            x,
            df[mean_col],
            yerr=df[std_col] if std_col in df.columns else None,
            capsize=4,
            color=colors[: len(methods)],
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"Ablation: {metric_label(metric)}")
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, os.path.join(out_dir, metric))
    print(f"Figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
