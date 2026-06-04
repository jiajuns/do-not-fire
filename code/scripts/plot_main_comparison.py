#!/usr/bin/env python3
"""Bar plots for main_comparison summary."""

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
    summary = os.path.join(PROJECT_ROOT, "results/summary/main_comparison_summary_numeric.csv")
    out_dir = os.path.join(PROJECT_ROOT, "results/figures/main_comparison")
    df = pd.read_csv(summary)
    methods = df["method_name"].astype(str).tolist()
    x = range(len(methods))

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in df.columns:
            continue
        means = df[mean_col].values
        stds = df[std_col].values if std_col in df.columns else [0] * len(means)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x, means, yerr=stds, capsize=4, color=["#4C72B0", "#DD8452"][: len(methods)])
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=15, ha="right")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"Main comparison: {metric_label(metric)}")
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, os.path.join(out_dir, metric))
    print(f"Figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
