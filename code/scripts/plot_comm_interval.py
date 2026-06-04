#!/usr/bin/env python3
"""Line plots vs communication_interval."""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import metric_label, save_figure

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
METRICS = [
    "final_coverage_ratio",
    "total_communication_volume",
    "communication_count",
    "total_travel_distance",
    "overlap_ratio",
]


def main():
    summary = os.path.join(PROJECT_ROOT, "results/summary/communication_interval_summary_numeric.csv")
    out_dir = os.path.join(PROJECT_ROOT, "results/figures/communication_interval")
    df = pd.read_csv(summary)
    df = df.sort_values("communication_interval")
    x = df["communication_interval"].values

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            x,
            df[mean_col],
            yerr=df[std_col] if std_col in df.columns else None,
            marker="o",
            capsize=4,
            linewidth=2,
        )
        ax.set_xlabel("Communication interval")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"Communication interval: {metric_label(metric)}")
        ax.grid(alpha=0.3)
        save_figure(fig, os.path.join(out_dir, metric))
    print(f"Figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
