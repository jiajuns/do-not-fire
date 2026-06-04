#!/usr/bin/env python3
"""Shared plotting helpers for experiment groups (supports Stage 9 *_small groups)."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import metric_label, save_figure

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

BAR_METRICS = [
    "final_coverage_ratio",
    "total_travel_distance",
    "max_single_robot_distance",
    "total_communication_volume",
    "overlap_ratio",
    "episode_steps",
]

COMM_METRICS = [
    "final_coverage_ratio",
    "total_communication_volume",
    "communication_count",
    "total_travel_distance",
    "overlap_ratio",
]

OVERLAP_METRICS = [
    "overlap_ratio",
    "final_coverage_ratio",
    "total_travel_distance",
    "max_single_robot_distance",
    "episode_steps",
]


def paths_for_group(group: str) -> tuple:
    summary = os.path.join(PROJECT_ROOT, "results/summary", f"{group}_summary_numeric.csv")
    out_dir = os.path.join(PROJECT_ROOT, "results/figures", group)
    return summary, out_dir


def plot_bars(group: str, title_prefix: str, colors=None) -> int:
    summary, out_dir = paths_for_group(group)
    if not os.path.isfile(summary):
        print(f"Missing summary: {summary}")
        return 1
    df = pd.read_csv(summary)
    methods = df["method_name"].astype(str).tolist()
    x = range(len(methods))
    if colors is None:
        colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for metric in BAR_METRICS:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7 if len(methods) > 2 else 6, 4))
        ax.bar(
            x,
            df[mean_col],
            yerr=df[std_col] if std_col in df.columns else None,
            capsize=4,
            color=colors[: len(methods)],
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(methods, rotation=20 if len(methods) > 2 else 15, ha="right")
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"{title_prefix}: {metric_label(metric)}")
        ax.grid(axis="y", alpha=0.3)
        save_figure(fig, os.path.join(out_dir, metric))
    print(f"Figures saved to {out_dir}")
    return 0


def plot_vs_x(group: str, x_col: str, x_label: str, title_prefix: str, metrics, marker="o") -> int:
    summary, out_dir = paths_for_group(group)
    if not os.path.isfile(summary):
        print(f"Missing summary: {summary}")
        return 1
    df = pd.read_csv(summary).sort_values(x_col)
    x = df[x_col].values
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(
            x,
            df[mean_col],
            yerr=df[std_col] if std_col in df.columns else None,
            marker=marker,
            capsize=4,
            linewidth=2,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_label(metric))
        ax.set_title(f"{title_prefix}: {metric_label(metric)}")
        ax.grid(alpha=0.3)
        save_figure(fig, os.path.join(out_dir, metric))
    print(f"Figures saved to {out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--plot-type", choices=["bar", "comm", "overlap"], required=True)
    parser.add_argument("--title", default="Experiment")
    args = parser.parse_args()
    if args.plot_type == "bar":
        return plot_bars(args.group, args.title)
    if args.plot_type == "comm":
        return plot_vs_x(
            args.group,
            "communication_interval",
            "Communication interval",
            args.title,
            COMM_METRICS,
        )
    return plot_vs_x(
        args.group,
        "overlap_penalty_weight",
        "Overlap penalty weight",
        args.title,
        OVERLAP_METRICS,
        marker="s",
    )


if __name__ == "__main__":
    sys.exit(main())
