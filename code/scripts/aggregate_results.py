#!/usr/bin/env python3
"""Aggregate raw episode CSVs into mean/std summary tables."""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

NUMERIC_METRICS = [
    "final_coverage_ratio",
    "success",
    "total_travel_distance",
    "max_single_robot_distance",
    "average_single_robot_distance",
    "episode_steps",
    "communication_count",
    "upload_data_volume",
    "download_data_volume",
    "total_communication_volume",
    "overlap_ratio",
    "action_change_count",
    "total_decision_count",
    "action_change_rate",
    "average_selected_overlap_cost_before_penalty",
    "average_selected_overlap_cost_after_penalty",
    "total_planning_time",
    "average_planning_time_per_step",
    "computation_time",
    "spike_total",
    "spike_per_robot_mean",
    "mean_broadcast_interval",
]

# experiment_name encodes seed; exclude it so seeds aggregate into one row per method/settings.
GROUP_CANDIDATES = [
    "method_name",
    "num_agents",
    "sensor_range",
    "communication_interval",
    "overlap_penalty_weight",
    "map_set",
]


def load_raw_dir(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files in {input_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["_source_file"] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def aggregate(df: pd.DataFrame) -> tuple:
    group_cols = [c for c in GROUP_CANDIDATES if c in df.columns]
    metrics = [c for c in NUMERIC_METRICS if c in df.columns]
    df = coerce_numeric(df, metrics)

    if not group_cols:
        group_cols = ["method_name"] if "method_name" in df.columns else []

    grouped = df.groupby(group_cols, dropna=False)
    rows = []
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["count"] = len(g)
        for m in metrics:
            s = g[m].dropna()
            row[f"{m}_mean"] = s.mean() if len(s) else np.nan
            row[f"{m}_std"] = s.std(ddof=0) if len(s) > 1 else (0.0 if len(s) == 1 else np.nan)
            row[f"{m}_count"] = int(s.count())
        rows.append(row)
    numeric = pd.DataFrame(rows)
    paper = numeric.copy()
    for m in metrics:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in paper.columns:
            paper[m] = paper.apply(
                lambda r, mc=mean_col, sc=std_col: format_mean_std(r[mc], r[sc]),
                axis=1,
            )
    keep_paper = group_cols + ["count"] + [m for m in metrics if m in paper.columns]
    paper = paper[[c for c in keep_paper if c in paper.columns]]
    return numeric, paper, group_cols, metrics


def format_mean_std(mean, std) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std) or std == 0:
        return f"{mean:.4g}"
    return f"{mean:.4g} ± {std:.4g}"


def main():
    parser = argparse.ArgumentParser(description="Aggregate raw CSV results.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="results/summary")
    parser.add_argument("--group", required=True, help="Group name prefix for output files.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(PROJECT_ROOT, input_dir)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = load_raw_dir(input_dir)
    numeric, paper, group_cols, metrics = aggregate(df)

    num_path = os.path.join(output_dir, f"{args.group}_summary_numeric.csv")
    paper_path = os.path.join(output_dir, f"{args.group}_summary_paper.csv")
    numeric.to_csv(num_path, index=False)
    paper.to_csv(paper_path, index=False)

    print(f"Loaded {len(df)} rows from {input_dir}")
    print(f"Group columns: {group_cols}")
    print(f"Metrics: {metrics}")
    skipped = [c for c in GROUP_CANDIDATES + NUMERIC_METRICS if c not in df.columns]
    if skipped:
        print(f"Skipped missing columns: {skipped}")
    print(f"Wrote {num_path}")
    print(f"Wrote {paper_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
