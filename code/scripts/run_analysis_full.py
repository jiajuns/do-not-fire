#!/usr/bin/env python3
"""Aggregate and plot full paper-scale experiment groups (*_full)."""

import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FULL_GROUPS = {
    "main": {
        "raw": "results/raw/main_comparison_full",
        "group": "main_comparison_full",
        "plot_type": "bar",
        "title": "Main comparison (full)",
    },
    "comm": {
        "raw": "results/raw/communication_interval_full",
        "group": "communication_interval_full",
        "plot_type": "comm",
        "title": "Communication interval (full)",
    },
    "ablation": {
        "raw": "results/raw/ablation_full",
        "group": "ablation_full",
        "plot_type": "bar",
        "title": "Ablation diagnostic (full)",
    },
    "overlap_diag": {
        "raw": "results/raw/overlap_diagnostic_full_lite",
        "group": "overlap_diagnostic_full_lite",
        "plot_type": "overlap",
        "title": "Overlap diagnostic full-lite",
    },
    "setcomm": {
        "raw": "results/raw/setcomm_full",
        "group": "setcomm_full",
        "plot_type": "bar",
        "title": "SET-Comm (full)",
    },
}


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default="main,comm,ablation")
    parser.add_argument("--include-overlap-diag", action="store_true")
    args = parser.parse_args()
    keys = [k.strip() for k in args.groups.split(",")]
    if args.include_overlap_diag and "overlap_diag" not in keys:
        keys.append("overlap_diag")
    py = sys.executable
    worst = 0
    for key in keys:
        if key not in FULL_GROUPS:
            print(f"Unknown: {key}")
            worst = 1
            continue
        g = FULL_GROUPS[key]
        raw = os.path.join(PROJECT_ROOT, g["raw"])
        if not os.path.isdir(raw):
            print(f"Skip {key}: missing {raw}")
            worst = 1
            continue
        worst = max(
            worst,
            run(
                [
                    py,
                    "scripts/aggregate_results.py",
                    "--input_dir",
                    g["raw"],
                    "--output_dir",
                    "results/summary",
                    "--group",
                    g["group"],
                ]
            ),
        )
        worst = max(
            worst,
            run(
                [
                    py,
                    "scripts/plot_group.py",
                    "--group",
                    g["group"],
                    "--plot-type",
                    g["plot_type"],
                    "--title",
                    g["title"],
                ]
            ),
        )
    if "main" in keys or "comm" in keys or "ablation" in keys:
        print("\nNote: run plot_coverage_curves.py manually after full runs if coverage CSVs exist.")
    print("\n=== Full analysis outputs ===")
    print("Summaries: results/summary/*_full_*")
    print("Figures:   results/figures/*_full/")
    return worst


if __name__ == "__main__":
    sys.exit(main())
