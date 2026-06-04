#!/usr/bin/env python3
"""Plot coverage ratio vs step from coverage curve CSV files."""

import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import save_figure

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

METHOD_PATTERNS = [
    ("Proposed", re.compile(r"Proposed", re.I)),
    ("Original", re.compile(r"Original", re.I)),
    ("CommOnly", re.compile(r"CommOnly", re.I)),
    ("OverlapOnly", re.compile(r"OverlapOnly", re.I)),
    ("OverlapPenalty_0.1", re.compile(r"OverlapPenalty_0[\.p]1", re.I)),
    ("CommInterval_5", re.compile(r"CommInterval_5", re.I)),
]


def infer_method(filename: str) -> str:
    base = os.path.basename(filename)
    for name, pat in METHOD_PATTERNS:
        if pat.search(base):
            return name
    return "unknown"


def find_curve_files() -> list:
    patterns = [
        os.path.join(PROJECT_ROOT, "results/raw", "*", "coverage_curves", "*.csv"),
        os.path.join(PROJECT_ROOT, "results/raw", "coverage_curves", "*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))


def load_curves(files: list) -> dict:
    by_method = {}
    for f in files:
        method = infer_method(f)
        if method == "unknown":
            continue
        df = pd.read_csv(f)
        if "step" not in df.columns or "coverage_ratio" not in df.columns:
            continue
        by_method.setdefault(method, []).append(df[["step", "coverage_ratio"]].copy())
    return by_method


def mean_curve(curves: list) -> tuple:
    max_step = max(c["step"].max() for c in curves)
    steps = np.arange(int(max_step) + 1)
    mats = []
    for c in curves:
        aligned = c.set_index("step").reindex(steps)["coverage_ratio"].interpolate().values
        mats.append(aligned)
    arr = np.array(mats)
    return steps, arr.mean(axis=0), arr.std(axis=0)


def main():
    out_dir = os.path.join(PROJECT_ROOT, "results/figures/coverage_curves")
    files = find_curve_files()
    if not files:
        print("No coverage curve CSV files found; skipping plots.")
        return 0

    by_method = load_curves(files)
    if not by_method:
        print("Coverage files found but no method could be inferred; skipping.")
        return 0

    preferred = ["Original", "Proposed", "CommOnly", "OverlapOnly"]
    methods = [m for m in preferred if m in by_method] + [
        m for m in sorted(by_method) if m not in preferred
    ]

    # Combined overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        steps, mean, std = mean_curve(by_method[method])
        ax.plot(steps, mean, label=method, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage ratio")
    ax.set_title("Coverage over time (mean ± std over episodes)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, os.path.join(out_dir, "coverage_all_methods"))

    for method in methods:
        steps, mean, std = mean_curve(by_method[method])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, mean, linewidth=2, color="#4C72B0")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Coverage ratio")
        ax.set_title(f"Coverage: {method}")
        ax.grid(alpha=0.3)
        safe = method.replace(".", "p")
        save_figure(fig, os.path.join(out_dir, f"coverage_{safe}"))

    print(f"Plotted methods: {methods}")
    print(f"Figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
