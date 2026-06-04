#!/usr/bin/env python3
"""
Visualize saved trajectory bundles for representative full-scale episodes.

Requires prior run with save_trajectory=True (main_comparison_full sets
Original + CommScheduled seed0 episode0).

Reads: results/trajectories/main_comparison_full/raw/*.npz
Writes: results/trajectories/figures/full_main_comparison/
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAJ_DIR = os.path.join(PROJECT_ROOT, "results/trajectories/main_comparison_full/raw")
OUT_DIR = os.path.join(PROJECT_ROOT, "results/trajectories/figures/full_main_comparison")


def plot_npz(path: str, out_base: str) -> bool:
    data = np.load(path, allow_pickle=True)
    if "robot_trajectories" not in data:
        return False
    trajs = data["robot_trajectories"]
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajs)))
    for i, traj in enumerate(trajs):
        if len(traj) == 0:
            continue
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], linewidth=1.5, label=f"robot {i}")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(os.path.basename(path))
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    fig.savefig(out_base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out_base + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    pattern = os.path.join(TRAJ_DIR, "*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No trajectory .npz files in {TRAJ_DIR}")
        print("Run main_comparison_full with representative trajectory saving first.")
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for f in files:
        base = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(f))[0])
        if plot_npz(f, base):
            n += 1
            print(f"Wrote {base}.png")
    print(f"Saved {n} figures to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
