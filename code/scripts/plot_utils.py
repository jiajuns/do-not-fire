"""Shared helpers for Stage 8 plotting (matplotlib only)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_figure(fig, path_base: str) -> None:
    """Save PNG and PDF from path without extension."""
    ensure_dir(os.path.dirname(path_base) or ".")
    fig.savefig(path_base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(path_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def metric_label(name: str) -> str:
    labels = {
        "final_coverage_ratio": "Final coverage ratio",
        "total_travel_distance": "Total travel distance (m)",
        "max_single_robot_distance": "Max single-robot distance (m)",
        "total_communication_volume": "Total communication volume (bytes)",
        "overlap_ratio": "Overlap ratio",
        "episode_steps": "Episode steps",
        "communication_count": "Communication count",
    }
    return labels.get(name, name.replace("_", " "))
