"""
Episode-level evaluation logger for Sensors experiments.
Appends one CSV row per episode without modifying algorithm behavior.
"""

import csv
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, Optional

CSV_COLUMNS = [
    "experiment_name",
    "method_name",
    "seed",
    "episode_id",
    "map_id",
    "map_name",
    "num_agents",
    "sensor_range",
    "communication_interval",
    "overlap_penalty_weight",
    "max_episode_steps",
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
    "checkpoint_path",
    "comm_mode",
    "spike_total",
    "spike_per_robot_mean",
    "mean_broadcast_interval",
]

RESULTS_ROOT = "results"


def get_experiment_group() -> Optional[str]:
    group = os.environ.get("BLMRE_EXPERIMENT_GROUP", "").strip()
    return group or None


def group_paths(results_root: str = RESULTS_ROOT, group: Optional[str] = None):
    if group is None:
        group = get_experiment_group()
    raw_dir = os.path.join(results_root, "raw", group) if group else os.path.join(results_root, "raw")
    config_dir = os.path.join(results_root, "configs", group) if group else os.path.join(results_root, "configs")
    log_dir = os.path.join(results_root, "logs", group) if group else os.path.join(results_root, "logs")
    coverage_dir = os.path.join(raw_dir, "coverage_curves")
    traj_dir = (
        os.path.join(results_root, "trajectories", group, "raw")
        if group
        else os.path.join(results_root, "trajectories", "raw")
    )
    return {
        "group": group,
        "raw_dir": raw_dir,
        "config_dir": config_dir,
        "log_dir": log_dir,
        "coverage_dir": coverage_dir,
        "traj_dir": traj_dir,
    }


def ensure_results_layout(results_root: str = RESULTS_ROOT, group: Optional[str] = None) -> None:
    for sub in ("raw", "summary", "figures", "trajectories", "logs", "configs", "checkpoints_used"):
        os.makedirs(os.path.join(results_root, sub), exist_ok=True)
    paths = group_paths(results_root, group)
    for key in ("raw_dir", "config_dir", "log_dir", "coverage_dir", "traj_dir"):
        os.makedirs(paths[key], exist_ok=True)


class EvaluationLogger:
    def __init__(self, experiment_name: str, results_root: str = RESULTS_ROOT, group: Optional[str] = None):
        self.group = group if group is not None else get_experiment_group()
        ensure_results_layout(results_root, self.group)
        self.experiment_name = experiment_name
        self.results_root = results_root
        paths = group_paths(results_root, self.group)
        self.raw_dir = paths["raw_dir"]
        self.csv_path = os.path.join(self.raw_dir, f"{experiment_name}.csv")
        self._write_header_if_needed()

    def _write_header_if_needed(self) -> None:
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    def log_episode(self, row: Dict[str, Any]) -> None:
        out = {col: "" for col in CSV_COLUMNS}
        out["experiment_name"] = self.experiment_name
        for key, value in row.items():
            if key in out:
                out[key] = "" if value is None else value
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(out)

    @property
    def path(self) -> str:
        return self.csv_path


def save_run_config(
    experiment_name: str,
    config: Dict[str, Any],
    results_root: str = RESULTS_ROOT,
    group: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    group = group if group is not None else get_experiment_group()
    ensure_results_layout(results_root, group)
    paths = group_paths(results_root, group)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(paths["config_dir"], f"{experiment_name}_{stamp}.json")
    payload = {
        "experiment_name": experiment_name,
        "experiment_group": group,
        "saved_at": datetime.now().isoformat(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def record_checkpoint_used(
    checkpoint_path: str,
    experiment_name: str,
    results_root: str = RESULTS_ROOT,
) -> str:
    ensure_results_layout(results_root)
    abs_path = os.path.abspath(checkpoint_path)
    dest = os.path.join(results_root, "checkpoints_used", f"{experiment_name}_checkpoint.pth")
    if os.path.isfile(abs_path) and not os.path.exists(dest):
        try:
            os.symlink(abs_path, dest)
        except OSError:
            shutil.copy2(abs_path, dest)
    meta_path = os.path.join(results_root, "checkpoints_used", f"{experiment_name}_checkpoint_path.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(abs_path + "\n")
    return abs_path


def build_episode_row(
    perf_metrics: Dict[str, Any],
    episode_id: int,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    travel = perf_metrics.get("travel_dist", "")
    success = perf_metrics.get("success_rate", perf_metrics.get("success", ""))
    if success is True:
        success = 1
    elif success is False:
        success = 0

    row = dict(defaults)
    row.update(
        {
            "episode_id": episode_id,
            "map_id": perf_metrics.get("map_id", ""),
            "map_name": perf_metrics.get("map_name", ""),
            "final_coverage_ratio": perf_metrics.get("final_coverage_ratio", ""),
            "success": success,
            "total_travel_distance": perf_metrics.get(
                "total_travel_distance", travel
            ),
            "max_single_robot_distance": perf_metrics.get("max_single_robot_distance", ""),
            "average_single_robot_distance": perf_metrics.get(
                "average_single_robot_distance", ""
            ),
            "episode_steps": perf_metrics.get("episode_steps", ""),
            "communication_count": perf_metrics.get("communication_count", ""),
            "upload_data_volume": perf_metrics.get("upload_data_volume", ""),
            "download_data_volume": perf_metrics.get("download_data_volume", ""),
            "total_communication_volume": perf_metrics.get(
                "total_communication_volume", ""
            ),
            "overlap_ratio": perf_metrics.get("overlap_ratio", ""),
            "action_change_count": perf_metrics.get("action_change_count", ""),
            "total_decision_count": perf_metrics.get("total_decision_count", ""),
            "action_change_rate": perf_metrics.get("action_change_rate", ""),
            "average_selected_overlap_cost_before_penalty": perf_metrics.get(
                "average_selected_overlap_cost_before_penalty", ""
            ),
            "average_selected_overlap_cost_after_penalty": perf_metrics.get(
                "average_selected_overlap_cost_after_penalty", ""
            ),
            "computation_time": perf_metrics.get("computation_time", ""),
            "comm_mode": perf_metrics.get("comm_mode", ""),
            "spike_total": perf_metrics.get("spike_total", ""),
            "spike_per_robot_mean": perf_metrics.get("spike_per_robot_mean", ""),
            "mean_broadcast_interval": perf_metrics.get("mean_broadcast_interval", ""),
        }
    )
    return row
