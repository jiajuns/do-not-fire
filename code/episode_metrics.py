"""
Episode-level metric instrumentation for evaluation (Sensors experiments).

Does not modify policy, actions, or environment dynamics — observation only.

Metric definitions:
- coverage_ratio: env.explored_rate = (# belief cells == 255) / (# ground_truth free cells == 255)
- success: explored_rate >= 0.95 at episode end (official test_worker criterion)
- overlap_ratio: repeatedly_visited_grid_cells / total_visited_grid_cells (visit_count_map)
- travel distance: per-robot sum of Euclidean distance between consecutive positions (meters);
  total_travel_distance = sum over robots; max/average over per-robot totals
- communication volume: float32 message tensor numel * 4 bytes per clone;
  each send_msg appends one clone per robot mailbox (n_agents deliveries per event)
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def tensor_message_bytes(msg) -> int:
    import torch

    if isinstance(msg, torch.Tensor):
        return int(msg.numel() * msg.element_size())
    return int(np.prod(np.asarray(msg).shape) * 4)


class OverlapDiagnosticStats:
    """Greedy before/after overlap-penalty selection stats (evaluation only)."""

    def __init__(self):
        self.action_change_count = 0
        self.total_decision_count = 0
        self._sum_cost_before = 0.0
        self._sum_cost_after = 0.0

    def record(self, diag: dict) -> None:
        if diag.get("candidate_count", 0) <= 0:
            return
        self.total_decision_count += 1
        if diag.get("action_changed"):
            self.action_change_count += 1
        self._sum_cost_before += float(diag.get("original_overlap_cost", 0.0))
        self._sum_cost_after += float(diag.get("adjusted_overlap_cost", 0.0))

    def summary(self) -> Dict[str, Any]:
        n = self.total_decision_count
        rate = float(self.action_change_count / n) if n > 0 else 0.0
        return {
            "action_change_count": self.action_change_count,
            "total_decision_count": n,
            "action_change_rate": rate,
            "average_selected_overlap_cost_before_penalty": (
                float(self._sum_cost_before / n) if n > 0 else 0.0
            ),
            "average_selected_overlap_cost_after_penalty": (
                float(self._sum_cost_after / n) if n > 0 else 0.0
            ),
        }


class EpisodeMetricsTracker:
    def __init__(self, n_agents: int, ground_truth: np.ndarray, save_trajectory: bool = False):
        self.n_agents = n_agents
        self.ground_truth = ground_truth
        self.save_trajectory = save_trajectory
        self.overlap_diag = OverlapDiagnosticStats()

        self.position_history: List[List[np.ndarray]] = [[] for _ in range(n_agents)]
        self.coverage_steps: List[int] = []
        self.coverage_values: List[float] = []

        self.visit_count_map = np.zeros(ground_truth.shape, dtype=np.int32)
        self.observed_count_map = self.visit_count_map
        self.communication_count = 0
        self.upload_data_volume = 0
        self.download_data_volume = 0

        self.trajectory_positions: Optional[List[List[np.ndarray]]] = (
            [[] for _ in range(n_agents)] if save_trajectory else None
        )
        self.final_belief: Optional[np.ndarray] = None

    def record_initial_positions(self, robot_locations: np.ndarray) -> None:
        for i in range(self.n_agents):
            pos = np.array(robot_locations[i], dtype=np.float64)
            self.position_history[i].append(pos.copy())
            if self.trajectory_positions is not None:
                self.trajectory_positions[i].append(pos.copy())

    def record_positions_after_step(self, robot_locations: np.ndarray) -> None:
        for i in range(self.n_agents):
            pos = np.array(robot_locations[i], dtype=np.float64)
            self.position_history[i].append(pos.copy())
            if self.trajectory_positions is not None:
                self.trajectory_positions[i].append(pos.copy())

    def record_coverage(self, step: int, coverage_ratio: float) -> None:
        self.coverage_steps.append(step)
        self.coverage_values.append(float(coverage_ratio))

    def record_robot_visits(self, belief_info, robot_locations: np.ndarray) -> None:
        from utils import get_cell_position_from_coords

        cells = get_cell_position_from_coords(robot_locations, belief_info)
        h, w = self.visit_count_map.shape
        for i in range(cells.shape[0]):
            x, y = int(cells[i, 0]), int(cells[i, 1])
            if 0 <= y < h and 0 <= x < w:
                self.visit_count_map[y, x] += 1

    def record_send_msg(self, msg, n_mailboxes: int) -> None:
        msg_bytes = tensor_message_bytes(msg)
        self.communication_count += 1
        delivered = msg_bytes * (n_mailboxes - 1)
        self.upload_data_volume += delivered
        self.download_data_volume += delivered

    def record_targeted_msg(self, msg) -> None:
        """One point-to-point delivery on a single directed link (targeted comm):
        the message reaches exactly one receiver, so volume is counted x1 (vs the
        x(n_agents-1) of a broadcast). Keeps total_communication_volume comparable
        across the broadcast and targeted models on the same byte axis."""
        msg_bytes = tensor_message_bytes(msg)
        self.communication_count += 1
        self.upload_data_volume += msg_bytes
        self.download_data_volume += msg_bytes

    @property
    def total_communication_volume(self) -> int:
        return int(self.upload_data_volume + self.download_data_volume)

    def compute_travel_distances(self) -> Tuple[float, float, float]:
        per_robot = []
        for history in self.position_history:
            dist = 0.0
            for k in range(1, len(history)):
                dist += float(np.linalg.norm(history[k] - history[k - 1]))
            per_robot.append(dist)
        if not per_robot:
            return 0.0, 0.0, 0.0
        return float(sum(per_robot)), float(max(per_robot)), float(np.mean(per_robot))

    def compute_overlap_ratio(self) -> float:
        visited = self.visit_count_map > 0
        n_visited = int(np.sum(visited))
        if n_visited == 0:
            return 0.0
        n_repeat = int(np.sum(self.visit_count_map > 1))
        return float(n_repeat / n_visited)

    def finalize(self, env) -> Dict[str, Any]:
        total_d, max_d, avg_d = self.compute_travel_distances()
        self.final_belief = env.robot_belief.copy()
        out = {
            "total_travel_distance": total_d,
            "max_single_robot_distance": max_d,
            "average_single_robot_distance": avg_d,
            "travel_dist": max_d,
            "overlap_ratio": self.compute_overlap_ratio(),
            "communication_count": self.communication_count,
            "upload_data_volume": self.upload_data_volume,
            "download_data_volume": self.download_data_volume,
            "total_communication_volume": self.total_communication_volume,
            "coverage_steps": self.coverage_steps,
            "coverage_values": self.coverage_values,
        }
        out.update(self.overlap_diag.summary())
        return out


def save_coverage_curve(
    experiment_name: str,
    seed: int,
    episode_id: int,
    steps: List[int],
    coverage_values: List[float],
    results_root: str = "results",
) -> str:
    from experiment_logger import group_paths

    out_dir = group_paths(results_root)["coverage_dir"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{experiment_name}_seed{seed}_episode{episode_id}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "coverage_ratio"])
        for s, c in zip(steps, coverage_values):
            writer.writerow([s, c])
    return path


def save_trajectory_bundle(
    experiment_name: str,
    seed: int,
    episode_id: int,
    tracker: EpisodeMetricsTracker,
    map_name: str,
    map_id: int,
    perf_metrics: Dict[str, Any],
    results_root: str = "results",
) -> str:
    from experiment_logger import group_paths

    out_dir = group_paths(results_root)["traj_dir"]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{experiment_name}_seed{seed}_episode{episode_id}.npz")

    traj_arrays = []
    for i in range(tracker.n_agents):
        if tracker.trajectory_positions and tracker.trajectory_positions[i]:
            traj_arrays.append(np.stack(tracker.trajectory_positions[i]))
        else:
            traj_arrays.append(np.empty((0, 2)))

    np.savez(
        path,
        map_name=map_name,
        map_id=map_id,
        robot_trajectories=np.array(traj_arrays, dtype=object),
        final_belief=tracker.final_belief,
        observed_count_map=tracker.observed_count_map,
        overlap_ratio=perf_metrics.get("overlap_ratio", 0.0),
        final_coverage_ratio=perf_metrics.get("final_coverage_ratio", 0.0),
        success=perf_metrics.get("success_rate", 0),
    )
    return path
