"""
Shared utilities for batch evaluation experiments (Stage 7+).
Evaluation only — invokes test_driver.py, never training.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "model", "clean", "checkpoint.pth")


@dataclass
class ModeSettings:
    seeds: List[int]
    num_test: int
    num_meta_agent: int
    save_trajectory: bool

    @staticmethod
    def from_mode(mode: str, seeds_override: Optional[List[int]] = None, num_test_override: Optional[int] = None) -> "ModeSettings":
        if mode == "quick":
            base = ModeSettings(seeds=[0], num_test=2, num_meta_agent=1, save_trajectory=True)
        elif mode == "small":
            base = ModeSettings(seeds=[0, 1, 2], num_test=5, num_meta_agent=4, save_trajectory=False)
        elif mode in ("normal", "full"):
            # Paper-scale default (Stage 10)
            base = ModeSettings(seeds=[0, 1, 2, 3, 4], num_test=10, num_meta_agent=8, save_trajectory=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        if seeds_override is not None:
            base.seeds = list(seeds_override)
        if num_test_override is not None:
            base.num_test = int(num_test_override)
        return base


def resolve_experiment_group(base_group: str, mode: str, group_override: Optional[str] = None) -> str:
    if group_override:
        return group_override
    if mode == "full":
        return f"{base_group}_full"
    if mode == "small":
        return f"{base_group}_small"
    return base_group


@dataclass
class RunSpec:
    experiment_group: str
    experiment_name: str
    method_name: str
    communication_interval: int
    overlap_penalty_weight: float
    seed: int
    mode: str = "quick"
    save_trajectory: Optional[bool] = None
    save_trajectory_episode_ids: Optional[str] = None
    # SET-Comm (spiking event-triggered) options; None -> periodic baseline behaviour
    comm_mode: Optional[str] = None
    spike_threshold: Optional[float] = None
    spike_leak: Optional[float] = None
    spike_refractory: Optional[int] = None
    spike_w_drift: Optional[float] = None
    spike_w_cov: Optional[float] = None
    random_comm_prob: Optional[float] = None
    spike_gate_path: Optional[str] = None
    map_dir: Optional[str] = None

    def env(self, settings: ModeSettings) -> Dict[str, str]:
        env = os.environ.copy()
        env["BLMRE_COMM_INTERVAL"] = str(self.communication_interval)
        env["BLMRE_OVERLAP_PENALTY_WEIGHT"] = str(self.overlap_penalty_weight)
        env["BLMRE_METHOD_NAME"] = self.method_name
        env["BLMRE_EXPERIMENT_NAME"] = self.experiment_name
        env["BLMRE_EXPERIMENT_GROUP"] = self.experiment_group
        env["BLMRE_SEED"] = str(self.seed)
        env["BLMRE_NUM_TEST"] = str(settings.num_test)
        env["BLMRE_NUM_META_AGENT"] = str(settings.num_meta_agent)
        traj = settings.save_trajectory if self.save_trajectory is None else self.save_trajectory
        env["BLMRE_SAVE_TRAJECTORY"] = "1" if traj else "0"
        if self.save_trajectory_episode_ids:
            env["BLMRE_SAVE_TRAJECTORY_EPISODE_IDS"] = self.save_trajectory_episode_ids
        env["BLMRE_SMOKE_TEST"] = "0"
        # spiking event-triggered controls (only set when used)
        spike_env = {
            "BLMRE_COMM_MODE": self.comm_mode,
            "BLMRE_SPIKE_THRESHOLD": self.spike_threshold,
            "BLMRE_SPIKE_LEAK": self.spike_leak,
            "BLMRE_SPIKE_REFRACTORY": self.spike_refractory,
            "BLMRE_SPIKE_W_DRIFT": self.spike_w_drift,
            "BLMRE_SPIKE_W_COV": self.spike_w_cov,
            "BLMRE_RANDOM_COMM_PROB": self.random_comm_prob,
            "BLMRE_SPIKE_GATE_PATH": self.spike_gate_path,
            "BLMRE_MAP_DIR": self.map_dir,
        }
        for k, v in spike_env.items():
            if v is not None:
                env[k] = str(v)
        return env

    def cli_args(self) -> List[str]:
        return [
            "--communication-interval",
            str(self.communication_interval),
            "--overlap-penalty-weight",
            str(self.overlap_penalty_weight),
            "--experiment-name",
            self.experiment_name,
            "--method-name",
            self.method_name,
        ]


def ensure_group_dirs(group: str) -> Dict[str, str]:
    paths = {
        "raw": os.path.join(RESULTS_ROOT, "raw", group),
        "configs": os.path.join(RESULTS_ROOT, "configs", group),
        "logs": os.path.join(RESULTS_ROOT, "logs", group),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def planned_run_label(spec: RunSpec, settings: ModeSettings) -> str:
    traj = settings.save_trajectory if spec.save_trajectory is None else spec.save_trajectory
    return (
        f"[{spec.experiment_group}] {spec.method_name} seed={spec.seed} "
        f"comm={spec.communication_interval} overlap={spec.overlap_penalty_weight} "
        f"episodes={settings.num_test} traj={traj}"
    )


def save_planned_config(spec: RunSpec, settings: ModeSettings) -> str:
    paths = ensure_group_dirs(spec.experiment_group)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(paths["configs"], f"planned_{spec.experiment_name}_seed{spec.seed}_{stamp}.json")
    payload = {
        "experiment_group": spec.experiment_group,
        "experiment_name": spec.experiment_name,
        "method_name": spec.method_name,
        "seed": spec.seed,
        "seeds_planned": settings.seeds,
        "communication_interval": spec.communication_interval,
        "overlap_penalty_weight": spec.overlap_penalty_weight,
        "num_test_maps": settings.num_test,
        "num_meta_agent": settings.num_meta_agent,
        "save_trajectory": settings.save_trajectory if spec.save_trajectory is None else spec.save_trajectory,
        "save_trajectory_episode_ids": spec.save_trajectory_episode_ids,
        "checkpoint_path": CHECKPOINT_PATH,
        "max_episode_steps": 128,
        "num_agents": 8,
        "sensor_range": 20,
        "evaluation_only": True,
        "timestamp": datetime.now().isoformat(),
        "status": "planned",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def execute_run(spec: RunSpec, settings: ModeSettings, dry_run: bool = False) -> int:
    print(planned_run_label(spec, settings))
    save_planned_config(spec, settings)
    if dry_run:
        print("  (dry-run: skipped execution)")
        return 0

    paths = ensure_group_dirs(spec.experiment_group)
    log_path = os.path.join(paths["logs"], f"{spec.experiment_name}.log")
    env = spec.env(settings)
    cmd = [sys.executable, "test_driver.py"] + spec.cli_args()

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# {planned_run_label(spec, settings)}\n")
        logf.write(f"# cmd: {' '.join(cmd)}\n")
        logf.write(f"# checkpoint: {CHECKPOINT_PATH}\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logf.write(proc.stdout)
    print(f"  log -> {log_path}")
    print(f"  exit code -> {proc.returncode}")
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
    return proc.returncode


def run_batch(
    specs: List[RunSpec],
    mode: str,
    dry_run: bool = False,
    stop_on_failure: bool = False,
    seeds_override: Optional[List[int]] = None,
    num_test_override: Optional[int] = None,
) -> int:
    settings = ModeSettings.from_mode(mode, seeds_override, num_test_override)
    worst = 0
    for spec in specs:
        spec.mode = mode
        code = execute_run(spec, settings, dry_run=dry_run)
        worst = max(worst, code)
        if stop_on_failure and code != 0:
            print(f"ERROR: stopping batch after failed run: {spec.experiment_name}")
            break
    return worst


def mark_representative_trajectory(
    specs: List[RunSpec],
    method_name: str,
    seed: int,
    episode_ids: str = "0",
) -> None:
    for spec in specs:
        if spec.method_name == method_name and spec.seed == seed:
            spec.save_trajectory = True
            spec.save_trajectory_episode_ids = episode_ids


def print_batch_summary(
    title: str,
    group: str,
    specs: List[RunSpec],
    settings: ModeSettings,
) -> None:
    n_runs = len(specs)
    n_episodes = n_runs * settings.num_test
    methods = sorted({s.method_name for s in specs})
    comms = sorted({s.communication_interval for s in specs})
    overlaps = sorted({s.overlap_penalty_weight for s in specs})
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"experiment_group: {group}")
    print(f"methods: {methods}")
    print(f"seeds: {settings.seeds}")
    print(f"num_test_maps: {settings.num_test}")
    print(f"communication_interval values: {comms}")
    print(f"overlap_penalty_weight values: {overlaps}")
    print(f"test_driver invocations: {n_runs}")
    print(f"estimated episodes: {n_episodes}")
    print(f"raw:    results/raw/{group}/")
    print(f"configs: results/configs/{group}/")
    print(f"logs:   results/logs/{group}/")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print("-" * 72)
    for spec in specs:
        traj = ""
        if spec.save_trajectory:
            traj = f" [traj ep={spec.save_trajectory_episode_ids}]"
        print(
            f"  {spec.method_name} seed={spec.seed} "
            f"comm={spec.communication_interval} overlap={spec.overlap_penalty_weight}{traj}"
        )
    print("=" * 72)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=["quick", "small", "normal", "full"],
        default="quick",
        help="full/normal = paper scale (5 seeds, 10 maps).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs only.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required to execute full/normal mode (not dry-run).",
    )
    parser.add_argument("--group", type=str, default=None, help="Override experiment group folder name.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Override seed list.")
    parser.add_argument("--num-test-maps", type=int, default=None, help="Override NUM_TEST maps per run.")


def execution_guard(mode: str, dry_run: bool, confirm: bool) -> bool:
    """Block accidental full-scale runs without --confirm."""
    if mode in ("full", "normal") and not dry_run and not confirm:
        print("ERROR: full/normal execution requires --confirm (evaluation only, no training).")
        print("Preview with: --mode full --dry-run")
        return False
    return True


def parse_settings_from_args(args) -> tuple:
    return ModeSettings.from_mode(args.mode, args.seeds, getattr(args, "num_test_maps", None))


def make_experiment_name(group: str, method_name: str, seed: int) -> str:
    safe_method = method_name.replace(".", "p")
    return f"{group}_{safe_method}_seed{seed}"
