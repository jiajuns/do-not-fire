#!/usr/bin/env python3
"""SET-Comm (spiking event-triggered communication) evaluation batches.

Plugs into the same batch pipeline as run_main_comparison.py (evaluation only,
invokes test_driver.py). Select a sub-matrix with --matrix:

  main     : Original(interval1) | Periodic5 | SETComm(fixed) | SETComm-L(learned)
  pareto   : SETComm fixed across a threshold sweep (comm-vs-coverage front)
  random   : equal-budget random scheduler across firing probabilities
  ablation : trigger-signal / leak / refractory variants at a fixed threshold
  learned  : SETComm-L only (requires a trained gate, see train_spiking_gate.py)
  all      : union of the above

Eval uses the test map folder by default; pass --map-dir to point at a held-out
folder (keep it DISJOINT from the gate training maps to avoid leakage).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from experiment_utils import (
    RunSpec,
    add_common_args,
    execution_guard,
    make_experiment_name,
    parse_settings_from_args,
    print_batch_summary,
    resolve_experiment_group,
    run_batch,
)

BASE_GROUP = "setcomm"

PARETO_THRESHOLDS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
RANDOM_PROBS = [0.1, 0.2, 0.35, 0.5, 0.7]


def _spec(group, method, seed, **kw):
    return RunSpec(
        experiment_group=group,
        experiment_name=make_experiment_name(group, method, seed),
        method_name=method,
        communication_interval=kw.pop("interval", 1),
        overlap_penalty_weight=0.0,
        seed=seed,
        **kw,
    )


def build_specs(matrix, group, settings, args):
    specs = []
    seeds = settings.seeds
    th = args.threshold
    gate = args.gate_path

    def add(method, seed, **kw):
        specs.append(_spec(group, method, seed, map_dir=args.map_dir, **kw))

    for seed in seeds:
        if matrix in ("main", "all"):
            add("Original", seed, comm_mode="periodic", interval=1)
            add("Periodic_5", seed, comm_mode="periodic", interval=5)
            add(f"SETComm_th{th:g}", seed, comm_mode="spiking", spike_threshold=th)
            add("SETComm_L", seed, comm_mode="spiking_learned", spike_gate_path=gate)
        if matrix in ("pareto", "all"):
            for t in PARETO_THRESHOLDS:
                add(f"SETComm_th{t:g}", seed, comm_mode="spiking", spike_threshold=t)
        if matrix in ("random", "all"):
            for p in RANDOM_PROBS:
                add(f"Random_p{p:g}", seed, comm_mode="random", random_comm_prob=p)
        if matrix in ("ablation", "all"):
            add(f"SETComm_th{th:g}_driftonly", seed, comm_mode="spiking",
                spike_threshold=th, spike_w_drift=1.0, spike_w_cov=0.0)
            add(f"SETComm_th{th:g}_driftcov", seed, comm_mode="spiking",
                spike_threshold=th, spike_w_drift=1.0, spike_w_cov=1.0)
            add(f"SETComm_th{th:g}_leak0.9", seed, comm_mode="spiking",
                spike_threshold=th, spike_leak=0.9)
            add(f"SETComm_th{th:g}_refr2", seed, comm_mode="spiking",
                spike_threshold=th, spike_refractory=2)
        if matrix == "learned":
            add("SETComm_L", seed, comm_mode="spiking_learned", spike_gate_path=gate)

    # de-duplicate (main+pareto+all can repeat the same fixed threshold)
    seen, unique = set(), []
    for s in specs:
        key = (s.method_name, s.seed)
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Run SET-Comm evaluation batches.")
    add_common_args(parser)
    parser.add_argument("--matrix", choices=["main", "pareto", "random", "ablation", "learned", "all"],
                        default="main")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="fixed threshold used by 'main'/'ablation' SETComm variants")
    parser.add_argument("--gate-path", type=str, default="model/spiking_gate.pth",
                        help="learned gate weights for SETComm-L")
    parser.add_argument("--map-dir", type=str, default=None,
                        help="held-out eval map folder (sets BLMRE_MAP_DIR)")
    args = parser.parse_args()

    if not execution_guard(args.mode, args.dry_run, args.confirm):
        return 1
    group = resolve_experiment_group(BASE_GROUP, args.mode, args.group)
    settings = parse_settings_from_args(args)
    specs = build_specs(args.matrix, group, settings, args)
    if args.dry_run:
        print_batch_summary(f"SET-COMM [{args.matrix}]", group, specs, settings)
    return run_batch(
        specs, args.mode, dry_run=args.dry_run,
        seeds_override=args.seeds, num_test_override=args.num_test_maps,
    )


if __name__ == "__main__":
    sys.exit(main())
