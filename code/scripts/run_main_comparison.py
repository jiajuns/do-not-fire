#!/usr/bin/env python3
"""Main comparison: Original vs CommScheduled (communication scheduling only)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from experiment_utils import (
    RunSpec,
    add_common_args,
    execution_guard,
    make_experiment_name,
    mark_representative_trajectory,
    parse_settings_from_args,
    print_batch_summary,
    resolve_experiment_group,
    run_batch,
)

BASE_GROUP = "main_comparison"

# Stage 10 direction: core method = comm interval 5, no overlap penalty
VARIANTS = [
    ("Original", 1, 0.0),
    ("CommScheduled", 5, 0.0),
]


def build_specs(mode: str, group: str, settings) -> list:
    specs = []
    for method_name, comm, overlap in VARIANTS:
        for seed in settings.seeds:
            specs.append(
                RunSpec(
                    experiment_group=group,
                    experiment_name=make_experiment_name(group, method_name, seed),
                    method_name=method_name,
                    communication_interval=comm,
                    overlap_penalty_weight=overlap,
                    seed=seed,
                    mode=mode,
                )
            )
    if mode == "full":
        mark_representative_trajectory(specs, "Original", seed=settings.seeds[0], episode_ids="0")
        mark_representative_trajectory(specs, "CommScheduled", seed=settings.seeds[0], episode_ids="0")
    return specs


def main():
    parser = argparse.ArgumentParser(description="Run main comparison batch experiments.")
    add_common_args(parser)
    args = parser.parse_args()
    if not execution_guard(args.mode, args.dry_run, args.confirm):
        return 1
    group = resolve_experiment_group(BASE_GROUP, args.mode, args.group)
    settings = parse_settings_from_args(args)
    specs = build_specs(args.mode, group, settings)
    if args.dry_run:
        print_batch_summary("MAIN COMPARISON", group, specs, settings)
    return run_batch(
        specs,
        args.mode,
        dry_run=args.dry_run,
        seeds_override=args.seeds,
        num_test_override=args.num_test_maps,
    )


if __name__ == "__main__":
    sys.exit(main())
