#!/usr/bin/env python3
"""Communication interval sweep (overlap penalty = 0)."""

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

BASE_GROUP = "communication_interval"
INTERVALS = [1, 2, 5, 10, 20]


def build_specs(mode: str, group: str, settings, intervals=None) -> list:
    use_intervals = intervals if intervals is not None else INTERVALS
    if mode == "quick":
        use_intervals = [i for i in use_intervals if i in (1, 5)] or [1, 5]
    specs = []
    for interval in use_intervals:
        method = f"CommInterval_{interval}"
        for seed in settings.seeds:
            specs.append(
                RunSpec(
                    experiment_group=group,
                    experiment_name=make_experiment_name(group, method, seed),
                    method_name=method,
                    communication_interval=interval,
                    overlap_penalty_weight=0.0,
                    seed=seed,
                    mode=mode,
                )
            )
    return specs


def main():
    parser = argparse.ArgumentParser(description="Communication interval experiments.")
    add_common_args(parser)
    parser.add_argument(
        "--intervals",
        type=int,
        nargs="+",
        default=None,
        help="Override interval list (default: [1,2,5,10,20]; quick uses 1 and 5).",
    )
    args = parser.parse_args()
    if not execution_guard(args.mode, args.dry_run, args.confirm):
        return 1
    group = resolve_experiment_group(BASE_GROUP, args.mode, args.group)
    settings = parse_settings_from_args(args)
    specs = build_specs(args.mode, group, settings, args.intervals)
    if args.dry_run:
        print_batch_summary("COMMUNICATION INTERVAL SWEEP", group, specs, settings)
    return run_batch(
        specs,
        args.mode,
        dry_run=args.dry_run,
        seeds_override=args.seeds,
        num_test_override=args.num_test_maps,
    )


if __name__ == "__main__":
    sys.exit(main())
