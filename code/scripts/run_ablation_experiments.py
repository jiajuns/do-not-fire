#!/usr/bin/env python3
"""Diagnostic ablation: comm scheduling vs overlap penalty (not main contribution)."""

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

BASE_GROUP = "ablation"

VARIANTS = [
    ("Original", 1, 0.0),
    ("CommOnly", 5, 0.0),
    ("OverlapOnly", 1, 0.1),
    ("CommPlusOverlap", 5, 0.1),
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
    return specs


def main():
    parser = argparse.ArgumentParser(description="Ablation / diagnostic experiments.")
    add_common_args(parser)
    args = parser.parse_args()
    if not execution_guard(args.mode, args.dry_run, args.confirm):
        return 1
    group = resolve_experiment_group(BASE_GROUP, args.mode, args.group)
    settings = parse_settings_from_args(args)
    specs = build_specs(args.mode, group, settings)
    if args.dry_run:
        print_batch_summary("ABLATION / DIAGNOSTIC", group, specs, settings)
    return run_batch(
        specs,
        args.mode,
        dry_run=args.dry_run,
        seeds_override=args.seeds,
        num_test_override=args.num_test_maps,
    )


if __name__ == "__main__":
    sys.exit(main())
