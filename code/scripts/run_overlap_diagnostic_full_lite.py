#!/usr/bin/env python3
"""Optional full-lite overlap diagnostic (Stage 9.5 confirmation at larger sample)."""

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

BASE_GROUP = "overlap_diagnostic"
WEIGHTS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]


def build_specs(mode: str, group: str, settings) -> list:
    specs = []
    for weight in WEIGHTS:
        method = "Original" if weight == 0.0 else f"OverlapPenalty_{weight}"
        for seed in settings.seeds:
            specs.append(
                RunSpec(
                    experiment_group=group,
                    experiment_name=make_experiment_name(group, method, seed),
                    method_name=method,
                    communication_interval=1,
                    overlap_penalty_weight=weight,
                    seed=seed,
                    mode=mode,
                )
            )
    return specs


def main():
    parser = argparse.ArgumentParser(description="Overlap diagnostic full-lite.")
    add_common_args(parser)
    args = parser.parse_args()
    if not execution_guard(args.mode, args.dry_run, args.confirm):
        return 1
    group = args.group or "overlap_diagnostic_full_lite"
    mode = args.mode if args.mode in ("small", "quick", "full") else "small"
    settings = parse_settings_from_args(args)
    specs = build_specs(mode, group, settings)
    if args.dry_run:
        print_batch_summary("OVERLAP DIAGNOSTIC FULL-LITE (optional)", group, specs, settings)
    return run_batch(
        specs,
        mode,
        dry_run=args.dry_run,
        seeds_override=args.seeds,
        num_test_override=args.num_test_maps,
    )


if __name__ == "__main__":
    sys.exit(main())
