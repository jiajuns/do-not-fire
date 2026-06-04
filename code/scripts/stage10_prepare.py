#!/usr/bin/env python3
"""Stage 10: dry-run all full paper-scale experiment plans and write preparation log."""

import os
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_PATH = os.path.join(PROJECT_ROOT, "results/logs/stage10_full_evaluation_preparation_log.txt")
PY = sys.executable

DRY_RUNS = [
    ([PY, "scripts/run_main_comparison.py", "--mode", "full", "--dry-run"], "A. main_comparison_full"),
    ([PY, "scripts/run_comm_interval_experiments.py", "--mode", "full", "--dry-run"], "B. communication_interval_full"),
    ([PY, "scripts/run_ablation_experiments.py", "--mode", "full", "--dry-run"], "C. ablation_full"),
    (
        [PY, "scripts/run_overlap_diagnostic_full_lite.py", "--mode", "small", "--dry-run",
         "--group", "overlap_diagnostic_full_lite"],
        "D. overlap_diagnostic_full_lite (optional, small scale)",
    ),
]


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    lines = []
    lines.append("STAGE 10 FULL EVALUATION PREPARATION LOG")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append("NO FULL EXPERIMENTS WERE EXECUTED.")
    lines.append("NO TRAINING. NO MODEL WEIGHT CHANGES.")
    lines.append("")

    for cmd, title in DRY_RUNS:
        lines.append("\n" + "#" * 72)
        lines.append(f"# {title}")
        lines.append("#" * 72 + "\n")
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines.append(proc.stdout)
        lines.append(f"\nexit_code: {proc.returncode}\n")

    lines.append("\n" + "=" * 72)
    lines.append("MANUAL EXECUTION COMMANDS (require --confirm)")
    lines.append("=" * 72)
    manual = """
# 1) Full main comparison (10 runs, 100 episodes)
python scripts/run_main_comparison.py --mode full --confirm

# 2) Full communication interval sweep (25 runs, 250 episodes)
python scripts/run_comm_interval_experiments.py --mode full --confirm

# 3) Full ablation diagnostic (20 runs, 200 episodes)
python scripts/run_ablation_experiments.py --mode full --confirm

# 4) Optional overlap diagnostic full-lite (18 runs, 90 episodes @ small maps)
python scripts/run_overlap_diagnostic_full_lite.py --mode small --confirm \\
  --group overlap_diagnostic_full_lite --seeds 0 1 2 --num-test-maps 5

# 5) After all runs complete — aggregate and plot
python scripts/run_analysis_full.py --groups main,comm,ablation
python scripts/run_analysis_full.py --groups main,comm,ablation --include-overlap-diag

# 6) Coverage curves (if coverage_curves/ exists under *_full raw folders)
python scripts/plot_coverage_curves.py   # may need group filter; see EXPERIMENTS.md

# 7) Trajectory figures (only if representative .npz saved)
python scripts/plot_trajectories_full.py
"""
    lines.append(manual)

    lines.append("\nESTIMATED EPISODES (full mode: 5 seeds x 10 maps)")
    lines.append("  main_comparison_full:        2 methods x 5 seeds x 10 = 100")
    lines.append("  communication_interval_full: 5 intervals x 5 seeds x 10 = 250")
    lines.append("  ablation_full:               4 variants x 5 seeds x 10 = 200")
    lines.append("  TOTAL (required):            550 episodes")
    lines.append("  overlap_diagnostic_full_lite:  6 weights x 3 seeds x 5 = 90 (optional)")
    lines.append("")
    lines.append("Rough runtime: ~6-8 min/episode on RTX 4060 class GPU => ~55-75 hours for 550 episodes.")
    lines.append("Run groups sequentially; use stop_on_failure in scripts if a group crashes.")

    text = "\n".join(lines)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nLog written to {LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
