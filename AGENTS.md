# Repository Guidelines

## Project Structure & Module Organization

Core Python simulation code lives in `code/`. Communication policies are implemented in `code/spiking_comm.py`; environment and policy components use files such as `env.py`, `agent.py`, and `model.py`. Experiment entry points follow `run_*.py`, analyses use `analyze_*.py`, and figure generation uses `gen_*.py` or `regen_*.py`. Shared experiment and plotting utilities are under `code/scripts/`. The ROS 2 Humble package is in `ros2_ws/src/svoi_comm/`, with nodes in `svoi_comm/` and launch files in `launch/`. See `EXPERIMENT.md` for the hardware protocol.

Generated results belong in `results/` and are ignored. The frozen checkpoint expected at `model/clean/checkpoint.pth` and map data are external artifacts; do not commit them.

## Build, Test, and Development Commands

Create a Python 3.10 environment, then install simulation dependencies:

```bash
pip install -r requirements.txt
python code/run_sync.py
python code/owval_master.py
python code/regen_oldfigs.py
```

These commands run the matched-rate synchronization comparison, validate OW-Desync, and regenerate legacy figures, respectively. For a small local check, run `python code/run_smoke_local.py --comm-mode spiking --threshold 0.8 --map-dir <maps>` after supplying maps and the checkpoint.

Build and exercise the ROS package with:

```bash
cd ros2_ws
colcon build --packages-select svoi_comm
source install/setup.bash
colcon test --packages-select svoi_comm
```

## Coding Style & Naming Conventions

Use four-space indentation and standard PEP 8 conventions. Name modules, functions, and variables with `snake_case`, classes with `PascalCase`, and configuration constants with `UPPER_SNAKE_CASE` (including `BLMRE_*` environment variables). Keep runner scripts thin, put reusable logic in modules, and add concise docstrings to public functions and non-obvious algorithms. No formatter or linter is configured, so preserve nearby style and run `python -m compileall code ros2_ws/src/svoi_comm/svoi_comm` before submitting.

## Testing Guidelines

No coverage threshold or dedicated unit-test suite is currently committed. Add pytest tests under `code/tests/` or `ros2_ws/src/svoi_comm/test/`, named `test_*.py`. For scheduler changes, include deterministic seeds and compare communication volume, coverage, travel, and overlap at matched budgets. Run the smoke check and the smallest relevant experiment; document required external artifacts.

## Commit & Pull Request Guidelines

History favors concise, imperative subjects, optionally scoped (`Docs: ...`). Keep each commit focused. Pull requests should explain the hypothesis or behavior changed, list exact validation commands, link related issues, and report seeds/configuration. Include plots or screenshots for figure, trajectory, launch, or robot-behavior changes, but exclude checkpoints, raw results, logs, and ROS build artifacts.
