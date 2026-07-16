# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

"Don't Fire Together" is a **test-time communication-scheduling layer** for bandwidth-limited
multi-robot exploration, plus its real-robot validation package. The learned graph-attention
explorer (from Ma et al.) is used **frozen and unmodified** — nothing here trains or changes the
policy or the planner. The research contribution is only *when/which* robots broadcast their state.

Central thesis (drives the whole experiment design): under scarce communication the controllable
cost is **synchronisation, not volume**. The decisive control is `periodic` (synchronised) vs
`staggered` (even phase-offset) broadcasting at a *matched* per-robot rate; if staggered wins, the
cost is synchronisation specifically. See `code/run_sync.py` for that headline experiment.

Two independent halves:
- `code/` — the Python simulation (frozen explorer + scheduling layer + evaluation + figures).
- `ros2_ws/src/svoi_comm/` — a ROS 2 Humble package deploying the same scheduler on 4 real robots.

`AGENTS.md` holds contributor conventions (style, commit/PR norms) and complements this file.

## Environment note

Always invoke Python as `python3` (not `python`) in this environment.

## Simulation (`code/`)

```bash
conda create -n desync python=3.10 && conda activate desync
pip install -r requirements.txt        # torch, numpy, pandas, scipy, matplotlib, imageio, rosbags
```

The frozen weights are **not in the repo**; obtain them from the upstream project and place at
`model/clean/checkpoint.pth` (`run_eval_one.py` resolves this relative to `code/`). Map images and
all `results/` are external / gitignored.

### The one atomic job

Everything routes through a single per-process job unit; the many `run_*.py` are just orchestrators
that fan this out over methods/seeds with `ThreadPoolExecutor` and then aggregate:

```bash
python3 code/run_eval_one.py --method-name NAME --comm-mode MODE --seed S \
    --num-maps N --group G --map-dir /abs/path/to/maps
```

It runs one `(method, seed)` across `N` maps and writes a per-episode CSV + coverage curves under
`results/`. `--comm-mode` is one of: `periodic staggered spiking spiking_learned random evtrig cwd
submod vlw desync stagfix owdesync`. Add `--targeted` for the budgeted pairwise link scheduler
(`--target-mode voi|voi_lif|err|prox|stale|whittle|decision_voi|round_robin|random`).

### Common entry points

```bash
python3 code/run_sync.py         # decisive sync vs even-staggered at matched rate, N=8
python3 code/run_syncN.py ; python3 code/run_ploss.py ; python3 code/run_srange.py   # generality
python3 code/owval_master.py     # OW-Desync standalone validation (peak-AoI vs round-robin)
python3 code/run_overnight.py    # full pipeline: split maps -> smoke -> train gates -> eval matrix -> REPORT.md
# quick stack check (needs maps + checkpoint):
python3 code/run_smoke_local.py --comm-mode spiking --threshold 0.8 --map-dir <maps>
# regenerate paper figures:
python3 code/regen_oldfigs.py ; python3 code/regen_owfigs.py ; python3 code/gen_meaningful_figs.py
```

Working directory matters: `results/`, `maps_test/`, `model/` resolve against the current directory
(orchestrators run children with `cwd=code/`), while `--map-dir` takes an absolute path. Run from
the repo root as the README shows, or from `code/`.

## Architecture — how the scheduling layer plugs into the frozen explorer

- **`code/spiking_comm.py`** is the scheduling layer. `CommGate.should_broadcast(...)` decides *when*
  a robot broadcasts (all the broadcast modes live here as one big dispatch); `LinkScheduler.select(...)`
  is the kWTA *targeted/pairwise* scheduler under a hard per-step link budget. `make_comm_gate()` is
  the factory that reads config and builds the gate.
- **`code/test_worker.py`** (`TestWorker.run_episode`) is the actual simulator loop and the single
  integration point. Each step: every robot computes its 64-D policy embedding; the gate decides
  whether to broadcast; `send_msg` distributes state into teammates' mailboxes (range-limited by
  `R_COMM` for OW-Desync); the frozen `PolicyNet` then selects the next waypoint from the (possibly
  stale) team state. The targeted path (`_run_targeted_step`) and predictive/Kalman belief
  maintenance also live here.
- The frozen explorer itself — `model.py` (`PolicyNet`), `agent.py`, `env.py`, `sensor.py`,
  `local_node_manager_quadtree.py`, `ground_truth_node_manager.py`, `quads.py`, `utils.py` — is the
  upstream framework; treat it as fixed.
- `experiment_logger.py` + `episode_metrics.py` produce the per-episode CSVs, coverage curves, and
  trajectory bundles; `scripts/aggregate_results.py` + `scripts/analyze_setcomm.py` roll them into
  Pareto fronts and paired Wilcoxon stats.

### Configuration is via `BLMRE_*` environment variables, read at import time

This is the most important and least obvious fact. `test_parameter.py` and `parameter.py` read all
knobs from `BLMRE_*` env vars **at module import**. `run_eval_one.py` therefore sets
`os.environ["BLMRE_*"] = ...` **before** `import test_parameter`. When adding a config knob, follow
that order or the value will be ignored. Key groups: `BLMRE_COMM_MODE` and `BLMRE_SPIKE_*`
(broadcast gate), `BLMRE_COMM_TARGETED`/`BLMRE_LINK_BUDGET`/`BLMRE_TARGET_MODE`/`BLMRE_VOI_*`
(targeted scheduler), `BLMRE_R_COMM`/`BLMRE_HETERO_SRANGE` (range-limited / heterogeneous channels),
`BLMRE_PREDICTIVE_BELIEF`/`BLMRE_KALMAN_BELIEF` (belief maintenance), `BLMRE_FRONTIER*`
(non-learned backbone for the generality study), `BLMRE_N_AGENTS`/`BLMRE_SENSOR_RANGE`/`BLMRE_SEED`.

### Naming gotcha

`test_parameter.py` is the **config module** and `test_worker.py` is the **simulator** — despite the
`test_` prefix, neither is a unit test. There is no pytest suite under `code/`. `test_driver.py` is a
Ray-based driver (the Ray-free `run_eval_one.py` is what the orchestrators use).

## ROS 2 package (`ros2_ws/src/svoi_comm/`)

The hardware and simulation deliberately **share the same scheduler modes** (`periodic`, `spiking`,
`evtrig`, `random`) so results transfer.

```bash
cd ros2_ws && colcon build --packages-select svoi_comm && source install/setup.bash
colcon test --packages-select svoi_comm
# or run the unit test directly (plain unittest, no colcon needed):
python3 -m pytest src/svoi_comm/test/test_svoi_core.py
```

- **`svoi_comm/svoi_core.py`** — ROS-agnostic scheduler core (no ROS imports): `EventTriggeredGate`
  (per-robot LIF gate mirroring the sim modes) and `PeriodicPhaseSchedule` (wall-clock phase schedule
  for independently started robots; late timer callbacks emit at most one event, never a burst).
- **`comm_node.py`** — one node per robot: gates broadcasts of `[id, x, y, yaw, emb...]` to the shared
  `/svoi/state`, aggregates teammates' last states, logs `comm_stats`.
- **`map_gate_node.py`** — gates the `/map` republish on a common wall-clock schedule to
  `/svoi/gated_map`. This is what actually gates the *real planner's* map path (task-level sync/desync
  experiment); `comm_node` alone only measures state/AoI.
- Console entry points (`setup.py`): `comm_node`, `map_gate_node`, `pose_file_writer`.

For the physical 4-robot rig, `run_all.sh sync|desync` SSHes into the vehicles and starts
serial/nav/gate/recording; `run_real.sh` controls the map gates. `code/parse_real_bags.py` parses the
recorded rosbags **offline without a ROS install** (uses the `rosbags` pip package). The full
hardware protocol is in `EXPERIMENT.md`.

## Before submitting

No formatter/linter is configured. Byte-compile as a minimal check and preserve nearby style:

```bash
python3 -m compileall code ros2_ws/src/svoi_comm/svoi_comm
```
