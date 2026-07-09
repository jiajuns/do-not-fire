**English** | [简体中文](EXPERIMENT.zh-CN.md)

# Real-Robot Validation Protocol

**What you are testing.** The *synchrony tax* on real hardware: at a matched per-robot broadcast rate,
de-synchronised broadcasting should reduce team travel and sensing overlap versus synchronised, on the
frozen explorer, over a real WiFi channel.

## 0. REVISION CHECKLIST — repeat trials required for the Sensors major revision (2026-07)

The reviewers require **task-level evidence on hardware**: repeated paired trials measuring travel and
sensing overlap, not only communication staleness. Sections 1–5 below apply unchanged; this checklist
marks what the existing protocol already covers `[x]` and what is **new or must be fixed** `[ ]`:

- [x] Conditions: synchronised vs even-staggered, `mode:=periodic interval:=T` (Section 2)
- [x] Record per trial: per-robot odometry (travel), occupancy maps / explored cells (coverage,
      overlap), broadcast + receive events with bytes (`comm_stats`, `/svoi/state`), WiFi stats,
      coverage-over-time, top-down video (Section 4 table)
- [x] Same arena, paired start layout, run to a coverage target or time cap (Section 3)
- [ ] **At least 5 paired trials** (the previous session produced a single pair; 5 is the minimum
      for a paired Wilcoxon test, 8 or more preferred)
- [ ] **Fix robot1's bag recording before the session**: in BOTH previous runs its bag contained no
      `/Odometry` or `/odom` (only costmap/rosout topics). Add its odometry topic to the record list
      and verify with a 10-second smoke recording + `ros2 bag info`
- [ ] **Verify all four robots actually move in every trial**: in the previous synchronised run
      robot3 travelled 0.26 m in total (effectively stationary). In the first 30 s of each trial,
      check every robot has a goal and is moving; otherwise abort and redo that trial
- [ ] Keep `T = 4 s` (matches the published demonstration); synchronised = common phase,
      staggered = offsets `{0, 1, 2, 3}` s for robots 0–3
- [ ] Alternate the condition order (sync, desync, desync, sync, ...) to cancel battery and scene
      drift across the session
- [ ] One run-ID directory per trial (`sync_<stamp>` / `desync_<stamp>`) and a note of which two
      runs form a pair
- [ ] Optional: log the battery level at the start of each trial (useful covariate)

**What happens with the data:** parsed offline with `code/parse_real_bags.py` (no ROS installation
needed); per-trial travel from odometry, coverage/overlap from the maps, staleness from the public
bag; a paired Wilcoxon over the trials then goes into the paper's hardware section.

## 1. Hardware
4 wheeled robots; each with: an onboard computer, a Livox Mid-360 LiDAR with FAST-LIO2 for
odometry/mapping, ROS 2 (Humble) + Nav2, and WiFi (one shared network). Optional: motion-capture or a
surveyed map for ground-truth coverage.

## 2. The code you run (this repo)
ROS 2 package **`ros2_ws/src/svoi_comm`**:
- **`svoi_comm/svoi_core.py` — `EventTriggeredGate`**: the per-robot scheduler that decides *when* to
  broadcast. Modes mirror the simulation:
  - `periodic` (fixed `interval`) — use **in phase** for the **synchronised** baseline, or **phase-offset**
    per robot for **even-staggered**;
  - `spiking` / `evtrig` / `random` — distributed, self-de-synchronising triggers (no global clock).
- **`svoi_comm/comm_node.py`** — one node per robot. Subscribes its own pose (`/robotI/Odometry` from
  FAST-LIO2) and optional 64-D policy embedding; runs the gate; **broadcasts** its state
  `[id, x, y, yaw, emb...]` to the shared WiFi topic `/svoi/state` **only when the gate fires**; holds the
  last received teammate states in between; republishes the aggregated `team_state` to the planner; and
  logs `comm_stats = [fire_rate, bytes_sent, fires, ticks]`.
- **`launch/team.launch.py`** — single-machine test (spawns N namespaced nodes). On real hardware each
  robot instead runs **one** `comm_node` with its own `robot_id`.
- The **frozen explorer policy** runs per robot (from the upstream framework; weights at
  `model/clean/checkpoint.pth`) and publishes its action and 64-D embedding; the planner consumes
  `team_state`.

**Scheduling conditions to compare** (set via the `mode` / `interval` params, all at the *same rate*):
1. **Synchronised** — `mode:=periodic interval:=T`, all robots in phase (identical fire steps).
2. **Even-staggered (de-sync)** — `mode:=periodic interval:=T`, robot *k* offset by a balanced
   `phase_offset` (stagger node start, or add a `phase_offset` param).
3. *(optional)* **Event-triggered de-sync** — `mode:=spiking` or `evtrig` (self-de-synchronising).

OW-Desync needs the centralised activation coordinator (a TODO node); the minimal hardware demo uses
conditions **1 vs 2**, which already test the synchrony tax.

## 3. How to operate (step by step)
1. **Build** on each robot: `cd ros2_ws && colcon build && source install/setup.bash`.
2. **Bring up per robot**: FAST-LIO2 (publishes `/robotI/Odometry`), the frozen explorer + Nav2, and
   `ros2 run svoi_comm comm_node --ros-args -p robot_id:=I -p n_agents:=4 -p mode:=periodic -p interval:=T`.
3. **Network**: put all robots on one WiFi network and the same `ROS_DOMAIN_ID` so `/svoi/state` crosses
   the network; sync clocks (chrony/PTP) or log clock offsets.
4. **Trials**: for each (arena, start layout), run **each schedule** (synchronised, even-staggered,
   [event-triggered]) **≥5 times**, paired. Record everything: `ros2 bag record -a` (or a topic list).
5. Run until a coverage target (e.g.\ 95%) or a time cap; keep everything identical except the schedule.

**Minimal viable demo**: 1 arena, synchronised vs even-staggered, N=4, fixed T, ~5 paired repeats.

## 4. What to record (one rosbag per trial)
| Signal | Used for |
|---|---|
| per-robot FAST-LIO2 pose | total travel distance |
| per-robot occupancy map + explored cells | coverage, **sensing overlap** (redundant coverage) |
| broadcast events `t, sender, payload_bytes` (`comm_stats`) | broadcast **count** + realised **volume (kB)** |
| receive events `t, receiver, sender` | belief **staleness** age τ_i, **peak team staleness** maxₜ Σ τ_i |
| WiFi stats: per-message latency, delivery/loss, throughput | channel realism |
| coverage-over-time, wall-clock | time/steps to target coverage |
| top-down video | qualitative figure (herd vs spread) |

Log raw timestamps on **both** send and receive so τ and realised volume are computed offline.

## 5. What you finally get
An offline parse of the bags (reuse the repo's analysis pipeline) yields, per schedule: total team
travel, sensing overlap, time/steps to coverage; realised comm **volume (kB)** + **broadcast count**
(showing the rate is matched, volume is not); **peak team staleness** (the mechanism); and WiFi
throughput/latency/loss. A paired **Wilcoxon** test (synchronised vs de-sync) then gives the paper
deliverables:
1. a **real-robot results table** — schedule vs {travel %, overlap %, p, n};
2. a **timing / staleness plot** — aligned vs staggered ticks; lower peak staleness for de-sync;
3. a **trajectory comparison figure** (herd vs spread) from the video/bags;
4. **WiFi link numbers** (showing the channel was real);
5. the released **rosbags + parsing script** for reproducibility.

**Expected result**: at an equal broadcast count, de-synchronisation cuts travel and overlap and shows a
lower peak team staleness — the synchrony tax, demonstrated on hardware.
