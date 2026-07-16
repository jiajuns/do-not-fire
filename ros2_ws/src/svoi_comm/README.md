# svoi_comm — real-robot SVoI communication scheduling (ROS2 Humble)

Deploys the simulation's communication-scheduling thesis onto **4 custom wheeled robots**
(Livox Mid-360 + FAST-LIO2, onboard compute, real WiFi). Validates, on real hardware, that
**de-synchronised / value-of-information communication scheduling improves bandwidth-limited
multi-robot exploration efficiency** (less travel / overlap at matched comm).

## Architecture (per robot, distributed)
```
livox_ros_driver2 ──► FAST-LIO2 ──► /Odometry (pose) ─┐
        (already running on your robots)              │
                                                      ▼
   /cloud_registered ─► occupancy_projector ─► /robotI/occ (2D grid)  [TODO node]
                                                      │
                                /robotI/occ + team_state ─► policy_node ─► /cmd_vel  [TODO node]
                                                      │                 └─► /robotI/embedding (64-D)
                                                      ▼
                                  comm_node  ◄── this package (DONE)
                                   • LIF event-triggered gate (Pillar A): decide WHEN to
                                     broadcast own state over WiFi (/svoi/state)
                                   • aggregate neighbours' last (stale) states ─► ~/team_state
                                   • log real comm volume / fire rate ─► ~/comm_stats
```
A central `coordinator_node` (Pillar B: kWTA value-of-information link scheduler / learned
weights `[2,0,1,1,0]`) is the next node to add for targeted point-to-point scheduling.

## Status
- **DONE**: `svoi_core.py` (LIF gate, sim-matched modes), `comm_node.py` (per-robot
  distributed event-triggered broadcasting + neighbour aggregation + comm logging),
  packaging, `team.launch.py`.
- **TODO**: `occupancy_projector` (Livox 3D cloud → scaled 2D occupancy), `policy_node`
  (wrap DARS PolicyNet → waypoint → Nav2/cmd_vel), `metrics_node` (coverage/travel/overlap),
  `coordinator_node` (Pillar B targeted/learned scheduler).

## Build & run
```bash
cd ros2_ws && colcon build --packages-select svoi_comm && source install/setup.bash
# test (single machine, 4 namespaced robots):
ros2 launch svoi_comm team.launch.py n_agents:=4 mode:=spiking
# on each real robot (onboard):
ros2 run svoi_comm comm_node --ros-args -p robot_id:=0 -p n_agents:=4 \
    -p odom_topic:=/Odometry -p state_topic:=/svoi/state -p mode:=spiking
```
Modes (match the paper): `periodic` (synchronised baseline) · `spiking` (LIF de-sync) ·
`evtrig` (distance-triggered) · `random`.

## Gate the map path used by the real planner

`comm_node` is useful for state/AoI measurements, but it does not by itself alter the
existing UDP map/TF uploader. For the task-level sync/desync experiment, run
`map_gate_node` on the same `ROS_DOMAIN_ID` as SLAM. It caches `/map` and publishes
`/svoi/gated_map` at a common 4 s period. Remap `robot_communication`'s `map` input to
that output and gate its continuous TF stream using
`integration/robot_communication_map_gate.patch`. See `integration/SENSEBETTLE.md`
for the pinned recursive upstream baseline and vehicle inventory.

All robots must receive one common future wall-clock timestamp. At a 4 s period,
sync uses phase `0` on every robot; desync uses phases `0, 1, 2, 3` seconds:

```bash
export ROS_DOMAIN_ID=2
ros2 launch svoi_comm map_gate.launch.py robot_id:=1 period_sec:=4 \
  phase_sec:=1 start_time:=1784090000.0
```

Use `scripts/start_map_gate.sh` to calculate the balanced phase,
`scripts/record_trial.sh` for each vehicle bag, and `scripts/record_server_trial.sh`
for merged maps/frontiers/goals. `scripts/capture_wifi.sh` records actual UDP/TCP wire
traffic. The gate event byte count alone is only the ROS-serialized map size.

## Key adaptation for the 4m×6m arena (IMPORTANT)
Full Livox range sees the whole room at once → exploration is trivial. Cap the **effective
sensor range to ~1.0–1.5 m** in `occupancy_projector` and scale the policy's spatial params
~10× (NODE_RESOLUTION 4→~0.3 m, SENSOR_RANGE 20→~1.5 m) so a small space becomes a real
multi-step exploration + coordination problem. Add a few obstacles for occlusion.

## Sim-to-real de-risk (P1, no driving needed)
Record a teleop rosbag of one robot (`/Odometry` + Livox cloud), replay through
`occupancy_projector` + `policy_node` offline to confirm the scaled occupancy graph is dense
enough and waypoints are sane BEFORE live multi-robot trials. Fallback if the learned policy
does not transfer: use a frontier explorer for the hardware trials and validate just the
communication scheduling (keep the learned-policy efficiency results in simulation).
