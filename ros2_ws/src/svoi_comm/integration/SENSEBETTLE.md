# SenseBettle Integration Baseline

The vehicle/server baseline is `SUNNYsyy2005/SenseBettle` at commit
`58f89ee58dceedf059d011360ec923a063ebb571` (checked 2026-07-15). Fetch linked
repositories recursively:

```bash
git clone --recurse-submodules git@github.com:SUNNYsyy2005/SenseBettle.git
cd SenseBettle
git submodule update --init --recursive
```

Relevant directories are `SenseBeetle/` (class-1 vehicle stack) and
`real_robots_explore/` (server-side map merger, frontier detector, allocator,
UDP map receiver, and reliable TCP goal server).

## Experiment Vehicle Set

| Experiment index | Vehicle ID | Address | ROS domain | Desync phase |
|---:|---:|---|---:|---:|
| 0 | 1 | `192.168.8.48` | 1 | 0 s |
| 1 | 2 | `192.168.8.43` | 2 | 1 s |
| 2 | 7 | `192.168.8.141` | 7 | 2 s |
| 3 | 8 | `192.168.8.3` | 8 | 3 s |

Use `use_explore:=False` on these class-1 vehicles. The server CURE allocator
must be the only source of exploration goals.

## Apply the Experiment Hook

After deploying `svoi_comm` to `~/communication/src/svoi_comm` on a vehicle:

```bash
cd ~/SenseBeetle
git apply --check ~/communication/src/svoi_comm/integration/robot_communication_map_gate.patch
git apply ~/communication/src/svoi_comm/integration/robot_communication_map_gate.patch
colcon build --packages-select robot_communication
```

The patch remaps the UDP map uploader from `/map` to `/svoi/gated_map` and
gates its continuous TF stream with `/svoi/fire`. Navigation status and the
reliable goal channel remain live. Do not start the patched communication node
without `map_gate_node`. To remove the hook, run `git apply -R` with the same
patch and rebuild.

The deployed robot-2 communication source currently matches this upstream
baseline. Its `bringup_real.launch.py` differs only in LiDAR `range_max`
(`20.0` deployed versus `10.0` upstream); preserve that deliberate local
setting unless the sensing protocol changes.

Robot 1 at `192.168.8.48` uses an older local communication variant whose map
subscription is named `map_postprocessed` and whose message dependencies differ.
Apply `robot_communication_map_gate_robot1.patch` there, but feed
`map_gate_node` from slam_toolbox's actual `/map` topic; the deployed workspace
contains no publisher for `/map_postprocessed`. The patched subscription still
receives `/svoi/gated_map`. Do not replace this vehicle-specific source tree
with robot 2's version.

## Laptop Controller

From the repository root, the shortest two-vehicle smoke/demo workflow is:

```bash
ros2_ws/src/svoi_comm/scripts/control_map_gates.sh start sync
ros2_ws/src/svoi_comm/scripts/control_map_gates.sh stop
ros2_ws/src/svoi_comm/scripts/control_map_gates.sh start desync
```

The controller supplies the inventory values in the table, one shared start
time, and robot 1's special map topic. Use `status` or `logs` in place of
`start`; append `all` for the formal four-vehicle set. Formal mode fails before
starting anything if SSH or clock preflight fails.
