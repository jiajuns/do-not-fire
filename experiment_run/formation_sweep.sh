#!/usr/bin/env bash
set -uo pipefail
# Formation sweep controller.
#
# The cars are placed in a physical row ~1 m apart (order 1,2,7,8 along y), all facing the
# same direction (+x). This sends each car a synchronized navigate_to_pose goal in that car's
# OWN `map` frame at (X, 0) -- i.e. X metres FORWARD of its own start pose. Because every car
# drives forward in its own frame from a parallel start, the row stays in formation and the
# lanes never cross => collisions are impossible, and no cross-robot frame calibration is needed.
#
# Map merging (the CURE server) is fed the same formation via robot_poses:="0,0,0;0,1,0;0,2,0;0,3,0"
# so it fuses the per-car maps into one global map.
#
# Usage:  formation_sweep.sh [X_metres_forward_from_start]   (default 2.0)
#   Call repeatedly with increasing X (2, 4, ...) to advance the sweep across the arena.

X="${1:-2.0}"
targets=(nvidia@192.168.8.49 nvidia@192.168.8.43 nvidia@192.168.8.141 nvidia@192.168.8.28)
vids=(1 2 7 8)
domains=(1 2 7 8)
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=5)

send_one() {
  local i="$1" ip="${targets[$i]}" dom="${domains[$i]}" vid="${vids[$i]}"
  echo "car $vid (dom $dom): sending goal x=$X m (forward of start)"
  timeout 150 ssh "${ssh_opts[@]}" "$ip" bash -s -- "$dom" "$X" "$vid" <<'REMOTE'
set +u
dom="$1"; X="$2"; vid="$3"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID="$dom"
# odometry sanity: present and not diverged (|pos| < 50 m). Guards against FAST-LIO blow-ups.
px=$(timeout 5 ros2 topic echo --once --field pose.pose.position.x /Odometry 2>/dev/null | head -1)
py=$(timeout 5 ros2 topic echo --once --field pose.pose.position.y /Odometry 2>/dev/null | head -1)
if [ -z "$px" ] || [ -z "$py" ]; then echo "  car$vid: NO odometry - not sending"; exit 2; fi
ok=$(awk -v a="$px" -v b="$py" 'BEGIN{print (a*a+b*b < 2500)?"OK":"BAD"}')
if [ "$ok" != "OK" ]; then echo "  car$vid: odom diverged (pos=$px,$py) - NOT sending"; exit 2; fi
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: 'map'}, pose: {position: {x: $X, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}" 2>&1 \
  | grep -iE 'Goal accepted|Goal finished|succeeded|aborted|rejected|canceled|Result' | head -4 | sed "s/^/  car$vid: /"
REMOTE
}

echo "=== formation sweep -> advance row to x=$X m (all cars in parallel) ==="
pids=()
for i in 0 1 2 3; do send_one "$i" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "=== sweep to x=$X m done ==="
