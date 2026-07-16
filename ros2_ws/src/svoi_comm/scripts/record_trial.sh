#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: ROS_DOMAIN_ID=<local-domain> $0 sync|desync ROBOT_LABEL [RUN_ID]"
}

[[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
condition="$1"
robot_label="$2"
[[ "$condition" == "sync" || "$condition" == "desync" ]] || { usage; exit 2; }
run_id="${3:-${condition}_$(date +%Y%m%d_%H%M%S)}"
root="${RESULTS_ROOT:-$HOME/svoi_runs}"
out_dir="$root/$run_id/$robot_label"
mkdir -p "$out_dir"

{
  echo "run_id: $run_id"
  echo "condition: $condition"
  echo "robot: $robot_label"
  echo "ros_domain_id: ${ROS_DOMAIN_ID:-unset}"
  echo "odom_topics: [/Odometry, /odom]"
  echo "map_topics: [/map, /map_postprocessed, /svoi/gated_map]"
  echo "period_sec: ${PERIOD_SEC:-4.0}"
  echo "phase_sec: ${PHASE_SEC:-unset}"
  echo "start_time: ${START_TIME:-unset}"
  echo "pair_id: ${PAIR_ID:-unset}"
  echo "started_at: $(date --iso-8601=seconds)"
} > "$out_dir/metadata.yaml"

echo "Recording $run_id/$robot_label to $out_dir/bag (Ctrl-C to stop)"
exec ros2 bag record --include-hidden-topics -o "$out_dir/bag" \
  /Odometry /odom /map /map_postprocessed /map_updates \
  /tf /tf_static /plan /cmd_vel /goal_pose \
  /navigate_to_pose/_action/goal /navigate_to_pose/_action/feedback \
  /navigate_to_pose/_action/result /navigate_to_pose/_action/status \
  /svoi/gated_map /svoi/fire /svoi/comm_events /svoi/comm_stats
