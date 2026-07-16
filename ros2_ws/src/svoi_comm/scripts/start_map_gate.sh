#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: ROS_DOMAIN_ID=<local-domain> $0 ROBOT_ID sync|desync START_TIME"
  echo "Example: ROS_DOMAIN_ID=2 $0 0 desync 1784090000.0"
}

[[ $# -eq 3 ]] || { usage; exit 2; }
robot_id="$1"
condition="$2"
start_time="$3"
n_agents="${N_AGENTS:-4}"
period_sec="${PERIOD_SEC:-4.0}"

case "$condition" in
  sync) phase_sec="0.0" ;;
  desync) phase_sec="$(python3 -c 'import sys; print(float(sys.argv[1]) * float(sys.argv[2]) / int(sys.argv[3]))' "$robot_id" "$period_sec" "$n_agents")" ;;
  *) usage; exit 2 ;;
esac

: "${ROS_DOMAIN_ID:?Set ROS_DOMAIN_ID to the robot navigation domain}"

exec ros2 launch svoi_comm map_gate.launch.py \
  robot_id:="$robot_id" n_agents:="$n_agents" \
  period_sec:="$period_sec" phase_sec:="$phase_sec" \
  start_time:="$start_time" map_topic:="${MAP_TOPIC:-/map}" \
  gated_map_topic:="${GATED_MAP_TOPIC:-/svoi/gated_map}"
