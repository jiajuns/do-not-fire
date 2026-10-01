#!/usr/bin/env bash
set -uo pipefail
# start_server_manual.sh -- laptop CURE server in MANUAL-DRIVE mode.
#
# Sources ROS + the in-repo underlay + the cure_fix_ws overlay (fixed map merger),
# then launches real_manual.launch.py: map merge + reliable_communication + the
# RViz goal relay + RViz, with NO autonomous frontier explorer. You drive by
# clicking 2D Nav Goals in RViz; pick the target car with exp.sh target <ID>.
#
# Usage:  start_server_manual.sh start|stop|status

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$root/.." && pwd)"
conf="$root/cars.conf"; [[ -f "$conf" ]] && source "$conf"
: "${ROBOT_IDS:=1,2,7}"; : "${ROBOT_POSES:=0,0,0;0,0.7,0;0,1.4,0}"

underlay="$repo/SenseBettle/real_robots_explore/install/setup.bash"
overlay="$repo/cure_fix_ws/install/setup.bash"
run_dir="$root/.run"; mkdir -p "$run_dir"
pid_file="$run_dir/server.pid"; log_file="$run_dir/server.log"

alive() { [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; }

case "${1:-}" in
  start)
    if alive; then echo "server already running, PID=$(<"$pid_file")"; exit 0; fi
    [[ -f "$underlay" ]] || { echo "missing underlay: $underlay" >&2; exit 1; }
    [[ -f "$overlay"  ]] || { echo "missing overlay: $overlay"  >&2; exit 1; }
    set +u
    source /opt/ros/humble/setup.bash
    source "$underlay"
    source "$overlay"
    set -u
    export ROS_DOMAIN_ID=0 DISPLAY="${DISPLAY:-:0}"
    setsid ros2 launch real_cure_bringup real_manual.launch.py \
      robot_ids:="$ROBOT_IDS" robot_poses:="$ROBOT_POSES" \
      >"$log_file" 2>&1 </dev/null &
    pid=$!; echo "$pid" > "$pid_file"; sleep 6
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "server launch failed:" >&2; tail -40 "$log_file" >&2; rm -f "$pid_file"; exit 1; fi
    echo "manual server up, PID=$pid (robot_ids=$ROBOT_IDS, poses=$ROBOT_POSES)"
    echo "log: $log_file"
    ;;
  stop)
    if alive; then
      pid="$(<"$pid_file")"
      kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
      for _ in {1..10}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
      kill -9 -- "-$pid" 2>/dev/null || true
      echo "server stopped, PID=$pid"
    else
      echo "server was not running"
    fi
    # sweep any stragglers from this launch (laptop only runs the CURE server on ROS)
    pkill -9 -f 'real_manual.launch.py' 2>/dev/null || true
    pkill -9 -f 'map_merger_simple_node|rviz_goal_relay|reliable_server_node|robot_position_publisher' 2>/dev/null || true
    rm -f "$pid_file"
    ;;
  status)
    if alive; then echo "server: running, PID=$(<"$pid_file")"; else echo "server: down"; fi
    ;;
  *) echo "Usage: $0 start|stop|status" >&2; exit 2 ;;
esac
