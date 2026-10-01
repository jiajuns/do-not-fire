#!/usr/bin/env bash
set -uo pipefail
# car_stack.sh -- one-key start / clean-kill for ONE car's motion stack.
#
# Deployed by exp.sh to ~/svoi_experiment/car_stack.sh on each car and run there
# (directly or over SSH). Reads its own ~/svoi_experiment/car.conf (VID/DOMAIN/WS/
# SERVER_IP/HELPER). Does NOT modify any of the car's original code: the serial
# driver is started through the car's OWN helper, and the nav launch + tuned
# configs are COPIES under ~/svoi_experiment.
#
# Usage:  car_stack.sh start | kill | status
#
#   start : serial driver (via car helper) + SLAM/Nav2/FAST-LIO (manual mode,
#           use_explore:=False, short lidar range, CPU-relief configs).
#   kill  : graceful helper stop, then a thorough -9 sweep of every ROS node so
#           nothing survives between repeated trials.
#   status: prints serial / nav / TF readiness.

conf="$HOME/svoi_experiment/car.conf"
[[ -f "$conf" ]] || { echo "missing $conf" >&2; exit 2; }
# shellcheck disable=SC1090
source "$conf"
: "${VID:?}"; : "${DOMAIN:?}"; : "${WS:?}"; : "${SERVER_IP:=192.168.8.30}"
: "${HELPER:=$HOME/communication/src/svoi_comm/scripts/control_robot_stack.sh}"

run_dir="$HOME/svoi_runs/control_stack"; mkdir -p "$run_dir"
nav_pid_file="$run_dir/navigation.pid"; nav_log="$run_dir/navigation.log"
exp_dir="$HOME/svoi_experiment"; mkdir -p "$exp_dir"

alive() { [[ -f "$1" ]] && kill -0 "$(<"$1")" 2>/dev/null; }

start_nav() {
  if alive "$nav_pid_file"; then echo "nav already running, PID=$(<"$nav_pid_file")"; return 0; fi
  if pgrep -f 'bringup_real.*\.launch\.py' >/dev/null; then
    echo "unmanaged nav launch present, refusing to double-start" >&2; return 1; fi
  set +u
  source /opt/ros/humble/setup.bash
  source "$WS/install/setup.bash"
  [[ -f "$HOME/communication/install/setup.bash" ]] && source "$HOME/communication/install/setup.bash"
  set -u
  export ROS_DOMAIN_ID="$DOMAIN"
  # Open the car's own nav RViz (nav2.rviz, has the 2D Nav Goal tool) when a display is present
  # -- i.e. when you run this on the car's desktop / NoMachine. Headless SSH (exp.sh) starts no RViz.
  local rviz_arg=False
  [ -n "${DISPLAY:-}" ] && rviz_arg=True
  # Copy the car's own launch (edits below are on the COPY only). Lidar range is left at the car's
  # own default -- manual driving needs the full range/map, no artificial arena cap.
  local src_launch="$WS/install/nav_bringup/share/nav_bringup/launch/bringup_real.launch.py"
  local sr_launch="$exp_dir/bringup_real_shortrange.launch.py"
  cp "$src_launch" "$sr_launch"
  # CPU-relief so the livox tf message-filter does not overflow ("queue is full") and
  # drop every scan. All edits are on COPIES under ~/svoi_experiment.
  local np_dst="$exp_dir/nav2_params.yaml"
  cp "$WS/install/nav_bringup/share/nav_bringup/config/reality/nav2_params_real.yaml" "$np_dst"
  sed -i -E 's/(transform_tolerance:[[:space:]]*)[0-9.]+/\13.0/' "$np_dst"
  sed -i -E 's/(update_frequency:[[:space:]]*)20\.0/\14.0/' "$np_dst"
  sed -i -E 's/(publish_frequency:[[:space:]]*)10\.0/\12.0/' "$np_dst"
  sed -i -E 's/(update_frequency:[[:space:]]*)5\.0/\12.0/' "$np_dst"
  sed -i -E 's/^([[:space:]]*)resolution:[[:space:]]*0\.02/\1resolution: 0.05/' "$np_dst"
  sed -i 's|"nav2_params_real.yaml"|"'"$np_dst"'"|' "$sr_launch"
  local sm_dst="$exp_dir/mapper_params.yaml"
  cp "$WS/install/nav_bringup/share/nav_bringup/config/reality/mapper_params_online_async_real.yaml" "$sm_dst"
  sed -i -E 's/^([[:space:]]*)resolution:[[:space:]]*0\.02/\1resolution: 0.05/' "$sm_dst"
  sed -i -E 's/(minimum_time_interval:[[:space:]]*)[0-9.]+/\10.8/' "$sm_dst"
  sed -i -E 's/(map_update_interval:[[:space:]]*)[0-9.]+/\12.0/' "$sm_dst"
  sed -i 's|"mapper_params_online_async_real.yaml"|"'"$sm_dst"'"|' "$sr_launch"
  # FAST-LIO downsample: at stock settings odom lags ~1.5s on these Jetsons (can't keep up with the
  # livox stream), which freezes slam -> no localization. Coarsen a COPY of the LIO config and
  # repoint the fastlio include to it. Big levers for a 4x6m arena: fewer input points
  # (point_filter_num 3->8), coarser map voxel (filter_size_map ->0.3), tiny map radius
  # (det_range 100->10 m). The car's original mid360.yaml is untouched.
  local lio_src="$WS/install/fast_lio/share/fast_lio/config/mid360.yaml"
  local lio_dst="$exp_dir/mid360_light.yaml"
  if [[ -f "$lio_src" ]]; then
    cp "$lio_src" "$lio_dst"
    sed -i -E 's/(point_filter_num:[[:space:]]*)[0-9]+/\18/' "$lio_dst"
    sed -i -E 's/(filter_size_surf:[[:space:]]*)[0-9.]+/\10.15/' "$lio_dst"
    sed -i -E 's/(filter_size_map:[[:space:]]*)[0-9.]+/\10.3/' "$lio_dst"
    sed -i -E 's/(det_range:[[:space:]]*)[0-9.]+/\110.0/' "$lio_dst"
    sed -i -E 's/(cube_side_length:[[:space:]]*)[0-9.]+/\150.0/' "$lio_dst"
    # repoint the fastlio include (mapping.launch.py) at the coarsened copy
    sed -i 's#{"lidar_type": lidar_type}\.items()#{"lidar_type": lidar_type, "config_path": "'"$exp_dir"'", "config_file": "mid360_light.yaml"}.items()#g' "$sr_launch"
  fi
  setsid env DISPLAY="${DISPLAY:-}" ros2 launch "$sr_launch" \
    mode:=mapping lio:=fastlio lio_rviz:=False nav_rviz:="$rviz_arg" \
    use_sim_time:=False use_foxglove_bridge:=False \
    use_explore:=False use_hik_camera:=False \
    robot_id:="$VID" server_ip:="$SERVER_IP" \
    >"$nav_log" 2>&1 </dev/null &
  local pid=$!; echo "$pid" > "$nav_pid_file"; sleep 6
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "nav launch failed (ws=$WS):" >&2; tail -40 "$nav_log" >&2
    rm -f "$nav_pid_file"; return 1; fi
  echo "nav up, PID=$pid, robot_id=$VID, ROS_DOMAIN_ID=$DOMAIN"
}

tf_ready() {
  # True when map->base_link exists (SLAM initialised -> car can accept goals).
  set +u; source /opt/ros/humble/setup.bash
  [[ -f "$HOME/communication/install/setup.bash" ]] && source "$HOME/communication/install/setup.bash"
  set -u; export ROS_DOMAIN_ID="$DOMAIN"
  # 12s: tf2_echo needs a few seconds to discover /tf + connect the chain over a fresh node.
  timeout 12 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q 'Translation'
}

do_kill() {
  # 1) graceful: the car helper flushes nav (SIGINT) + serial (SIGTERM) by pid file.
  "$HELPER" stop 2>/dev/null || true
  # 2) any ros2 launch PARENTS respawn children -> kill them first.
  pkill -9 -f 'ros2 launch' 2>/dev/null || true
  # 3) sledgehammer: every ROS node carries --ros-args. Three passes for stragglers.
  for _ in 1 2 3; do pkill -9 -f -- '--ros-args' 2>/dev/null || true; sleep 0.4; done
  # 4) a few driver processes may lack --ros-args.
  pkill -9 -f 'livox_ros_driver2|fast_lio|pointcloud_to_laserscan|async_slam_toolbox|rm_serial_driver' 2>/dev/null || true
  rm -f "$nav_pid_file" "$run_dir/serial.pid"
  sleep 0.5
  local left
  left="$(ps -eo cmd | grep -E '/opt/ros/humble/lib/|/install/.*/lib/.*node' | grep -v grep | wc -l)"
  echo "kill done; remaining ROS node processes: $left"
  (( left == 0 ))
}

case "${1:-}" in
  start)
    echo "[$VID] starting serial via car helper..."
    "$HELPER" start-serial "$VID" "$DOMAIN" "$SERVER_IP" || echo "[$VID] serial start returned non-zero (may already be up)"
    echo "[$VID] starting nav..."
    start_nav || exit 1
    ;;
  kill)  do_kill ;;
  restart)   # one-key clean restart: kill everything, then start fresh
    echo "[restart] killing old stack..."; do_kill || true; sleep 3
    echo "[restart] starting serial via car helper..."
    "$HELPER" start-serial "$VID" "$DOMAIN" "$SERVER_IP" || echo "[$VID] serial start returned non-zero (may already be up)"
    echo "[restart] starting nav..."; start_nav || exit 1 ;;
  tf)    tf_ready ;;   # exit 0 iff map->base_link is available (quiet; for exp.sh polling)
  status)
    alive "$run_dir/serial.pid" && echo "[$VID] serial: running" || echo "[$VID] serial: down"
    alive "$nav_pid_file" && echo "[$VID] nav: running" || echo "[$VID] nav: down"
    if tf_ready; then echo "[$VID] TF map->base_link: READY"; else echo "[$VID] TF map->base_link: not ready"; fi
    ;;
  *) echo "Usage: $0 start|restart|kill|status" >&2; exit 2 ;;
esac
