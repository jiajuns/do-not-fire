#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 start-serial|start-nav|stop|status|logs [VEHICLE_ID ROS_DOMAIN_ID SERVER_IP]"
}

[[ $# -ge 1 ]] || { usage; exit 2; }
action="$1"
vehicle_id="${2:-}"
domain="${3:-}"
server_ip="${4:-192.168.8.30}"
run_dir="$HOME/svoi_runs/control_stack"
serial_pid_file="$run_dir/serial.pid"
nav_pid_file="$run_dir/navigation.pid"
serial_log="$run_dir/serial.log"
nav_log="$run_dir/navigation.log"
mkdir -p "$run_dir"

alive() {
  [[ -f "$1" ]] && kill -0 "$(<"$1")" 2>/dev/null
}

source_ros() {
  set +u
  source /opt/ros/humble/setup.bash
  [[ -f "$HOME/SenseBeetle/install/setup.bash" ]] && \
    source "$HOME/SenseBeetle/install/setup.bash"
  [[ -f "$HOME/communication/install/setup.bash" ]] && \
    source "$HOME/communication/install/setup.bash"
  set -u
}

start_serial() {
  [[ -n "$domain" ]] || { usage; exit 2; }
  if alive "$serial_pid_file"; then
    echo "串口驱动已运行，PID=$(<"$serial_pid_file")" >&2
    exit 1
  fi
  if pgrep -f 'ros2 launch rm_serial_driver serial_driver.launch.py' >/dev/null; then
    echo "发现非本脚本管理的串口驱动，拒绝重复启动" >&2
    exit 1
  fi

  set +u
  source /opt/ros/humble/setup.bash
  source "$HOME/sensebeetle-control/install/setup.bash"
  set -u
  export ROS_DOMAIN_ID="$domain"
  setsid ros2 launch rm_serial_driver serial_driver.launch.py \
    >"$serial_log" 2>&1 </dev/null &
  pid=$!
  echo "$pid" > "$serial_pid_file"
  sleep 3
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "串口驱动启动失败：" >&2
    tail -40 "$serial_log" >&2
    rm -f "$serial_pid_file"
    exit 1
  fi
  echo "串口驱动已启动，PID=$pid，ROS_DOMAIN_ID=$domain"
}

start_nav() {
  [[ -n "$vehicle_id" && -n "$domain" ]] || { usage; exit 2; }
  if alive "$nav_pid_file"; then
    echo "导航已运行，PID=$(<"$nav_pid_file")" >&2
    exit 1
  fi
  if pgrep -f 'ros2 launch nav_bringup bringup_real.launch.py' >/dev/null; then
    echo "发现非本脚本管理的导航进程，拒绝重复启动" >&2
    exit 1
  fi

  source_ros
  export ROS_DOMAIN_ID="$domain"
  setsid ros2 launch nav_bringup bringup_real.launch.py \
    mode:=mapping lio:=fastlio lio_rviz:=False nav_rviz:=False \
    use_sim_time:=False use_foxglove_bridge:=False \
    use_explore:=True use_hik_camera:=False \
    robot_id:="$vehicle_id" server_ip:="$server_ip" \
    >"$nav_log" 2>&1 </dev/null &
  pid=$!
  echo "$pid" > "$nav_pid_file"
  sleep 5
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "导航/自主探索启动失败：" >&2
    tail -60 "$nav_log" >&2
    rm -f "$nav_pid_file"
    exit 1
  fi
  echo "自主探索已启动，PID=$pid，robot_id=$vehicle_id，ROS_DOMAIN_ID=$domain"
}

stop_group() {
  local pid_file="$1" label="$2" signal="${3:-TERM}" pid
  if alive "$pid_file"; then
    pid="$(<"$pid_file")"
    kill -"$signal" -- "-$pid" 2>/dev/null || kill -"$signal" "$pid" 2>/dev/null || true
    for _ in {1..10}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -TERM -- "-$pid" 2>/dev/null || true
    echo "$label 已停止，PID=$pid"
  else
    echo "$label 原本未运行"
  fi
  rm -f "$pid_file"
}

case "$action" in
  start-serial) start_serial ;;
  start-nav) start_nav ;;
  stop)
    stop_group "$nav_pid_file" "导航/自主探索" INT
    stop_group "$serial_pid_file" "串口驱动" TERM
    ;;
  status)
    alive "$serial_pid_file" && echo "串口驱动运行中，PID=$(<"$serial_pid_file")" || echo "串口驱动未运行"
    alive "$nav_pid_file" && echo "导航/自主探索运行中，PID=$(<"$nav_pid_file")" || echo "导航/自主探索未运行"
    ;;
  logs)
    echo "--- serial ---"
    [[ -f "$serial_log" ]] && tail -30 "$serial_log" || echo "暂无日志"
    echo "--- navigation ---"
    [[ -f "$nav_log" ]] && tail -50 "$nav_log" || echo "暂无日志"
    ;;
  *) usage; exit 2 ;;
esac
