#!/usr/bin/env bash
set -euo pipefail

# Run from the control laptop. It starts one detached map gate and recorder per
# vehicle with a shared wall-clock epoch and vehicle-specific ROS configuration.

usage() {
  cat <<'EOF'
Usage:
  control_map_gates.sh start sync|desync [demo|all]
  control_map_gates.sh status|stop|logs [demo|all]

Sets:
  demo  robots 1@192.168.8.48 and 2@192.168.8.43 (default)
  all   formal four-robot set; aborts unless every robot is reachable

Examples:
  ./control_map_gates.sh start sync
  ./control_map_gates.sh start desync
  ./control_map_gates.sh status
  ./control_map_gates.sh stop
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
action="$1"
condition=""
vehicle_set="demo"

case "$action" in
  start)
    [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
    condition="$2"
    vehicle_set="${3:-demo}"
    [[ "$condition" == "sync" || "$condition" == "desync" ]] || {
      usage
      exit 2
    }
    ;;
  status|stop|logs)
    [[ $# -le 2 ]] || { usage; exit 2; }
    vehicle_set="${2:-demo}"
    ;;
  *) usage; exit 2 ;;
esac

case "$vehicle_set" in
  demo) indices=(0 1) ;;
  three) indices=(0 1 2) ;;   # car1,car2,car7 (car8/idx3 excluded: NUC livox timestamp fault)
  all) indices=(0 1 2 3) ;;
  *) usage; exit 2 ;;
esac
n_agents_total="${#indices[@]}"

# IPs are overridable so exp.sh can pass the cars.conf table (handles DHCP churn).
targets=(
  "${SVOI_CAR0_SSH:-nvidia@192.168.8.49}"
  "${SVOI_CAR1_SSH:-nvidia@192.168.8.43}"
  "${SVOI_CAR2_SSH:-nvidia@192.168.8.141}"
  "${SVOI_CAR3_SSH:-nvidia@192.168.8.28}"
)
vehicle_ids=(1 2 7 8)
domains=(1 2 7 8)
map_topics=(/map /map /map /map)
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=4)

reachable() {
  ssh "${ssh_opts[@]}" "$1" true >/dev/null 2>&1
}

preflight() {
  # Skip any car that is offline or clock-skewed and record the rest (an offline "record-only" car
  # like car8 must not abort the whole group). Filters `indices` in place; fails only if none remain.
  local index target remote_now local_now offset keep=()
  echo "检查 SSH 和系统时钟……"
  for index in "${indices[@]}"; do
    target="${targets[$index]}"
    if ! remote_now="$(ssh "${ssh_opts[@]}" "$target" 'date +%s' 2>/dev/null)"; then
      echo "  警告：车 ${vehicle_ids[$index]} ($target) 无法连接，跳过（本组不录这台）" >&2
      continue
    fi
    local_now="$(date +%s)"
    offset=$((remote_now - local_now))
    if (( offset < -2 || offset > 2 )); then
      echo "  警告：车 ${vehicle_ids[$index]} 时钟偏差 ${offset}s 超过 2 秒，跳过（请先对时）" >&2
      continue
    fi
    echo "  车 ${vehicle_ids[$index]}: 在线，时钟偏差 ${offset}s"
    keep+=("$index")
  done
  indices=("${keep[@]}")
  if (( ${#indices[@]} == 0 )); then
    echo "错误：没有任何车可用" >&2
    return 1
  fi
  echo "本组将录制 ${#indices[@]} 台：${indices[*]}"
}

start_one() {
  local index="$1" start_time="$2" run_id="$3" phase target
  target="${targets[$index]}"
  if [[ "$condition" == "sync" ]]; then
    phase=0.0
  else
    phase="${index}.0"
  fi

  ssh "${ssh_opts[@]}" "$target" bash -s -- \
    "$index" "${domains[$index]}" "$condition" "$phase" "$start_time" \
    "${map_topics[$index]}" "$run_id" "$n_agents_total" <<'REMOTE'
set -euo pipefail
index="$1"
domain="$2"
condition="$3"
phase="$4"
start_time="$5"
map_topic="$6"
run_id="$7"
n_agents="$8"
run_dir="$HOME/svoi_runs/control"
pid_file="$run_dir/map_gate.pid"
log_file="$run_dir/map_gate.log"
bag_pid_file="$run_dir/recorder.pid"
bag_log_file="$run_dir/recorder.log"
bag_dir="$HOME/svoi_runs/$run_id/robot$index/bag"
mkdir -p "$run_dir"

if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
  echo "已有 map gate 在运行，PID=$(<"$pid_file")" >&2
  exit 1
fi
if [[ -f "$bag_pid_file" ]] && kill -0 "$(<"$bag_pid_file")" 2>/dev/null; then
  echo "已有录包进程在运行，PID=$(<"$bag_pid_file")" >&2
  exit 1
fi

# ROS 2 Humble setup scripts probe optional variables that may be unset.
set +u
source /opt/ros/humble/setup.bash
source "$HOME/communication/install/setup.bash"
set -u
export ROS_DOMAIN_ID="$domain"

{
  echo "condition=$condition"
  echo "index=$index"
  echo "domain=$domain"
  echo "phase=$phase"
  echo "start_time=$start_time"
  echo "map_topic=$map_topic"
  echo "run_id=$run_id"
  echo "bag_dir=$bag_dir"
} > "$run_dir/map_gate.env"

export PHASE_SEC="$phase" START_TIME="$start_time"
# Re-running a trial with the same run_id overwrites it: delete any prior recording for this
# car/run so ros2 bag does not refuse with "Output folder already exists".
rm -rf "$bag_dir"
# De-prioritise the recorder (CPU nice + idle IO class). On the CPU/IO-marginal cars the rosbag's
# continuous disk writes cause IO-wait stalls that block the tf message-filter callbacks, so slam +
# costmaps drop every livox scan ("queue is full") -> no map->odom -> car never moves. ionice -c3
# makes the bag yield IO to slam/nav so the TF pipeline keeps up. nice/ionice are inherited by the
# ros2 bag child. Falls back gracefully if ionice is unavailable. record_trial.sh itself untouched.
rec_prefix="nice -n 15"
command -v ionice >/dev/null 2>&1 && rec_prefix="$rec_prefix ionice -c3"
setsid $rec_prefix "$HOME/communication/src/svoi_comm/scripts/record_trial.sh" \
  "$condition" "robot$index" "$run_id" >"$bag_log_file" 2>&1 </dev/null &
bag_pid=$!
echo "$bag_pid" > "$bag_pid_file"
sleep 2
if ! kill -0 "$bag_pid" 2>/dev/null; then
  echo "录包启动失败：" >&2
  tail -30 "$bag_log_file" >&2
  rm -f "$bag_pid_file"
  exit 1
fi

setsid ros2 launch svoi_comm map_gate.launch.py \
  robot_id:="$index" n_agents:="$n_agents" period_sec:=4.0 phase_sec:="$phase" \
  start_time:="$start_time" map_topic:="$map_topic" \
  >"$log_file" 2>&1 </dev/null &
pid=$!
echo "$pid" > "$pid_file"
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "map gate 启动失败：" >&2
  tail -30 "$log_file" >&2
  kill -INT -- "-$bag_pid" 2>/dev/null || true
  rm -f "$bag_pid_file"
  exit 1
fi
echo "gate PID=$pid, bag PID=$bag_pid, phase=${phase}s, map=$map_topic"
REMOTE
}

show_one() {
  local index="$1" mode="$2" target
  target="${targets[$index]}"
  echo "车 ${vehicle_ids[$index]} ($target):"
  if ! reachable "$target"; then
    echo "  离线"
    return 1
  fi
  ssh "${ssh_opts[@]}" "$target" bash -s -- "$mode" <<'REMOTE'
set -euo pipefail
mode="$1"
run_dir="$HOME/svoi_runs/control"
pid_file="$run_dir/map_gate.pid"
log_file="$run_dir/map_gate.log"
bag_pid_file="$run_dir/recorder.pid"
bag_log_file="$run_dir/recorder.log"
case "$mode" in
  status)
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
      echo "  运行中，PID=$(<"$pid_file")"
      [[ -f "$run_dir/map_gate.env" ]] && sed 's/^/  /' "$run_dir/map_gate.env"
    else
      echo "  map gate 未运行"
    fi
    if [[ -f "$bag_pid_file" ]] && kill -0 "$(<"$bag_pid_file")" 2>/dev/null; then
      echo "  录包运行中，PID=$(<"$bag_pid_file")"
    else
      echo "  录包未运行"
    fi
    ;;
  logs)
    echo "  --- map gate ---"
    [[ -f "$log_file" ]] && tail -20 "$log_file" || echo "  暂无日志"
    echo "  --- recorder ---"
    [[ -f "$bag_log_file" ]] && tail -20 "$bag_log_file" || echo "  暂无日志"
    ;;
  stop)
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
      pid="$(<"$pid_file")"
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      echo "  已停止 PID=$pid"
    else
      echo "  map gate 原本未运行"
    fi
    rm -f "$pid_file"
    if [[ -f "$bag_pid_file" ]] && kill -0 "$(<"$bag_pid_file")" 2>/dev/null; then
      bag_pid="$(<"$bag_pid_file")"
      # SIGINT lets rosbag flush metadata and close sqlite cleanly.
      kill -INT -- "-$bag_pid" 2>/dev/null || kill -INT "$bag_pid" 2>/dev/null || true
      for _ in {1..10}; do
        kill -0 "$bag_pid" 2>/dev/null || break
        sleep 0.5
      done
      kill -TERM -- "-$bag_pid" 2>/dev/null || true
      echo "  已保存录包"
    else
      echo "  录包原本未运行"
    fi
    rm -f "$bag_pid_file"
    ;;
esac
REMOTE
}

if [[ "$action" == "start" ]]; then
  preflight
  start_time="${START_TIME:-$(( $(date +%s) + 20 )).0}"
  run_id="${RUN_ID:-${condition}_$(date +%Y%m%d_%H%M%S)}"
  echo "条件：$condition；run ID：$run_id；共同开始时间：$start_time（约 20 秒后）"
  started=()
  for index in "${indices[@]}"; do
    echo -n "启动车 ${vehicle_ids[$index]} (${targets[$index]})："
    if start_one "$index" "$start_time" "$run_id"; then
      started+=("$index")
    else
      echo "启动失败，回滚已启动的 map gate" >&2
      for rollback_index in "${started[@]}"; do
        show_one "$rollback_index" stop || true
      done
      exit 1
    fi
  done
  echo "启动完成。查看状态：$0 status $vehicle_set"
else
  rc=0
  for index in "${indices[@]}"; do
    show_one "$index" "$action" || rc=1
  done
  exit "$rc"
fi
