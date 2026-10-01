#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
helper="/home/nvidia/communication/src/svoi_comm/scripts/control_robot_stack.sh"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=5)
targets=(
  "nvidia@192.168.8.49"
  "nvidia@192.168.8.43"
  "nvidia@192.168.8.141"
  "nvidia@192.168.8.28"
)
vehicle_ids=(1 2 7 8)
domains=(1 2 7 8)
# Active car subset (indices into the arrays above). N=3: car1/car2/car7. car8 (index 3, an Intel
# NUC) floods livox scan-drops under full-run load (fine only nav-only) -> excluded. The recorder
# is de-prioritised (see control_map_gates.sh) which keeps these 3 at 0 drops under load.
# To try N=4 again once car8's livox time-sync is fixed: active_indices=(0 1 2 3) + run_real "all".
active_indices=(0 1 2)
server_ip="${SERVER_IP:-192.168.8.30}"
# Per-car nav workspace. Car 1's ~/SenseBeetle is a stale build missing hik_camera;
# its current stack is ~/SenBettle_latest. Cars 2/7/8 use ~/SenseBeetle. This is sourced
# only to LAUNCH nav; no files on any car are modified.
nav_ws=(
  "/home/nvidia/SenBettle_latest"
  "/home/nvidia/SenseBeetle"
  "/home/nvidia/SenseBeetle"
  "/home/nvidia/SenseBeetle"
)

usage() {
  cat <<EOF
用法:
  $0 sync|desync [--yes]
  $0 stop|status|logs

启动会让四台实体车自主运动；--yes 表示已确认场地安全。
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
action="$1"

remote() {
  local index="$1" command="$2"
  shift 2
  echo "车 ${vehicle_ids[$index]} (${targets[$index]}):"
  ssh "${ssh_opts[@]}" "${targets[$index]}" "$helper" "$command" "$@"
}

run_parallel() {
  local command="$1" failed=0 index pid
  shift
  pids=()
  for index in "${active_indices[@]}"; do
    remote "$index" "$command" "${vehicle_ids[$index]}" "${domains[$index]}" "$server_ip" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 ))
}

# Launch one car's nav/explore inline, sourcing that car's own workspace, so car 1's
# ~/SenBettle_latest (which has hik_camera) is used instead of the stale ~/SenseBeetle.
# Writes the same navigation.pid the car-side helper's `stop`/`status` expect.
start_nav_one() {
  local index="$1" target="${targets[$index]}"
  echo "车 ${vehicle_ids[$index]} ($target):"
  ssh "${ssh_opts[@]}" "$target" bash -s -- \
    "${vehicle_ids[$index]}" "${domains[$index]}" "$server_ip" "${nav_ws[$index]}" <<'REMOTE'
set -uo pipefail
vid="$1"; dom="$2"; server_ip="$3"; ws="$4"
run_dir="$HOME/svoi_runs/control_stack"; mkdir -p "$run_dir"
nav_pid_file="$run_dir/navigation.pid"; nav_log="$run_dir/navigation.log"
if [[ -f "$nav_pid_file" ]] && kill -0 "$(<"$nav_pid_file")" 2>/dev/null; then
  echo "导航已运行，PID=$(<"$nav_pid_file")" >&2; exit 1; fi
if pgrep -f 'bringup_real.*\.launch\.py' >/dev/null; then
  echo "发现非本脚本管理的导航进程，拒绝重复启动" >&2; exit 1; fi
set +u
source /opt/ros/humble/setup.bash
source "$ws/install/setup.bash"
[[ -f "$HOME/communication/install/setup.bash" ]] && source "$HOME/communication/install/setup.bash"
set -u
export ROS_DOMAIN_ID="$dom"
# Short-range launch: copy the car's own launch and cap the lidar range (range_max) so the
# small 4x6m arena needs real exploration. The ORIGINAL launch is untouched; only the copy
# under ~/svoi_experiment is edited. Regex handles per-car values (SenseBeetle 20.0, latest 10.0).
src_launch="$ws/install/nav_bringup/share/nav_bringup/launch/bringup_real.launch.py"
sr_launch="$HOME/svoi_experiment/bringup_real_shortrange.launch.py"
mkdir -p "$HOME/svoi_experiment"
cp "$src_launch" "$sr_launch"
sed -i -E 's/("range_max": )[0-9.]+/\11.5/' "$sr_launch"
# CPU-relief for the marginal cars (car7/car8): under the full stack their livox tf message-filter
# overflows ("queue is full") and drops EVERY scan -> starved costmap/SLAM -> car never moves.
# All edits are on COPIES under ~/svoi_experiment; the car's original configs are untouched.
np_dst="$HOME/svoi_experiment/nav2_params.yaml"
cp "$ws/install/nav_bringup/share/nav_bringup/config/reality/nav2_params_real.yaml" "$np_dst"
sed -i -E 's/(transform_tolerance:[[:space:]]*)[0-9.]+/\13.0/' "$np_dst"
# Local costmap was 20 Hz update / 10 Hz publish at 2 cm res -- by far the biggest CPU hog. Drop
# it to 4/2 Hz and 5 cm so the scan pipeline keeps up. Values are unique (local=20/10, global=5/2).
sed -i -E 's/(update_frequency:[[:space:]]*)20\.0/\14.0/' "$np_dst"
sed -i -E 's/(publish_frequency:[[:space:]]*)10\.0/\12.0/' "$np_dst"
sed -i -E 's/(update_frequency:[[:space:]]*)5\.0/\12.0/' "$np_dst"
sed -i -E 's/^([[:space:]]*)resolution:[[:space:]]*0\.02/\1resolution: 0.05/' "$np_dst"
# os.path.join(..., "<abs>") returns the abs path, so this cleanly repoints the launch's params.
sed -i 's|"nav2_params_real.yaml"|"'"$np_dst"'"|' "$sr_launch"
# SLAM (async_slam_toolbox): coarser map (2->5 cm) + slower scan cadence so scan-matching keeps up.
sm_dst="$HOME/svoi_experiment/mapper_params.yaml"
cp "$ws/install/nav_bringup/share/nav_bringup/config/reality/mapper_params_online_async_real.yaml" "$sm_dst"
sed -i -E 's/^([[:space:]]*)resolution:[[:space:]]*0\.02/\1resolution: 0.05/' "$sm_dst"
sed -i -E 's/(minimum_time_interval:[[:space:]]*)[0-9.]+/\10.8/' "$sm_dst"
sed -i -E 's/(map_update_interval:[[:space:]]*)[0-9.]+/\12.0/' "$sm_dst"
sed -i 's|"mapper_params_online_async_real.yaml"|"'"$sm_dst"'"|' "$sr_launch"
setsid ros2 launch "$sr_launch" \
  mode:=mapping lio:=fastlio lio_rviz:=False nav_rviz:=False \
  use_sim_time:=False use_foxglove_bridge:=False \
  use_explore:=False use_hik_camera:=False \
  robot_id:="$vid" server_ip:="$server_ip" \
  >"$nav_log" 2>&1 </dev/null &
pid=$!; echo "$pid" > "$nav_pid_file"; sleep 6
if ! kill -0 "$pid" 2>/dev/null; then
  echo "导航/自主探索启动失败（ws=$ws）：" >&2; tail -50 "$nav_log" >&2
  rm -f "$nav_pid_file"; exit 1; fi
echo "自主探索已启动，PID=$pid，robot_id=$vid，ROS_DOMAIN_ID=$dom，ws=$ws"
REMOTE
}

start_nav_all() {
  local failed=0 index pid pids=()
  for index in "${active_indices[@]}"; do
    start_nav_one "$index" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 ))
}

stop_everything() {
  run_parallel stop || true
  "$root/run_real.sh" stop three || true
}

case "$action" in
  sync|desync)
    [[ $# -le 2 ]] || { usage; exit 2; }
    if [[ "${2:-}" != "--yes" && "${ROBOT_RUN_CONFIRMED:-0}" != "1" ]]; then
      [[ -t 0 ]] || {
        echo "拒绝启动：非交互模式请添加 --yes" >&2
        exit 2
      }
      echo "即将启动四台实体车自主探索。请确认场地清空、急停可用。"
      read -r -p "输入 RUN 继续: " answer
      [[ "$answer" == "RUN" ]] || { echo "已取消"; exit 1; }
    fi

    echo "第一步：启动四车串口驱动"
    if ! run_parallel start-serial; then
      echo "串口启动失败，正在回滚" >&2
      stop_everything
      exit 1
    fi

    echo "第二步：启动统一门控和自动录包"
    common_start="$(( $(date +%s) + 35 )).0"
    if ! START_TIME="$common_start" "$root/run_real.sh" "$action" three; then
      echo "门控/录包启动失败，正在回滚" >&2
      stop_everything
      exit 1
    fi

    echo "第三步：启动四车建图、Nav2 和 explore_lite"
    if ! start_nav_all; then
      echo "导航启动失败，正在回滚" >&2
      stop_everything
      exit 1
    fi
    echo "四车自主探索已启动。停止命令：$0 stop"
    ;;
  nav-only)
    # Health-check mode: bring up mapping/Nav2 on all 4 cars only -- NO serial, NO gates, NO
    # recording, so nothing moves. Used to verify the CPU-relief config before a real run.
    [[ $# -eq 1 ]] || { usage; exit 2; }
    echo "健康检查：仅启动四车建图/Nav2（无串口、无门控、无录包、不运动）"
    if ! start_nav_all; then
      echo "导航启动失败" >&2
      exit 1
    fi
    echo "四车导航已启动（健康检查模式，不运动）。停止：$0 stop"
    ;;
  stop)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    stop_everything
    ;;
  status|logs)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    run_parallel "$action" || true
    "$root/run_real.sh" "$action" three || true
    ;;
  *) usage; exit 2 ;;
esac
