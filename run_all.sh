#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
helper="/home/nvidia/communication/src/svoi_comm/scripts/control_robot_stack.sh"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=5)
targets=(
  "nvidia@192.168.8.48"
  "nvidia@192.168.8.43"
  "nvidia@192.168.8.141"
  "nvidia@192.168.8.3"
)
vehicle_ids=(1 2 7 8)
domains=(1 2 7 8)
server_ip="${SERVER_IP:-192.168.8.30}"

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
  for index in "${!targets[@]}"; do
    remote "$index" "$command" "${vehicle_ids[$index]}" "${domains[$index]}" "$server_ip" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  (( failed == 0 ))
}

stop_everything() {
  run_parallel stop || true
  "$root/run_real.sh" stop all || true
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
    if ! START_TIME="$common_start" "$root/run_real.sh" "$action" all; then
      echo "门控/录包启动失败，正在回滚" >&2
      stop_everything
      exit 1
    fi

    echo "第三步：启动四车建图、Nav2 和 explore_lite"
    if ! run_parallel start-nav; then
      echo "导航启动失败，正在回滚" >&2
      stop_everything
      exit 1
    fi
    echo "四车自主探索已启动。停止命令：$0 stop"
    ;;
  stop)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    stop_everything
    ;;
  status|logs)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    run_parallel "$action" || true
    "$root/run_real.sh" "$action" all || true
    ;;
  *) usage; exit 2 ;;
esac
