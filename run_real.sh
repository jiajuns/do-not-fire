#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
controller="$root/ros2_ws/src/svoi_comm/scripts/control_map_gates.sh"

case "${1:-}" in
  sync|desync)
    exec "$controller" start "$1" "${2:-demo}"
    ;;
  status|stop|logs)
    exec "$controller" "$1" "${2:-demo}"
    ;;
  *)
    echo "用法: $0 sync|desync|status|logs|stop [demo|all]" >&2
    exit 2
    ;;
esac
