#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 sync|desync RUN_ID"
}

[[ $# -eq 2 ]] || { usage; exit 2; }
condition="$1"
run_id="$2"
[[ "$condition" == "sync" || "$condition" == "desync" ]] || { usage; exit 2; }
robot_count="${ROBOT_COUNT:-4}"
root="${RESULTS_ROOT:-$HOME/svoi_runs}"
out_dir="$root/$run_id/server"
mkdir -p "$out_dir"

topics=(/merged_map /frontiers /tf /tf_static)
for ((slot = 1; slot <= robot_count; slot++)); do
  topics+=(
    "/robot_${slot}/map"
    "/robot_${slot}/map_synced_tf"
    "/robot_${slot}/way_point"
    "/robot_${slot}/nav2_status"
    "/robot_${slot}/nav2_path"
  )
done

{
  echo "run_id: $run_id"
  echo "condition: $condition"
  echo "robot_count: $robot_count"
  echo "started_at: $(date --iso-8601=seconds)"
} > "$out_dir/metadata.yaml"

echo "Recording server topics to $out_dir/bag (Ctrl-C to stop)"
exec ros2 bag record -o "$out_dir/bag" "${topics[@]}"
