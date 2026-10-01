#!/usr/bin/env bash
set -uo pipefail
# exp.sh -- one-key controller for the manual-drive CURE sync/desync experiment.
#
# Workflow for one paired trial:
#   ./exp.sh cars-on            # serial + nav on every car; waits for TF ready (no motion)
#   ./exp.sh server-on          # laptop map-merge + RViz + goal relay (no autonomous explorer)
#   ./exp.sh run sync  p1       # start sync gate + recording (common wall-clock epoch)
#   ./exp.sh target 2           # aim your RViz clicks at car 2, then click 2D Nav Goals
#   ...  drive each car by clicking; record your video  ...
#   ./exp.sh stop               # flush + close the bag (cars & server stay up)
#   ./exp.sh run desync p1      # repeat for the matched desync half
#   ./exp.sh target 7 ; ... ; ./exp.sh stop
#   ./exp.sh sync-bags          # pull every car's bags to this laptop, organised per run
#   ./exp.sh kill-all           # clean everything (cars + server) when finished
#
# Config (IPs / poses / spacing) lives in cars.conf next to this script.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conf="$root/cars.conf"
[[ -f "$conf" ]] || { echo "missing $conf" >&2; exit 2; }
# shellcheck disable=SC1090
source "$conf"
: "${SERVER_IP:=192.168.8.30}"
n="${#CAR_NAMES[@]}"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=6)

# vehicle_set name understood by control_map_gates.sh, derived from active car count.
vehicle_set() { case "$n" in 2) echo demo;; 3) echo three;; 4) echo all;; *) echo all;; esac; }

# Export the cars.conf IP table so control_map_gates.sh (via run_real.sh) targets the same hosts.
export_gate_ips() {
  local i
  for i in "${!CAR_IPS[@]}"; do export "SVOI_CAR${i}_SSH=nvidia@${CAR_IPS[$i]}"; done
}

ssh_car() { local i="$1"; shift; ssh "${ssh_opts[@]}" "nvidia@${CAR_IPS[$i]}" "$@"; }

# --- deploy the per-car script + its own car.conf to ~/svoi_experiment on each car ---
deploy_one() {
  local i="$1"
  ssh_car "$i" 'mkdir -p ~/svoi_experiment' || return 1
  scp "${ssh_opts[@]}" -q "$root/car_stack.sh" "nvidia@${CAR_IPS[$i]}:~/svoi_experiment/car_stack.sh" || return 1
  ssh_car "$i" "cat > ~/svoi_experiment/car.conf" <<EOF
VID=${CAR_VIDS[$i]}
DOMAIN=${CAR_DOMAINS[$i]}
WS=${CAR_WSES[$i]}
SERVER_IP=${SERVER_IP}
HELPER=\$HOME/communication/src/svoi_comm/scripts/control_robot_stack.sh
EOF
  ssh_car "$i" 'chmod +x ~/svoi_experiment/car_stack.sh'
}

cars_on() {
  local i pids=() rc=0
  echo "== deploying + starting serial/nav on $n cars =="
  for i in "${!CAR_NAMES[@]}"; do
    ( deploy_one "$i" && ssh_car "$i" '~/svoi_experiment/car_stack.sh start' \
        && echo "  [${CAR_NAMES[$i]}] started" || echo "  [${CAR_NAMES[$i]}] START FAILED" ) &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  echo "== waiting for SLAM/TF (map->base_link) on all cars =="
  local deadline=$((SECONDS + 120)) ready
  declare -A done=()
  while (( SECONDS < deadline )); do
    ready=0
    for i in "${!CAR_NAMES[@]}"; do
      [[ -n "${done[$i]:-}" ]] && { ready=$((ready+1)); continue; }
      if ssh_car "$i" '~/svoi_experiment/car_stack.sh tf' >/dev/null 2>&1; then
        done[$i]=1; ready=$((ready+1)); echo "  [${CAR_NAMES[$i]}] TF READY"
      fi
    done
    (( ready == n )) && break
    sleep 5
  done
  if (( ${#done[@]} == n )); then
    echo "== all $n cars TF-ready. Safe to server-on + run. =="
  else
    echo "== WARNING: only ${#done[@]}/$n cars TF-ready. Check: ./exp.sh status =="
  fi
}

cars_off() {
  local i pids=()
  echo "== clean-killing all car stacks =="
  for i in "${!CAR_NAMES[@]}"; do
    ( ssh_car "$i" '~/svoi_experiment/car_stack.sh kill' | sed "s/^/  [${CAR_NAMES[$i]}] /" ) &
    pids+=("$!")
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
}

server_on()  { "$root/start_server_manual.sh" start; }
server_off() { "$root/start_server_manual.sh" stop; }

# set which car your RViz 2D-Nav-Goal clicks drive (domain 0, on the laptop)
set_target() {
  local id="$1"
  set +u; source /opt/ros/humble/setup.bash; set -u
  export ROS_DOMAIN_ID=0
  if ros2 param set /rviz_goal_relay target_robot_id "$id" 2>/dev/null; then
    echo "RViz clicks now drive robot_$id"
  else
    echo "could not reach /rviz_goal_relay (is the server up?)" >&2; return 1
  fi
}

# Localized? Fresh 20s tf2_echo on car $1 (generous: under record+RViz load tf2_echo is slow to
# connect). Exit 0 iff map->base_link resolves.
tf_ok_remote() {
  local i="$1"
  ssh_car "$i" "bash -lc 'source /opt/ros/humble/setup.bash >/dev/null 2>&1; \
    source \$HOME/communication/install/setup.bash >/dev/null 2>&1; \
    export ROS_DOMAIN_ID=${CAR_DOMAINS[$i]}; \
    timeout 20 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -m1 -q Translation'" \
    >/dev/null 2>&1
}

# Poll the cars flagged CAR_WAIT=1 until each is localized. Cars flagged 0 (e.g. car8) are still
# recorded but not waited on. Checks run in PARALLEL each round so one slow car can't starve others.
wait_all_data() {
  local deadline=$((SECONDS + ${1:-200})) i waitset=() pid
  for i in "${!CAR_NAMES[@]}"; do [[ "${CAR_WAIT[$i]:-1}" == 1 ]] && waitset+=("$i"); done
  local nw=${#waitset[@]}
  declare -A ok=()
  while (( SECONDS < deadline )); do
    declare -A pidmap=()
    for i in "${waitset[@]}"; do
      [[ -n "${ok[$i]:-}" ]] && continue
      tf_ok_remote "$i" & pidmap[$!]=$i
    done
    for pid in "${!pidmap[@]}"; do
      if wait "$pid"; then i=${pidmap[$pid]}; ok[$i]=1; echo "  [${CAR_NAMES[$i]}] data OK (localized)"; fi
    done
    (( ${#ok[@]} == nw )) && return 0
    echo "  ... ${#ok[@]}/$nw driveable cars have data (waiting; record-only cars not counted)"
    sleep 2
  done
  return 1
}

# Loud, hard-to-miss prompt: terminal bell + desktop notification + banner.
notify_user() {
  local msg="$1"
  printf '\a' >&2
  command -v notify-send >/dev/null 2>&1 && notify-send -u critical "SVOI experiment" "$msg" 2>/dev/null || true
  echo
  echo "################################################################"
  echo "#  $msg"
  echo "################################################################"
  echo
}

run_trial() {
  local cond="$1" label="${2:-}"
  [[ "$cond" == sync || "$cond" == desync ]] || { echo "run needs: sync|desync [label]" >&2; exit 2; }
  echo "== [1/3] ensure laptop server (merged map) is up =="
  server_on >/dev/null 2>&1 || true    # idempotent
  export_gate_ips
  [[ -n "$label" ]] && export RUN_ID="${cond}_${label}"
  echo "== [2/3] start $cond gate + recording on all $n cars =="
  "$root/run_real.sh" "$cond" "$(vehicle_set)" || { echo "gate/recording start FAILED" >&2; return 1; }
  echo "== [3/3] recording. waiting until ALL $n cars have data (localized) =="
  if wait_all_data 150; then
    notify_user "DRIVEABLE CARS HAVE DATA -- START DRIVING NOW ($cond${label:+ $label}). Send goals in each car's RViz (car8 records only). End trial: ./exp.sh stop"
  else
    notify_user "TIMEOUT: not all driveable cars localized yet (FAST-LIO lag?). Recording IS running. Run ./exp.sh status; once localized, drive; then ./exp.sh stop."
  fi
}

stop_trial() {
  export_gate_ips
  "$root/run_real.sh" stop "$(vehicle_set)"
  echo "trial stopped; bag flushed. Cars + server still up for the next run."
}

sync_bags() {
  local i dst="$HOME/svoi_runs"; mkdir -p "$dst"
  echo "== pulling bags from $n cars to $dst =="
  for i in "${!CAR_NAMES[@]}"; do
    echo "  [${CAR_NAMES[$i]}] rsync ..."
    rsync -a -e "ssh ${ssh_opts[*]}" "nvidia@${CAR_IPS[$i]}:svoi_runs/" "$dst/" \
      && echo "    ok" || echo "    FAILED"
  done
  echo "local runs:"; ls -1dt "$dst"/*/ 2>/dev/null | head
}

status_all() {
  local i
  for i in "${!CAR_NAMES[@]}"; do
    echo "== ${CAR_NAMES[$i]} (${CAR_IPS[$i]}) =="
    ssh_car "$i" '~/svoi_experiment/car_stack.sh status' 2>&1 | sed 's/^/  /' || echo "  offline"
  done
  echo "== server =="; "$root/start_server_manual.sh" status | sed 's/^/  /'
}

scan() {
  echo "reachable nvidia@ hosts on 192.168.8.0/24:"
  command -v nmap >/dev/null || { echo "nmap not installed" >&2; return 1; }
  timeout 45 nmap -n -p22 --open 192.168.8.0/24 2>/dev/null | grep 'Nmap scan report' | awk '{print "  "$NF}'
}

case "${1:-}" in
  scan)       scan ;;
  cars-on)    cars_on ;;
  cars-off)   cars_off ;;
  server-on)  server_on ;;
  server-off) server_off ;;
  run)        run_trial "${2:-}" "${3:-}" ;;
  stop)       stop_trial ;;
  target)     set_target "${2:?usage: exp.sh target <ID>}" ;;
  sync-bags)  sync_bags ;;
  status)     status_all ;;
  kill-all)
    stop_trial || true
    cars_off
    server_off
    echo "== everything down =="
    ;;
  *)
    cat >&2 <<EOF
Usage: $0 <command>
  scan            list reachable cars on the LAN
  cars-on         deploy + start serial/nav on all cars, wait for TF ready (no motion)
  cars-off        clean-kill all car stacks
  server-on       start laptop manual server (merge + RViz + goal relay)
  server-off      stop laptop manual server
  run sync|desync [label]   start gate + recording for one trial
  target <ID>     aim your RViz clicks at robot <ID>
  stop            end current trial (flush bag); cars + server stay up
  sync-bags       pull all cars' bags to ~/svoi_runs
  status          show per-car + server status
  kill-all        stop trial + cars-off + server-off
EOF
    exit 2 ;;
esac
