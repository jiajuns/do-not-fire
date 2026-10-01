#!/usr/bin/env bash
set -uo pipefail
# Kickstart + stuck-watchdog for the CURE runs. Runs on the control laptop, one rclpy node per
# car (over SSH, in the car's own ROS_DOMAIN_ID). Does NOT modify any car-side code.
#
# WHY: a car whose first server goal arrives before its base_link->map TF is ready deadlocks --
# the goal's stamp ages out of the ~10 s TF buffer, Nav2 can never transform it, so the car never
# moves, so SLAM never initialises, so the TF never appears. Permanent stall ("2 cars won't move").
#
# WHAT: whenever a car has been ~stationary for STUCK_SEC, publish a SHORT fresh-stamped
# /goal_pose (STEP m ahead of its current pose, forward-biased) straight to Nav2. That breaks the
# deadlock; once it moves, SLAM inits, TF appears, and CURE's own coordinated frontier goals take
# over (distinct frontiers -> no collisions). Kicks fire ONLY while stuck, so a car that CURE is
# already driving is never touched. Steps are short + forward to stay inside the arena.
#
# Usage: goal_watchdog.sh [DURATION_s=220] [STUCK_SEC=18] [STEP_m=0.8]

DUR="${1:-220}"; STUCK="${2:-18}"; STEP="${3:-0.8}"
targets=(nvidia@192.168.8.49 nvidia@192.168.8.43 nvidia@192.168.8.141 nvidia@192.168.8.28)
domains=(1 2 7 8); vids=(1 2 7 8)
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=6)

watch_one() {
  local i="$1" ip="${targets[$i]}" dom="${domains[$i]}" vid="${vids[$i]}"
  timeout $((DUR + 45)) ssh "${ssh_opts[@]}" "$ip" bash -s -- "$dom" "$DUR" "$STUCK" "$STEP" "$vid" <<'REMOTE'
set +u
dom="$1"; dur="$2"; stuck="$3"; step="$4"; vid="$5"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID="$dom"
python3 - "$dur" "$stuck" "$step" "$vid" <<'PY'
import sys, time, math
from collections import deque
import rclpy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
dur=float(sys.argv[1]); stuck=float(sys.argv[2]); step=float(sys.argv[3]); vid=sys.argv[4]
rclpy.init()
n=rclpy.create_node('kick_%s'%vid)
cur={'x':None,'y':None,'yaw':0.0}
hist=deque()
def oc(m):
    cur['x']=m.pose.pose.position.x; cur['y']=m.pose.pose.position.y
    q=m.pose.pose.orientation
    cur['yaw']=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
n.create_subscription(Odometry,'/Odometry',oc,10)
pub=n.create_publisher(PoseStamped,'/goal_pose',10)
# forward-biased kick headings (rad, relative to current yaw): straight, then gentle L/R fans.
offsets=[0.0, 0.6, -0.6, 0.3, -0.3]
kidx=0; last_kick=0.0; kicks=0
t0=time.time()
while cur['x'] is None and time.time()-t0 < 8: rclpy.spin_once(n, timeout_sec=0.1)
while time.time()-t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.1)
    now=time.time()
    if cur['x'] is None: continue
    hist.append((now, cur['x'], cur['y']))
    while hist and now-hist[0][0] > stuck: hist.popleft()
    if len(hist) >= 2 and now-hist[0][0] >= stuck-1 and now-last_kick > stuck:
        disp=math.hypot(cur['x']-hist[0][1], cur['y']-hist[0][2])
        if disp < 0.12:   # stationary -> deadlocked; kick it
            th=cur['yaw']+offsets[kidx % len(offsets)]
            gx=cur['x']+step*math.cos(th); gy=cur['y']+step*math.sin(th)
            for _ in range(4):
                g=PoseStamped(); g.header.frame_id='map'
                g.header.stamp=n.get_clock().now().to_msg()   # FRESH stamp -> transforms fine
                g.pose.position.x=gx; g.pose.position.y=gy
                g.pose.orientation.z=math.sin(th/2); g.pose.orientation.w=math.cos(th/2)
                pub.publish(g); time.sleep(0.15)
            kidx+=1; kicks+=1; last_kick=now; hist.clear()
            print('car%s KICK #%d from (%.2f,%.2f) -> (%.2f,%.2f)'%(vid,kicks,cur['x'],cur['y'],gx,gy), flush=True)
print('car%s watchdog done: %d kicks'%(vid,kicks), flush=True)
n.destroy_node(); rclpy.shutdown()
PY
REMOTE
}

echo "=== goal watchdog: dur=${DUR}s stuck=${STUCK}s step=${STEP}m (kicks only while a car is stuck) ==="
# N=3: 0=car1 1=car2 2=car7 (car8/idx3 excluded, floods under load). Use `for i in 0 1 2 3` for N=4.
pids=()
for i in 0 1 2; do watch_one "$i" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "=== watchdog finished ==="
