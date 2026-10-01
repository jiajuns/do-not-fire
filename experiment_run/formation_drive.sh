#!/usr/bin/env bash
set -uo pipefail
# Direct-velocity formation drive (bypasses Nav2 entirely) -- robust version.
#
# Runs a small rclpy node on each car that publishes a fixed FORWARD Twist to /cmd_vel_nav --
# the twist_relay input. This is the ONE topic+type that works on every car: cars 1/2's serial
# driver takes Twist but cars 7/8's takes TwistStamped (different builds); both cars' twist_relay
# subscribes Twist on /cmd_vel_nav and converts to whatever its serial driver needs. The node
# waits for the relay subscriber to be discovered, publishes steadily at 20 Hz for DURATION
# seconds, then sends zeros to stop. Every car drives dead-straight at the same speed, so a row
# started parallel stays rigid -> lanes never cross -> no collisions, no Nav2 planning/recovery.
# Nav2 must be idle (no active goal) since its output is remapped to /cmd_vel_nav too.
# NO obstacle avoidance: keep it slow, short, clear space ahead, E-stop in hand.
# Note: firmware ramps speed slowly -- short windows undershoot; prefer longer windows.
#
# Usage:  formation_drive.sh [SPEED_mps] [DURATION_s]     (default 0.15 m/s for 6 s ~= 0.9 m)

SPEED="${1:-0.15}"
DUR="${2:-6}"
targets=(nvidia@192.168.8.49 nvidia@192.168.8.43 nvidia@192.168.8.141 nvidia@192.168.8.28)
domains=(1 2 7 8)
vids=(1 2 7 8)
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=5)

drive_one() {
  local i="$1" ip="${targets[$i]}" dom="${domains[$i]}" vid="${vids[$i]}"
  timeout $((DUR + 30)) ssh "${ssh_opts[@]}" "$ip" bash -s -- "$dom" "$SPEED" "$DUR" "$vid" <<'REMOTE'
set +u
dom="$1"; speed="$2"; dur="$3"; vid="$4"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
export ROS_DOMAIN_ID="$dom"
python3 - "$speed" "$dur" "$vid" <<'PY'
import sys, time
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist

speed = float(sys.argv[1]); dur = float(sys.argv[2]); vid = sys.argv[3]
rclpy.init()
node = rclpy.create_node('formation_driver_%s' % vid)
# twist_relay subscribes Twist on /cmd_vel_nav with default QoS (RELIABLE, VOLATILE) on all cars.
qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST, depth=10,
                 durability=DurabilityPolicy.VOLATILE)
pub = node.create_publisher(Twist, '/cmd_vel_nav', qos)
# wait up to 8s for the twist_relay subscription to be discovered
t0 = time.time()
while pub.get_subscription_count() < 1 and (time.time() - t0) < 8.0:
    rclpy.spin_once(node, timeout_sec=0.05)
subs = pub.get_subscription_count()
fwd = Twist(); fwd.linear.x = speed
stop = Twist()
# steady 20 Hz forward for `dur` seconds
t0 = time.time()
while (time.time() - t0) < dur:
    pub.publish(fwd); time.sleep(0.05)
# stop: zeros for 0.6s
t0 = time.time()
while (time.time() - t0) < 0.6:
    pub.publish(stop); time.sleep(0.05)
print("  car%s: subs=%d, drove %.2f m/s for %.1fs (~%.2f m), stopped" % (vid, subs, speed, dur, speed*dur))
node.destroy_node(); rclpy.shutdown()
PY
REMOTE
}

echo "=== FORMATION DRIVE (rclpy node): ${SPEED} m/s x ${DUR}s, all cars synchronized, straight ==="
pids=()
for i in 0 1 2 3; do drive_one "$i" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "=== drive done ==="
