"""Per-robot SVoI communication node (ROS2 Humble, distributed, one per robot).

Validates Pillar A (de-synchronised event-triggered communication) on real WiFi:
each robot integrates a 'novelty' signal (embedding drift from policy_node, else pose
change) on an LIF membrane and only BROADCASTS its state when the gate fires; teammates
hold the last received (stale) state in between. Logs real comm volume / fire rate.

Topics (all remappable via params):
  sub  odom_topic        nav_msgs/Odometry        own pose (e.g. FAST-LIO2 /Odometry)
  sub  embedding_topic   std_msgs/Float32MultiArray  own 64-D policy embedding (optional)
  sub  state_topic       std_msgs/Float32MultiArray  shared team channel (over WiFi)
  pub  state_topic       std_msgs/Float32MultiArray  own gated state -> team (over WiFi)
  pub  ~/team_state      std_msgs/Float32MultiArray  aggregated neighbour states -> planner
  pub  ~/comm_stats      std_msgs/Float32MultiArray  [fire_rate, bytes_sent, fires, ticks]

State vector layout (Float32MultiArray.data): [robot_id, x, y, yaw, e0..e_{D-1}]
"""
import math
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from svoi_comm.svoi_core import EventTriggeredGate, embedding_drift


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class CommNode(Node):
    def __init__(self):
        super().__init__("svoi_comm_node")
        p = self.declare_parameter
        self.robot_id = p("robot_id", 0).value
        self.n_agents = p("n_agents", 4).value
        self.emb_dim = p("emb_dim", 64).value
        self.rate_hz = float(p("rate_hz", 5.0).value)
        self.start_time = float(p("start_time", 0.0).value)
        mode = p("mode", "spiking").value
        self.gate = EventTriggeredGate(
            mode=mode, threshold=float(p("threshold", 1.0).value),
            leak=float(p("leak", 1.0).value), refractory=int(p("refractory", 0).value),
            interval=int(p("interval", 1).value),
            phase_offset=int(p("phase_offset", 0).value),
            random_prob=float(p("random_prob", 0.2).value),
            evtrig_delta=float(p("evtrig_delta", 0.3).value), seed=int(self.robot_id),
        )
        odom_topic = p("odom_topic", "/Odometry").value
        emb_topic = p("embedding_topic", "embedding").value
        self.state_topic = p("state_topic", "/svoi/state").value
        self.pose_file = p("pose_file", "").value

        self.pose = None            # (x, y, yaw)
        self.pose_file_mtime = 0.0
        self.emb = None             # latest embedding (list)
        self.last_sent_emb = None
        self.last_sent_pose = None
        self.tick = 0
        self.next_step = 0
        self.bytes_sent = 0
        self.neighbors = {}         # robot_id -> (stamp_sec, [x,y,yaw,emb...])

        if self.pose_file:
            self.get_logger().info(f"Reading pose from file: {self.pose_file}")
        else:
            self.create_subscription(Odometry, odom_topic, self._odom_cb, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, emb_topic, self._emb_cb, 10)
        self.create_subscription(Float32MultiArray, self.state_topic, self._state_cb, 50)
        self.pub_state = self.create_publisher(Float32MultiArray, self.state_topic, 50)
        self.pub_team = self.create_publisher(Float32MultiArray, "~/team_state", 10)
        self.pub_stats = self.create_publisher(Float32MultiArray, "~/comm_stats", 10)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f"SVoI comm_node robot {self.robot_id}/{self.n_agents} mode={mode} "
            f"rate={self.rate_hz}Hz interval={self.gate.interval} "
            f"phase_offset={self.gate.phase_offset} start_time={self.start_time:.3f} "
            f"odom={odom_topic} state={self.state_topic}")

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        self.pose = (pos.x, pos.y, yaw_from_quat(msg.pose.pose.orientation))

    def _emb_cb(self, msg):
        self.emb = list(msg.data)

    def _state_cb(self, msg):
        d = list(msg.data)
        if len(d) < 4:
            return
        rid = int(round(d[0]))
        if rid == self.robot_id:
            return  # ignore own echo
        self.neighbors[rid] = (self.get_clock().now().nanoseconds * 1e-9, d[1:])
        self._publish_team()

    def _signal(self):
        """Novelty current since last broadcast: embedding drift if available, else pose move."""
        if self.emb is not None:
            return embedding_drift(self.emb, self.last_sent_emb)
        if self.pose is None:
            return 0.0
        if self.last_sent_pose is None:
            return float("inf")
        dx = self.pose[0] - self.last_sent_pose[0]; dy = self.pose[1] - self.last_sent_pose[1]
        return math.hypot(dx, dy)

    def _tick(self):
        self._load_pose_file()
        if self.pose is None:
            return
        step_idx = self._current_step()
        if step_idx is None:
            return
        fire = self.gate.should_fire(step_idx, self._signal())
        self.tick = step_idx + 1
        if fire:
            data = [float(self.robot_id), float(self.pose[0]), float(self.pose[1]), float(self.pose[2])]
            data += list(self.emb) if self.emb is not None else [0.0] * self.emb_dim
            msg = Float32MultiArray(); msg.data = data
            self.pub_state.publish(msg)
            self.bytes_sent += len(data) * 4
            self.last_sent_emb = list(self.emb) if self.emb is not None else None
            self.last_sent_pose = self.pose
        st = Float32MultiArray()
        st.data = [self.gate.fire_rate, float(self.bytes_sent), float(self.gate.fire_count), float(self.tick)]
        self.pub_stats.publish(st)

    def _current_step(self):
        if self.start_time <= 0.0:
            step_idx = self.next_step
            self.next_step += 1
            return step_idx
        now = self.get_clock().now().nanoseconds * 1e-9
        if now < self.start_time:
            return None
        step_idx = int((now - self.start_time) * self.rate_hz)
        if step_idx < self.next_step:
            return None
        self.next_step = step_idx + 1
        return step_idx

    def _publish_team(self):
        """Flatten neighbour last-known states for the planner: [n, (id,x,y,yaw,staleness)...]."""
        now = self.get_clock().now().nanoseconds * 1e-9
        out = [float(len(self.neighbors))]
        for rid, (stamp, vec) in sorted(self.neighbors.items()):
            out += [float(rid), float(vec[0]), float(vec[1]), float(vec[2]), float(now - stamp)]
        msg = Float32MultiArray(); msg.data = out
        self.pub_team.publish(msg)

    def _load_pose_file(self):
        if not self.pose_file:
            return
        try:
            st = os.stat(self.pose_file)
            if st.st_mtime <= self.pose_file_mtime:
                return
            with open(self.pose_file, "r", encoding="utf-8") as f:
                parts = f.read().strip().split()
            if len(parts) < 3:
                return
            self.pose = (float(parts[0]), float(parts[1]), float(parts[2]))
            self.pose_file_mtime = st.st_mtime
        except OSError:
            return
        except ValueError:
            self.get_logger().warn(f"Invalid pose file contents: {self.pose_file}")


def main():
    rclpy.init()
    node = CommNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
