"""Gate occupancy-map uploads on a common periodic wall-clock schedule.

The local SLAM map remains available on ``map_topic``.  This node holds its
latest sample and republishes it on ``gated_map_topic`` only when the schedule
fires.  Remap robot_communication's ``map`` input to the gated topic so the
schedule controls information that actually reaches the team planner.

``comm_events`` layout (Float64MultiArray):
  [event_type, robot_id, sequence, event_time, source_time,
   serialized_map_bytes, schedule_cycle]
where event_type is 1 for a published map and 0 for a missed slot (no map yet).
"""
import math

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.serialization import serialize_message
from std_msgs.msg import Bool, Float64MultiArray

from svoi_comm.svoi_core import PeriodicPhaseSchedule


def _map_stamp(msg):
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


class MapGateNode(Node):
    def __init__(self):
        super().__init__("svoi_map_gate")
        p = self.declare_parameter
        self.robot_id = int(p("robot_id", 0).value)
        self.n_agents = max(1, int(p("n_agents", 4).value))
        self.period_sec = float(p("period_sec", 4.0).value)
        phase_sec = float(p("phase_sec", -1.0).value)
        if phase_sec < 0.0:
            phase_sec = self.robot_id * self.period_sec / self.n_agents
        start_time = float(p("start_time", 0.0).value)
        self.check_rate_hz = float(p("check_rate_hz", 20.0).value)
        map_topic = p("map_topic", "/map").value
        gated_map_topic = p("gated_map_topic", "/svoi/gated_map").value
        event_topic = p("event_topic", "/svoi/comm_events").value
        stats_topic = p("stats_topic", "/svoi/comm_stats").value
        fire_topic = p("fire_topic", "/svoi/fire").value

        now = self._now()
        if start_time <= 0.0:
            # Convenient smoke-test default.  Formal trials should pass one
            # explicit timestamp to all robots from the laptop coordinator.
            start_time = math.ceil((now + 2.0) / self.period_sec) * self.period_sec
        self.schedule = PeriodicPhaseSchedule(self.period_sec, phase_sec, start_time)

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.latest_map = None
        self.sequence = 0
        self.fire_count = 0
        self.missed_count = 0
        self.bytes_published = 0
        self.last_fire_time = 0.0

        self.create_subscription(OccupancyGrid, map_topic, self._map_cb, map_qos)
        self.pub_map = self.create_publisher(OccupancyGrid, gated_map_topic, map_qos)
        self.pub_events = self.create_publisher(Float64MultiArray, event_topic, 100)
        self.pub_stats = self.create_publisher(Float64MultiArray, stats_topic, 10)
        self.pub_fire = self.create_publisher(Bool, fire_topic, 10)
        self.create_timer(1.0 / max(1.0, self.check_rate_hz), self._tick)
        self.get_logger().info(
            f"map gate robot={self.robot_id}/{self.n_agents} period={self.period_sec:.3f}s "
            f"phase={self.schedule.phase_sec:.3f}s start={start_time:.3f} "
            f"input={map_topic} output={gated_map_topic}"
        )

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _map_cb(self, msg):
        self.latest_map = msg

    def _tick(self):
        now = self._now()
        cycle = self.schedule.due_cycle(now)
        if cycle is None:
            return
        if self.latest_map is None:
            self.missed_count += 1
            self._publish_event(0, now, 0.0, 0, cycle)
            self._publish_stats()
            self.get_logger().warn(f"missed schedule cycle {cycle}: no map received")
            return

        payload_bytes = len(serialize_message(self.latest_map))
        pulse = Bool()
        pulse.data = True
        self.pub_fire.publish(pulse)
        self.pub_map.publish(self.latest_map)
        self.sequence += 1
        self.fire_count += 1
        self.bytes_published += payload_bytes
        self.last_fire_time = now
        self._publish_event(
            1, now, _map_stamp(self.latest_map), payload_bytes, cycle
        )
        self._publish_stats()

    def _publish_event(self, event_type, event_time, source_time, payload_bytes, cycle):
        msg = Float64MultiArray()
        msg.data = [
            float(event_type), float(self.robot_id), float(self.sequence),
            float(event_time), float(source_time), float(payload_bytes), float(cycle),
        ]
        self.pub_events.publish(msg)

    def _publish_stats(self):
        msg = Float64MultiArray()
        msg.data = [
            float(self.fire_count), float(self.missed_count),
            float(self.bytes_published), float(self.last_fire_time),
        ]
        self.pub_stats.publish(msg)


def main():
    rclpy.init()
    node = MapGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
