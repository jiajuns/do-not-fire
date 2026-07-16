"""Write local odometry pose atomically for a comm node in another domain."""
import math
import os
import tempfile

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PoseFileWriter(Node):
    def __init__(self):
        super().__init__("svoi_pose_file_writer")
        p = self.declare_parameter
        self.pose_file = p("pose_file", "/tmp/svoi_pose.txt").value
        odom_topic = p("odom_topic", "/Odometry").value
        self.create_subscription(Odometry, odom_topic, self._odom_cb, qos_profile_sensor_data)
        self.get_logger().info(f"Writing {odom_topic} pose to {self.pose_file}")

    def _odom_cb(self, msg):
        pos = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        directory = os.path.dirname(self.pose_file) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".svoi_pose_", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{pos.x:.9f} {pos.y:.9f} {yaw:.9f}\n")
            os.replace(tmp, self.pose_file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


def main():
    rclpy.init()
    node = PoseFileWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
