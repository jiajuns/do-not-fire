#!/usr/bin/env python3
"""
external_robot_tracker.py
第三方小车位置追踪器（独立通道，不影响已有三车逻辑）

UDP 接收 bingda 等外部小车位置 -> 发布 /external_robot/trajectory (MarkerArray):
  - 小方块 (CUBE) 标记当前位置
  - LINE_STRIP 历史路径
  橙色，与已有三车区分
"""

import socket
import math
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class ExternalRobotTracker(Node):
    def __init__(self):
        super().__init__('external_robot_tracker',
                         automatically_declare_parameters_from_overrides=True)

        self.udp_port = self.get_parameter('external_robot_port').value
        self.global_frame = self.get_parameter('external_robot_frame').value
        self.max_history = self.get_parameter_or('max_history', 5000)

        self.trajectory = []  # [(x, y), ...]

        # 发布 MarkerArray（小方块当前位置 + 轨迹线）
        self.traj_pub = self.create_publisher(MarkerArray, '/external_robot/trajectory', 10)

        # UDP 监听
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.udp_port))
        self.sock.settimeout(0.5)

        self._running = True
        self._thread = threading.Thread(target=self._udp_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(f'ExternalRobotTracker listening on UDP:{self.udp_port}')

    def _udp_loop(self):
        while self._running:
            try:
                data, addr = self.sock.recvfrom(1024)
                line = data.decode('utf-8', errors='ignore').strip()
                parts = line.split(',')
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    yaw = float(parts[2]) if len(parts) > 2 else 0.0
                    self._update_position(x, y, yaw)
            except socket.timeout:
                continue
            except Exception:
                continue

    def _update_position(self, x, y, yaw):
        self.trajectory.append((x, y))
        if len(self.trajectory) > self.max_history:
            self.trajectory = self.trajectory[-self.max_history:]

        now = self.get_clock().now()
        ma = MarkerArray()

        # 1. 小方块 (CUBE) - 当前位置
        cube = Marker()
        cube.header.frame_id = self.global_frame
        cube.header.stamp = now.to_msg()
        cube.ns = 'external_robot'
        cube.id = 1
        cube.type = Marker.CUBE
        cube.action = Marker.ADD
        cube.pose.position.x = x
        cube.pose.position.y = y
        cube.pose.position.z = 0.05
        cube.pose.orientation.z = math.sin(yaw / 2)
        cube.pose.orientation.w = math.cos(yaw / 2)
        cube.scale.x = 0.15
        cube.scale.y = 0.15
        cube.scale.z = 0.15
        cube.color = ColorRGBA(r=1.0, g=0.4, b=0.0, a=1.0)
        ma.markers.append(cube)

        # 2. LINE_STRIP - 历史轨迹
        line = Marker()
        line.header.frame_id = self.global_frame
        line.header.stamp = now.to_msg()
        line.ns = 'external_robot'
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.05
        line.color = ColorRGBA(r=1.0, g=0.6, b=0.0, a=1.0)
        for tx, ty in self.trajectory:
            line.points.append(Point(x=tx, y=ty, z=0.02))
        ma.markers.append(line)

        self.traj_pub.publish(ma)

    def destroy_node(self):
        self._running = False
        self.sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExternalRobotTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
