#!/usr/bin/env python3
"""发布所有机器人历史轨迹到 /robot_trajectories (MarkerArray)，订阅 /robots_positions (PoseArray)"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher',
                         automatically_declare_parameters_from_overrides=True)

        self.robot_count = self.get_parameter('robot_count').value
        self.global_frame = self.get_parameter('global_frame').value

        self.trajectories = {}  # slot -> [(x, y), ...]
        self.max_history = 5000

        # 订阅 /robots_positions (PoseArray 类型)
        self.poses_sub = self.create_subscription(
            PoseArray, '/robots_positions', self.poses_callback, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/robot_trajectories', 10)

        self.colors = [
            ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0),
            ColorRGBA(r=0.2, g=0.6, b=1.0, a=1.0),
            ColorRGBA(r=0.2, g=1.0, b=0.2, a=1.0),
        ]

        self.get_logger().info(f'TrajectoryPublisher ready for {self.robot_count} robots')

    def poses_callback(self, msg: PoseArray):
        """从 /robots_positions (PoseArray) 获取位置并更新轨迹"""
        for i, pose in enumerate(msg.poses):
            slot = i + 1  # PoseArray 按 robot_1, robot_2, ... 顺序排列
            key = f'slot_{slot}'
            if key not in self.trajectories:
                self.trajectories[key] = []
            self.trajectories[key].append((pose.position.x, pose.position.y))
            if len(self.trajectories[key]) > self.max_history:
                self.trajectories[key] = self.trajectories[key][-self.max_history:]

        # 发布轨迹
        self._publish()

    def _publish(self):
        markers = MarkerArray()
        now = self.get_clock().now()

        for slot in range(1, self.robot_count + 1):
            key = f'slot_{slot}'
            if key not in self.trajectories:
                continue

            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = now.to_msg()
            marker.ns = 'trajectories'
            marker.id = slot
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color = self.colors[(slot - 1) % len(self.colors)]

            for x, y in self.trajectories[key]:
                marker.points.append(Point(x=x, y=y, z=0.01))

            markers.markers.append(marker)

        if markers.markers:
            self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
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
