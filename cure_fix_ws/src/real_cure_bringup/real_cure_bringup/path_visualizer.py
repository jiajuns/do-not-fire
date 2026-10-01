#!/usr/bin/env python3
"""
path_visualizer.py

订阅所有机器人的 /robot_{id}/nav2_path 话题 (nav_msgs/Path)。
将它们转换成一个单一的 /planned_paths_vis (MarkerArray) 话题，
每个机器人的路径使用不同颜色。
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from functools import partial
import colorsys # 用于生成不同颜色

class PathVisualizer(Node):
    def __init__(self):
        super().__init__('path_visualizer', automatically_declare_parameters_from_overrides=True)
        
        # 所有参数由 launch 文件自动声明
        self.robot_count = self.get_parameter_or('robot_count', rclpy.Parameter('robot_count', rclpy.Parameter.Type.INTEGER, 3)).value
        self.robot_namespace = self.get_parameter_or('robot_namespace', rclpy.Parameter('robot_namespace', rclpy.Parameter.Type.STRING, 'robot_')).value
        self.vis_topic = self.get_parameter_or('vis_topic', rclpy.Parameter('vis_topic', rclpy.Parameter.Type.STRING, '/planned_paths_vis')).value
        
        self.path_vis_pub = self.create_publisher(MarkerArray, self.vis_topic, 10)
        
        self.path_subs = []
        self.path_cache = {} # {robot_id: Path}
        self.robot_colors = {} # {robot_id: ColorRGBA}
        
        for i in range(self.robot_count):
            robot_id = i + 1
            topic_name = f'/{self.robot_namespace}{robot_id}/nav2_path'
            
            # 生成不同的颜色
            h = i / float(self.robot_count)
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            self.robot_colors[robot_id] = ColorRGBA(r=r, g=g, b=b, a=1.0)
            
            sub = self.create_subscription(
                Path,
                topic_name,
                partial(self.path_callback, robot_id=robot_id),
                10
            )
            self.path_subs.append(sub)
            
        # 使用定时器统一发布，避免过于频繁
        self.timer = self.create_timer(0.5, self.publish_visuals)
        self.get_logger().info('Path Visualizer initialized.')

    def path_callback(self, msg: Path, robot_id: int):
        # 如果收到空路径，从缓存中删除该机器人的路径（清除显示）
        if not msg.poses:
            if robot_id in self.path_cache:
                del self.path_cache[robot_id]
                self.get_logger().debug(f'Cleared path cache for robot_{robot_id}')
        else:
            self.path_cache[robot_id] = msg

    def publish_visuals(self):
        stamp = self.get_clock().now().to_msg()

        # 只发布当前有效的路径marker (不使用DELETE)
        marker_array = MarkerArray()
        
        for robot_id, path in self.path_cache.items():
            if not path.poses:
                continue

            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = path.header.frame_id  # 使用路径的 frame_id
            marker.ns = "planned_paths"
            marker.id = robot_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            marker.points = [pose.pose.position for pose in path.poses]
            marker.color = self.robot_colors[robot_id]

            marker.scale.x = 0.05 # 路径线条粗细
            marker.pose.orientation.w = 1.0

            marker_array.markers.append(marker)

        # 对于不在缓存中的机器人，发布空marker来清除
        for i in range(self.robot_count):
            robot_id = i + 1
            if robot_id not in self.path_cache:
                delete_marker = Marker()
                delete_marker.header.stamp = stamp
                delete_marker.header.frame_id = "global_map"
                delete_marker.ns = "planned_paths"
                delete_marker.id = robot_id
                delete_marker.action = Marker.DELETE
                marker_array.markers.append(delete_marker)

        self.path_vis_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PathVisualizer()
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
