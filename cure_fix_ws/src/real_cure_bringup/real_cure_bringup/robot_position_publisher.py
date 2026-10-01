#!/usr/bin/env python3
"""
robot_position_publisher.py

(CURE1 架构所需)

监听所有机器人的TF ( /map -> /robot_{id}/base_link )。
将它们的位置发布到两个话题：
- `/robots_positions` (PoseArray) - 用于其他节点订阅机器人位置
- `/robots_positions_vis` (MarkerArray) - 用于 RViz 可视化
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Point, Pose, PoseArray
from geometry_msgs.msg import PoseArray as PoseArrayMsg
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf2_ros
from tf2_ros import TransformException

class RobotPositionPublisher(Node):
    """发布所有机器人位置的节点"""
    
    def __init__(self):
        super().__init__('robot_position_publisher', automatically_declare_parameters_from_overrides=True)
        
        # 所有参数由 launch 文件自动声明
        self.robot_count = self.get_parameter_or('robot_count', rclpy.Parameter('robot_count', rclpy.Parameter.Type.INTEGER, 3)).value
        self.robot_namespace = self.get_parameter_or('robot_namespace', rclpy.Parameter('robot_namespace', rclpy.Parameter.Type.STRING, 'robot_')).value
        self.global_frame = self.get_parameter_or('global_frame', rclpy.Parameter('global_frame', rclpy.Parameter.Type.STRING, 'global_map')).value
        self.update_rate = self.get_parameter_or('update_rate', rclpy.Parameter('update_rate', rclpy.Parameter.Type.DOUBLE, 5.0)).value
        
        # TF 监听
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 发布机器人位置 (PoseArray - 用于其他节点订阅)
        self.positions_pub = self.create_publisher(
            PoseArray,
            '/robots_positions',
            10
        )
        
        # 发布机器人位置可视化 (MarkerArray - 用于 RViz)
        self.positions_vis_pub = self.create_publisher(
            MarkerArray,
            '/robots_positions_vis',
            10
        )

        # 订阅初始位姿（TRANSIENT_LOCAL），用于换图/重置后TF短暂不可用时回退
        qos_transient = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            depth=1
        )
        self.initial_pose_sub = self.create_subscription(
            PoseArrayMsg,
            '/robot_initial_poses',
            self.initial_poses_callback,
            qos_transient
        )
        
        # 机器人帧ID列表
        self.robot_frames = []
        for i in range(self.robot_count):
            robot_id = i + 1
            frame_id = f'{self.robot_namespace}{robot_id}/base_link'
            self.robot_frames.append(frame_id)

        self.last_valid_poses = {}
        self.initial_poses = {}
            
        # 定时器
        self.timer = self.create_timer(1.0 / self.update_rate, self.timer_callback)
        
        self.get_logger().info(f'Robot Position Publisher initialized for {self.robot_count} robots.')

    def initial_poses_callback(self, msg: PoseArrayMsg):
        for i, pose in enumerate(msg.poses):
            robot_id = i + 1
            if robot_id > self.robot_count:
                break
            self.initial_poses[robot_id] = pose
        self.get_logger().info(f'Received initial poses for {len(self.initial_poses)} robots')

    def _fallback_pose(self, robot_id: int) -> Pose:
        if robot_id in self.last_valid_poses:
            return self.last_valid_poses[robot_id]
        if robot_id in self.initial_poses:
            return self.initial_poses[robot_id]

        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        return pose

    def timer_callback(self):
        """定时获取所有机器人的位置并发布"""
        robot_poses = []
        robot_points = []
        
        for i, frame_id in enumerate(self.robot_frames):
            robot_id = i + 1
            try:
                # 查找从 map -> robot_X/base_link 的变换
                transform = self.tf_buffer.lookup_transform(
                    self.global_frame,
                    frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                # 创建 Pose 用于 PoseArray
                pose = Pose()
                pose.position.x = transform.transform.translation.x
                pose.position.y = transform.transform.translation.y
                pose.position.z = 0.0
                pose.orientation = transform.transform.rotation
                self.last_valid_poses[robot_id] = pose
                
                # 创建 Point 用于 MarkerArray 可视化
                point = Point()
                point.x = transform.transform.translation.x
                point.y = transform.transform.translation.y
                point.z = 0.0
            except TransformException as ex:
                self.get_logger().warn(
                    f'Could not transform {frame_id} to {self.global_frame}, using fallback pose: {ex}'
                )
                pose = self._fallback_pose(robot_id)
                point = Point()
                point.x = pose.position.x
                point.y = pose.position.y
                point.z = 0.0
                
            robot_poses.append(pose)
            robot_points.append(point)
        
        if len(robot_poses) == self.robot_count:
            # 发布 PoseArray 给其他节点使用
            pose_array = PoseArray()
            pose_array.header.frame_id = self.global_frame
            pose_array.header.stamp = self.get_clock().now().to_msg()
            pose_array.poses = robot_poses
            self.positions_pub.publish(pose_array)
            
            # 发布 MarkerArray 给 RViz 可视化
            marker_array = MarkerArray()
            
            # 机器人位置点标记
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "robot_positions"
            marker.id = 0
            marker.type = Marker.POINTS
            marker.action = Marker.ADD
            marker.points = robot_points
            
            # 可视化样式
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            
            marker_array.markers.append(marker)
            self.positions_vis_pub.publish(marker_array)
            
            # self.get_logger().debug(f'Published {len(robot_poses)} robot positions')
        else:
            self.get_logger().warn('Failed to build complete robots_positions array.')

def main(args=None):
    rclpy.init(args=args)
    node = RobotPositionPublisher()
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
