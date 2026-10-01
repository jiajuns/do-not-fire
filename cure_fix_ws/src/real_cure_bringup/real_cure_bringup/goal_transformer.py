#!/usr/bin/env python3
"""
goal_transformer.py

(基于您提供的 task_receiver.py)

动态启动 N 个 (每个机器人一个)。
命名空间: /robot_{id}

1. 订阅 /robot_{id}/assigned_goal (PoseStamped, 全局坐标系)
2. 查找 TF (global_frame -> robot_{id}/map)
3. 将全局坐标转换为局部坐标
4. 发布 /robot_{id}/way_point (PointStamped, 局部坐标系)
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
import math
import tf2_ros
from rclpy.duration import Duration
from tf2_ros import TransformException
import tf2_geometry_msgs  # 添加：用于点变换

class GoalTransformer(Node):
    def __init__(self):
        # 节点名称由 launch 文件设置，命名空间也由 launch 设置
        super().__init__('goal_transformer', automatically_declare_parameters_from_overrides=True)
        
        # 所有参数由 launch 文件自动声明
        self.robot_id = self.get_parameter_or('robot_id', rclpy.Parameter('robot_id', rclpy.Parameter.Type.INTEGER, 0)).value
        self.global_frame = self.get_parameter_or('global_frame', rclpy.Parameter('global_frame', rclpy.Parameter.Type.STRING, 'map')).value
        
        if self.robot_id == 0:
            self.get_logger().error("robot_id parameter is not set or is 0.")
            return

        # 订阅服务器分配的目标点 (全局坐标)
        self.goal_sub = self.create_subscription(
            PoseStamped,
            'assigned_goal', # 话题在 /robot_{id} 命名空间下
            self.goal_callback,
            10
        )
        
        # 订阅恢复管理器的恢复目标点 (优先级更高)
        self.recovery_goal_sub = self.create_subscription(
            PoseStamped,
            'recovery_goal', # 话题在 /robot_{id} 命名空间下
            self.recovery_goal_callback,
            10
        )

        # TF 缓存与监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 目标局部坐标系 (e.g., "robot_1/map")
        self.local_map_frame = f'robot_{self.robot_id}/map'

        # 发布 /robot_{id}/way_point (局部坐标)
        self.way_point_pub = self.create_publisher(
            PointStamped, 
            'way_point', # 话题在 /robot_{id} 命名空间下
            10
        )

        # 缓存上次的目标点，避免重复发送
        self.last_goal_pos = None
        self.goal_tolerance = 0.01  # 目标点变化阈值（米）

        self.get_logger().info(f'Goal Transformer ready for robot {self.robot_id}')
        self.get_logger().info(f'Transforming from {self.global_frame} to {self.local_map_frame}')

    def goal_callback(self, goal_msg: PoseStamped):
        
        in_point = goal_msg.pose.position
        
        # 检查是否与上次目标点几乎一致
        if self.last_goal_pos:
            if (abs(in_point.x - self.last_goal_pos.x) < self.goal_tolerance and
                abs(in_point.y - self.last_goal_pos.y) < self.goal_tolerance):
                # self.get_logger().debug(f'Goal for robot {self.robot_id} is same as last one. Skipping.')
                return
                
        self.last_goal_pos = in_point

        # 目标 frame_id (e.g., robot_1/map)
        target_frame = self.local_map_frame

        try:
            # 1. 创建输入点 (在全局坐标系中)
            input_point = PointStamped()
            input_point.header.frame_id = self.global_frame  # 源坐标系 (map)
            # Zero stamp = latest-available TF; propagating the goal's creation stamp makes the
            # transformed goal age past the cars' 10s TF buffer and Nav2 stalls on it.
            input_point.header.stamp = rclpy.time.Time().to_msg()
            input_point.point.x = in_point.x
            input_point.point.y = in_point.y
            input_point.point.z = 0.0
            
            # 2. 使用 TF2 直接转换点坐标
            # tf_buffer.transform() 会自动查找变换并应用
            # 从 global_frame (map) 转换到 target_frame (robot_1/map)
            out_point = self.tf_buffer.transform(
                input_point,
                target_frame,
                timeout=Duration(seconds=1.0)
            )
            
            # 3. 发布转换后的局部目标点
            self.way_point_pub.publish(out_point)
            self.get_logger().info(
                f"✅ Transformed goal for robot {self.robot_id}: "
                f"Global({self.global_frame}): ({in_point.x:.2f}, {in_point.y:.2f}) → "
                f"Local({target_frame}): ({out_point.point.x:.2f}, {out_point.point.y:.2f})"
            )
            
        except TransformException as ex:
            self.get_logger().warn(
                f'⚠️ Failed to transform goal from {self.global_frame} to {target_frame}: {ex}'
            )
        except Exception as e:
            self.get_logger().error(f'❌ GoalTransformer failed: {e}')

    def recovery_goal_callback(self, goal_msg: PoseStamped):
        """处理恢复目标点（优先级高于普通目标）"""
        self.get_logger().info(f'🔧 Received RECOVERY goal for robot {self.robot_id}')
        
        # 直接调用相同的转换逻辑
        self.goal_callback(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    
    # robot_id 从参数服务器获取
    # (注意：Spinning 必须在节点实例化后)
    
    # 临时的 Node 仅用于读取参数
    temp_node = rclpy.create_node('temp_param_reader', allow_undeclared_parameters=True)
    temp_node.declare_parameter('robot_id', 0)
    robot_id = temp_node.get_parameter('robot_id').value
    temp_node.destroy_node()
    
    if robot_id == 0:
        print("Error: robot_id not provided to goal_transformer")
        return

    node = GoalTransformer()
    
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
