#!/usr/bin/env python3
"""
rviz_goal_relay.py

RViz目标点转发节点
订阅RViz的2D Goal Pose（/goal_pose），转换坐标系后通过reliable_communication发送给指定机器人
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from reliable_communication_interfaces.action import NavigateToGoal
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from rclpy.duration import Duration


class RVizGoalRelayNode(Node):
    """RViz目标点转发节点（通过reliable_communication）"""
    
    def __init__(self):
        super().__init__('rviz_goal_relay')
        
        # 声明参数
        self.declare_parameter('target_robot_id', 1)
        self.declare_parameter('global_frame', 'global_map')
        self.declare_parameter('tolerance', 0.5)  # 到达容差
        
        # 获取参数
        self.target_robot_id = self.get_parameter('target_robot_id').value
        self.global_frame = self.get_parameter('global_frame').value
        self.tolerance = self.get_parameter('tolerance').value
        
        self.get_logger().info(f'RViz Goal Relay initialized for robot_{self.target_robot_id}')
        self.get_logger().info(f'Global frame: {self.global_frame}, Tolerance: {self.tolerance}m')
        
        # TF2缓冲区和监听器
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 订阅RViz的目标点（RViz2默认发布到/goal_pose）
        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # 创建Action客户端（指向reliable_communication_server）
        self.action_client = ActionClient(
            self,
            NavigateToGoal,
            f'/robot_{self.target_robot_id}/navigate_to_goal'
        )
        
        # 支持动态修改目标机器人ID
        self.add_on_set_parameters_callback(self.parameter_callback)
        
        self.get_logger().info('RViz Goal Relay started, waiting for /goal_pose...')
    
    def parameter_callback(self, params):
        """参数修改回调"""
        from rcl_interfaces.msg import SetParametersResult
        
        result = SetParametersResult()
        result.successful = True
        
        for param in params:
            if param.name == 'target_robot_id':
                old_id = self.target_robot_id
                new_id = param.value
                if new_id > 0:
                    self.target_robot_id = new_id
                    # 重新创建Action客户端
                    self.action_client = ActionClient(
                        self,
                        NavigateToGoal,
                        f'/robot_{self.target_robot_id}/navigate_to_goal'
                    )
                    self.get_logger().info(f'Target robot ID changed: {old_id} -> {new_id}')
                else:
                    result.successful = False
                    result.reason = 'target_robot_id must be positive'
            elif param.name == 'tolerance':
                self.tolerance = param.value
                self.get_logger().info(f'Tolerance changed to {self.tolerance}m')
        
        return result
    
    def goal_callback(self, msg: PoseStamped):
        """
        处理RViz发布的目标点
        1. 从global_map坐标转换到robot_N/map坐标
        2. 通过reliable_communication Action发送给目标机器人
        
        Args:
            msg: RViz发布的PoseStamped消息（通常在global_map坐标系）
        """
        self.get_logger().info(
            f'📍 Received RViz goal: x={msg.pose.position.x:.2f}, '
            f'y={msg.pose.position.y:.2f}, frame="{msg.header.frame_id}"'
        )
        
        try:
            # 确定源坐标系和目标坐标系
            source_frame = msg.header.frame_id if msg.header.frame_id else self.global_frame
            target_frame = f'robot_{self.target_robot_id}/map'  # 机器人本地地图坐标系
            
            self.get_logger().info(
                f'🔄 Transforming: "{source_frame}" -> "{target_frame}"'
            )
            
            # 等待TF变换可用
            if not self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=2.0)
            ):
                self.get_logger().warn(
                    f'⚠️  TF not available: {source_frame} -> {target_frame}'
                )
                return
            
            # 执行TF变换（全局坐标 -> 本地坐标）
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )
            
            # 转换PoseStamped
            transformed_pose = tf2_geometry_msgs.do_transform_pose_stamped(msg, transform)
            transformed_pose.header.frame_id = 'map'  # 设置为机器人本地frame名称
            
            self.get_logger().info(
                f'✅ Transformed goal (local): x={transformed_pose.pose.position.x:.2f}, '
                f'y={transformed_pose.pose.position.y:.2f}'
            )
            
            # 通过Action发送给reliable_communication_server
            self.send_goal_via_action(transformed_pose)
        
        except TransformException as ex:
            self.get_logger().error(f'❌ TF transform failed: {ex}')
        except Exception as ex:
            self.get_logger().error(f'❌ Error in goal_callback: {ex}')
    
    def send_goal_via_action(self, goal_pose: PoseStamped):
        """
        通过Action发送目标点到reliable_communication_server
        
        Args:
            goal_pose: 已转换到机器人本地坐标系的目标位姿
        """
        # 等待Action服务器
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(
                f'❌ Action server not available: /robot_{self.target_robot_id}/navigate_to_goal'
            )
            return
        
        # 构造Action Goal
        goal_msg = NavigateToGoal.Goal()
        goal_msg.target_pose = goal_pose
        goal_msg.tolerance = self.tolerance
        
        self.get_logger().info(
            f'🚀 Sending goal to robot_{self.target_robot_id} via Action...'
        )
        
        # 发送Goal（异步）
        send_goal_future = self.action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        """Goal响应回调"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'❌ Goal rejected by robot_{self.target_robot_id}')
            return
        
        self.get_logger().info(f'✅ Goal accepted by robot_{self.target_robot_id}')
        
        # 等待结果（异步）
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)
    
    def feedback_callback(self, feedback_msg):
        """反馈回调"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'📊 Progress: {feedback.progress_percent:.1f}%, '
            f'distance: {feedback.distance_remaining:.2f}m',
            throttle_duration_sec=2.0  # 节流：每2秒最多输出一次
        )
    
    def result_callback(self, future):
        """结果回调"""
        result = future.result().result
        if result.success:
            self.get_logger().info(
                f'🎉 Navigation succeeded! Time: {result.elapsed_time_seconds:.1f}s'
            )
        else:
            self.get_logger().warn(
                f'⚠️  Navigation failed: {result.message}'
            )


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    node = RVizGoalRelayNode()
    
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
