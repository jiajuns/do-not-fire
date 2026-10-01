#!/usr/bin/env python3
"""
pose_config_server.py

位置配置服务器 - 管理机器人初始位置配置

功能:
1. 发布 /robot_initial_poses (geometry_msgs/PoseArray, TRANSIENT_LOCAL)
2. 提供 /set_robot_poses 服务用于动态更新位置
3. 支持通过 launch 参数或服务动态设置位置

生命周期状态:
- configure: 加载初始配置
- activate: 开始发布位置
- deactivate: 停止发布
- cleanup: 清理资源
"""

import json
import math
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher as LifecyclePublisher
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import PoseArray, Pose
from cure_explore_interfaces.srv import SetRobotPoses
from cure_explore_interfaces.msg import RobotPose


class PoseConfigServer(LifecycleNode):
    def __init__(self):
        super().__init__('pose_config_server')
        self.get_logger().info('PoseConfigServer 节点已创建')

        # 内部状态
        self.poses = []
        self.poses_pub = None
        self.poses_outbound_pub = None
        self.set_poses_srv = None

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """配置阶段：声明参数，加载配置"""
        self.get_logger().info('配置 PoseConfigServer...')

        # 声明参数
        self.declare_parameter('robot_count', 2)
        self.declare_parameter('robot_poses', '')  # JSON格式
        self.declare_parameter('global_frame', 'global_map')

        # 解析位置配置
        self.robot_count = self.get_parameter('robot_count').value
        self.global_frame = self.get_parameter('global_frame').value
        self.poses = self._parse_poses()

        # 创建 TRANSIENT_LOCAL 发布者（确保新订阅者能收到最新消息）
        qos = QoSProfile(
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
            depth=1
        )
        self.poses_pub = self.create_lifecycle_publisher(
            PoseArray, '/robot_initial_poses', qos)
        self.poses_outbound_pub = self.create_lifecycle_publisher(
            PoseArray, '/robot_initial_poses_outbound', qos)

        # 创建服务
        self.set_poses_srv = self.create_service(
            SetRobotPoses, '/set_robot_poses', self.set_poses_callback)
        self.get_logger().info(f'配置完成: robot_count={self.robot_count}')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """激活阶段：开始发布位置"""
        self.get_logger().info('激活 PoseConfigServer...')
        self.get_logger().info(f'当前有 {len(self.poses)} 个机器人位置待发布')

        # 先激活 publisher，然后再发布数据
        ret = super().on_activate(state)
        self.get_logger().info('Publisher 已激活')

        # 发布当前位置配置
        self._publish_poses(publish_outbound=False)
        self.get_logger().info('已调用 _publish_poses()')

        return ret

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """停用阶段：停止发布"""
        self.get_logger().info('停用 PoseConfigServer...')
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """清理阶段：释放资源"""
        self.get_logger().info('清理 PoseConfigServer...')
        self.poses = []
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """关闭阶段"""
        self.get_logger().info('关闭 PoseConfigServer...')
        return TransitionCallbackReturn.SUCCESS

    def _parse_poses(self):
        """解析 robot_poses 参数"""
        poses_str = self.get_parameter('robot_poses').value
        robot_count = self.get_parameter('robot_count').value

        if not poses_str:
            # 使用默认位置（线性排列）
            self.get_logger().info('使用默认位置配置')
            return self._generate_default_poses(robot_count)

        try:
            # 先尝试分号格式: "x,y,yaw;x,y,yaw"
            if ';' in poses_str:
                poses_data = []
                for item in poses_str.split(';'):
                    item = item.strip()
                    if not item: continue
                    parts = [p.strip() for p in item.split(',')]
                    if len(parts) >= 2:
                        poses_data.append([float(parts[0]), float(parts[1]),
                                          float(parts[2]) if len(parts) > 2 else 0.0])
            else:
                poses_data = json.loads(poses_str)
            poses = []

            for i, p in enumerate(poses_data):
                pose = Pose()
                if isinstance(p, list):
                    # 格式: [x, y, yaw]
                    pose.position.x = float(p[0])
                    pose.position.y = float(p[1])
                    pose.position.z = 0.0
                    yaw = float(p[2]) if len(p) > 2 else 0.0
                elif isinstance(p, dict):
                    # 格式: {"x": ..., "y": ..., "yaw": ...}
                    pose.position.x = float(p.get('x', 0))
                    pose.position.y = float(p.get('y', 0))
                    pose.position.z = 0.0
                    yaw = float(p.get('yaw', 0))
                else:
                    self.get_logger().warn(f'无法解析位置 {i}: {p}')
                    continue

                # 设置四元数 (绕Z轴旋转)
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = math.sin(yaw / 2)
                pose.orientation.w = math.cos(yaw / 2)

                poses.append(pose)

            # 验证数量
            if len(poses) < robot_count:
                self.get_logger().warn(
                    f'位置数量不足: 需要 {robot_count}，提供了 {len(poses)}，'
                    f'将使用默认位置补充')
                while len(poses) < robot_count:
                    default_poses = self._generate_default_poses(robot_count)
                    poses.append(default_poses[len(poses)])

            return poses

        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON解析错误: {e}')
            return self._generate_default_poses(robot_count)

    def _generate_default_poses(self, robot_count):
        """生成默认位置（与 robot_offsets.yaml 保持一致）"""
        # 硬编码的默认位置，与 config/robot_offsets.yaml 保持一致
        default_positions = [
            (0.0, 0.0, 0.0),    # robot_1
            (0.0, -0.36, 0.0),#(0.86, 6.23, 0.0),    # robot_2
            (0.0, 0.36, 0.0),# (-4.57, -9.54, 0.0),   # robot_3
            (-1.0, 0.0, 0.0),   # robot_4
            (-1.5, 0.0, 0.0),   # robot_5
        ]
        
        poses = []
        for i in range(robot_count):
            pose = Pose()
            if i < len(default_positions):
                # 使用预定义的位置
                x, y, yaw = default_positions[i]
                pose.position.x = x
                pose.position.y = y
                pose.position.z = 0.0
                # 设置朝向
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = math.sin(yaw / 2)
                pose.orientation.w = math.cos(yaw / 2)
            else:
                # 超出预定义数量时，继续线性排列
                pose.position.x = -0.5 * i
                pose.position.y = 0.0
                pose.position.z = 0.0
                pose.orientation.w = 1.0
            poses.append(pose)
        return poses

    def _publish_poses(self, *, publish_outbound: bool):
        """发布位置配置。

        publish_outbound=False:
            只在服务器域本地话题 /robot_initial_poses 发布，用于正常启动展示。
        publish_outbound=True:
            额外在 /robot_initial_poses_outbound 发布，供跨域 relay 把更新后的位姿
            推送到仿真域和机器人域，避免被仿真启动时的默认位姿覆盖。
        """
        self.get_logger().info(
            f'_publish_poses() 被调用，poses_pub={self.poses_pub}, '
            f'poses_outbound_pub={self.poses_outbound_pub}, '
            f'publish_outbound={publish_outbound}'
        )

        if self.poses_pub is None:
            self.get_logger().warn('poses_pub 为 None，无法发布')
            return

        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.global_frame
        msg.poses = self.poses

        self.get_logger().info(
            f'准备发布到 /robot_initial_poses，frame_id={self.global_frame}, poses数量={len(self.poses)}'
        )
        self.poses_pub.publish(msg)
        self.get_logger().info('publish() 调用完成')

        if publish_outbound:
            if self.poses_outbound_pub is None:
                self.get_logger().warn('poses_outbound_pub 为 None，无法发布 outbound 位姿')
            else:
                self.poses_outbound_pub.publish(msg)
                self.get_logger().info('已发布到 /robot_initial_poses_outbound')

        self.get_logger().info(f'已发布 {len(self.poses)} 个机器人位置')
        for i, pose in enumerate(self.poses):
            yaw = 2 * math.atan2(pose.orientation.z, pose.orientation.w)
            self.get_logger().info(
                f'  robot_{i+1}: x={pose.position.x:.2f}, '
                f'y={pose.position.y:.2f}, yaw={yaw:.2f}')

    def set_poses_callback(self, request, response):
        """处理设置位置服务请求"""
        self.get_logger().info(f'收到设置位置请求: {len(request.poses)} 个机器人')

        try:
            # 如果请求为空，保持当前配置并重新发布
            if len(request.poses) == 0:
                self.get_logger().info('请求为空，保持当前配置并重新发布')
                self._publish_poses(publish_outbound=True)
                response.success = True
                response.message = f'已重新发布 {len(self.poses)} 个机器人位置'
                return response

            new_poses = []
            for robot_pose in request.poses:
                pose = Pose()
                pose.position.x = robot_pose.x
                pose.position.y = robot_pose.y
                pose.position.z = 0.0

                yaw = robot_pose.yaw
                pose.orientation.z = math.sin(yaw / 2)
                pose.orientation.w = math.cos(yaw / 2)

                new_poses.append(pose)

            # 更新位置
            self.poses = new_poses
            self.robot_count = len(new_poses)

            # 重新发布
            self._publish_poses(publish_outbound=True)

            response.success = True
            response.message = f'已更新 {len(new_poses)} 个机器人位置'

        except Exception as e:
            response.success = False
            response.message = f'更新位置失败: {str(e)}'
            self.get_logger().error(response.message)

        return response


def main(args=None):
    rclpy.init(args=args)

    node = PoseConfigServer()

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
