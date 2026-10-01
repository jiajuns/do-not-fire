#!/usr/bin/env python3
"""Real-robot CURE exploration with robot_ids support.
Usage: ros2 launch real_cure_bringup real_explore.launch.py robot_ids:=2, robot_poses:="0,0,0"
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import (DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node, LifecycleNode
import launch_ros.actions
from lifecycle_msgs.msg import Transition


def launch_all_nodes(context, *args, **kwargs):
    """Compute robot_count from robot_ids, create all nodes."""
    ids_str = LaunchConfiguration('robot_ids').perform(context)
    ids_str = ids_str.rstrip(',')
    robot_ids_list = [x.strip() for x in ids_str.split(',') if x.strip()]
    rc = len(robot_ids_list)

    common = {'use_sim_time': False, 'robot_count': rc,
              'robot_namespace': LaunchConfiguration('robot_namespace'),
              'global_frame': LaunchConfiguration('global_frame')}

    nodes = []

    # Core C++ nodes
    nodes.append(Node(package='real_robot_communication', executable='real_robot_communication_node',
        name='robot_communication', output='screen',
        parameters=[{'robot_count': rc, 'robot_ids': LaunchConfiguration('robot_ids'),
                     'listen_localhost': False, 'listen_lan': True}]))
    nodes.append(Node(package='real_reliable_communication_server',
        executable='reliable_server_node', name='reliable_communication_server', output='screen',
        parameters=[{'robot_count': rc, 'robot_ids': LaunchConfiguration('robot_ids'),
                     'tcp_port': LaunchConfiguration('tcp_port'), 'listen_localhost': False, 'listen_lan': True}]))
    nodes.append(Node(package='real_cure_cpp', executable='map_merger_simple_node',
        name='map_merger', output='screen', parameters=[{**common, 'update_rate': 2.0}]))
    nodes.append(Node(package='real_cure_cpp', executable='frontier_detector_node',
        name='frontier_detector', output='screen',
        parameters=[{'use_sim_time': False, 'update_rate': 0.5, 'min_frontier_size': 10,
                     'info_radius': 2.0, 'use_reachability_filter': True}]))
    nodes.append(Node(package='real_cure_cpp', executable='simple_greedy_explorer_node',
        name='simple_greedy_explorer', output='screen',
        parameters=[{**common, 'decision_frequency': 0.333, 'stuck_distance_threshold': 0.3,
                     'stuck_time_threshold': 5.0, 'progress_stuck_time_threshold': 10.0,
                     'progress_distance_threshold': 0.1}]))

    # Python nodes
    nodes.append(Node(package='real_cure_bringup', executable='robot_position_publisher',
        name='robot_position_publisher', output='screen', parameters=[{**common, 'update_rate': 5.0}]))
    nodes.append(Node(package='real_cure_bringup', executable='trajectory_publisher',
        name='trajectory_publisher', output='screen', parameters=[common]))
    nodes.append(Node(package='real_cure_bringup', executable='external_robot_tracker',
        name='external_robot_tracker', output='screen',
        parameters=[{'use_sim_time': False,
                     'external_robot_frame': LaunchConfiguration('external_robot_frame'),
                     'external_robot_port': LaunchConfiguration('external_robot_port')}]))
    # 外部小车坐标系偏移 (global_map -> external_robot)，纯可视化，不影响已有 TF
    nodes.append(Node(package='tf2_ros', executable='static_transform_publisher',
        name='external_robot_tf',
        arguments=LaunchConfiguration('external_robot_offset').perform(context).split(',') + [
            LaunchConfiguration('global_frame'), LaunchConfiguration('external_robot_frame')],
        parameters=[{'use_sim_time': False}]))
    nodes.append(Node(package='real_cure_bringup', executable='external_map_receiver',
        name='external_map_receiver', output='screen',
        parameters=[{'use_sim_time': False,
                     'external_robot_frame': LaunchConfiguration('external_robot_frame'),
                     'external_map_port': LaunchConfiguration('external_map_port')}]))
    nodes.append(Node(package='real_cure_bringup', executable='rviz_goal_relay',
        name='rviz_goal_relay', output='screen',
        parameters=[{'use_sim_time': False, 'global_frame': LaunchConfiguration('global_frame'),
                     'target_robot_id': LaunchConfiguration('rviz_target_robot_id'),
                     'tolerance': LaunchConfiguration('arrival_tolerance')}]))
    nodes.append(Node(package='real_cure_bringup', executable='path_visualizer',
        name='path_visualizer', output='screen', parameters=[{**common, 'vis_topic': '/planned_paths_vis'}]))

    # Lifecycle nodes
    pose_node = LifecycleNode(package='real_cure_bringup', executable='pose_config_server',
        namespace='', name='pose_config_server', output='screen',
        parameters=[{**common, 'robot_poses': LaunchConfiguration('robot_poses')}])
    tf_node = LifecycleNode(package='real_cure_bringup', executable='dynamic_tf_broadcaster',
        namespace='', name='dynamic_tf_broadcaster', output='screen', parameters=[common])

    pose_idx = len(nodes)
    nodes.append(pose_node)
    nodes.append(tf_node)

    nodes.append(RegisterEventHandler(OnProcessStart(target_action=nodes[pose_idx],
        on_start=[TimerAction(period=1.0, actions=[launch_ros.actions.LifecycleTransition(
            lifecycle_node_names=['pose_config_server'], transition_ids=[Transition.TRANSITION_CONFIGURE])])])))
    nodes.append(RegisterEventHandler(OnProcessStart(target_action=nodes[pose_idx],
        on_start=[TimerAction(period=2.0, actions=[launch_ros.actions.LifecycleTransition(
            lifecycle_node_names=['pose_config_server'], transition_ids=[Transition.TRANSITION_ACTIVATE])])])))
    nodes.append(RegisterEventHandler(OnProcessStart(target_action=nodes[pose_idx+1],
        on_start=[TimerAction(period=3.0, actions=[launch_ros.actions.LifecycleTransition(
            lifecycle_node_names=['dynamic_tf_broadcaster'], transition_ids=[Transition.TRANSITION_CONFIGURE])])])))
    nodes.append(RegisterEventHandler(OnProcessStart(target_action=nodes[pose_idx+1],
        on_start=[TimerAction(period=5.0, actions=[launch_ros.actions.LifecycleTransition(
            lifecycle_node_names=['dynamic_tf_broadcaster'], transition_ids=[Transition.TRANSITION_ACTIVATE])])])))

    # Goal transformers
    enable = LaunchConfiguration('enable_exploration').perform(context).lower() == 'true'
    if enable:
        ns = LaunchConfiguration('robot_namespace').perform(context)
        gf = LaunchConfiguration('global_frame').perform(context)
        for i in range(rc):
            s = i + 1
            nodes.append(Node(package='real_cure_bringup', executable='goal_transformer',
                name=f'goal_transformer_{s}', namespace=f'{ns}{s}', output='screen',
                parameters=[{'use_sim_time': False, 'robot_id': s, 'global_frame': gf}]))

    return nodes


def generate_launch_description():
    pkg_share = get_package_share_directory('real_cure_bringup')

    args = [
        DeclareLaunchArgument('robot_ids', default_value='2,7,8',
            description='Comma-separated actual robot IDs'),
        DeclareLaunchArgument('robot_namespace', default_value='robot_'),
        DeclareLaunchArgument('global_frame', default_value='global_map'),
        DeclareLaunchArgument('enable_exploration', default_value='true'),
        DeclareLaunchArgument('tcp_port', default_value='12140'),
        DeclareLaunchArgument('rviz_config',
            default_value=os.path.join(pkg_share, 'rviz', 'real_explore.rviz')),
        DeclareLaunchArgument('rviz_target_robot_id', default_value='1'),
        DeclareLaunchArgument('arrival_tolerance', default_value='0.8'),
        DeclareLaunchArgument('robot_poses', default_value='',
            description='Poses: x,y,yaw;x,y,yaw'),
        DeclareLaunchArgument('external_robot_port', default_value='12141',
            description='UDP port for external robot position data (bingda car)'),
        DeclareLaunchArgument('external_robot_frame', default_value='external_robot',
            description='Frame id for external robot visualization'),
        DeclareLaunchArgument('external_robot_offset', default_value='0,0,0,0,0,0',
            description='Offset x,y,z,roll,pitch,yaw from global_map to external_robot frame'),
        DeclareLaunchArgument('external_map_port', default_value='12142',
            description='TCP port for external robot map data'),
    ]

    return LaunchDescription(args + [
        OpaqueFunction(function=launch_all_nodes),
        Node(package='rviz2', executable='rviz2', name='rviz2', output='log',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': False}]),
    ])
