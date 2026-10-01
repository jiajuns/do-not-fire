"""
Reliable Communication Server Launch File

启动可靠通信服务器（TCP服务器、Action管理、QoS监控）
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # 获取包的共享目录
    pkg_share = FindPackageShare('reliable_communication_server')
    
    # 默认配置文件路径
    default_config_file = PathJoinSubstitution([
        pkg_share,
        'config',
        'reliable_server_config.yaml'
    ])

    # 声明launch参数
    robot_count_arg = DeclareLaunchArgument(
        'robot_count',
        default_value='2',
        description='Number of robots'
    )

    tcp_port_arg = DeclareLaunchArgument(
        'tcp_port',
        default_value='12140',
        description='TCP server port'
    )

    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to YAML config file (optional)'
    )

    # 可靠通信服务器节点
    reliable_server_node = Node(
        package='reliable_communication_server',
        executable='reliable_server_node',
        name='reliable_communication_server',
        output='screen',
        respawn=True,
        respawn_delay=2.0,
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'listen_localhost': False,
                'listen_lan': True,
            },
        ]
    )

    return LaunchDescription([
        robot_count_arg,
        tcp_port_arg,
        config_file_arg,
        reliable_server_node,
    ])
