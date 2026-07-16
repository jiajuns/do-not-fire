"""Launch the map gate on the same ROS_DOMAIN_ID as SLAM/navigation."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("robot_id", default_value="0"),
        DeclareLaunchArgument("n_agents", default_value="4"),
        DeclareLaunchArgument("period_sec", default_value="4.0"),
        DeclareLaunchArgument("phase_sec", default_value="-1.0"),
        DeclareLaunchArgument("start_time", default_value="0.0"),
        DeclareLaunchArgument("map_topic", default_value="/map"),
        DeclareLaunchArgument("gated_map_topic", default_value="/svoi/gated_map"),
        DeclareLaunchArgument("event_topic", default_value="/svoi/comm_events"),
        DeclareLaunchArgument("stats_topic", default_value="/svoi/comm_stats"),
        DeclareLaunchArgument("fire_topic", default_value="/svoi/fire"),
    ]
    node = Node(
        package="svoi_comm",
        executable="map_gate_node",
        name=["map_gate_", LaunchConfiguration("robot_id")],
        output="screen",
        parameters=[{
            "robot_id": LaunchConfiguration("robot_id"),
            "n_agents": LaunchConfiguration("n_agents"),
            "period_sec": LaunchConfiguration("period_sec"),
            "phase_sec": LaunchConfiguration("phase_sec"),
            "start_time": LaunchConfiguration("start_time"),
            "map_topic": LaunchConfiguration("map_topic"),
            "gated_map_topic": LaunchConfiguration("gated_map_topic"),
            "event_topic": LaunchConfiguration("event_topic"),
            "stats_topic": LaunchConfiguration("stats_topic"),
            "fire_topic": LaunchConfiguration("fire_topic"),
        }],
    )
    return LaunchDescription(arguments + [node])
