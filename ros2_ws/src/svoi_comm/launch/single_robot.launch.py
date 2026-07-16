"""Launch one real-robot state communication node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    names_defaults = [
        ("robot_id", "0"), ("n_agents", "4"), ("mode", "periodic"),
        ("rate_hz", "5.0"), ("start_time", "0.0"), ("interval", "20"),
        ("phase_offset", "0"), ("odom_topic", "/Odometry"),
        ("embedding_topic", "embedding"), ("state_topic", "/svoi/state"),
        ("pose_file", ""),
    ]
    arguments = [DeclareLaunchArgument(name, default_value=default)
                 for name, default in names_defaults]
    parameters = {name: LaunchConfiguration(name) for name, _ in names_defaults}
    node = Node(
        package="svoi_comm", executable="comm_node",
        name=["comm_", LaunchConfiguration("robot_id")],
        namespace=["robot", LaunchConfiguration("robot_id")],
        output="screen", parameters=[parameters],
    )
    return LaunchDescription(arguments + [node])
