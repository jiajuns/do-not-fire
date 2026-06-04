"""Single-machine / test launch: spawn N per-robot SVoI comm nodes (namespaced robot0..N-1).
On real hardware each robot instead runs ONE `comm_node` with its own robot_id param.
Usage:  ros2 launch svoi_comm team.launch.py n_agents:=4 mode:=spiking
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn(context, *args, **kwargs):
    n = int(LaunchConfiguration("n_agents").perform(context))
    mode = LaunchConfiguration("mode").perform(context)
    nodes = []
    for i in range(n):
        nodes.append(Node(
            package="svoi_comm", executable="comm_node", name=f"comm_{i}",
            namespace=f"robot{i}", output="screen",
            parameters=[{
                "robot_id": i, "n_agents": n, "mode": mode,
                "odom_topic": f"/robot{i}/Odometry",
                "embedding_topic": f"/robot{i}/embedding",
                "state_topic": "/svoi/state",
                "rate_hz": 5.0,
            }],
        ))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("n_agents", default_value="4"),
        DeclareLaunchArgument("mode", default_value="spiking"),
        OpaqueFunction(function=_spawn),
    ])
