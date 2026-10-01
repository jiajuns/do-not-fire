from setuptools import setup, find_packages
package_name = "real_cure_bringup"
setup(
    name=package_name, version="1.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/real_explore.launch.py",
                                                 "launch/real_manual.launch.py"]),
        ("share/" + package_name + "/rviz", ["rviz/real_explore.rviz"]),
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "pose_config_server = real_cure_bringup.pose_config_server:main",
            "dynamic_tf_broadcaster = real_cure_bringup.dynamic_tf_broadcaster:main",
            "robot_position_publisher = real_cure_bringup.robot_position_publisher:main",
            "rviz_goal_relay = real_cure_bringup.rviz_goal_relay:main",
            "path_visualizer = real_cure_bringup.path_visualizer:main",
            "goal_transformer = real_cure_bringup.goal_transformer:main",
            "trajectory_publisher = real_cure_bringup.trajectory_publisher:main",
            "external_robot_tracker = real_cure_bringup.external_robot_tracker:main",
            "external_map_receiver = real_cure_bringup.external_map_receiver:main",
        ],
    },
)
