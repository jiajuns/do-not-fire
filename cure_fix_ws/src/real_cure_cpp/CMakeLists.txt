cmake_minimum_required(VERSION 3.8)
project(real_cure_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cure_explore_interfaces REQUIRED)
find_package(reliable_communication_interfaces REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(include)

add_executable(map_merger_simple_node src/map_merger_simple.cpp src/map_merger_simple_node.cpp)
ament_target_dependencies(map_merger_simple_node rclcpp nav_msgs geometry_msgs std_msgs tf2 tf2_ros cure_explore_interfaces reliable_communication_interfaces)

add_executable(frontier_detector_node src/frontier_detector.cpp src/frontier_detector_node.cpp)
ament_target_dependencies(frontier_detector_node rclcpp nav_msgs std_msgs geometry_msgs visualization_msgs cure_explore_interfaces OpenCV)

add_executable(simple_greedy_explorer_node src/simple_greedy_explorer_node.cpp)
ament_target_dependencies(simple_greedy_explorer_node rclcpp rclcpp_action nav_msgs std_msgs geometry_msgs tf2_ros tf2_geometry_msgs cure_explore_interfaces reliable_communication_interfaces)

install(TARGETS map_merger_simple_node frontier_detector_node simple_greedy_explorer_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
ament_package()
