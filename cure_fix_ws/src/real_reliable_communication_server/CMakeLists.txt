cmake_minimum_required(VERSION 3.8)
project(real_reliable_communication_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(reliable_communication_interfaces REQUIRED)
find_package(cure_explore_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(communication_simulation REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Library
add_library(${PROJECT_NAME} SHARED
  src/message_protocol.cpp
  src/tcp_server.cpp
  src/action_manager.cpp
  src/path_planner.cpp
  src/qos_monitor.cpp
  src/reliable_server_node.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rclcpp_action
  rclcpp_components
  reliable_communication_interfaces
  cure_explore_interfaces
  geometry_msgs
  std_msgs
  nav_msgs
  visualization_msgs
  communication_simulation
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(${PROJECT_NAME}
  yaml-cpp
)

# Register as a component
rclcpp_components_register_nodes(${PROJECT_NAME} "reliable_communication::ReliableServerNode")

# Executable
add_executable(reliable_server_node src/main.cpp)
target_link_libraries(reliable_server_node ${PROJECT_NAME})

# Install
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS reliable_server_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
