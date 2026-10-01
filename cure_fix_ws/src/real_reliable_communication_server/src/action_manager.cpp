#include "reliable_communication_server/action_manager.hpp"
#include <rclcpp/serialization.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <nav_msgs/msg/path.hpp>
#include <thread>
#include <cmath>

namespace reliable_communication {

ActionManager::ActionManager(rclcpp::Node* node, TCPServer* tcp_server)
  : node_(node), tcp_server_(tcp_server), next_request_id_(1) {

  // 注册消息处理器
  tcp_server_->registerMessageHandler(
    MessageType::ACTION_FEEDBACK,
    std::bind(&ActionManager::handleFeedback, this,
              std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4)
  );

  tcp_server_->registerMessageHandler(
    MessageType::ACTION_RESULT,
    std::bind(&ActionManager::handleResult, this,
              std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4)
  );

  tcp_server_->registerMessageHandler(
    MessageType::PATH_PLANNING_REQUEST,
    std::bind(&ActionManager::handlePathPlanningRequest, this,
              std::placeholders::_1, std::placeholders::_2,
              std::placeholders::_3, std::placeholders::_4)
  );

  // 创建路径规划器
  path_planner_ = std::make_unique<PathPlanner>(node_);

  RCLCPP_INFO(node_->get_logger(), "ActionManager initialized with PathPlanner");
}

uint64_t ActionManager::sendGoal(
  int robot_id,
  const geometry_msgs::msg::PoseStamped& goal,
  float tolerance
) {
  // 生成请求ID
  uint64_t request_id;
  {
    std::lock_guard<std::mutex> lock(request_id_mutex_);
    request_id = next_request_id_++;
  }

  // 构造Goal消息
  NavigateToGoal::Goal action_goal;
  action_goal.target_pose = goal;
  action_goal.tolerance = tolerance;

  // 序列化
  rclcpp::SerializedMessage serialized_msg;
  rclcpp::Serialization<NavigateToGoal::Goal> serializer;
  serializer.serialize_message(&action_goal, &serialized_msg);

  std::vector<uint8_t> payload(
    serialized_msg.get_rcl_serialized_message().buffer,
    serialized_msg.get_rcl_serialized_message().buffer +
    serialized_msg.get_rcl_serialized_message().buffer_length
  );

  // 发送
  bool success = tcp_server_->sendMessage(
    robot_id,
    MessageType::ACTION_GOAL,
    request_id,
    payload
  );

  if (success) {
    // 记录活跃目标
    {
      std::lock_guard<std::mutex> lock(active_goals_mutex_);
      active_goals_[robot_id] = request_id;
    }
    // 记录目标点信息
    {
      std::lock_guard<std::mutex> lock(active_goal_info_mutex_);
      active_goal_info_[robot_id] = std::make_pair(goal, tolerance);
    }
    RCLCPP_INFO(node_->get_logger(),
                "Sent goal to robot %d: (%.2f, %.2f) [request_id=%lu]",
                robot_id, goal.pose.position.x, goal.pose.position.y, request_id);
  } else {
    RCLCPP_ERROR(node_->get_logger(),
                 "🚨🚨🚨 [SERVER ERROR] Failed to send goal to robot %d - TCP send failed! 🚨🚨🚨",
                 robot_id);
  }

  return request_id;
}

uint64_t ActionManager::sendGoalWithAutoCancel(
  int robot_id,
  const geometry_msgs::msg::PoseStamped& goal,
  float tolerance
) {
  // 检查是否有活跃目标，先取消
  uint64_t old_request_id = 0;
  {
    std::lock_guard<std::mutex> lock(active_goals_mutex_);
    auto it = active_goals_.find(robot_id);
    if (it != active_goals_.end()) {
      old_request_id = it->second;
    }
  }

  if (old_request_id > 0) {
    RCLCPP_DEBUG(node_->get_logger(),
                 "Robot %d has active goal [request_id=%lu], canceling...",
                 robot_id, old_request_id);
    cancelGoal(robot_id, old_request_id);

    // 移除sleep：TCP可靠传输保证取消消息会到达
    // 机器人端的action handler会正确处理goal preemption
    // 不需要在服务器端阻塞等待
  }

  // 发送新目标
  return sendGoal(robot_id, goal, tolerance);
}

uint64_t ActionManager::sendGoalWithDeduplication(
  int robot_id,
  const geometry_msgs::msg::PoseStamped& goal,
  float tolerance
) {
  // 生成新的请求ID
  uint64_t new_request_id;
  {
    std::lock_guard<std::mutex> lock(request_id_mutex_);
    new_request_id = next_request_id_++;
  }

  // 检查是否有活跃目标且目标点相同
  uint64_t old_request_id = 0;
  bool has_same_goal = false;
  {
    std::lock_guard<std::mutex> lock_goals(active_goals_mutex_);
    std::lock_guard<std::mutex> lock_info(active_goal_info_mutex_);
    
    auto it_goals = active_goals_.find(robot_id);
    auto it_info = active_goal_info_.find(robot_id);
    
    if (it_goals != active_goals_.end() && it_info != active_goal_info_.end()) {
      old_request_id = it_goals->second;
      const auto& [old_goal, old_tolerance] = it_info->second;
      
      // 比较目标点是否相同
      if (isSameGoal(goal, old_goal) && std::abs(tolerance - old_tolerance) < 0.001f) {
        has_same_goal = true;
        RCLCPP_INFO(node_->get_logger(),
                    "Robot %d: Same goal detected! Reusing old goal [old_req=%lu, new_req=%lu]",
                    robot_id, old_request_id, new_request_id);
      }
    }
  }

  if (has_same_goal) {
    // 目标点相同，建立映射关系，不发送新目标
    {
      std::lock_guard<std::mutex> lock(request_id_mapping_mutex_);
      request_id_mapping_[new_request_id] = old_request_id;
    }
    RCLCPP_INFO(node_->get_logger(),
                "Robot %d: Goal deduplicated, mapping new_req %lu -> old_req %lu",
                robot_id, new_request_id, old_request_id);
    return new_request_id;
  }

  // 目标点不同，使用AutoCancel发送新目标
  uint64_t actual_request_id = sendGoalWithAutoCancel(robot_id, goal, tolerance);
  
  // 建立映射：new_request_id -> actual_request_id
  {
    std::lock_guard<std::mutex> lock(request_id_mapping_mutex_);
    request_id_mapping_[new_request_id] = actual_request_id;
  }
  
  return new_request_id;
}

bool ActionManager::cancelGoal(int robot_id, uint64_t request_id) {
  // 发送空负载的取消消息
  std::vector<uint8_t> empty_payload;
  bool success = tcp_server_->sendMessage(
    robot_id,
    MessageType::ACTION_CANCEL,
    request_id,
    empty_payload
  );

  if (success) {
    RCLCPP_INFO(node_->get_logger(), "Sent cancel to robot %d [request_id=%lu]",
                robot_id, request_id);
    // 清除活跃目标记录
    clearActiveGoal(robot_id);
  }

  return success;
}

uint64_t ActionManager::getActiveGoalId(int robot_id) const {
  std::lock_guard<std::mutex> lock(active_goals_mutex_);
  auto it = active_goals_.find(robot_id);
  return (it != active_goals_.end()) ? it->second : 0;
}

bool ActionManager::hasActiveGoal(int robot_id) const {
  std::lock_guard<std::mutex> lock(active_goals_mutex_);
  return active_goals_.find(robot_id) != active_goals_.end();
}

void ActionManager::clearActiveGoal(int robot_id) {
  std::lock_guard<std::mutex> lock(active_goals_mutex_);
  active_goals_.erase(robot_id);
  
  // 同时清除目标点信息
  {
    std::lock_guard<std::mutex> lock_info(active_goal_info_mutex_);
    active_goal_info_.erase(robot_id);
  }
}

void ActionManager::setFeedbackCallback(
  std::function<void(int, const NavigateToGoal::Feedback&)> callback
) {
  feedback_callback_ = callback;
}

void ActionManager::setResultCallback(
  std::function<void(int, const NavigateToGoal::Result&)> callback
) {
  result_callback_ = callback;
}

void ActionManager::reset() {
  RCLCPP_INFO(node_->get_logger(), "🔄 ActionManager reset starting...");

  // 收集所有需要发送取消的机器人ID
  std::set<int> robot_ids_to_cancel;

  // 1. 从活跃路径中收集机器人ID并取消
  {
    std::lock_guard<std::mutex> lock(active_paths_mutex_);
    for (const auto& [robot_id, active_path] : active_paths_) {
      robot_ids_to_cancel.insert(robot_id);
      if (active_path.request_id > 0) {
        RCLCPP_INFO(node_->get_logger(),
                    "🛑 Canceling path navigation for robot %d [request_id=%lu]",
                    robot_id, active_path.request_id);
        std::vector<uint8_t> empty_payload;
        tcp_server_->sendMessage(robot_id, MessageType::ACTION_CANCEL, active_path.request_id, empty_payload);
      }
    }
    active_paths_.clear();
    RCLCPP_INFO(node_->get_logger(), "Cleared all active paths");
  }

  // 2. 从活跃目标中收集机器人ID并取消
  {
    std::lock_guard<std::mutex> lock(active_goals_mutex_);
    for (const auto& [robot_id, request_id] : active_goals_) {
      robot_ids_to_cancel.insert(robot_id);
      RCLCPP_INFO(node_->get_logger(),
                  "🛑 Canceling active goal for robot %d [request_id=%lu]",
                  robot_id, request_id);
      std::vector<uint8_t> empty_payload;
      tcp_server_->sendMessage(robot_id, MessageType::ACTION_CANCEL, request_id, empty_payload);
    }
    active_goals_.clear();
  }

  // 3. 对所有涉及的机器人发送全局取消消息（request_id=0 表示清除所有状态）
  for (int robot_id : robot_ids_to_cancel) {
    RCLCPP_INFO(node_->get_logger(),
                "📤 Sending global reset cancel to robot %d [request_id=0]", robot_id);
    std::vector<uint8_t> empty_payload;
    tcp_server_->sendMessage(robot_id, MessageType::ACTION_CANCEL, 0, empty_payload);
  }

  // 4. 清除目标点信息和映射
  {
    std::lock_guard<std::mutex> lock_info(active_goal_info_mutex_);
    active_goal_info_.clear();
  }
  {
    std::lock_guard<std::mutex> lock_mapping(request_id_mapping_mutex_);
    request_id_mapping_.clear();
  }

  // 5. 清除缓存的局部地图（重置后地图会过时）
  {
    std::lock_guard<std::mutex> lock(local_maps_mutex_);
    robot_local_maps_.clear();
    RCLCPP_INFO(node_->get_logger(), "Cleared cached local maps");
  }

  // 6. 重置request_id计数器
  {
    std::lock_guard<std::mutex> lock(request_id_mutex_);
    next_request_id_ = 1;
  }

  // 7. 清除路径可视化
  if (path_marker_pub_) {
    visualization_msgs::msg::MarkerArray clear_markers;
    visualization_msgs::msg::Marker delete_all;
    delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
    clear_markers.markers.push_back(delete_all);
    path_marker_pub_->publish(clear_markers);
    RCLCPP_INFO(node_->get_logger(), "Cleared path visualization markers");
  }

  RCLCPP_INFO(node_->get_logger(), "✅ ActionManager reset completed");
}

void ActionManager::handleFeedback(
  int robot_id,
  uint64_t request_id,
  uint64_t timestamp,
  const std::vector<uint8_t>& payload
) {
  // 反序列化
  rclcpp::SerializedMessage serialized_msg;
  serialized_msg.reserve(payload.size());
  std::memcpy(
    serialized_msg.get_rcl_serialized_message().buffer,
    payload.data(),
    payload.size()
  );
  serialized_msg.get_rcl_serialized_message().buffer_length = payload.size();

  NavigateToGoal::Feedback feedback;
  rclcpp::Serialization<NavigateToGoal::Feedback> serializer;
  serializer.deserialize_message(&serialized_msg, &feedback);

  RCLCPP_DEBUG(node_->get_logger(),
               "Received feedback from robot %d: distance=%.2fm, progress=%.1f%% [req=%lu]",
               robot_id, feedback.distance_remaining, feedback.progress_percent, request_id);

  // 调用回调（回调会收到实际的request_id，无需转换）
  if (feedback_callback_) {
    feedback_callback_(robot_id, feedback);
  }
}

void ActionManager::handleResult(
  int robot_id,
  uint64_t request_id,
  uint64_t timestamp,
  const std::vector<uint8_t>& payload
) {
  // 反序列化
  rclcpp::SerializedMessage serialized_msg;
  serialized_msg.reserve(payload.size());
  std::memcpy(
    serialized_msg.get_rcl_serialized_message().buffer,
    payload.data(),
    payload.size()
  );
  serialized_msg.get_rcl_serialized_message().buffer_length = payload.size();

  NavigateToGoal::Result result;
  rclcpp::Serialization<NavigateToGoal::Result> serializer;
  serializer.deserialize_message(&serialized_msg, &result);

  RCLCPP_INFO(node_->get_logger(),
              "Received result from robot %d: success=%d, message='%s', time=%.2fs [req=%lu]",
              robot_id, result.success, result.message.c_str(), result.elapsed_time_seconds, request_id);

  // 处理活跃路径（使用单一锁避免竞态条件）
  if (result.success) {
    std::lock_guard<std::mutex> lock(active_paths_mutex_);
    auto it = active_paths_.find(robot_id);
    if (it != active_paths_.end()) {
      ActivePath& active_path = it->second;

      // 检查是否已到达最终目标
      bool at_final_goal = false;
      if (active_path.current_path_index >= active_path.full_astar_path.size() - 1) {
        at_final_goal = true;
      } else {
        // 检查距离最终目标的距离
        const auto& final_goal = active_path.final_goal_global;
        const auto& current_pos = active_path.full_astar_path[active_path.current_path_index];
        double dist_to_final = std::sqrt(
          std::pow(current_pos.x - final_goal.x, 2) +
          std::pow(current_pos.y - final_goal.y, 2)
        );
        if (dist_to_final < 0.5) {  // 容差
          at_final_goal = true;
        }
      }

      if (at_final_goal) {
        RCLCPP_INFO(node_->get_logger(),
                    "🎯 Robot %d reached final goal! Clearing active path.", robot_id);
        active_paths_.erase(it);
      } else {
        // 还有更多路径要走，选择下一个中继点
        geometry_msgs::msg::Point current_waypoint_pos =
          active_path.full_astar_path[active_path.current_path_index];
        RCLCPP_INFO(node_->get_logger(),
                    "📍 Robot %d reached intermediate waypoint at index %zu, selecting next...",
                    robot_id, active_path.current_path_index);

        bool success = selectAndSendNextWaypoint(robot_id, current_waypoint_pos);
        if (!success) {
          RCLCPP_WARN(node_->get_logger(),
                      "Robot %d: Failed to select next waypoint, clearing path", robot_id);
          active_paths_.erase(robot_id);
        }
      }
    }
  } else {
    // 导航失败，清除活跃路径
    std::lock_guard<std::mutex> lock(active_paths_mutex_);
    auto it = active_paths_.find(robot_id);
    if (it != active_paths_.end()) {
      RCLCPP_WARN(node_->get_logger(),
                  "Robot %d navigation failed, clearing active path", robot_id);
      active_paths_.erase(it);
    }
  }

  // 清除活跃目标记录（目标已完成）
  clearActiveGoal(robot_id);

  // 清除所有指向这个request_id的映射
  {
    std::lock_guard<std::mutex> lock(request_id_mapping_mutex_);
    auto it = request_id_mapping_.begin();
    while (it != request_id_mapping_.end()) {
      if (it->second == request_id) {
        RCLCPP_DEBUG(node_->get_logger(),
                    "Removing mapping: virtual_req %lu -> actual_req %lu",
                    it->first, it->second);
        it = request_id_mapping_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // 调用回调
  if (result_callback_) {
    result_callback_(robot_id, result);
  }
}

bool ActionManager::isSameGoal(
  const geometry_msgs::msg::PoseStamped& goal1,
  const geometry_msgs::msg::PoseStamped& goal2,
  double tolerance_diff
) const {
  // 检查frame_id是否相同
  if (goal1.header.frame_id != goal2.header.frame_id) {
    return false;
  }

  // 检查位置是否相同（使用欧氏距离）
  double dx = goal1.pose.position.x - goal2.pose.position.x;
  double dy = goal1.pose.position.y - goal2.pose.position.y;
  double dz = goal1.pose.position.z - goal2.pose.position.z;
  double distance = std::sqrt(dx*dx + dy*dy + dz*dz);

  if (distance > tolerance_diff) {
    return false;
  }

  // 检查朝向是否相同（四元数点积接近1表示朝向相同）
  const auto& q1 = goal1.pose.orientation;
  const auto& q2 = goal2.pose.orientation;
  double dot_product = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z;

  return std::abs(dot_product) > 0.99;  // 允许小的朝向差异
}

void ActionManager::setGlobalMap(const nav_msgs::msg::OccupancyGrid::SharedPtr& map) {
  if (path_planner_) {
    path_planner_->setGlobalMap(map);
  }

  // 地图更新后检查所有活跃路径是否冲突
  checkPathsForCollision();
}

void ActionManager::setTfBuffer(std::shared_ptr<tf2_ros::Buffer> tf_buffer) {
  tf_buffer_ = tf_buffer;
  RCLCPP_INFO(node_->get_logger(), "TF buffer set in ActionManager");
}

void ActionManager::updateTFCache(
  int robot_id,
  const geometry_msgs::msg::TransformStamped& robot_to_global,
  const geometry_msgs::msg::TransformStamped& global_to_robot,
  const geometry_msgs::msg::TransformStamped& robot_base_to_global
) {
  std::lock_guard<std::mutex> lock(tf_cache_mutex_);
  tf_cache_[robot_id].robot_to_global = robot_to_global;
  tf_cache_[robot_id].global_to_robot = global_to_robot;
  tf_cache_[robot_id].robot_base_to_global = robot_base_to_global;
  tf_cache_[robot_id].valid = true;
}

void ActionManager::setRobotLocalMap(int robot_id, const nav_msgs::msg::OccupancyGrid::SharedPtr& map) {
  std::lock_guard<std::mutex> lock(local_maps_mutex_);
  robot_local_maps_[robot_id] = map;
  RCLCPP_DEBUG(node_->get_logger(),
               "Robot %d local map updated: %dx%d, resolution=%.3f",
               robot_id, map->info.width, map->info.height, map->info.resolution);
}

void ActionManager::handlePathPlanningRequest(
  int robot_id,
  uint64_t request_id,
  uint64_t timestamp,
  const std::vector<uint8_t>& payload
) {
  RCLCPP_INFO(node_->get_logger(),
              "🗺️ Received path planning request from robot %d [request_id=%lu]",
              robot_id, request_id);

  // 反序列化请求（包含start和goal点）
  // 请求格式：start_x(double) + start_y(double) + goal_x(double) + goal_y(double) + waypoint_distance(double)
  // 注意：这些坐标是 robot-local 坐标系中的坐标
  if (payload.size() != sizeof(double) * 5) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Invalid path planning request payload size: %zu", payload.size());

    // 发送空路径响应
    std::vector<uint8_t> response;
    tcp_server_->sendMessage(robot_id, MessageType::PATH_PLANNING_RESPONSE, request_id, response);
    return;
  }

  const double* data = reinterpret_cast<const double*>(payload.data());
  geometry_msgs::msg::Point start_local, goal_local;
  start_local.x = data[0];
  start_local.y = data[1];
  start_local.z = 0.0;
  goal_local.x = data[2];
  goal_local.y = data[3];
  goal_local.z = 0.0;
  double waypoint_distance = data[4];

  RCLCPP_INFO(node_->get_logger(),
              "Received path planning request (robot-local coords): start(%.2f, %.2f) -> goal(%.2f, %.2f)",
              start_local.x, start_local.y, goal_local.x, goal_local.y);

  // 将 robot-local 坐标转换为 global_map 坐标
  geometry_msgs::msg::Point start_global, goal_global;

  if (!transformPointToGlobal(robot_id, start_local, start_global)) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Failed to transform start point from robot_%d/map to global_map", robot_id);

    // 发送失败响应
    std::vector<uint8_t> response;
    response.push_back(0);  // success = false
    uint32_t num_waypoints = 0;
    response.insert(response.end(),
                    reinterpret_cast<uint8_t*>(&num_waypoints),
                    reinterpret_cast<uint8_t*>(&num_waypoints) + sizeof(uint32_t));
    tcp_server_->sendMessage(robot_id, MessageType::PATH_PLANNING_RESPONSE, request_id, response);
    return;
  }

  if (!transformPointToGlobal(robot_id, goal_local, goal_global)) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Failed to transform goal point from robot_%d/map to global_map", robot_id);

    // 发送失败响应
    std::vector<uint8_t> response;
    response.push_back(0);  // success = false
    uint32_t num_waypoints = 0;
    response.insert(response.end(),
                    reinterpret_cast<uint8_t*>(&num_waypoints),
                    reinterpret_cast<uint8_t*>(&num_waypoints) + sizeof(uint32_t));
    tcp_server_->sendMessage(robot_id, MessageType::PATH_PLANNING_RESPONSE, request_id, response);
    return;
  }

  RCLCPP_INFO(node_->get_logger(),
              "Transformed to global_map coords: start(%.2f, %.2f) -> goal(%.2f, %.2f)",
              start_global.x, start_global.y, goal_global.x, goal_global.y);

  // 获取机器人的局部地图
  nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
  {
    std::lock_guard<std::mutex> lock(local_maps_mutex_);
    auto it = robot_local_maps_.find(robot_id);
    if (it != robot_local_maps_.end()) {
      local_map = it->second;
    }
  }

  // 使用新架构：规划完整A*路径，只发送第一个中继点
  PathPlanner::PathPlanResult plan_result = path_planner_->planPathWithFullResult(
    start_global, goal_global, local_map, tf_buffer_
  );

  // 序列化路径响应
  // 新架构响应格式：success(uint8_t) + num_waypoints(uint32_t) + waypoints[](x, y, yaw)
  // 只发送一个中继点，后续中继点在机器人到达时再发送
  std::vector<uint8_t> response;

  if (!plan_result.success || plan_result.full_astar_path.empty()) {
    // 规划失败
    response.push_back(0);  // success = false
    uint32_t num_waypoints = 0;
    response.insert(response.end(),
                    reinterpret_cast<uint8_t*>(&num_waypoints),
                    reinterpret_cast<uint8_t*>(&num_waypoints) + sizeof(uint32_t));

    RCLCPP_WARN(node_->get_logger(),
                "Path planning failed for robot %d: %s | local: start(%.2f,%.2f)->goal(%.2f,%.2f) | global: start(%.2f,%.2f)->goal(%.2f,%.2f)",
                robot_id, path_planner_->getLastError().c_str(),
                start_local.x, start_local.y, goal_local.x, goal_local.y,
                start_global.x, start_global.y, goal_global.x, goal_global.y);
  } else {
    // 规划成功，只发送第一个中继点
    geometry_msgs::msg::Point first_waypoint_local;
    bool transform_success = transformPointToLocal(
      robot_id, plan_result.first_waypoint.pose.position, first_waypoint_local
    );

    if (!transform_success) {
      // 转换失败
      response.push_back(0);  // success = false
      uint32_t num_waypoints = 0;
      response.insert(response.end(),
                      reinterpret_cast<uint8_t*>(&num_waypoints),
                      reinterpret_cast<uint8_t*>(&num_waypoints) + sizeof(uint32_t));
      RCLCPP_WARN(node_->get_logger(),
                  "Path planning succeeded but first waypoint transformation failed");
    } else {
      response.push_back(1);  // success = true
      uint32_t num_waypoints = 1;  // 只发送一个中继点
      response.insert(response.end(),
                      reinterpret_cast<uint8_t*>(&num_waypoints),
                      reinterpret_cast<uint8_t*>(&num_waypoints) + sizeof(uint32_t));

      // 添加第一个中继点（x, y, yaw）- 使用 robot-local 坐标
      double x = first_waypoint_local.x;
      double y = first_waypoint_local.y;

      // 计算 yaw 角（指向下一个路径点）
      double yaw = 0.0;
      size_t wp_idx = plan_result.first_waypoint_path_index;
      if (wp_idx + 1 < plan_result.full_astar_path.size()) {
        const auto& next_pt = plan_result.full_astar_path[wp_idx + 1];
        const auto& curr_pt = plan_result.full_astar_path[wp_idx];
        yaw = std::atan2(next_pt.y - curr_pt.y, next_pt.x - curr_pt.x);
      }

      response.insert(response.end(),
                      reinterpret_cast<uint8_t*>(&x),
                      reinterpret_cast<uint8_t*>(&x) + sizeof(double));
      response.insert(response.end(),
                      reinterpret_cast<uint8_t*>(&y),
                      reinterpret_cast<uint8_t*>(&y) + sizeof(double));
      response.insert(response.end(),
                      reinterpret_cast<uint8_t*>(&yaw),
                      reinterpret_cast<uint8_t*>(&yaw) + sizeof(double));

      RCLCPP_INFO(node_->get_logger(),
                  "✅ Path planning succeeded: full path %zu points, first waypoint at (%.2f, %.2f) [idx=%zu]",
                  plan_result.full_astar_path.size(),
                  plan_result.first_waypoint.pose.position.x,
                  plan_result.first_waypoint.pose.position.y,
                  plan_result.first_waypoint_path_index);

      // 保存活跃路径（包含完整A*路径）
      {
        std::lock_guard<std::mutex> lock(active_paths_mutex_);
        ActivePath active_path;
        active_path.full_astar_path = plan_result.full_astar_path;
        active_path.selected_waypoints.push_back(plan_result.first_waypoint);
        active_path.final_goal_global = goal_global;
        active_path.current_path_index = plan_result.first_waypoint_path_index;
        active_path.current_waypoint_index = 0;
        active_path.request_id = request_id;
        active_paths_[robot_id] = active_path;

        RCLCPP_INFO(node_->get_logger(),
                    "📍 Saved active path for robot %d: %zu A* points, target waypoint at idx %zu",
                    robot_id, plan_result.full_astar_path.size(), plan_result.first_waypoint_path_index);
      }

      // 发布完整路径可视化（A*路径 + 当前中继点）
      publishFullPathVisualization(robot_id, plan_result.full_astar_path,
                                    {plan_result.first_waypoint}, 0);
    }
  }

  // 发送路径响应
  tcp_server_->sendMessage(robot_id, MessageType::PATH_PLANNING_RESPONSE, request_id, response);
}

bool ActionManager::transformPointToGlobal(
  int robot_id,
  const geometry_msgs::msg::Point& point_local,
  geometry_msgs::msg::Point& point_global
) {
  // 优先使用TF缓存
  {
    std::lock_guard<std::mutex> lock(tf_cache_mutex_);
    if (tf_cache_[robot_id].valid) {
      geometry_msgs::msg::PointStamped ps_local, ps_global;
      ps_local.header.frame_id = "robot_" + std::to_string(robot_id) + "/map";
      ps_local.header.stamp = rclcpp::Time(0);
      ps_local.point = point_local;

      try {
        tf2::doTransform(ps_local, ps_global, tf_cache_[robot_id].robot_to_global);
        point_global = ps_global.point;

        RCLCPP_DEBUG(node_->get_logger(),
                     "Transform (cached) robot_%d/map(%.2f, %.2f) -> global_map(%.2f, %.2f)",
                     robot_id, point_local.x, point_local.y,
                     point_global.x, point_global.y);
        return true;
      } catch (const std::exception& ex) {
        RCLCPP_WARN(node_->get_logger(), "Failed to apply cached transform: %s", ex.what());
        // 缓存失效，继续到回退逻辑
      }
    }
  }

  // 回退到直接查询TF
  if (!tf_buffer_) {
    RCLCPP_ERROR(node_->get_logger(), "TF buffer not available for coordinate transformation");
    return false;
  }

  std::string source_frame = "robot_" + std::to_string(robot_id) + "/map";
  std::string target_frame = "global_map";

  try {
    geometry_msgs::msg::PointStamped ps_local, ps_global;
    ps_local.header.frame_id = source_frame;
    ps_local.header.stamp = rclcpp::Time(0);
    ps_local.point = point_local;

    ps_global = tf_buffer_->transform(ps_local, target_frame, tf2::durationFromSec(0.02));
    point_global = ps_global.point;

    RCLCPP_DEBUG(node_->get_logger(),
                 "Transform %s(%.2f, %.2f) -> %s(%.2f, %.2f)",
                 source_frame.c_str(), point_local.x, point_local.y,
                 target_frame.c_str(), point_global.x, point_global.y);

    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Failed to transform from '%s' to '%s': %s",
                 source_frame.c_str(), target_frame.c_str(), ex.what());
    return false;
  }
}

bool ActionManager::transformPointToLocal(
  int robot_id,
  const geometry_msgs::msg::Point& point_global,
  geometry_msgs::msg::Point& point_local
) {
  // 优先使用TF缓存
  {
    std::lock_guard<std::mutex> lock(tf_cache_mutex_);
    if (tf_cache_[robot_id].valid) {
      geometry_msgs::msg::PointStamped ps_global, ps_local;
      ps_global.header.frame_id = "global_map";
      ps_global.header.stamp = rclcpp::Time(0);
      ps_global.point = point_global;

      try {
        tf2::doTransform(ps_global, ps_local, tf_cache_[robot_id].global_to_robot);
        point_local = ps_local.point;

        RCLCPP_DEBUG(node_->get_logger(),
                     "Transform (cached) global_map(%.2f, %.2f) -> robot_%d/map(%.2f, %.2f)",
                     point_global.x, point_global.y,
                     robot_id, point_local.x, point_local.y);
        return true;
      } catch (const std::exception& ex) {
        RCLCPP_WARN(node_->get_logger(), "Failed to apply cached transform: %s", ex.what());
        // 缓存失效，继续到回退逻辑
      }
    }
  }

  // 回退到直接查询TF
  if (!tf_buffer_) {
    RCLCPP_ERROR(node_->get_logger(), "TF buffer not available for coordinate transformation");
    return false;
  }

  std::string source_frame = "global_map";
  std::string target_frame = "robot_" + std::to_string(robot_id) + "/map";

  try {
    geometry_msgs::msg::PointStamped ps_global, ps_local;
    ps_global.header.frame_id = source_frame;
    ps_global.header.stamp = rclcpp::Time(0);
    ps_global.point = point_global;

    ps_local = tf_buffer_->transform(ps_global, target_frame, tf2::durationFromSec(0.02));
    point_local = ps_local.point;

    RCLCPP_DEBUG(node_->get_logger(),
                 "Transform %s(%.2f, %.2f) -> %s(%.2f, %.2f)",
                 source_frame.c_str(), point_global.x, point_global.y,
                 target_frame.c_str(), point_local.x, point_local.y);

    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Failed to transform from '%s' to '%s': %s",
                 source_frame.c_str(), target_frame.c_str(), ex.what());
    return false;
  }
}

void ActionManager::setPathVisualizationPublishers(
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub,
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub
) {
  path_vis_pub_ = path_pub;
  path_marker_pub_ = marker_pub;
  RCLCPP_INFO(node_->get_logger(), "Path visualization publishers set");
}

void ActionManager::publishPathVisualization(
  int robot_id,
  const std::vector<geometry_msgs::msg::PoseStamped>& path
) {
  RCLCPP_INFO(node_->get_logger(),
              "📊 publishPathVisualization called for robot %d with %zu waypoints, "
              "path_vis_pub_=%s, path_marker_pub_=%s",
              robot_id, path.size(),
              path_vis_pub_ ? "valid" : "null",
              path_marker_pub_ ? "valid" : "null");

  if (!path_vis_pub_ && !path_marker_pub_) {
    RCLCPP_WARN(node_->get_logger(), "Path visualization publishers not set!");
    return;
  }

  if (path.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Empty path, skipping visualization");
    return;
  }

  auto stamp = node_->get_clock()->now();

  // 发布 Path 消息
  if (path_vis_pub_) {
    nav_msgs::msg::Path path_msg;
    path_msg.header.stamp = stamp;
    path_msg.header.frame_id = "global_map";

    for (const auto& pose : path) {
      geometry_msgs::msg::PoseStamped ps;
      ps.header = path_msg.header;
      ps.pose = pose.pose;
      path_msg.poses.push_back(ps);
    }

    path_vis_pub_->publish(path_msg);
  }

  // 发布 MarkerArray 可视化
  if (path_marker_pub_) {
    visualization_msgs::msg::MarkerArray marker_array;

    // 颜色：根据机器人ID生成不同颜色（HSV转RGB）
    float h = (robot_id - 1) / 5.0f;  // 假设最多5个机器人
    float r, g, b;
    int hi = static_cast<int>(h * 6.0f);
    float f = h * 6.0f - hi;
    float q = 1.0f - f;
    switch (hi % 6) {
      case 0: r = 1.0f; g = f; b = 0.0f; break;
      case 1: r = q; g = 1.0f; b = 0.0f; break;
      case 2: r = 0.0f; g = 1.0f; b = f; break;
      case 3: r = 0.0f; g = q; b = 1.0f; break;
      case 4: r = f; g = 0.0f; b = 1.0f; break;
      case 5: r = 1.0f; g = 0.0f; b = q; break;
      default: r = 1.0f; g = 1.0f; b = 1.0f; break;
    }

    // Marker 1: 路径线（LINE_STRIP）
    visualization_msgs::msg::Marker line_marker;
    line_marker.header.stamp = stamp;
    line_marker.header.frame_id = "global_map";
    line_marker.ns = "planned_path_line";
    line_marker.id = robot_id;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.scale.x = 0.1;  // 线宽
    line_marker.color.r = r;
    line_marker.color.g = g;
    line_marker.color.b = b;
    line_marker.color.a = 0.8;
    line_marker.lifetime = rclcpp::Duration(30, 0);  // 30秒后消失

    for (const auto& pose : path) {
      line_marker.points.push_back(pose.pose.position);
    }
    marker_array.markers.push_back(line_marker);

    // Marker 2: 路径点（SPHERE_LIST）
    visualization_msgs::msg::Marker points_marker;
    points_marker.header.stamp = stamp;
    points_marker.header.frame_id = "global_map";
    points_marker.ns = "planned_path_points";
    points_marker.id = robot_id;
    points_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    points_marker.action = visualization_msgs::msg::Marker::ADD;
    points_marker.scale.x = 0.2;
    points_marker.scale.y = 0.2;
    points_marker.scale.z = 0.2;
    points_marker.color.r = r;
    points_marker.color.g = g;
    points_marker.color.b = b;
    points_marker.color.a = 1.0;
    points_marker.lifetime = rclcpp::Duration(30, 0);

    for (const auto& pose : path) {
      points_marker.points.push_back(pose.pose.position);
    }
    marker_array.markers.push_back(points_marker);

    // Marker 3: 路径点编号（TEXT_VIEW_FACING）
    for (size_t i = 0; i < path.size(); ++i) {
      visualization_msgs::msg::Marker text_marker;
      text_marker.header.stamp = stamp;
      text_marker.header.frame_id = "global_map";
      text_marker.ns = "planned_path_labels";
      text_marker.id = robot_id * 100 + static_cast<int>(i);  // 确保唯一ID
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose.position = path[i].pose.position;
      text_marker.pose.position.z += 0.3;  // 文本在球体上方
      text_marker.scale.z = 0.15;  // 文本高度
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 1.0;
      text_marker.color.a = 1.0;
      text_marker.text = "R" + std::to_string(robot_id) + "-WP" + std::to_string(i);
      text_marker.lifetime = rclcpp::Duration(30, 0);

      marker_array.markers.push_back(text_marker);
    }

    path_marker_pub_->publish(marker_array);

    RCLCPP_INFO(node_->get_logger(),
                "📍 Published path visualization for robot %d with %zu waypoints",
                robot_id, path.size());
  }
}

void ActionManager::checkPathsForCollision() {
  std::lock_guard<std::mutex> lock(active_paths_mutex_);

  if (active_paths_.empty()) {
    return;
  }

  RCLCPP_DEBUG(node_->get_logger(),
               "🔍 Checking %zu active paths for collision with updated global map",
               active_paths_.size());

  for (auto& [robot_id, active_path] : active_paths_) {
    // 只检查从当前中继点开始的接下来20个路径点，而不是全部
    // 这样可以大幅减少CPU占用，同时保证近期路径的安全性
    size_t check_end = std::min(
        active_path.current_path_index + 20,
        active_path.full_astar_path.size()
    );
    
    bool has_collision = false;
    for (size_t i = active_path.current_path_index; i < check_end; ++i) {
      if (!path_planner_->isPointReachable(active_path.full_astar_path[i])) {
        has_collision = true;
        RCLCPP_WARN(node_->get_logger(),
                    "⚠️ Robot %d: A* path collision at index %zu (%.2f, %.2f)",
                    robot_id, i,
                    active_path.full_astar_path[i].x, active_path.full_astar_path[i].y);
        break;
      }
    }

    if (has_collision) {
      RCLCPP_WARN(node_->get_logger(),
                  "🛑 Robot %d: Path collision detected! Canceling current navigation...",
                  robot_id);

      // 1. 先取消机器人当前的导航
      uint64_t current_request_id = active_path.request_id;
      if (current_request_id > 0) {
        std::vector<uint8_t> empty_payload;
        tcp_server_->sendMessage(robot_id, MessageType::ACTION_CANCEL, current_request_id, empty_payload);
        RCLCPP_INFO(node_->get_logger(),
                    "📤 Sent cancel command to robot %d [request_id=%lu]",
                    robot_id, current_request_id);
      }

      // 2. 获取机器人当前实际位置作为重新规划的起点
      geometry_msgs::msg::Point robot_current_pos;
      bool got_position = getRobotCurrentPosition(robot_id, robot_current_pos);

      geometry_msgs::msg::Point start;
      if (got_position) {
        // 使用机器人的实际位置
        start = robot_current_pos;
        RCLCPP_INFO(node_->get_logger(),
                    "🤖 Robot %d actual position: (%.2f, %.2f)",
                    robot_id, start.x, start.y);
      } else {
        // 如果无法获取实际位置，回退到使用上一个中继点位置
        RCLCPP_WARN(node_->get_logger(),
                    "⚠️ Robot %d: Cannot get actual position, using last waypoint as fallback",
                    robot_id);
        if (active_path.current_path_index < active_path.full_astar_path.size()) {
          start = active_path.full_astar_path[active_path.current_path_index];
        } else if (!active_path.full_astar_path.empty()) {
          start = active_path.full_astar_path.front();
        } else {
          continue;
        }
      }

      // 3. 触发重新规划（从机器人当前位置到最终目标）
      triggerReplan(robot_id, start, active_path.final_goal_global);
    }
  }
}

int ActionManager::checkPathCollision(
  const std::vector<geometry_msgs::msg::PoseStamped>& path,
  size_t from_index
) {
  if (!path_planner_) {
    return -1;
  }

  // 检查路径上的每个点是否可达
  for (size_t i = from_index; i < path.size(); ++i) {
    const auto& pose = path[i];
    if (!path_planner_->isPointReachable(pose.pose.position)) {
      return static_cast<int>(i);
    }

    // 如果不是最后一个点，检查到下一个点的线段
    if (i + 1 < path.size()) {
      const auto& next_pose = path[i + 1];
      // 简单的线段检查：检查中点是否可达
      geometry_msgs::msg::Point midpoint;
      midpoint.x = (pose.pose.position.x + next_pose.pose.position.x) / 2.0;
      midpoint.y = (pose.pose.position.y + next_pose.pose.position.y) / 2.0;
      midpoint.z = 0.0;

      if (!path_planner_->isPointReachable(midpoint)) {
        return static_cast<int>(i);
      }
    }
  }

  return -1;  // 没有冲突
}

void ActionManager::triggerReplan(
  int robot_id,
  const geometry_msgs::msg::Point& start,
  const geometry_msgs::msg::Point& goal
) {
  RCLCPP_INFO(node_->get_logger(),
              "🔄 Re-planning path for robot %d: (%.2f, %.2f) -> (%.2f, %.2f)",
              robot_id, start.x, start.y, goal.x, goal.y);

  // 获取机器人的局部地图
  nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
  {
    std::lock_guard<std::mutex> lock(local_maps_mutex_);
    auto it = robot_local_maps_.find(robot_id);
    if (it != robot_local_maps_.end()) {
      local_map = it->second;
    }
  }

  // 使用新架构重新规划路径
  PathPlanner::PathPlanResult plan_result = path_planner_->planPathWithFullResult(
    start, goal, local_map, tf_buffer_
  );

  if (!plan_result.success || plan_result.full_astar_path.empty()) {
    RCLCPP_WARN(node_->get_logger(),
                "❌ Re-plan failed for robot %d: %s",
                robot_id, path_planner_->getLastError().c_str());
    // 清除活跃路径
    active_paths_.erase(robot_id);
    return;
  }

  // 更新活跃路径
  {
    // 注意: active_paths_mutex_ 已在调用者中持有
    ActivePath new_active_path;
    new_active_path.full_astar_path = plan_result.full_astar_path;
    new_active_path.selected_waypoints.push_back(plan_result.first_waypoint);
    new_active_path.final_goal_global = goal;
    new_active_path.current_path_index = plan_result.first_waypoint_path_index;
    new_active_path.current_waypoint_index = 0;
    // 保持原来的request_id
    auto it = active_paths_.find(robot_id);
    if (it != active_paths_.end()) {
      new_active_path.request_id = it->second.request_id;
    }
    active_paths_[robot_id] = new_active_path;
  }

  // 发布完整路径可视化
  publishFullPathVisualization(robot_id, plan_result.full_astar_path,
                                {plan_result.first_waypoint}, 0);

  // 只发送第一个中继点给机器人
  geometry_msgs::msg::Point first_waypoint_local;
  if (!transformPointToLocal(robot_id, plan_result.first_waypoint.pose.position, first_waypoint_local)) {
    RCLCPP_WARN(node_->get_logger(),
                "Failed to transform first waypoint for replan");
    return;
  }

  // 发送新路径给机器人（只有一个中继点）
  std::vector<uint8_t> response;
  response.push_back(1);  // success = true
  uint32_t num_waypoints = 1;  // 只发送一个中继点
  response.insert(response.end(),
                  reinterpret_cast<uint8_t*>(&num_waypoints),
                  reinterpret_cast<uint8_t*>(&num_waypoints) + sizeof(uint32_t));

  double x = first_waypoint_local.x;
  double y = first_waypoint_local.y;
  double yaw = 0.0;
  size_t wp_idx = plan_result.first_waypoint_path_index;
  if (wp_idx + 1 < plan_result.full_astar_path.size()) {
    const auto& next_pt = plan_result.full_astar_path[wp_idx + 1];
    const auto& curr_pt = plan_result.full_astar_path[wp_idx];
    yaw = std::atan2(next_pt.y - curr_pt.y, next_pt.x - curr_pt.x);
  }

  response.insert(response.end(),
                  reinterpret_cast<uint8_t*>(&x),
                  reinterpret_cast<uint8_t*>(&x) + sizeof(double));
  response.insert(response.end(),
                  reinterpret_cast<uint8_t*>(&y),
                  reinterpret_cast<uint8_t*>(&y) + sizeof(double));
  response.insert(response.end(),
                  reinterpret_cast<uint8_t*>(&yaw),
                  reinterpret_cast<uint8_t*>(&yaw) + sizeof(double));

  // 发送重新规划的路径（使用特殊的request_id=0表示重新规划）
  tcp_server_->sendMessage(robot_id, MessageType::PATH_PLANNING_RESPONSE, 0, response);

  RCLCPP_INFO(node_->get_logger(),
              "✅ Re-plan succeeded for robot %d: new A* path %zu points, first waypoint at (%.2f, %.2f)",
              robot_id, plan_result.full_astar_path.size(),
              plan_result.first_waypoint.pose.position.x,
              plan_result.first_waypoint.pose.position.y);
}

void ActionManager::publishFullPathVisualization(
  int robot_id,
  const std::vector<geometry_msgs::msg::Point>& full_path,
  const std::vector<geometry_msgs::msg::PoseStamped>& waypoints,
  size_t current_waypoint_idx
) {
  if (!path_marker_pub_) {
    return;
  }

  if (full_path.empty()) {
    return;
  }

  auto stamp = node_->get_clock()->now();
  visualization_msgs::msg::MarkerArray marker_array;

  // 颜色：根据机器人ID生成不同颜色
  float h = (robot_id - 1) / 5.0f;
  float r, g, b;
  int hi = static_cast<int>(h * 6.0f);
  float f = h * 6.0f - hi;
  float q = 1.0f - f;
  switch (hi % 6) {
    case 0: r = 1.0f; g = f; b = 0.0f; break;
    case 1: r = q; g = 1.0f; b = 0.0f; break;
    case 2: r = 0.0f; g = 1.0f; b = f; break;
    case 3: r = 0.0f; g = q; b = 1.0f; break;
    case 4: r = f; g = 0.0f; b = 1.0f; break;
    case 5: r = 1.0f; g = 0.0f; b = q; break;
    default: r = 1.0f; g = 1.0f; b = 1.0f; break;
  }

  // Marker 1: 完整A*路径（LINE_STRIP）- 细线
  visualization_msgs::msg::Marker astar_line_marker;
  astar_line_marker.header.stamp = stamp;
  astar_line_marker.header.frame_id = "global_map";
  astar_line_marker.ns = "full_astar_path";
  astar_line_marker.id = robot_id;
  astar_line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  astar_line_marker.action = visualization_msgs::msg::Marker::ADD;
  astar_line_marker.scale.x = 0.03;  // 细线
  astar_line_marker.color.r = r * 0.5;  // 较淡的颜色
  astar_line_marker.color.g = g * 0.5;
  astar_line_marker.color.b = b * 0.5;
  astar_line_marker.color.a = 0.6;
  astar_line_marker.lifetime = rclcpp::Duration(60, 0);  // 60秒后消失

  for (const auto& pt : full_path) {
    astar_line_marker.points.push_back(pt);
  }
  marker_array.markers.push_back(astar_line_marker);

  // Marker 2: 中继点（SPHERE_LIST）- 大球
  if (!waypoints.empty()) {
    visualization_msgs::msg::Marker waypoint_marker;
    waypoint_marker.header.stamp = stamp;
    waypoint_marker.header.frame_id = "global_map";
    waypoint_marker.ns = "selected_waypoints";
    waypoint_marker.id = robot_id;
    waypoint_marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    waypoint_marker.action = visualization_msgs::msg::Marker::ADD;
    waypoint_marker.scale.x = 0.4;
    waypoint_marker.scale.y = 0.4;
    waypoint_marker.scale.z = 0.4;
    waypoint_marker.color.r = r;
    waypoint_marker.color.g = g;
    waypoint_marker.color.b = b;
    waypoint_marker.color.a = 1.0;
    waypoint_marker.lifetime = rclcpp::Duration(60, 0);

    for (const auto& wp : waypoints) {
      waypoint_marker.points.push_back(wp.pose.position);
    }
    marker_array.markers.push_back(waypoint_marker);

    // Marker 3: 当前中继点高亮（SPHERE）- 更大的球
    if (current_waypoint_idx < waypoints.size()) {
      visualization_msgs::msg::Marker current_wp_marker;
      current_wp_marker.header.stamp = stamp;
      current_wp_marker.header.frame_id = "global_map";
      current_wp_marker.ns = "current_waypoint";
      current_wp_marker.id = robot_id;
      current_wp_marker.type = visualization_msgs::msg::Marker::SPHERE;
      current_wp_marker.action = visualization_msgs::msg::Marker::ADD;
      current_wp_marker.pose = waypoints[current_waypoint_idx].pose;
      current_wp_marker.scale.x = 0.6;
      current_wp_marker.scale.y = 0.6;
      current_wp_marker.scale.z = 0.6;
      current_wp_marker.color.r = 1.0;  // 白色高亮
      current_wp_marker.color.g = 1.0;
      current_wp_marker.color.b = 1.0;
      current_wp_marker.color.a = 0.8;
      current_wp_marker.lifetime = rclcpp::Duration(60, 0);
      marker_array.markers.push_back(current_wp_marker);
    }

    // Marker 4: 中继点标签
    for (size_t i = 0; i < waypoints.size(); ++i) {
      visualization_msgs::msg::Marker text_marker;
      text_marker.header.stamp = stamp;
      text_marker.header.frame_id = "global_map";
      text_marker.ns = "waypoint_labels";
      text_marker.id = robot_id * 100 + static_cast<int>(i);
      text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
      text_marker.action = visualization_msgs::msg::Marker::ADD;
      text_marker.pose.position = waypoints[i].pose.position;
      text_marker.pose.position.z += 0.5;
      text_marker.scale.z = 0.2;
      text_marker.color.r = 1.0;
      text_marker.color.g = 1.0;
      text_marker.color.b = 1.0;
      text_marker.color.a = 1.0;
      text_marker.text = "R" + std::to_string(robot_id) + "-WP" + std::to_string(i);
      text_marker.lifetime = rclcpp::Duration(60, 0);
      marker_array.markers.push_back(text_marker);
    }
  }

  // Marker 5: 最终目标点（ARROW）
  if (!full_path.empty()) {
    visualization_msgs::msg::Marker goal_marker;
    goal_marker.header.stamp = stamp;
    goal_marker.header.frame_id = "global_map";
    goal_marker.ns = "final_goal";
    goal_marker.id = robot_id;
    goal_marker.type = visualization_msgs::msg::Marker::ARROW;
    goal_marker.action = visualization_msgs::msg::Marker::ADD;
    goal_marker.pose.position = full_path.back();
    goal_marker.pose.position.z = 0.5;  // 悬浮
    goal_marker.pose.orientation.x = -0.707;  // 指向下方
    goal_marker.pose.orientation.w = 0.707;
    goal_marker.scale.x = 0.5;  // 箭头长度
    goal_marker.scale.y = 0.15;  // 箭身宽度
    goal_marker.scale.z = 0.15;  // 箭头宽度
    goal_marker.color.r = r;
    goal_marker.color.g = g;
    goal_marker.color.b = b;
    goal_marker.color.a = 1.0;
    goal_marker.lifetime = rclcpp::Duration(60, 0);
    marker_array.markers.push_back(goal_marker);
  }

  path_marker_pub_->publish(marker_array);

  RCLCPP_DEBUG(node_->get_logger(),
               "📍 Published full path visualization for robot %d: %zu A* points, %zu waypoints",
               robot_id, full_path.size(), waypoints.size());
}

void ActionManager::handleWaypointReached(int robot_id, const geometry_msgs::msg::Point& robot_position) {
  std::lock_guard<std::mutex> lock(active_paths_mutex_);

  auto it = active_paths_.find(robot_id);
  if (it == active_paths_.end()) {
    RCLCPP_DEBUG(node_->get_logger(),
                 "Robot %d: No active path found for waypoint reached", robot_id);
    return;
  }

  ActivePath& active_path = it->second;

  // 检查是否到达最终目标
  double dist_to_goal = std::sqrt(
    std::pow(robot_position.x - active_path.final_goal_global.x, 2) +
    std::pow(robot_position.y - active_path.final_goal_global.y, 2)
  );

  if (dist_to_goal < 0.5) {  // 到达最终目标的容差
    RCLCPP_INFO(node_->get_logger(),
                "🎯 Robot %d reached final goal at (%.2f, %.2f)",
                robot_id, active_path.final_goal_global.x, active_path.final_goal_global.y);
    // 清除活跃路径
    active_paths_.erase(robot_id);
    return;
  }

  // 检查是否到达当前中继点
  if (active_path.current_path_index < active_path.full_astar_path.size()) {
    const auto& current_target = active_path.full_astar_path[active_path.current_path_index];
    double dist_to_waypoint = std::sqrt(
      std::pow(robot_position.x - current_target.x, 2) +
      std::pow(robot_position.y - current_target.y, 2)
    );

    if (dist_to_waypoint < 1.0) {  // 到达中继点的容差
      RCLCPP_INFO(node_->get_logger(),
                  "📍 Robot %d reached waypoint at path index %zu (%.2f, %.2f)",
                  robot_id, active_path.current_path_index,
                  current_target.x, current_target.y);

      // 选择并发送下一个中继点
      selectAndSendNextWaypoint(robot_id, robot_position);
    }
  }
}

bool ActionManager::selectAndSendNextWaypoint(int robot_id, const geometry_msgs::msg::Point& robot_position) {
  // 注意: 调用者已持有 active_paths_mutex_
  auto it = active_paths_.find(robot_id);
  if (it == active_paths_.end()) {
    return false;
  }

  ActivePath& active_path = it->second;

  // 获取机器人的局部地图
  nav_msgs::msg::OccupancyGrid::SharedPtr local_map;
  {
    std::lock_guard<std::mutex> lock(local_maps_mutex_);
    auto map_it = robot_local_maps_.find(robot_id);
    if (map_it != robot_local_maps_.end()) {
      local_map = map_it->second;
    }
  }

  if (!local_map) {
    RCLCPP_WARN(node_->get_logger(),
                "Robot %d: No local map available for waypoint selection", robot_id);
    return false;
  }

  // 使用PathPlanner选择下一个中继点
  auto [waypoint, path_idx] = path_planner_->selectNextWaypointFromPath(
    active_path.full_astar_path,
    active_path.current_path_index,
    local_map,
    tf_buffer_
  );

  if (path_idx <= active_path.current_path_index) {
    // 无法选择更远的中继点，可能需要等待局部地图更新
    RCLCPP_WARN(node_->get_logger(),
                "Robot %d: Cannot select next waypoint (current_idx=%zu, selected_idx=%zu)",
                robot_id, active_path.current_path_index, path_idx);
    return false;
  }

  // 更新活跃路径
  active_path.current_path_index = path_idx;
  active_path.selected_waypoints.push_back(waypoint);
  active_path.current_waypoint_index++;

  RCLCPP_INFO(node_->get_logger(),
              "📍 Robot %d: Selected next waypoint at path index %zu (%.2f, %.2f)",
              robot_id, path_idx, waypoint.pose.position.x, waypoint.pose.position.y);

  // 转换为机器人本地坐标
  geometry_msgs::msg::Point waypoint_local;
  if (!transformPointToLocal(robot_id, waypoint.pose.position, waypoint_local)) {
    RCLCPP_ERROR(node_->get_logger(),
                 "Robot %d: Failed to transform next waypoint to local", robot_id);
    return false;
  }

  // 发送新的中继点给机器人
  std::vector<uint8_t> response;
  response.push_back(1);  // success = true
  uint32_t num_waypoints = 1;
  response.insert(response.end(),
                  reinterpret_cast<uint8_t*>(&num_waypoints),
                  reinterpret_cast<uint8_t*>(&num_waypoints) + sizeof(uint32_t));

  double x = waypoint_local.x;
  double y = waypoint_local.y;
  double yaw = 0.0;
  if (path_idx + 1 < active_path.full_astar_path.size()) {
    const auto& next_pt = active_path.full_astar_path[path_idx + 1];
    yaw = std::atan2(next_pt.y - waypoint.pose.position.y,
                     next_pt.x - waypoint.pose.position.x);
  }

  response.insert(response.end(),
                  reinterpret_cast<uint8_t*>(&x),
                  reinterpret_cast<uint8_t*>(&x) + sizeof(double));
  response.insert(response.end(),
                  reinterpret_cast<uint8_t*>(&y),
                  reinterpret_cast<uint8_t*>(&y) + sizeof(double));
  response.insert(response.end(),
                  reinterpret_cast<uint8_t*>(&yaw),
                  reinterpret_cast<uint8_t*>(&yaw) + sizeof(double));

  // 发送（使用特殊的request_id=0表示这是中继点更新）
  tcp_server_->sendMessage(robot_id, MessageType::PATH_PLANNING_RESPONSE, 0, response);

  // 更新可视化
  publishFullPathVisualization(robot_id, active_path.full_astar_path,
                                active_path.selected_waypoints, active_path.current_waypoint_index);

  RCLCPP_INFO(node_->get_logger(),
              "✅ Robot %d: Sent next waypoint (local: %.2f, %.2f)",
              robot_id, waypoint_local.x, waypoint_local.y);

  return true;
}

bool ActionManager::getRobotCurrentPosition(int robot_id, geometry_msgs::msg::Point& position) {
  // 优先使用TF缓存
  {
    std::lock_guard<std::mutex> lock(tf_cache_mutex_);
    if (tf_cache_[robot_id].valid) {
      const auto& transform = tf_cache_[robot_id].robot_base_to_global;
      position.x = transform.transform.translation.x;
      position.y = transform.transform.translation.y;
      position.z = transform.transform.translation.z;

      RCLCPP_DEBUG(node_->get_logger(),
                   "Robot %d position (cached) in global_map: (%.2f, %.2f)",
                   robot_id, position.x, position.y);
      return true;
    }
  }

  // 回退到直接查询TF
  if (!tf_buffer_) {
    RCLCPP_ERROR(node_->get_logger(), "TF buffer not available for getting robot position");
    return false;
  }

  std::string robot_frame = "robot_" + std::to_string(robot_id) + "/base_link";
  std::string target_frame = "global_map";

  try {
    geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
      target_frame, robot_frame, rclcpp::Time(0), tf2::durationFromSec(0.02)
    );

    position.x = transform.transform.translation.x;
    position.y = transform.transform.translation.y;
    position.z = transform.transform.translation.z;

    RCLCPP_DEBUG(node_->get_logger(),
                 "Robot %d position in global_map: (%.2f, %.2f)",
                 robot_id, position.x, position.y);

    return true;
  } catch (const tf2::TransformException& ex) {
    RCLCPP_WARN(node_->get_logger(),
                "Failed to get robot %d position: %s", robot_id, ex.what());
    return false;
  }
}

}  // namespace reliable_communication
