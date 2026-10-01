/**
 * @file map_merger_simple.cpp
 * @brief Simplified map merger implementation for real robots
 */

#include "real_cure_cpp/map_merger_simple.hpp"
#include <tf2/utils.h>

namespace real_cure_cpp
{

MapMergerSimple::MapMergerSimple(const rclcpp::NodeOptions & options)
: Node("map_merger_simple",
       rclcpp::NodeOptions(options).automatically_declare_parameters_from_overrides(true)),
  reset_in_progress_(false)
{
  // Get parameters
  robot_count_ = this->get_parameter_or("robot_count", 3);
  robot_namespace_ = this->get_parameter_or("robot_namespace", std::string("robot_"));
  global_frame_ = this->get_parameter_or("global_frame", std::string("global_map"));
  update_rate_ = this->get_parameter_or("update_rate", 2.0);

  // Initialize global map (全部设为未知)
  global_map_data_.resize(GLOBAL_MAP_SIZE_X * GLOBAL_MAP_SIZE_Y, -1);

  // Create publishers
  global_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/global_map", 10);
  reset_ack_pub_ = this->create_publisher<cure_explore_interfaces::msg::ResetAck>(
    "/cure_explore/internal_reset_ack", 10);

  // QoS for transient local topics
  rclcpp::QoS qos_transient(10);
  qos_transient.durability(rclcpp::DurabilityPolicy::TransientLocal);
  qos_transient.reliability(rclcpp::ReliabilityPolicy::Reliable);

  // Create subscriptions for each robot
  for (int i = 0; i < robot_count_; i++) {
    int robot_id = i + 1;
    std::string map_topic = "/" + robot_namespace_ + std::to_string(robot_id) + "/map";
    std::string map_tf_topic = "/" + robot_namespace_ + std::to_string(robot_id) + "/map_synced_tf";

    // Map subscription (收到新地图立即触发融合)
    auto map_sub = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      map_topic, 10,
      [this, robot_id](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        this->mapCallback(msg, robot_id);
      });
    map_subs_.push_back(map_sub);

    // Map->base_link transform subscription
    auto map_tf_sub = this->create_subscription<geometry_msgs::msg::TransformStamped>(
      map_tf_topic, qos_transient,
      [this, robot_id](const geometry_msgs::msg::TransformStamped::SharedPtr msg) {
        this->mapTfCallback(msg, robot_id);
      });
    map_tf_subs_.push_back(map_tf_sub);
  }

  // Reset subscription
  reset_sub_ = this->create_subscription<std_msgs::msg::Empty>(
    "/cure_explore/reset", 10,
    std::bind(&MapMergerSimple::resetCallback, this, std::placeholders::_1));

  // TF2 setup
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Timer for publishing global map
  double period = 1.0 / update_rate_;
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(period),
    std::bind(&MapMergerSimple::publishGlobalMap, this));

  RCLCPP_INFO(this->get_logger(),
              "Map Merger Simple (C++) initialized for %d robots", robot_count_);
  RCLCPP_INFO(this->get_logger(),
              "  Update rate: %.1f Hz (optimized for communication delays)", update_rate_);
}

void MapMergerSimple::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg, int robot_id)
{
  // Check reset lockout
  if (reset_in_progress_) {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - reset_lockout_time_).count();
    if (elapsed < RESET_LOCKOUT_DURATION) {
      return;  // 静默忽略
    } else {
      reset_in_progress_ = false;
      RCLCPP_INFO(this->get_logger(), "重置锁定期结束");
    }
  }

  // 只保留最新地图，直接覆盖旧数据
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_map_[robot_id] = msg;
  }

  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
              "收到 robot_%d 地图 (%dx%d)，立即融合",
              robot_id, msg->info.width, msg->info.height);

  // 立即触发融合
  mergeSingleRobotMap(robot_id);
}

void MapMergerSimple::mapTfCallback(
  const geometry_msgs::msg::TransformStamped::SharedPtr msg, int robot_id)
{
  if (reset_in_progress_) {
    return;
  }

  std::lock_guard<std::mutex> lock(data_mutex_);
  latest_tf_[robot_id] = msg;
}

void MapMergerSimple::resetCallback(const std_msgs::msg::Empty::SharedPtr /*msg*/)
{
  RCLCPP_INFO(this->get_logger(), "收到重置信号");

  reset_in_progress_ = true;
  reset_lockout_time_ = std::chrono::steady_clock::now();

  // Clear caches
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    latest_map_.clear();
    latest_tf_.clear();
  }

  // Reset global map
  std::fill(global_map_data_.begin(), global_map_data_.end(), -1);

  // Publish empty map
  publishGlobalMap();

  // Send acknowledgment
  cure_explore_interfaces::msg::ResetAck ack;
  ack.source = "map_merger_simple";
  ack.success = true;
  ack.message = "简化版地图融合器重置完成";
  reset_ack_pub_->publish(ack);

  RCLCPP_INFO(this->get_logger(), "地图融合器重置完成");
}

void MapMergerSimple::mergeSingleRobotMap(int robot_id)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if (latest_map_.find(robot_id) == latest_map_.end()) {
    return;
  }

  auto local_map = latest_map_[robot_id];

  try {
    // 1. Get TF: global_frame -> robot_{id}/map
    std::string local_map_frame = robot_namespace_ + std::to_string(robot_id) + "/map";

    if (!tf_buffer_->canTransform(global_frame_, local_map_frame, tf2::TimePointZero)) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                  "TF %s -> %s 不可用，跳过融合",
                  global_frame_.c_str(), local_map_frame.c_str());
      return;
    }

    auto transform_map = tf_buffer_->lookupTransform(
      global_frame_, local_map_frame, tf2::TimePointZero);

    // 2/3. Extract transform parameters.
    // BUGFIX: rotate each local map by the FIXED global<-robot_map orientation taken from
    // transform_map (the same transform the translation comes from). The previous code took yaw
    // from map->base_link -- the robot's LIVE heading -- so every time a robot turned, its whole
    // contributed map spun around the global origin, smearing the merged map and producing bogus
    // frontier goals (robots driven into walls). Translation and rotation must come from the SAME
    // global<-robot_map transform.
    double tx = transform_map.transform.translation.x;
    double ty = transform_map.transform.translation.y;
    double map_qz = transform_map.transform.rotation.z;
    double map_qw = transform_map.transform.rotation.w;
    double yaw = 2.0 * std::atan2(map_qz, map_qw);  // planar yaw of global<-robot_map
    double cos_yaw = std::cos(yaw);
    double sin_yaw = std::sin(yaw);

    // 4. Get local map parameters
    int local_width = local_map->info.width;
    int local_height = local_map->info.height;
    double local_res = local_map->info.resolution;
    double local_origin_x = local_map->info.origin.position.x;
    double local_origin_y = local_map->info.origin.position.y;

    // 5. 直接融合（简化版：直接覆盖，不做加权）
    int merged_count = 0;
    int known_cells = 0;

    for (int y_local = 0; y_local < local_height; y_local++) {
      for (int x_local = 0; x_local < local_width; x_local++) {
        int8_t local_val = local_map->data[y_local * local_width + x_local];

        // 只融合已知单元格
        if (local_val == -1) {
          continue;
        }
        known_cells++;

        // Convert local grid to world coordinates (cell center)
        double px_local = x_local * local_res + local_res / 2.0 + local_origin_x;
        double py_local = y_local * local_res + local_res / 2.0 + local_origin_y;

        // Transform to global coordinates
        double wx = tx + px_local * cos_yaw - py_local * sin_yaw;
        double wy = ty + px_local * sin_yaw + py_local * cos_yaw;

        // Convert to global grid coordinates
        int gx = static_cast<int>(std::round((wx - GLOBAL_MAP_ORIGIN_X) / GLOBAL_MAP_RESOLUTION));
        int gy = static_cast<int>(std::round((wy - GLOBAL_MAP_ORIGIN_Y) / GLOBAL_MAP_RESOLUTION));

        // Bounds check
        if (gx < 0 || gx >= GLOBAL_MAP_SIZE_X || gy < 0 || gy >= GLOBAL_MAP_SIZE_Y) {
          continue;
        }

        merged_count++;
        int idx = gy * GLOBAL_MAP_SIZE_X + gx;

        // 简化融合策略：
        // 1. 未知区域：直接设置
        // 2. 已知区域：障碍物优先（防止障碍物消失）
        if (global_map_data_[idx] == -1) {
          // 未知 -> 直接设置
          global_map_data_[idx] = local_val;
        } else {
          // 已知 -> 障碍物优先策略
          if (local_val >= 50) {
            // 新观测是障碍物：直接覆盖
            global_map_data_[idx] = 100;
          } else if (local_val < 50 && global_map_data_[idx] < 50) {
            // 都是自由空间：保持自由
            global_map_data_[idx] = 0;
          }
          // 如果旧数据是障碍物但新观测是自由，需要多次观测才清除（保守策略）
          // 这里简化为：保留障碍物（避免动态障碍物导致的误判）
        }
      }
    }

    if (known_cells > 0) {
      RCLCPP_DEBUG(this->get_logger(),
                   "Robot_%d: 融合完成，已知=%d, 合并=%d",
                   robot_id, known_cells, merged_count);
    }

    // 后处理：填充孤立未知区域
    if (merged_count > 0) {
      fillIsolatedUnknowns();
    }

  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Robot_%d TF异常: %s", robot_id, ex.what());
  } catch (const std::exception & e) {
    RCLCPP_ERROR(this->get_logger(),
                 "Robot_%d 地图融合错误: %s", robot_id, e.what());
  }
}

void MapMergerSimple::fillIsolatedUnknowns()
{
  // 更激进的未知区域填充策略
  // 如果一个未知单元格的8邻域中有>=4个自由空间且没有障碍物，则填充为自由空间
  
  std::vector<std::pair<int, int>> cells_to_fill;

  for (int y = 1; y < GLOBAL_MAP_SIZE_Y - 1; y++) {
    for (int x = 1; x < GLOBAL_MAP_SIZE_X - 1; x++) {
      int idx = y * GLOBAL_MAP_SIZE_X + x;

      if (global_map_data_[idx] != -1) {
        continue;
      }

      // Count neighbors
      int free_count = 0;
      int obstacle_count = 0;
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) {
            continue;
          }
          int nidx = (y + dy) * GLOBAL_MAP_SIZE_X + (x + dx);
          if (global_map_data_[nidx] == 0) {
            free_count++;
          } else if (global_map_data_[nidx] == 100) {
            obstacle_count++;
          }
        }
      }

      // 更激进的条件：>=4个自由邻居且无障碍物
      if (free_count >= 4 && obstacle_count == 0) {
        cells_to_fill.push_back({x, y});
      }
    }
  }

  // Fill cells
  for (const auto & [x, y] : cells_to_fill) {
    int idx = y * GLOBAL_MAP_SIZE_X + x;
    global_map_data_[idx] = 0;
  }

  if (!cells_to_fill.empty()) {
    RCLCPP_DEBUG(this->get_logger(),
                 "填充了 %zu 个孤立未知区域", cells_to_fill.size());
  }
}

void MapMergerSimple::publishGlobalMap()
{
  nav_msgs::msg::OccupancyGrid global_map_msg;
  global_map_msg.header.stamp = this->get_clock()->now();
  global_map_msg.header.frame_id = global_frame_;
  global_map_msg.info.resolution = GLOBAL_MAP_RESOLUTION;
  global_map_msg.info.width = GLOBAL_MAP_SIZE_X;
  global_map_msg.info.height = GLOBAL_MAP_SIZE_Y;
  global_map_msg.info.origin.position.x = GLOBAL_MAP_ORIGIN_X;
  global_map_msg.info.origin.position.y = GLOBAL_MAP_ORIGIN_Y;
  global_map_msg.info.origin.orientation.w = 1.0;
  global_map_msg.data = global_map_data_;

  global_map_pub_->publish(global_map_msg);
}

}  // namespace real_cure_cpp
