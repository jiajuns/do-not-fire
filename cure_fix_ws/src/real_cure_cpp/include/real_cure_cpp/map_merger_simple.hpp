/**
 * @file map_merger_simple.hpp
 * @brief Simplified map merger for real robots with communication delays
 *
 * 简化版地图融合节点 - 针对通信延迟优化
 * - 只保留最新的地图数据（不缓存历史）
 * - 放宽可见性检查条件
 * - 更激进的未知区域填充
 * - 移除复杂的后处理步骤
 */

#ifndef REAL_CURE_CPP__MAP_MERGER_SIMPLE_HPP_
#define REAL_CURE_CPP__MAP_MERGER_SIMPLE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/empty.hpp>
#include <cure_explore_interfaces/msg/reset_ack.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>

namespace real_cure_cpp
{

class MapMergerSimple : public rclcpp::Node
{
public:
  explicit MapMergerSimple(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~MapMergerSimple() override = default;

private:
  // Callbacks
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg, int robot_id);
  void mapTfCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg, int robot_id);
  void resetCallback(const std_msgs::msg::Empty::SharedPtr msg);
  void publishGlobalMap();

  // Map fusion (simplified)
  void mergeSingleRobotMap(int robot_id);
  void fillIsolatedUnknowns();  // 更激进的填充

  // Parameters
  int robot_count_;
  std::string robot_namespace_;
  std::string global_frame_;
  double update_rate_;

  // Global map constants (扩大地图范围)
  static constexpr int GLOBAL_MAP_SIZE_X = 1600;    // 800 -> 1600 (2倍)
  static constexpr int GLOBAL_MAP_SIZE_Y = 1600;    // 800 -> 1600 (2倍)
  static constexpr double GLOBAL_MAP_RESOLUTION = 0.05;
  static constexpr double GLOBAL_MAP_ORIGIN_X = -40.0;  // -20.0 -> -40.0 (对应1600格)
  static constexpr double GLOBAL_MAP_ORIGIN_Y = -40.0;  // -20.0 -> -40.0 (对应1600格)

  // Global map data (只保留简单版本)
  std::vector<int8_t> global_map_data_;  // -1=未知, 0=自由, 100=障碍物

  // 只缓存最新的地图和TF（不保留历史）
  std::mutex data_mutex_;
  std::unordered_map<int, nav_msgs::msg::OccupancyGrid::SharedPtr> latest_map_;
  std::unordered_map<int, geometry_msgs::msg::TransformStamped::SharedPtr> latest_tf_;

  // Reset state
  bool reset_in_progress_;
  std::chrono::steady_clock::time_point reset_lockout_time_;
  static constexpr double RESET_LOCKOUT_DURATION = 1.0;  // 缩短锁定时间

  // TF2
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Publishers & Subscribers
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr global_map_pub_;
  rclcpp::Publisher<cure_explore_interfaces::msg::ResetAck>::SharedPtr reset_ack_pub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;

  std::vector<rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr> map_subs_;
  std::vector<rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr> map_tf_subs_;

  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace real_cure_cpp

#endif  // REAL_CURE_CPP__MAP_MERGER_SIMPLE_HPP_
