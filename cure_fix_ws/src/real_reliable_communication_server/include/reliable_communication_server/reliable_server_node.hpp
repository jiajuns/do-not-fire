#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/int8.hpp>
#include <cure_explore_interfaces/msg/reset_command.hpp>
#include <cure_explore_interfaces/msg/reset_ack.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <memory>
#include <map>
#include <atomic>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <functional>

#include "reliable_communication_server/tcp_server.hpp"
#include "reliable_communication_server/action_manager.hpp"
#include "reliable_communication_server/qos_monitor.hpp"
#include "reliable_communication_interfaces/action/navigate_to_goal.hpp"

namespace reliable_communication {

using NavigateToGoal = reliable_communication_interfaces::action::NavigateToGoal;
using GoalHandleNavigate = rclcpp_action::ServerGoalHandle<NavigateToGoal>;

/**
 * @brief 可靠通信服务器节点
 *
 * 整合TCP服务器、Action管理和QoS监控
 */
class ReliableServerNode : public rclcpp::Node {
public:
  explicit ReliableServerNode(const rclcpp::NodeOptions& options);
  ~ReliableServerNode();

private:
  // 连接状态回调
  void onConnectionChange(int robot_id, bool connected);

  // TCP反馈/结果回调 - 转发到ROS Action
  void onGoalFeedback(int robot_id, const NavigateToGoal::Feedback& feedback);
  void onGoalResult(int robot_id, const NavigateToGoal::Result& result);

  // 话题订阅回调（用于测试）
  void onGoalPose(int robot_id, const geometry_msgs::msg::PoseStamped::SharedPtr msg);

  // ROS Action Server 回调
  rclcpp_action::GoalResponse handleGoal(
    const rclcpp_action::GoalUUID& uuid,
    std::shared_ptr<const NavigateToGoal::Goal> goal,
    int robot_id);
  rclcpp_action::CancelResponse handleCancel(
    const std::shared_ptr<GoalHandleNavigate> goal_handle,
    int robot_id);
  void handleAccepted(
    const std::shared_ptr<GoalHandleNavigate> goal_handle,
    int robot_id);

  // 系统重置回调
  void onResetCommand(const cure_explore_interfaces::msg::ResetCommand::SharedPtr msg);
  void handleShutdown();
  void handleStartup();

  // 可视化相关
  void publishGoalVisualization(int robot_id, const geometry_msgs::msg::PoseStamped& goal);

  // 全局地图回调
  void onGlobalMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

  // 机器人局部地图回调
  void onRobotLocalMap(int robot_id, const nav_msgs::msg::OccupancyGrid::SharedPtr msg);

  // 坐标转换：global_map -> robot_local (robot_X/map)
  // 返回 true 表示转换成功，result 中包含转换后的目标点
  bool transformGoalToRobotLocal(
    int robot_id,
    const geometry_msgs::msg::PoseStamped& goal_in_global,
    geometry_msgs::msg::PoseStamped& goal_out_local);

  int robot_count_;
  int tcp_port_;

  std::unique_ptr<TCPServer> tcp_server_;
  std::unique_ptr<ActionManager> action_manager_;
  std::unique_ptr<QoSMonitor> qos_monitor_;

  // TF2 用于坐标转换
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // TF变换缓存（避免频繁查询TF）
  struct TFCache {
    geometry_msgs::msg::TransformStamped transform;
    rclcpp::Time last_update;
    bool valid = false;
  };
  std::map<int, TFCache> robot_to_global_tf_;  // robot_X/map -> global_map
  std::map<int, TFCache> global_to_robot_tf_;  // global_map -> robot_X/map
  std::map<int, TFCache> robot_base_to_global_tf_;  // robot_X/base_link -> global_map
  std::mutex tf_cache_mutex_;
  rclcpp::TimerBase::SharedPtr tf_update_timer_;  // 定时更新TF缓存
  void updateTFCache();  // 更新TF缓存

  // 目标点订阅器 (用于测试/兼容)
  std::map<int, rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> goal_subscribers_;

  // 全局地图订阅器
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr global_map_sub_;

  // 各机器人局部地图订阅器: robot_id -> subscriber
  std::map<int, rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr> local_map_subscribers_;

  // ROS Action Servers (每个机器人一个)
  std::map<int, rclcpp_action::Server<NavigateToGoal>::SharedPtr> action_servers_;

  // 活跃的Goal Handles (用于转发TCP反馈/结果)
  std::map<int, std::shared_ptr<GoalHandleNavigate>> active_goal_handles_;
  std::map<int, uint64_t> active_request_ids_;  // robot_id -> TCP request_id 映射
  std::mutex goal_handles_mutex_;

  // 系统重置
  rclcpp::Subscription<cure_explore_interfaces::msg::ResetCommand>::SharedPtr reset_sub_;
  rclcpp::Publisher<cure_explore_interfaces::msg::ResetAck>::SharedPtr reset_ack_pub_;

  // 目标点可视化
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr goal_vis_pub_;
  std::map<int, geometry_msgs::msg::PoseStamped> current_goals_;  // 缓存当前目标
  rclcpp::TimerBase::SharedPtr vis_timer_;  // 定时发布可视化

  // 路径规划可视化
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_vis_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr path_marker_pub_;

  // 目标状态发布器（用于通知RL立即cancel等情况）
  // 状态值: -1=立即cancel(不可达), 0=失败, 1=成功
  std::map<int, rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr> goal_status_pubs_;

  // 每个机器人一个工作线程（避免频繁创建销毁线程）
  struct RobotWorker {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<std::function<void()>> tasks;
    bool running = true;
    std::atomic<bool> task_executing{false};
    std::chrono::steady_clock::time_point task_start_time;
  };
  std::map<int, std::unique_ptr<RobotWorker>> robot_workers_;
  std::atomic<bool> shutdown_flag_{false};
  void initRobotWorkers();
  void robotWorkerLoop(int robot_id);
  void submitTask(int robot_id, std::function<void()> task);
  void checkWorkerHealth();  // 健康检查
  rclcpp::TimerBase::SharedPtr health_check_timer_;  // 健康检查定时器
};

}  // namespace reliable_communication
