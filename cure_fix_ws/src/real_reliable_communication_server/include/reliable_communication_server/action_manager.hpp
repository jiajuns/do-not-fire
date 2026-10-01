#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <reliable_communication_interfaces/action/navigate_to_goal.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/buffer.h>
#include <map>
#include <memory>
#include <mutex>

#include "reliable_communication_server/tcp_server.hpp"
#include "reliable_communication_server/path_planner.hpp"

namespace reliable_communication {

using NavigateToGoal = reliable_communication_interfaces::action::NavigateToGoal;
using GoalHandle = rclcpp_action::ServerGoalHandle<NavigateToGoal>;

/**
 * @brief Action管理器
 *
 * 管理NavigateToGoal Action的发送、取消、反馈接收
 * 可扩展支持更多Action类型
 */
class ActionManager {
public:
  ActionManager(rclcpp::Node* node, TCPServer* tcp_server);

  /**
   * @brief 发送目标点给机器人
   * @param robot_id 机器人ID
   * @param goal 目标位姿
   * @param tolerance 到达容差
   * @return 请求ID
   */
  uint64_t sendGoal(
    int robot_id,
    const geometry_msgs::msg::PoseStamped& goal,
    float tolerance
  );

  /**
   * @brief 取消目标
   * @param robot_id 机器人ID
   * @param request_id 请求ID
   * @return 是否成功发送取消请求
   */
  bool cancelGoal(int robot_id, uint64_t request_id);

  /**
   * @brief 设置反馈回调
   * @param callback 反馈回调函数
   */
  void setFeedbackCallback(
    std::function<void(int robot_id, const NavigateToGoal::Feedback&)> callback
  );

  /**
   * @brief 设置结果回调
   * @param callback 结果回调函数
   */
  void setResultCallback(
    std::function<void(int robot_id, const NavigateToGoal::Result&)> callback
  );

  /**
   * @brief 重置Action Manager状态
   */
  void reset();

  /**
   * @brief 发送目标点（自动取消旧目标）
   * @param robot_id 机器人ID
   * @param goal 目标位姿
   * @param tolerance 到达容差
   * @return 请求ID
   * @note 会自动检查并取消该机器人的旧目标，避免goal preemption
   */
  uint64_t sendGoalWithAutoCancel(
    int robot_id,
    const geometry_msgs::msg::PoseStamped& goal,
    float tolerance
  );

  /**
   * @brief 发送目标点（支持去重）
   * @param robot_id 机器人ID
   * @param goal 目标位姿
   * @param tolerance 到达容差
   * @return 请求ID
   * @note 如果目标点与当前活跃目标相同，则不发送新目标，而是将旧目标的回调更新到新请求ID
   */
  uint64_t sendGoalWithDeduplication(
    int robot_id,
    const geometry_msgs::msg::PoseStamped& goal,
    float tolerance
  );

  /**
   * @brief 获取机器人当前活跃的目标ID
   * @param robot_id 机器人ID
   * @return 当前目标的request_id，0表示无活跃目标
   */
  uint64_t getActiveGoalId(int robot_id) const;

  /**
   * @brief 检查机器人是否有活跃目标
   * @param robot_id 机器人ID
   * @return 是否有活跃目标
   */
  bool hasActiveGoal(int robot_id) const;

  /**
   * @brief 设置全局地图用于路径规划
   * @param map 全局地图
   */
  void setGlobalMap(const nav_msgs::msg::OccupancyGrid::SharedPtr& map);

  /**
   * @brief 设置TF buffer用于坐标转换
   * @param tf_buffer TF2 buffer
   */
  void setTfBuffer(std::shared_ptr<tf2_ros::Buffer> tf_buffer);

  /**
   * @brief 设置TF缓存（由ReliableServerNode定期更新）
   * @param robot_id 机器人ID
   * @param robot_to_global robot_X/map -> global_map
   * @param global_to_robot global_map -> robot_X/map
   * @param robot_base_to_global robot_X/base_link -> global_map
   */
  void updateTFCache(
    int robot_id,
    const geometry_msgs::msg::TransformStamped& robot_to_global,
    const geometry_msgs::msg::TransformStamped& global_to_robot,
    const geometry_msgs::msg::TransformStamped& robot_base_to_global
  );

  /**
   * @brief 设置机器人的局部地图
   * @param robot_id 机器人ID
   * @param map 局部地图
   */
  void setRobotLocalMap(int robot_id, const nav_msgs::msg::OccupancyGrid::SharedPtr& map);

  /**
   * @brief 检查所有活跃路径是否与更新后的全局地图冲突
   * 如果冲突，触发重新规划
   */
  void checkPathsForCollision();

  /**
   * @brief 处理路径规划请求
   * @param robot_id 机器人ID
   * @param request_id 请求ID
   * @param timestamp 时间戳
   * @param payload 请求负载
   */
  void handlePathPlanningRequest(
    int robot_id,
    uint64_t request_id,
    uint64_t timestamp,
    const std::vector<uint8_t>& payload
  );

  /**
   * @brief 设置路径可视化发布器
   * @param path_pub 路径发布器
   * @param marker_pub Marker发布器
   */
  void setPathVisualizationPublishers(
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub,
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub
  );

private:
  void handleFeedback(
    int robot_id,
    uint64_t request_id,
    uint64_t timestamp,
    const std::vector<uint8_t>& payload
  );

  void handleResult(
    int robot_id,
    uint64_t request_id,
    uint64_t timestamp,
    const std::vector<uint8_t>& payload
  );

  /**
   * @brief 清除机器人的活跃目标记录
   * @param robot_id 机器人ID
   */
  void clearActiveGoal(int robot_id);

  /**
   * @brief 比较两个目标点是否相同
   * @param goal1 目标点1
   * @param goal2 目标点2
   * @param tolerance_diff 容差阈值（米）
   * @return 是否相同
   */
  bool isSameGoal(
    const geometry_msgs::msg::PoseStamped& goal1,
    const geometry_msgs::msg::PoseStamped& goal2,
    double tolerance_diff = 0.01
  ) const;

  /**
   * @brief 将点从 robot-local 坐标系转换到 global_map 坐标系
   * @param robot_id 机器人ID
   * @param point_local robot-local 坐标中的点
   * @param point_global 输出的 global_map 坐标中的点
   * @return 是否转换成功
   */
  bool transformPointToGlobal(
    int robot_id,
    const geometry_msgs::msg::Point& point_local,
    geometry_msgs::msg::Point& point_global
  );

  /**
   * @brief 将点从 global_map 坐标系转换到 robot-local 坐标系
   * @param robot_id 机器人ID
   * @param point_global global_map 坐标中的点
   * @param point_local 输出的 robot-local 坐标中的点
   * @return 是否转换成功
   */
  bool transformPointToLocal(
    int robot_id,
    const geometry_msgs::msg::Point& point_global,
    geometry_msgs::msg::Point& point_local
  );

  rclcpp::Node* node_;
  TCPServer* tcp_server_;

  uint64_t next_request_id_;
  std::mutex request_id_mutex_;

  // 每个机器人的活跃目标: robot_id -> request_id
  std::map<int, uint64_t> active_goals_;
  mutable std::mutex active_goals_mutex_;

  // 存储每个机器人的活跃目标点信息: robot_id -> (goal, tolerance)
  std::map<int, std::pair<geometry_msgs::msg::PoseStamped, float>> active_goal_info_;
  mutable std::mutex active_goal_info_mutex_;

  // 请求ID映射：新request_id -> 实际goal的request_id（用于去重时的回调转发）
  std::map<uint64_t, uint64_t> request_id_mapping_;
  mutable std::mutex request_id_mapping_mutex_;

  std::function<void(int, const NavigateToGoal::Feedback&)> feedback_callback_;
  std::function<void(int, const NavigateToGoal::Result&)> result_callback_;

  // 路径规划器
  std::unique_ptr<PathPlanner> path_planner_;

  // 各机器人的局部地图: robot_id -> local_map
  std::map<int, nav_msgs::msg::OccupancyGrid::SharedPtr> robot_local_maps_;
  mutable std::mutex local_maps_mutex_;

  // TF buffer用于坐标转换
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  // TF缓存（由ReliableServerNode定期推送）
  struct TFCache {
    geometry_msgs::msg::TransformStamped robot_to_global;  // robot_X/map -> global_map
    geometry_msgs::msg::TransformStamped global_to_robot;  // global_map -> robot_X/map
    geometry_msgs::msg::TransformStamped robot_base_to_global;  // robot_X/base_link -> global_map
    bool valid = false;
  };
  std::map<int, TFCache> tf_cache_;  // robot_id -> TFCache
  mutable std::mutex tf_cache_mutex_;

  // 路径可视化发布器
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_vis_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr path_marker_pub_;

  // 当前活跃的规划路径: robot_id -> ActivePath
  struct ActivePath {
    // 完整的A*栅格路径（global_map坐标）
    std::vector<geometry_msgs::msg::Point> full_astar_path;
    // 选择的中继点（global_map坐标）
    std::vector<geometry_msgs::msg::PoseStamped> selected_waypoints;
    // 最终目标点（global_map坐标）
    geometry_msgs::msg::Point final_goal_global;
    // 当前已到达的路径点在full_astar_path中的索引
    size_t current_path_index;
    // 当前正在导航的中继点索引
    size_t current_waypoint_index;
    // 请求ID
    uint64_t request_id;
  };
  std::map<int, ActivePath> active_paths_;
  mutable std::mutex active_paths_mutex_;

  /**
   * @brief 检查单条路径是否与障碍物冲突
   * @param path 路径点列表（global_map坐标）
   * @param from_index 从哪个索引开始检查
   * @return 冲突的路径点索引，如果没有冲突返回-1
   */
  int checkPathCollision(
    const std::vector<geometry_msgs::msg::PoseStamped>& path,
    size_t from_index = 0
  );

  /**
   * @brief 触发路径重新规划
   * @param robot_id 机器人ID
   * @param start 起点（global_map坐标）
   * @param goal 终点（global_map坐标）
   */
  void triggerReplan(
    int robot_id,
    const geometry_msgs::msg::Point& start,
    const geometry_msgs::msg::Point& goal
  );

  /**
   * @brief 发布规划路径的可视化
   * @param robot_id 机器人ID
   * @param path 规划的路径点（global_map坐标系）
   */
  void publishPathVisualization(
    int robot_id,
    const std::vector<geometry_msgs::msg::PoseStamped>& path
  );

  /**
   * @brief 发布完整路径可视化（A*路径 + 中继点）
   * @param robot_id 机器人ID
   * @param full_path 完整A*路径
   * @param waypoints 选择的中继点
   * @param current_waypoint_idx 当前中继点索引
   */
  void publishFullPathVisualization(
    int robot_id,
    const std::vector<geometry_msgs::msg::Point>& full_path,
    const std::vector<geometry_msgs::msg::PoseStamped>& waypoints,
    size_t current_waypoint_idx
  );

  /**
   * @brief 处理机器人到达中继点的反馈
   * @param robot_id 机器人ID
   * @param robot_position 机器人当前位置（global_map坐标）
   */
  void handleWaypointReached(int robot_id, const geometry_msgs::msg::Point& robot_position);

  /**
   * @brief 根据机器人局部地图选择下一个中继点
   * @param robot_id 机器人ID
   * @param robot_position 机器人当前位置
   * @return 是否成功选择并发送了新的中继点
   */
  bool selectAndSendNextWaypoint(int robot_id, const geometry_msgs::msg::Point& robot_position);

  /**
   * @brief 从TF获取机器人当前位置（global_map坐标系）
   * @param robot_id 机器人ID
   * @param position 输出的机器人位置
   * @return 是否成功获取位置
   */
  bool getRobotCurrentPosition(int robot_id, geometry_msgs::msg::Point& position);
};

}  // namespace reliable_communication
