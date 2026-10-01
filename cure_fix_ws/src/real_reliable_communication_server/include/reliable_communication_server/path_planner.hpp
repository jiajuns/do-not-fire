#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/buffer.h>
#include <vector>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <memory>

namespace reliable_communication {

/**
 * @brief 全局路径规划器
 * 
 * 在全局地图上进行A*路径规划，用于检查目标点是否可达
 * 并生成中间路径点
 */
class PathPlanner {
public:
  PathPlanner(rclcpp::Node* node);

  /**
   * @brief 设置全局地图
   * @param map 全局地图
   */
  void setGlobalMap(const nav_msgs::msg::OccupancyGrid::SharedPtr& map);

  /**
   * @brief 检查目标点是否在地图内
   * @param goal 目标点
   * @return true 如果在地图内
   */
  bool isPointInMap(const geometry_msgs::msg::Point& point) const;

  /**
   * @brief 检查目标点是否可达（不在障碍物上）
   * @param goal 目标点
   * @return true 如果可达
   */
  bool isPointReachable(const geometry_msgs::msg::Point& point) const;

  /**
   * @brief 在全局地图上规划路径
   * @param start 起点
   * @param goal 终点
   * @param waypoint_distance 路径点间隔距离
   * @return 路径点列表，如果规划失败返回空列表
   */
  std::vector<geometry_msgs::msg::PoseStamped> planPath(
    const geometry_msgs::msg::Point& start,
    const geometry_msgs::msg::Point& goal,
    double waypoint_distance = 2.0
  );

  /**
   * @brief 基于机器人局部地图边界选择路径中继点
   * @param start 起点（机器人当前位置）
   * @param goal 终点
   * @param local_map 机器人的局部地图
   * @param tf_buffer TF2 buffer用于坐标转换
   * @return 路径点列表，如果规划失败返回空列表
   * @note 沿着规划路径选择局部地图范围内的最远点作为中继目标
   */
  std::vector<geometry_msgs::msg::PoseStamped> planPathWithLocalMapBoundary(
    const geometry_msgs::msg::Point& start,
    const geometry_msgs::msg::Point& goal,
    const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer
  );

  /**
   * @brief 获取路径规划失败的原因
   * @return 失败原因字符串
   */
  std::string getLastError() const { return last_error_; }

  /**
   * @brief 路径规划结果，包含完整A*路径和首个中继点
   */
  struct PathPlanResult {
    bool success;
    std::vector<geometry_msgs::msg::Point> full_astar_path;  // 完整A*路径（世界坐标）
    geometry_msgs::msg::PoseStamped first_waypoint;  // 首个中继点
    size_t first_waypoint_path_index;  // 首个中继点在full_astar_path中的索引
  };

  /**
   * @brief 规划路径并返回完整A*路径和首个中继点
   * @param start 起点
   * @param goal 终点
   * @param local_map 机器人局部地图
   * @param tf_buffer TF buffer
   * @return 路径规划结果
   */
  PathPlanResult planPathWithFullResult(
    const geometry_msgs::msg::Point& start,
    const geometry_msgs::msg::Point& goal,
    const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer
  );

  /**
   * @brief 从完整A*路径中选择下一个中继点
   * @param full_path 完整A*路径
   * @param current_index 当前在路径中的位置
   * @param local_map 机器人局部地图
   * @param tf_buffer TF buffer
   * @return (waypoint, path_index) 中继点和对应的路径索引，如果失败返回空
   */
  std::pair<geometry_msgs::msg::PoseStamped, size_t> selectNextWaypointFromPath(
    const std::vector<geometry_msgs::msg::Point>& full_path,
    size_t current_index,
    const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer
  );

private:
  struct GridCell {
    int x, y;
    double g, h, f;
    GridCell* parent;

    GridCell(int x_, int y_, double g_, double h_, GridCell* p = nullptr)
      : x(x_), y(y_), g(g_), h(h_), f(g_ + h_), parent(p) {}

    bool operator>(const GridCell& other) const {
      return f > other.f;
    }
  };

  struct CellHash {
    size_t operator()(const std::pair<int, int>& cell) const {
      return std::hash<int>()(cell.first) ^ (std::hash<int>()(cell.second) << 1);
    }
  };

  /**
   * @brief 将世界坐标转换为地图坐标
   */
  bool worldToMap(const geometry_msgs::msg::Point& world_point, int& mx, int& my) const;

  /**
   * @brief 将地图坐标转换为世界坐标
   */
  geometry_msgs::msg::Point mapToWorld(int mx, int my) const;

  /**
   * @brief 检查地图单元是否被占用
   */
  bool isCellOccupied(int mx, int my) const;

  /**
   * @brief 获取栅格的移动代价
   * @note 未知栅格返回高代价（5.0），已知自由空间返回1.0
   */
  double getCellCost(int mx, int my) const;

  /**
   * @brief 计算启发式距离
   */
  double heuristic(int x1, int y1, int x2, int y2) const;

  /**
   * @brief 简化路径，抽取关键路径点
   */
  std::vector<geometry_msgs::msg::PoseStamped> simplifyPath(
    const std::vector<std::pair<int, int>>& grid_path,
    double waypoint_distance
  );

  /**
   * @brief 基于局部地图边界选择路径中继点
   * @param grid_path 栅格路径
   * @param robot_position 机器人位置（世界坐标）
   * @param local_map 局部地图
   * @param tf_buffer TF2 buffer用于坐标转换
   * @return 中继点列表
   */
  std::vector<geometry_msgs::msg::PoseStamped> selectWaypointsBasedOnLocalMap(
    const std::vector<std::pair<int, int>>& grid_path,
    const geometry_msgs::msg::Point& robot_position,
    const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer
  );

  /**
   * @brief 检查世界坐标点是否在局部地图范围内
   * @param point 世界坐标点（global_map坐标系）
   * @param local_map 局部地图
   * @param tf_buffer TF2 buffer用于坐标转换
   * @return true 如果在局部地图范围内
   */
  bool isPointInLocalMap(
    const geometry_msgs::msg::Point& point,
    const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer
  ) const;

  /**
   * @brief 检查世界坐标点是否在局部地图的已知区域内（非unknown）
   * @param point 世界坐标点（global_map坐标系）
   * @param local_map 局部地图
   * @param tf_buffer TF2 buffer用于坐标转换
   * @return true 如果在局部地图的已知区域内
   */
  bool isPointInKnownAreaOfLocalMap(
    const geometry_msgs::msg::Point& point,
    const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
    std::shared_ptr<tf2_ros::Buffer> tf_buffer
  ) const;

  /**
   * @brief 在指定位置周围寻找可用的起点
   * @param blocked_point 被阻塞的起点
   * @param search_radius 搜索半径（栅格数）
   * @return 找到的可用起点，如果失败返回原点
   */
  geometry_msgs::msg::Point findNearbyFreeCell(
    const geometry_msgs::msg::Point& blocked_point,
    int search_radius = 10
  ) const;

  /**
   * @brief 设置机器人半径用于障碍物膨胀
   * @param radius 机器人半径（米）
   */
  void setRobotRadius(double radius) { robot_radius_ = radius; }

  /**
   * @brief 获取当前机器人半径
   * @return 机器人半径（米）
   */
  double getRobotRadius() const { return robot_radius_; }

  rclcpp::Node* node_;
  nav_msgs::msg::OccupancyGrid::SharedPtr global_map_;
  std::string last_error_;

  // 地图参数缓存
  bool map_initialized_;
  double resolution_;
  double origin_x_;
  double origin_y_;
  int width_;
  int height_;

  // 机器人半径（用于障碍物膨胀），默认0.3米
  double robot_radius_ = 0.3;

  // A* 搜索的8个方向
  const std::vector<std::pair<int, int>> directions_ = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1},
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
  };
};

}  // namespace reliable_communication
