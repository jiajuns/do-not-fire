#include "reliable_communication_server/path_planner.hpp"
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <tuple>

namespace reliable_communication {

PathPlanner::PathPlanner(rclcpp::Node* node)
  : node_(node), map_initialized_(false),
    resolution_(0.0), origin_x_(0.0), origin_y_(0.0),
    width_(0), height_(0) {
  RCLCPP_INFO(node_->get_logger(), "PathPlanner initialized");
}

void PathPlanner::setGlobalMap(const nav_msgs::msg::OccupancyGrid::SharedPtr& map) {
  global_map_ = map;
  resolution_ = map->info.resolution;
  origin_x_ = map->info.origin.position.x;
  origin_y_ = map->info.origin.position.y;
  width_ = map->info.width;
  height_ = map->info.height;
  map_initialized_ = true;
}

bool PathPlanner::isPointInMap(const geometry_msgs::msg::Point& point) const {
  if (!map_initialized_) {
    return false;
  }

  int mx, my;
  if (!worldToMap(point, mx, my)) {
    return false;
  }

  return mx >= 0 && mx < width_ && my >= 0 && my < height_;
}

bool PathPlanner::isPointReachable(const geometry_msgs::msg::Point& point) const {
  if (!map_initialized_) {
    return false;
  }

  int mx, my;
  if (!worldToMap(point, mx, my)) {
    return false;
  }

  if (mx < 0 || mx >= width_ || my < 0 || my >= height_) {
    return false;
  }

  return !isCellOccupied(mx, my);
}

std::vector<geometry_msgs::msg::PoseStamped> PathPlanner::planPath(
  const geometry_msgs::msg::Point& start,
  const geometry_msgs::msg::Point& goal,
  double waypoint_distance
) {
  last_error_.clear();
  std::vector<geometry_msgs::msg::PoseStamped> result;

  if (!map_initialized_) {
    last_error_ = "Global map not initialized - ensure /global_map topic is being published (check map_merger)";
    RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
    return result;
  }

  // 转换为地图坐标
  int start_mx, start_my, goal_mx, goal_my;
  if (!worldToMap(start, start_mx, start_my)) {
    last_error_ = "Start point outside map bounds";
    RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
    return result;
  }

  if (!worldToMap(goal, goal_mx, goal_my)) {
    last_error_ = "Goal point outside map bounds";
    RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
    return result;
  }

  // 检查起点和终点是否可达
  // 注意：起点是机器人当前位置，不应该因为地图噪声而拒绝
  // 只有当起点确实是严重障碍物时才尝试寻找附近的可用栅格
  int start_index = start_my * width_ + start_mx;
  int8_t start_value = (start_index >= 0 && start_index < static_cast<int>(global_map_->data.size()))
                       ? global_map_->data[start_index] : 100;

  // 使用的实际起点（可能被调整）
  geometry_msgs::msg::Point actual_start = start;
  int actual_start_mx = start_mx;
  int actual_start_my = start_my;

  if (start_value > 70) {
    // 起点被阻塞，尝试在周围寻找可用栅格
    RCLCPP_WARN(node_->get_logger(),
                "Start point at (%.2f,%.2f)[%d,%d] is occupied (value=%d), searching nearby...",
                start.x, start.y, start_mx, start_my, static_cast<int>(start_value));

    actual_start = findNearbyFreeCell(start, 15);  // 搜索半径15栅格

    if (!worldToMap(actual_start, actual_start_mx, actual_start_my)) {
      last_error_ = "Failed to find valid start point";
      RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
      return result;
    }

    // 验证新起点
    int new_start_index = actual_start_my * width_ + actual_start_mx;
    int8_t new_start_value = (new_start_index >= 0 && new_start_index < static_cast<int>(global_map_->data.size()))
                             ? global_map_->data[new_start_index] : 100;

    if (new_start_value > 70) {
      last_error_ = "Could not find valid start point near robot position";
      RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
      return result;
    }

    RCLCPP_INFO(node_->get_logger(),
                "Using adjusted start point at (%.2f,%.2f)[%d,%d] (value=%d)",
                actual_start.x, actual_start.y, actual_start_mx, actual_start_my,
                static_cast<int>(new_start_value));
  }
  // 如果起点有轻微占用，发出警告但继续规划
  else if (start_value > 50) {
    RCLCPP_WARN(node_->get_logger(),
                "Start point has elevated occupancy (value=%d) at (%.2f,%.2f)[%d,%d], but proceeding",
                static_cast<int>(start_value), start.x, start.y, start_mx, start_my);
  }

  // 终点检查：允许轻微占用的目标点，但不能是明确障碍物
  int goal_index = goal_my * width_ + goal_mx;
  int8_t goal_value = (goal_index >= 0 && goal_index < static_cast<int>(global_map_->data.size()))
                      ? global_map_->data[goal_index] : 100;
  if (goal_value > 65) {
    last_error_ = "Goal point is occupied (value=" + std::to_string(static_cast<int>(goal_value)) + ")";
    RCLCPP_WARN(node_->get_logger(), "%s at (%.2f,%.2f)[%d,%d]",
                last_error_.c_str(), goal.x, goal.y, goal_mx, goal_my);
    return result;
  }
  // 如果终点有轻微占用，发出警告但继续规划
  if (goal_value > 40) {
    RCLCPP_WARN(node_->get_logger(),
                "Goal point has elevated occupancy (value=%d) at (%.2f,%.2f)[%d,%d], but proceeding",
                static_cast<int>(goal_value), goal.x, goal.y, goal_mx, goal_my);
  }

  RCLCPP_INFO(node_->get_logger(),
              "Planning path from (%.2f, %.2f) [%d, %d] to (%.2f, %.2f) [%d, %d]",
              actual_start.x, actual_start.y, actual_start_mx, actual_start_my,
              goal.x, goal.y, goal_mx, goal_my);

  // A* 路径规划 - 使用高效的数据结构
  // 优先队列存储 (f值, x, y)
  using PQElement = std::tuple<double, int, int>;
  std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> open_list;

  // 使用 unordered_map 存储每个格子的信息，O(1) 查找
  std::unordered_map<int, double> g_score;  // index -> g值
  std::unordered_map<int, int> came_from;   // index -> parent_index
  std::unordered_set<int> closed_set;       // 已访问集合

  auto getIndex = [this](int x, int y) { return y * width_ + x; };

  int start_idx = getIndex(actual_start_mx, actual_start_my);
  int goal_idx = getIndex(goal_mx, goal_my);

  g_score[start_idx] = 0.0;
  open_list.push({heuristic(actual_start_mx, actual_start_my, goal_mx, goal_my), actual_start_mx, actual_start_my});

  int iterations = 0;
  const int max_iterations = 100000;  // 减少最大迭代次数，避免长时间卡住

  while (!open_list.empty() && iterations < max_iterations) {
    iterations++;

    auto [f, cx, cy] = open_list.top();
    open_list.pop();

    int current_idx = getIndex(cx, cy);

    // 检查是否到达目标
    if (cx == goal_mx && cy == goal_my) {
      // 找到目标，重建路径
      std::vector<std::pair<int, int>> grid_path;
      int idx = current_idx;
      while (idx != -1) {
        int py = idx / width_;
        int px = idx % width_;
        grid_path.push_back({px, py});
        auto it = came_from.find(idx);
        idx = (it != came_from.end()) ? it->second : -1;
      }
      std::reverse(grid_path.begin(), grid_path.end());

      RCLCPP_INFO(node_->get_logger(),
                  "Path found with %zu grid cells after %d iterations",
                  grid_path.size(), iterations);

      result = simplifyPath(grid_path, waypoint_distance);
      return result;
    }

    // 跳过已访问的节点
    if (closed_set.count(current_idx)) {
      continue;
    }
    closed_set.insert(current_idx);

    // 扩展邻居
    for (const auto& dir : directions_) {
      int nx = cx + dir.first;
      int ny = cy + dir.second;

      // 检查边界
      if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {
        continue;
      }

      // 检查目标格子是否是障碍物
      if (isCellOccupied(nx, ny)) {
        continue;
      }

      // 对角线移动时，检查两侧格子是否都可通行
      // 防止"挤过"两个相邻障碍物的角落
      if (dir.first != 0 && dir.second != 0) {
        // 对角线移动，检查两个相邻格子
        if (isCellOccupied(cx + dir.first, cy) || isCellOccupied(cx, cy + dir.second)) {
          continue;  // 如果任一侧有障碍物，不允许对角线移动
        }
      }

      int neighbor_idx = getIndex(nx, ny);

      // 跳过已访问
      if (closed_set.count(neighbor_idx)) {
        continue;
      }

      // 计算代价：对角线移动代价为1.414，直线为1.0
      // 乘以栅格本身的代价系数（未知区域、接近障碍物等会有更高代价）
      double base_move_cost = (dir.first != 0 && dir.second != 0) ? 1.414 : 1.0;
      double cell_cost = getCellCost(nx, ny);
      double tentative_g = g_score[current_idx] + base_move_cost * cell_cost;

      // 如果这是新节点或找到更好的路径
      auto g_it = g_score.find(neighbor_idx);
      if (g_it == g_score.end() || tentative_g < g_it->second) {
        came_from[neighbor_idx] = current_idx;
        g_score[neighbor_idx] = tentative_g;
        double f_score = tentative_g + heuristic(nx, ny, goal_mx, goal_my);
        open_list.push({f_score, nx, ny});
      }
    }
  }

  if (iterations >= max_iterations) {
    last_error_ = "Path planning exceeded max iterations (" + std::to_string(max_iterations) + ")";
    RCLCPP_WARN(node_->get_logger(),
                "%s - start[%d,%d] goal[%d,%d]",
                last_error_.c_str(), start_mx, start_my, goal_mx, goal_my);
  } else {
    double direct_dist = std::sqrt(static_cast<double>((goal_mx - start_mx) * (goal_mx - start_mx) +
                                                       (goal_my - start_my) * (goal_my - start_my)));
    last_error_ = "No path found - goal may be blocked by obstacles";
    RCLCPP_WARN(node_->get_logger(),
                "%s - start[%d,%d] goal[%d,%d] iterations=%d closed=%zu dist=%.0f",
                last_error_.c_str(), start_mx, start_my, goal_mx, goal_my,
                iterations, closed_set.size(), direct_dist);
  }
  return result;
}

bool PathPlanner::worldToMap(const geometry_msgs::msg::Point& world_point, int& mx, int& my) const {
  if (!map_initialized_) {
    return false;
  }

  mx = static_cast<int>((world_point.x - origin_x_) / resolution_);
  my = static_cast<int>((world_point.y - origin_y_) / resolution_);

  // Validate that the computed pixel coordinates are within map bounds
  if (mx < 0 || mx >= width_ || my < 0 || my >= height_) {
    RCLCPP_DEBUG(node_->get_logger(),
                 "worldToMap: point (%.2f, %.2f) -> pixel [%d, %d] is outside map bounds [%d x %d]",
                 world_point.x, world_point.y, mx, my, width_, height_);
    return false;
  }

  return true;
}

geometry_msgs::msg::Point PathPlanner::mapToWorld(int mx, int my) const {
  geometry_msgs::msg::Point point;
  point.x = origin_x_ + (mx + 0.5) * resolution_;
  point.y = origin_y_ + (my + 0.5) * resolution_;
  point.z = 0.0;
  return point;
}

bool PathPlanner::isCellOccupied(int mx, int my) const {
  if (!map_initialized_) {
    return true;
  }

  if (mx < 0 || mx >= width_ || my < 0 || my >= height_) {
    return true;
  }

  // 计算机器人半径对应的栅格数（向上取整确保安全）
  int radius_cells = static_cast<int>(std::ceil(robot_radius_ / resolution_));

  // 检查以(mx, my)为中心、radius_cells为半径的圆形区域
  for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
      // 检查是否在圆形范围内
      if (dx * dx + dy * dy > radius_cells * radius_cells) {
        continue;
      }

      int nx = mx + dx;
      int ny = my + dy;

      // 边界检查
      if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {
        continue;  // 超出边界的格子不作为障碍物
      }

      int index = ny * width_ + nx;
      if (index < 0 || index >= static_cast<int>(global_map_->data.size())) {
        continue;
      }

      // OccupancyGrid: -1=unknown, 0=free, 100=occupied
      int8_t value = global_map_->data[index];
      // 严格阻止穿过障碍物：大于50认为是障碍物
      if (value > 50) {
        return true;  // 在机器人半径范围内发现障碍物
      }
    }
  }

  return false;  // 机器人半径范围内没有障碍物
}

double PathPlanner::getCellCost(int mx, int my) const {
  // 获取栅格的额外移动代价倍数
  // 用于引导路径偏好：尽量走已知自由空间，避免（但允许）未知区域
  if (!map_initialized_) {
    return 2.5;  // 地图未初始化，中等代价
  }

  if (mx < 0 || mx >= width_ || my < 0 || my >= height_) {
    return 5.0;  // 超出边界，高代价
  }

  int index = my * width_ + mx;
  if (index < 0 || index >= static_cast<int>(global_map_->data.size())) {
    return 5.0;
  }

  int8_t value = global_map_->data[index];

  // OccupancyGrid: -1=unknown, 0=free, 100=occupied
  // 注意：>50的区域已被isCellOccupied()阻挡，这里不会到达
  if (value == -1) {
    // 未知栅格：2.5倍代价，尽量避免但允许通过
    return 2.5;
  } else if (value <= 10) {
    // 确定的自由空间：基础代价
    return 1.0;
  } else if (value <= 30) {
    // 较自由空间：轻微增加
    return 1.2;
  } else if (value <= 50) {
    // 接近障碍物边缘：明显增加代价
    return 1.8;
  } else {
    // 不应该到这里（>50已被阻挡），但保险起见
    return 3.0;
  }
}

double PathPlanner::heuristic(int x1, int y1, int x2, int y2) const {
  // 使用欧几里得距离
  int dx = x2 - x1;
  int dy = y2 - y1;
  return std::sqrt(dx * dx + dy * dy);
}

std::vector<geometry_msgs::msg::PoseStamped> PathPlanner::simplifyPath(
  const std::vector<std::pair<int, int>>& grid_path,
  double waypoint_distance
) {
  std::vector<geometry_msgs::msg::PoseStamped> simplified_path;

  if (grid_path.empty()) {
    return simplified_path;
  }

  // 转换为世界坐标并按距离间隔抽取点
  std::vector<geometry_msgs::msg::Point> world_path;
  for (const auto& cell : grid_path) {
    world_path.push_back(mapToWorld(cell.first, cell.second));
  }

  // 总是包含第一个点
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = global_map_->header.frame_id;
  pose.pose.position = world_path[0];
  pose.pose.orientation.w = 1.0;
  simplified_path.push_back(pose);

  // 按距离间隔抽取中间点
  double accumulated_distance = 0.0;
  for (size_t i = 1; i < world_path.size(); ++i) {
    double dx = world_path[i].x - world_path[i-1].x;
    double dy = world_path[i].y - world_path[i-1].y;
    double segment_dist = std::sqrt(dx * dx + dy * dy);
    accumulated_distance += segment_dist;

    if (accumulated_distance >= waypoint_distance) {
      pose.pose.position = world_path[i];
      
      // 计算朝向（指向下一个点）
      if (i + 1 < world_path.size()) {
        double angle = std::atan2(
          world_path[i+1].y - world_path[i].y,
          world_path[i+1].x - world_path[i].x
        );
        pose.pose.orientation.z = std::sin(angle / 2.0);
        pose.pose.orientation.w = std::cos(angle / 2.0);
      }
      
      simplified_path.push_back(pose);
      accumulated_distance = 0.0;
    }
  }

  // 总是包含最后一个点（目标点）
  if (simplified_path.back().pose.position.x != world_path.back().x ||
      simplified_path.back().pose.position.y != world_path.back().y) {
    pose.pose.position = world_path.back();
    pose.pose.orientation.w = 1.0;
    simplified_path.push_back(pose);
  }

  RCLCPP_INFO(node_->get_logger(),
              "Simplified path from %zu to %zu waypoints (distance: %.2f m)",
              grid_path.size(), simplified_path.size(), waypoint_distance);

  return simplified_path;
}

std::vector<geometry_msgs::msg::PoseStamped> PathPlanner::planPathWithLocalMapBoundary(
  const geometry_msgs::msg::Point& start,
  const geometry_msgs::msg::Point& goal,
  const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer
) {
  last_error_.clear();
  std::vector<geometry_msgs::msg::PoseStamped> result;

  if (!map_initialized_) {
    last_error_ = "Global map not initialized - ensure /global_map topic is being published (check map_merger)";
    RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
    return result;
  }

  // 先在全局地图上规划完整路径
  int start_mx, start_my, goal_mx, goal_my;
  if (!worldToMap(start, start_mx, start_my) || !worldToMap(goal, goal_mx, goal_my)) {
    last_error_ = "Start or goal point outside global map bounds";
    RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
    return result;
  }

  // 检查起点和终点是否可达（放宽条件）
  int start_index = start_my * width_ + start_mx;
  int goal_index = goal_my * width_ + goal_mx;
  int8_t start_value = (start_index >= 0 && start_index < static_cast<int>(global_map_->data.size()))
                       ? global_map_->data[start_index] : 100;
  int8_t goal_value = (goal_index >= 0 && goal_index < static_cast<int>(global_map_->data.size()))
                      ? global_map_->data[goal_index] : 100;

  // 使用的实际起点（可能被调整）
  geometry_msgs::msg::Point actual_start = start;
  int actual_start_mx = start_mx;
  int actual_start_my = start_my;

  // 起点检查：如果在障碍物上，尝试寻找附近可用栅格
  if (start_value > 70) {
    RCLCPP_WARN(node_->get_logger(),
                "Start point at (%.2f,%.2f)[%d,%d] is occupied (value=%d), searching nearby...",
                start.x, start.y, start_mx, start_my, static_cast<int>(start_value));

    actual_start = findNearbyFreeCell(start, 15);

    if (!worldToMap(actual_start, actual_start_mx, actual_start_my)) {
      last_error_ = "Failed to find valid start point";
      RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
      return result;
    }

    int new_start_index = actual_start_my * width_ + actual_start_mx;
    int8_t new_start_value = (new_start_index >= 0 && new_start_index < static_cast<int>(global_map_->data.size()))
                             ? global_map_->data[new_start_index] : 100;

    if (new_start_value > 70) {
      last_error_ = "Could not find valid start point near robot position";
      RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
      return result;
    }

    RCLCPP_INFO(node_->get_logger(),
                "Using adjusted start point at (%.2f,%.2f)[%d,%d] (value=%d)",
                actual_start.x, actual_start.y, actual_start_mx, actual_start_my,
                static_cast<int>(new_start_value));
  }
  else if (start_value > 50) {
    RCLCPP_WARN(node_->get_logger(),
                "Start point has elevated occupancy (value=%d) at (%.2f,%.2f)[%d,%d], but proceeding",
                static_cast<int>(start_value), start.x, start.y, start_mx, start_my);
  }

  // 终点检查：允许轻微占用但不能是障碍物
  if (goal_value > 65) {
    last_error_ = "Goal point is occupied (value=" + std::to_string(static_cast<int>(goal_value)) + ")";
    RCLCPP_WARN(node_->get_logger(),
                "%s at (%.2f,%.2f)[%d,%d]",
                last_error_.c_str(), goal.x, goal.y, goal_mx, goal_my);
    return result;
  }
  if (goal_value > 40) {
    RCLCPP_WARN(node_->get_logger(),
                "Goal point has elevated occupancy (value=%d) at (%.2f,%.2f)[%d,%d], but proceeding",
                static_cast<int>(goal_value), goal.x, goal.y, goal_mx, goal_my);
  }

  RCLCPP_INFO(node_->get_logger(),
              "Planning path with local map boundary from (%.2f, %.2f) [%d, %d] to (%.2f, %.2f) [%d, %d]",
              actual_start.x, actual_start.y, actual_start_mx, actual_start_my,
              goal.x, goal.y, goal_mx, goal_my);

  // A* 路径规划 - 使用高效的数据结构
  using PQElement = std::tuple<double, int, int>;
  std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> open_list;

  std::unordered_map<int, double> g_score;
  std::unordered_map<int, int> came_from;
  std::unordered_set<int> closed_set;

  auto getIndex = [this](int x, int y) { return y * width_ + x; };

  int start_idx = getIndex(actual_start_mx, actual_start_my);
  int goal_idx = getIndex(goal_mx, goal_my);

  g_score[start_idx] = 0.0;
  open_list.push({heuristic(actual_start_mx, actual_start_my, goal_mx, goal_my), actual_start_mx, actual_start_my});

  int iterations = 0;
  const int max_iterations = 100000;

  while (!open_list.empty() && iterations < max_iterations) {
    iterations++;

    auto [f, cx, cy] = open_list.top();
    open_list.pop();

    int current_idx = getIndex(cx, cy);

    if (cx == goal_mx && cy == goal_my) {
      // 找到目标，重建路径
      std::vector<std::pair<int, int>> grid_path;
      int idx = current_idx;
      while (idx != -1) {
        int py = idx / width_;
        int px = idx % width_;
        grid_path.push_back({px, py});
        auto it = came_from.find(idx);
        idx = (it != came_from.end()) ? it->second : -1;
      }
      std::reverse(grid_path.begin(), grid_path.end());

      RCLCPP_INFO(node_->get_logger(),
                  "Full path found with %zu grid cells after %d iterations",
                  grid_path.size(), iterations);

      // 使用基于局部地图边界的waypoint选择
      result = selectWaypointsBasedOnLocalMap(grid_path, start, local_map, tf_buffer);
      return result;
    }

    if (closed_set.count(current_idx)) {
      continue;
    }
    closed_set.insert(current_idx);

    for (const auto& dir : directions_) {
      int nx = cx + dir.first;
      int ny = cy + dir.second;

      if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {
        continue;
      }

      // 检查目标格子是否是障碍物
      if (isCellOccupied(nx, ny)) {
        continue;
      }

      // 对角线移动时，检查两侧格子是否都可通行
      // 防止"挤过"两个相邻障碍物的角落
      if (dir.first != 0 && dir.second != 0) {
        // 对角线移动，检查两个相邻格子
        if (isCellOccupied(cx + dir.first, cy) || isCellOccupied(cx, cy + dir.second)) {
          continue;  // 如果任一侧有障碍物，不允许对角线移动
        }
      }

      int neighbor_idx = getIndex(nx, ny);

      if (closed_set.count(neighbor_idx)) {
        continue;
      }

      // 计算代价：基础移动代价 * 栅格代价系数
      // 未知区域代价更高，引导路径优先走已知自由空间
      double base_move_cost = (dir.first != 0 && dir.second != 0) ? 1.414 : 1.0;
      double cell_cost = getCellCost(nx, ny);
      double tentative_g = g_score[current_idx] + base_move_cost * cell_cost;

      auto g_it = g_score.find(neighbor_idx);
      if (g_it == g_score.end() || tentative_g < g_it->second) {
        came_from[neighbor_idx] = current_idx;
        g_score[neighbor_idx] = tentative_g;
        double f_score = tentative_g + heuristic(nx, ny, goal_mx, goal_my);
        open_list.push({f_score, nx, ny});
      }
    }
  }

  if (iterations >= max_iterations) {
    last_error_ = "Path planning exceeded max iterations (" + std::to_string(max_iterations) + ")";
    RCLCPP_WARN(node_->get_logger(),
                "%s - start[%d,%d] goal[%d,%d]",
                last_error_.c_str(), start_mx, start_my, goal_mx, goal_my);
  } else {
    double direct_dist = std::sqrt(static_cast<double>((goal_mx - start_mx) * (goal_mx - start_mx) +
                                                       (goal_my - start_my) * (goal_my - start_my)));
    last_error_ = "No path found - goal may be blocked by obstacles";
    RCLCPP_WARN(node_->get_logger(),
                "%s - start[%d,%d] goal[%d,%d] iterations=%d closed=%zu dist=%.0f",
                last_error_.c_str(), start_mx, start_my, goal_mx, goal_my,
                iterations, closed_set.size(), direct_dist);
  }
  return result;
}

std::vector<geometry_msgs::msg::PoseStamped> PathPlanner::selectWaypointsBasedOnLocalMap(
  const std::vector<std::pair<int, int>>& grid_path,
  const geometry_msgs::msg::Point& robot_position,
  const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer
) {
  std::vector<geometry_msgs::msg::PoseStamped> waypoints;

  if (grid_path.empty() || !local_map) {
    return waypoints;
  }

  // 转换为世界坐标
  std::vector<geometry_msgs::msg::Point> world_path;
  for (const auto& cell : grid_path) {
    world_path.push_back(mapToWorld(cell.first, cell.second));
  }

  RCLCPP_INFO(node_->get_logger(),
              "Selecting waypoints from %zu path points based on local map KNOWN areas",
              world_path.size());

  // 第一个waypoint是起点
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = global_map_->header.frame_id;
  pose.pose.position = world_path[0];
  pose.pose.orientation.w = 1.0;

  size_t current_index = 0;
  geometry_msgs::msg::Point current_position = robot_position;

  while (current_index < world_path.size()) {
    // 沿路径查找最远但仍在局部地图**已知区域**内的点
    size_t farthest_index = current_index;

    for (size_t i = current_index + 1; i < world_path.size(); ++i) {
      if (isPointInKnownAreaOfLocalMap(world_path[i], local_map, tf_buffer)) {
        farthest_index = i;
      } else {
        // 一旦遇到未知区域或超出局部地图范围，停止搜索
        break;
      }
    }

    // 如果找到了更远的点
    if (farthest_index > current_index) {
      pose.pose.position = world_path[farthest_index];
      
      // 计算朝向
      if (farthest_index + 1 < world_path.size()) {
        double angle = std::atan2(
          world_path[farthest_index + 1].y - world_path[farthest_index].y,
          world_path[farthest_index + 1].x - world_path[farthest_index].x
        );
        pose.pose.orientation.z = std::sin(angle / 2.0);
        pose.pose.orientation.w = std::cos(angle / 2.0);
      }
      
      waypoints.push_back(pose);
      current_index = farthest_index;
      current_position = world_path[farthest_index];
      
      RCLCPP_DEBUG(node_->get_logger(),
                   "Selected waypoint %zu at (%.2f, %.2f)",
                   waypoints.size(), pose.pose.position.x, pose.pose.position.y);
    } else {
      // 如果没有找到更远的点（可能已经到达终点或遇到问题）
      break;
    }

    // 如果已经到达路径终点
    if (farthest_index >= world_path.size() - 1) {
      break;
    }
  }

  // 确保最后一个点是目标点
  if (waypoints.empty() || 
      waypoints.back().pose.position.x != world_path.back().x ||
      waypoints.back().pose.position.y != world_path.back().y) {
    pose.pose.position = world_path.back();
    pose.pose.orientation.w = 1.0;
    waypoints.push_back(pose);
  }

  RCLCPP_INFO(node_->get_logger(),
              "✅ Selected %zu waypoints based on local map KNOWN areas",
              waypoints.size());

  return waypoints;
}

bool PathPlanner::isPointInLocalMap(
  const geometry_msgs::msg::Point& point,
  const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer
) const {
  if (!local_map) {
    return false;
  }

  geometry_msgs::msg::Point point_in_local_frame = point;

  // 如果提供了tf_buffer且地图frame不是global_map，则需要坐标转换
  if (tf_buffer && local_map->header.frame_id != "global_map" &&
      local_map->header.frame_id != global_map_->header.frame_id) {
    try {
      // 创建PointStamped进行转换
      geometry_msgs::msg::PointStamped point_global, point_local;
      point_global.header.frame_id = global_map_->header.frame_id;
      point_global.header.stamp = rclcpp::Time(0);  // 使用最新可用的transform
      point_global.point = point;

      // 转换到局部地图坐标系
      point_local = tf_buffer->transform(point_global, local_map->header.frame_id,
                                         tf2::durationFromSec(0.1));
      point_in_local_frame = point_local.point;

      RCLCPP_DEBUG(node_->get_logger(),
                   "Transformed point from '%s'(%.2f,%.2f) to '%s'(%.2f,%.2f)",
                   global_map_->header.frame_id.c_str(), point.x, point.y,
                   local_map->header.frame_id.c_str(),
                   point_in_local_frame.x, point_in_local_frame.y);
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                           "Failed to transform point from '%s' to '%s': %s",
                           global_map_->header.frame_id.c_str(),
                           local_map->header.frame_id.c_str(), ex.what());
      // 如果TF转换失败，尝试直接使用坐标（假设在同一坐标系）
      point_in_local_frame = point;
    }
  }

  // 转换为局部地图栅格坐标
  double local_resolution = local_map->info.resolution;
  double local_origin_x = local_map->info.origin.position.x;
  double local_origin_y = local_map->info.origin.position.y;
  int local_width = local_map->info.width;
  int local_height = local_map->info.height;

  int mx = static_cast<int>((point_in_local_frame.x - local_origin_x) / local_resolution);
  int my = static_cast<int>((point_in_local_frame.y - local_origin_y) / local_resolution);

  // 安全边距：1米（转换为栅格数）
  // 避免选择刚好在边界的点，防止TF精度问题导致导航失败
  const double safety_margin_meters = 1.0;
  int safety_margin_cells = static_cast<int>(safety_margin_meters / local_resolution);

  return mx >= safety_margin_cells &&
         mx < (local_width - safety_margin_cells) &&
         my >= safety_margin_cells &&
         my < (local_height - safety_margin_cells);
}

bool PathPlanner::isPointInKnownAreaOfLocalMap(
  const geometry_msgs::msg::Point& point,
  const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer
) const {
  if (!local_map) {
    return false;
  }

  geometry_msgs::msg::Point point_in_local_frame = point;

  // 如果提供了tf_buffer且地图frame不是global_map，则需要坐标转换
  if (tf_buffer && local_map->header.frame_id != "global_map" &&
      global_map_ && local_map->header.frame_id != global_map_->header.frame_id) {
    try {
      geometry_msgs::msg::PointStamped point_global, point_local;
      point_global.header.frame_id = global_map_->header.frame_id;
      point_global.header.stamp = rclcpp::Time(0);
      point_global.point = point;

      point_local = tf_buffer->transform(point_global, local_map->header.frame_id,
                                         tf2::durationFromSec(0.1));
      point_in_local_frame = point_local.point;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 5000,
                           "Failed to transform point for known area check: %s", ex.what());
      point_in_local_frame = point;
    }
  }

  // 转换为局部地图栅格坐标
  double local_resolution = local_map->info.resolution;
  double local_origin_x = local_map->info.origin.position.x;
  double local_origin_y = local_map->info.origin.position.y;
  int local_width = local_map->info.width;
  int local_height = local_map->info.height;

  int mx = static_cast<int>((point_in_local_frame.x - local_origin_x) / local_resolution);
  int my = static_cast<int>((point_in_local_frame.y - local_origin_y) / local_resolution);

  // 安全边距
  const double safety_margin_meters = 0.5;
  int safety_margin_cells = static_cast<int>(safety_margin_meters / local_resolution);

  // 首先检查是否在地图边界内
  if (mx < safety_margin_cells || mx >= (local_width - safety_margin_cells) ||
      my < safety_margin_cells || my >= (local_height - safety_margin_cells)) {
    return false;
  }

  // 检查该点是否在已知区域（不是unknown）
  int index = my * local_width + mx;
  if (index < 0 || index >= static_cast<int>(local_map->data.size())) {
    return false;
  }

  int8_t value = local_map->data[index];
  // OccupancyGrid: -1=unknown, 0-100=known (0=free, 100=occupied)
  // 我们只接受已知且可通行的区域（value >= 0 且 value < 50）
  if (value == -1) {
    // 未知区域
    return false;
  }

  if (value > 50) {
    // 障碍物区域
    return false;
  }

  // 已知的自由或低占用区域
  return true;
}

geometry_msgs::msg::Point PathPlanner::findNearbyFreeCell(
  const geometry_msgs::msg::Point& blocked_point,
  int search_radius
) const {
  if (!map_initialized_) {
    return blocked_point;
  }

  int center_mx, center_my;
  if (!worldToMap(blocked_point, center_mx, center_my)) {
    return blocked_point;
  }

  // 使用螺旋搜索找最近的可用栅格
  // 从中心向外扩展
  for (int radius = 1; radius <= search_radius; ++radius) {
    // 搜索当前半径的正方形边界上的所有点
    for (int dx = -radius; dx <= radius; ++dx) {
      for (int dy = -radius; dy <= radius; ++dy) {
        // 只检查边界上的点（跳过内部已检查过的点）
        if (std::abs(dx) != radius && std::abs(dy) != radius) {
          continue;
        }

        int nx = center_mx + dx;
        int ny = center_my + dy;

        // 边界检查
        if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {
          continue;
        }

        // 检查该栅格是否可用（使用膨胀检查）
        if (!isCellOccupied(nx, ny)) {
          // 额外检查：确保该点的占用值较低
          int index = ny * width_ + nx;
          if (index >= 0 && index < static_cast<int>(global_map_->data.size())) {
            int8_t value = global_map_->data[index];
            if (value >= 0 && value <= 50) {
              // 找到可用的栅格
              geometry_msgs::msg::Point free_point = mapToWorld(nx, ny);
              RCLCPP_INFO(node_->get_logger(),
                          "Found nearby free cell at (%.2f, %.2f) [%d, %d], "
                          "offset from blocked point by %d cells",
                          free_point.x, free_point.y, nx, ny, radius);
              return free_point;
            }
          }
        }
      }
    }
  }

  // 没有找到可用栅格，返回原点
  RCLCPP_WARN(node_->get_logger(),
              "Could not find free cell within %d cells of (%.2f, %.2f)",
              search_radius, blocked_point.x, blocked_point.y);
  return blocked_point;
}

PathPlanner::PathPlanResult PathPlanner::planPathWithFullResult(
  const geometry_msgs::msg::Point& start,
  const geometry_msgs::msg::Point& goal,
  const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer
) {
  PathPlanResult result;
  result.success = false;
  result.first_waypoint_path_index = 0;

  last_error_.clear();

  if (!map_initialized_) {
    last_error_ = "Global map not initialized";
    RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
    return result;
  }

  // 转换为地图坐标
  int start_mx, start_my, goal_mx, goal_my;
  if (!worldToMap(start, start_mx, start_my)) {
    // 起点在地图外，尝试钳位到最近边界（可能因TF漂移导致轻微越界）
    start_mx = static_cast<int>((start.x - origin_x_) / resolution_);
    start_my = static_cast<int>((start.y - origin_y_) / resolution_);
    int clamped_mx = std::max(0, std::min(start_mx, width_ - 1));
    int clamped_my = std::max(0, std::min(start_my, height_ - 1));
    int drift = std::abs(start_mx - clamped_mx) + std::abs(start_my - clamped_my);
    if (drift > 10) {  // 超过10个栅格（0.5m）的漂移，认为确实越界
      last_error_ = "Start point outside global map bounds: pixel [" +
                    std::to_string(start_mx) + ", " + std::to_string(start_my) +
                    "] world (" + std::to_string(start.x) + ", " + std::to_string(start.y) + ")";
      RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
      return result;
    }
    RCLCPP_WARN(node_->get_logger(),
                "Start point pixel [%d, %d] slightly outside map, clamped to [%d, %d]",
                start_mx, start_my, clamped_mx, clamped_my);
    start_mx = clamped_mx;
    start_my = clamped_my;
  }

  if (!worldToMap(goal, goal_mx, goal_my)) {
    // 终点在地图外，尝试钳位到最近边界
    goal_mx = static_cast<int>((goal.x - origin_x_) / resolution_);
    goal_my = static_cast<int>((goal.y - origin_y_) / resolution_);
    int clamped_mx = std::max(0, std::min(goal_mx, width_ - 1));
    int clamped_my = std::max(0, std::min(goal_my, height_ - 1));
    int drift = std::abs(goal_mx - clamped_mx) + std::abs(goal_my - clamped_my);
    if (drift > 10) {  // 超过10个栅格（0.5m）的漂移，认为确实越界
      last_error_ = "Goal point outside global map bounds: pixel [" +
                    std::to_string(goal_mx) + ", " + std::to_string(goal_my) +
                    "] world (" + std::to_string(goal.x) + ", " + std::to_string(goal.y) + ")";
      RCLCPP_ERROR(node_->get_logger(), "%s", last_error_.c_str());
      return result;
    }
    RCLCPP_WARN(node_->get_logger(),
                "Goal point pixel [%d, %d] slightly outside map, clamped to [%d, %d]",
                goal_mx, goal_my, clamped_mx, clamped_my);
    goal_mx = clamped_mx;
    goal_my = clamped_my;
  }

  // 检查起点和终点
  int start_index = start_my * width_ + start_mx;
  int goal_index = goal_my * width_ + goal_mx;
  int8_t start_value = (start_index >= 0 && start_index < static_cast<int>(global_map_->data.size()))
                       ? global_map_->data[start_index] : 100;
  int8_t goal_value = (goal_index >= 0 && goal_index < static_cast<int>(global_map_->data.size()))
                      ? global_map_->data[goal_index] : 100;

  // 使用的实际起点（可能被调整）
  geometry_msgs::msg::Point actual_start = start;
  int actual_start_mx = start_mx;
  int actual_start_my = start_my;

  if (start_value > 70) {
    RCLCPP_WARN(node_->get_logger(),
                "Start point at (%.2f,%.2f)[%d,%d] is occupied (value=%d), searching nearby...",
                start.x, start.y, start_mx, start_my, static_cast<int>(start_value));

    actual_start = findNearbyFreeCell(start, 15);

    if (!worldToMap(actual_start, actual_start_mx, actual_start_my)) {
      last_error_ = "Failed to find valid start point";
      return result;
    }

    int new_start_index = actual_start_my * width_ + actual_start_mx;
    int8_t new_start_value = (new_start_index >= 0 && new_start_index < static_cast<int>(global_map_->data.size()))
                             ? global_map_->data[new_start_index] : 100;

    if (new_start_value > 70) {
      last_error_ = "Could not find valid start point near robot position";
      return result;
    }

    RCLCPP_INFO(node_->get_logger(),
                "Using adjusted start point at (%.2f,%.2f)[%d,%d] (value=%d)",
                actual_start.x, actual_start.y, actual_start_mx, actual_start_my,
                static_cast<int>(new_start_value));
  }

  if (goal_value > 65) {
    last_error_ = "Goal point is occupied (value=" + std::to_string(static_cast<int>(goal_value)) + ")";
    return result;
  }

  RCLCPP_INFO(node_->get_logger(),
              "Planning full path from (%.2f, %.2f) [%d, %d] to (%.2f, %.2f) [%d, %d]",
              actual_start.x, actual_start.y, actual_start_mx, actual_start_my,
              goal.x, goal.y, goal_mx, goal_my);

  // A* 路径规划
  using PQElement = std::tuple<double, int, int>;
  std::priority_queue<PQElement, std::vector<PQElement>, std::greater<PQElement>> open_list;

  std::unordered_map<int, double> g_score;
  std::unordered_map<int, int> came_from;
  std::unordered_set<int> closed_set;

  auto getIndex = [this](int x, int y) { return y * width_ + x; };

  int start_idx = getIndex(actual_start_mx, actual_start_my);

  g_score[start_idx] = 0.0;
  open_list.push({heuristic(actual_start_mx, actual_start_my, goal_mx, goal_my), actual_start_mx, actual_start_my});

  int iterations = 0;
  const int max_iterations = 100000;

  while (!open_list.empty() && iterations < max_iterations) {
    iterations++;

    auto [f, cx, cy] = open_list.top();
    open_list.pop();

    int current_idx = getIndex(cx, cy);

    if (cx == goal_mx && cy == goal_my) {
      // 找到目标，重建路径
      std::vector<std::pair<int, int>> grid_path;
      int idx = current_idx;
      while (idx != -1) {
        int py = idx / width_;
        int px = idx % width_;
        grid_path.push_back({px, py});
        auto it = came_from.find(idx);
        idx = (it != came_from.end()) ? it->second : -1;
      }
      std::reverse(grid_path.begin(), grid_path.end());

      // 转换为世界坐标
      for (const auto& cell : grid_path) {
        result.full_astar_path.push_back(mapToWorld(cell.first, cell.second));
      }

      RCLCPP_INFO(node_->get_logger(),
                  "Full path found with %zu grid cells after %d iterations",
                  grid_path.size(), iterations);

      // 选择第一个中继点
      if (local_map && tf_buffer) {
        auto [waypoint, path_idx] = selectNextWaypointFromPath(
          result.full_astar_path, 0, local_map, tf_buffer
        );

        if (path_idx > 0) {
          result.first_waypoint = waypoint;
          result.first_waypoint_path_index = path_idx;
          result.success = true;

          RCLCPP_INFO(node_->get_logger(),
                      "✅ First waypoint selected at index %zu: (%.2f, %.2f)",
                      path_idx, waypoint.pose.position.x, waypoint.pose.position.y);
        } else {
          // 如果无法选择中继点，使用终点
          result.first_waypoint.header.frame_id = global_map_->header.frame_id;
          result.first_waypoint.pose.position = goal;
          result.first_waypoint.pose.orientation.w = 1.0;
          result.first_waypoint_path_index = result.full_astar_path.size() - 1;
          result.success = true;

          RCLCPP_INFO(node_->get_logger(),
                      "Using goal as first waypoint (local map may not be available)");
        }
      } else {
        // 没有局部地图信息，使用终点作为第一个中继点
        result.first_waypoint.header.frame_id = global_map_->header.frame_id;
        result.first_waypoint.pose.position = goal;
        result.first_waypoint.pose.orientation.w = 1.0;
        result.first_waypoint_path_index = result.full_astar_path.size() - 1;
        result.success = true;

        RCLCPP_WARN(node_->get_logger(),
                    "No local map available, using goal as first waypoint");
      }

      return result;
    }

    if (closed_set.count(current_idx)) {
      continue;
    }
    closed_set.insert(current_idx);

    for (const auto& dir : directions_) {
      int nx = cx + dir.first;
      int ny = cy + dir.second;

      if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) {
        continue;
      }

      if (isCellOccupied(nx, ny)) {
        continue;
      }

      if (dir.first != 0 && dir.second != 0) {
        if (isCellOccupied(cx + dir.first, cy) || isCellOccupied(cx, cy + dir.second)) {
          continue;
        }
      }

      int neighbor_idx = getIndex(nx, ny);

      if (closed_set.count(neighbor_idx)) {
        continue;
      }

      double base_move_cost = (dir.first != 0 && dir.second != 0) ? 1.414 : 1.0;
      double cell_cost = getCellCost(nx, ny);
      double tentative_g = g_score[current_idx] + base_move_cost * cell_cost;

      auto g_it = g_score.find(neighbor_idx);
      if (g_it == g_score.end() || tentative_g < g_it->second) {
        came_from[neighbor_idx] = current_idx;
        g_score[neighbor_idx] = tentative_g;
        double f_score = tentative_g + heuristic(nx, ny, goal_mx, goal_my);
        open_list.push({f_score, nx, ny});
      }
    }
  }

  if (iterations >= max_iterations) {
    last_error_ = "Path planning exceeded max iterations";
  } else {
    last_error_ = "No path found - goal may be blocked by obstacles";
  }
  RCLCPP_WARN(node_->get_logger(), "%s", last_error_.c_str());
  return result;
}

std::pair<geometry_msgs::msg::PoseStamped, size_t> PathPlanner::selectNextWaypointFromPath(
  const std::vector<geometry_msgs::msg::Point>& full_path,
  size_t current_index,
  const nav_msgs::msg::OccupancyGrid::SharedPtr& local_map,
  std::shared_ptr<tf2_ros::Buffer> tf_buffer
) {
  geometry_msgs::msg::PoseStamped waypoint;
  waypoint.header.frame_id = global_map_ ? global_map_->header.frame_id : "global_map";
  waypoint.pose.orientation.w = 1.0;

  if (full_path.empty() || current_index >= full_path.size() - 1) {
    return {waypoint, 0};
  }

  if (!local_map) {
    RCLCPP_WARN(node_->get_logger(), "No local map provided for waypoint selection");
    // 返回路径终点
    waypoint.pose.position = full_path.back();
    return {waypoint, full_path.size() - 1};
  }

  RCLCPP_DEBUG(node_->get_logger(),
               "Selecting next waypoint from path index %zu (total: %zu)",
               current_index, full_path.size());

  // 从当前位置开始，找到局部地图**已知区域**内最远的点
  size_t farthest_index = current_index;

  for (size_t i = current_index + 1; i < full_path.size(); ++i) {
    // 使用 isPointInKnownAreaOfLocalMap 检查点是否在已知区域
    if (isPointInKnownAreaOfLocalMap(full_path[i], local_map, tf_buffer)) {
      farthest_index = i;
    } else {
      // 一旦遇到未知区域或超出局部地图范围，停止搜索
      RCLCPP_DEBUG(node_->get_logger(),
                   "Path point %zu at (%.2f, %.2f) is NOT in known area, stopping search",
                   i, full_path[i].x, full_path[i].y);
      break;
    }
  }

  // 如果找到了更远的点
  if (farthest_index > current_index) {
    waypoint.pose.position = full_path[farthest_index];

    // 计算朝向（指向下一个点）
    if (farthest_index + 1 < full_path.size()) {
      double angle = std::atan2(
        full_path[farthest_index + 1].y - full_path[farthest_index].y,
        full_path[farthest_index + 1].x - full_path[farthest_index].x
      );
      waypoint.pose.orientation.z = std::sin(angle / 2.0);
      waypoint.pose.orientation.w = std::cos(angle / 2.0);
    }

    RCLCPP_INFO(node_->get_logger(),
                 "✅ Selected waypoint at path index %zu: (%.2f, %.2f) - farthest point in KNOWN area",
                 farthest_index, waypoint.pose.position.x, waypoint.pose.position.y);

    return {waypoint, farthest_index};
  }

  // 如果当前点之后都不在已知区域内，返回路径终点
  // 这种情况说明目标点在未知区域，机器人需要继续探索
  waypoint.pose.position = full_path.back();

  RCLCPP_WARN(node_->get_logger(),
               "No point in known area after index %zu, using goal at index %zu. "
               "Robot may need to explore further.",
               current_index, full_path.size() - 1);

  return {waypoint, full_path.size() - 1};
}

}  // namespace reliable_communication
