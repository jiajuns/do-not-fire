/**
 * @file frontier_detector.cpp
 * @brief Implementation of CURE1 frontier detection algorithm
 */

#include "real_cure_cpp/frontier_detector.hpp"

namespace real_cure_cpp
{

// Static color definitions
const std::vector<std::tuple<float, float, float>> FrontierDetector::COLORS = {
  {1.0f, 0.0f, 0.0f},  // Red
  {0.0f, 1.0f, 0.0f},  // Green
  {0.0f, 0.0f, 1.0f},  // Blue
  {1.0f, 1.0f, 0.0f},  // Yellow
  {1.0f, 0.0f, 1.0f},  // Magenta
  {0.0f, 1.0f, 1.0f},  // Cyan
  {1.0f, 0.5f, 0.0f},  // Orange
  {0.5f, 0.0f, 1.0f},  // Purple
  {0.0f, 1.0f, 0.5f},  // Spring green
  {1.0f, 0.0f, 0.5f},  // Rose
};

FrontierDetector::FrontierDetector(const rclcpp::NodeOptions & options)
: Node("frontier_detector",
       rclcpp::NodeOptions(options).automatically_declare_parameters_from_overrides(true)),
  map_received_(false)
{
  // Get parameters
  global_map_topic_ = this->get_parameter_or("global_map_topic", std::string("/global_map"));
  frontier_topic_ = this->get_parameter_or("frontier_topic", std::string("/frontiers"));
  cluster_vis_topic_ = this->get_parameter_or("cluster_vis_topic",
                                               std::string("/frontier_clusters_vis"));
  update_rate_ = this->get_parameter_or("update_rate", 0.5);
  min_frontier_size_ = this->get_parameter_or("min_frontier_size", 80);
  info_radius_ = this->get_parameter_or("info_radius", 2.0);
  canny_low_ = this->get_parameter_or("canny_low", 80);
  canny_high_ = this->get_parameter_or("canny_high", 240);
  dilation_radius_ = this->get_parameter_or("dilation_radius", 2);
  use_reachability_filter_ = this->get_parameter_or("use_reachability_filter", true);

  // Create subscriptions
  map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
    global_map_topic_, 10,
    std::bind(&FrontierDetector::mapCallback, this, std::placeholders::_1));

  reset_sub_ = this->create_subscription<std_msgs::msg::Empty>(
    "/cure_explore/reset", 10,
    std::bind(&FrontierDetector::resetCallback, this, std::placeholders::_1));

  // Create publishers
  frontier_pub_ = this->create_publisher<cure_explore_interfaces::msg::FrontierArray>(
    frontier_topic_, 10);
  cluster_vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    cluster_vis_topic_, 10);
  frontier_points_vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/frontier_points_vis", 10);

  // Create timer
  double period = 1.0 / update_rate_;
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(period),
    std::bind(&FrontierDetector::timerCallback, this));

  RCLCPP_INFO(this->get_logger(), "Frontier Detector (C++ CURE1) initialized");
  RCLCPP_INFO(this->get_logger(), "  Canny params: low=%d, high=%d", canny_low_, canny_high_);
  RCLCPP_INFO(this->get_logger(), "  Dilation radius: %d", dilation_radius_);
  RCLCPP_INFO(this->get_logger(), "  Reachability filter: %s",
              use_reachability_filter_ ? "ENABLED" : "disabled");
}

void FrontierDetector::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(map_mutex_);
  current_map_ = msg;
  map_received_ = true;
}

void FrontierDetector::resetCallback(const std_msgs::msg::Empty::SharedPtr /*msg*/)
{
  RCLCPP_INFO(this->get_logger(), "Received reset signal, clearing map state");

  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    current_map_ = nullptr;
    map_received_ = false;
  }

  // Publish empty frontiers
  cure_explore_interfaces::msg::FrontierArray empty_frontiers;
  empty_frontiers.header.stamp = this->get_clock()->now();
  empty_frontiers.header.frame_id = "global_map";
  frontier_pub_->publish(empty_frontiers);

  // Clear visualization
  visualization_msgs::msg::MarkerArray empty_markers;
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  empty_markers.markers.push_back(delete_marker);
  cluster_vis_pub_->publish(empty_markers);

  RCLCPP_INFO(this->get_logger(), "Frontier detector reset complete");
}

void FrontierDetector::timerCallback()
{
  std::lock_guard<std::mutex> lock(map_mutex_);

  if (!map_received_ || !current_map_) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Waiting for global map...");
    return;
  }

  detectAndPublishFrontiersCure1();
}

std::pair<double, double> FrontierDetector::gridToWorld(int gx, int gy)
{
  // Assumes map_mutex_ is held
  double origin_x = current_map_->info.origin.position.x;
  double origin_y = current_map_->info.origin.position.y;
  double resolution = current_map_->info.resolution;

  double wx = (gx + 0.5) * resolution + origin_x;
  double wy = (gy + 0.5) * resolution + origin_y;

  return {wx, wy};
}

std::pair<int, int> FrontierDetector::worldToGrid(double wx, double wy)
{
  // Assumes map_mutex_ is held
  if (!current_map_) {
    return {-1, -1};
  }

  double origin_x = current_map_->info.origin.position.x;
  double origin_y = current_map_->info.origin.position.y;
  double resolution = current_map_->info.resolution;

  int gx = static_cast<int>((wx - origin_x) / resolution);
  int gy = static_cast<int>((wy - origin_y) / resolution);

  return {gx, gy};
}

cv::Mat FrontierDetector::occupancyToImage(
  const std::vector<int8_t> & map_data, int height, int width)
{
  cv::Mat img(height, width, CV_8UC1);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int8_t val = map_data[y * width + x];
      if (val == -1) {
        img.at<uchar>(y, x) = 255;  // Unknown = white
      } else if (val == 0) {
        img.at<uchar>(y, x) = 0;    // Free = black
      } else {  // val >= 100
        img.at<uchar>(y, x) = 100;  // Occupied = gray
      }
    }
  }

  return img;
}

bool FrontierDetector::isValidFrontierCentroid(
  double cx, double cy,
  const std::vector<int8_t> & map_data,
  int width, int height)
{
  // Margin check
  int margin = 15;
  if (cx < margin || cx >= width - margin || cy < margin || cy >= height - margin) {
    return false;
  }

  int gx = static_cast<int>(cx);
  int gy = static_cast<int>(cy);

  if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
    return false;
  }

  // Check centroid is not on obstacle
  int8_t center_val = map_data[gy * width + gx];
  if (center_val == 100) {
    return false;
  }

  // Check for unknown cells nearby (5x5 neighborhood)
  int search_radius = 5;
  bool has_unknown_nearby = false;

  for (int dy = -search_radius; dy <= search_radius && !has_unknown_nearby; dy++) {
    for (int dx = -search_radius; dx <= search_radius && !has_unknown_nearby; dx++) {
      int nx = gx + dx;
      int ny = gy + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        if (map_data[ny * width + nx] == -1) {
          has_unknown_nearby = true;
        }
      }
    }
  }

  if (!has_unknown_nearby) {
    return false;
  }

  // Check for free space nearby (3x3 neighborhood)
  int search_radius_free = 3;
  bool has_free_nearby = false;

  for (int dy = -search_radius_free; dy <= search_radius_free && !has_free_nearby; dy++) {
    for (int dx = -search_radius_free; dx <= search_radius_free && !has_free_nearby; dx++) {
      int nx = gx + dx;
      int ny = gy + dy;
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        if (map_data[ny * width + nx] == 0) {
          has_free_nearby = true;
        }
      }
    }
  }

  return has_free_nearby;
}

std::vector<bool> FrontierDetector::computeReachableFreeSpace(
  const std::vector<int8_t> & map_data, int width, int height)
{
  // Find all connected components of free space and keep the largest one
  // This filters out unreachable frontiers on the other side of obstacles

  std::vector<bool> visited(width * height, false);
  std::vector<bool> reachable(width * height, false);

  // 8-connectivity directions
  static const int DX[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  static const int DY[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  int largest_component_size = 0;
  std::vector<int> largest_component_cells;

  // Find all connected components of free space
  for (int start_y = 0; start_y < height; start_y++) {
    for (int start_x = 0; start_x < width; start_x++) {
      int start_idx = start_y * width + start_x;

      // Skip if not free or already visited
      if (map_data[start_idx] != 0 || visited[start_idx]) {
        continue;
      }

      // BFS flood fill for this component
      std::queue<std::pair<int, int>> bfs_queue;
      std::vector<int> component_cells;

      bfs_queue.push({start_x, start_y});
      visited[start_idx] = true;
      component_cells.push_back(start_idx);

      while (!bfs_queue.empty()) {
        auto [cx, cy] = bfs_queue.front();
        bfs_queue.pop();

        for (int d = 0; d < 8; d++) {
          int nx = cx + DX[d];
          int ny = cy + DY[d];

          if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
            continue;
          }

          int nidx = ny * width + nx;

          // Only expand to free cells
          if (map_data[nidx] == 0 && !visited[nidx]) {
            visited[nidx] = true;
            bfs_queue.push({nx, ny});
            component_cells.push_back(nidx);
          }
        }
      }

      // Update largest component
      if (static_cast<int>(component_cells.size()) > largest_component_size) {
        largest_component_size = static_cast<int>(component_cells.size());
        largest_component_cells = std::move(component_cells);
      }
    }
  }

  // Mark largest component as reachable
  for (int idx : largest_component_cells) {
    reachable[idx] = true;
  }

  RCLCPP_DEBUG(this->get_logger(),
               "Reachability filter: largest free component has %d cells",
               largest_component_size);

  return reachable;
}

bool FrontierDetector::isFrontierAdjacentToReachable(
  double cx, double cy,
  const std::vector<bool> & reachable_mask,
  int width, int height)
{
  int gx = static_cast<int>(cx);
  int gy = static_cast<int>(cy);

  if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
    return false;
  }

  // Check 5x5 neighborhood for reachable free cells
  // Larger window to handle frontiers that are slightly offset from free space
  int search_radius = 5;

  for (int dy = -search_radius; dy <= search_radius; dy++) {
    for (int dx = -search_radius; dx <= search_radius; dx++) {
      int nx = gx + dx;
      int ny = gy + dy;

      if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        continue;
      }

      int nidx = ny * width + nx;
      if (reachable_mask[nidx]) {
        return true;
      }
    }
  }

  return false;
}

double FrontierDetector::calculateInformationGainAtCentroid(
  double cx, double cy,
  const std::vector<int8_t> & map_data,
  int width, int height)
{
  if (!current_map_) {
    return 0.0;
  }

  double resolution = current_map_->info.resolution;
  int radius_cells = static_cast<int>(info_radius_ / resolution);

  int gx = static_cast<int>(cx);
  int gy = static_cast<int>(cy);

  if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
    return 0.0;
  }

  int x_min = std::max(0, gx - radius_cells);
  int x_max = std::min(width, gx + radius_cells + 1);
  int y_min = std::max(0, gy - radius_cells);
  int y_max = std::min(height, gy + radius_cells + 1);

  int unknown_count = 0;
  int radius_sq = radius_cells * radius_cells;

  for (int y = y_min; y < y_max; y++) {
    for (int x = x_min; x < x_max; x++) {
      int dist_sq = (x - gx) * (x - gx) + (y - gy) * (y - gy);
      if (dist_sq <= radius_sq) {
        if (map_data[y * width + x] == -1) {
          unknown_count++;
        }
      }
    }
  }

  return static_cast<double>(unknown_count);
}

void FrontierDetector::detectAndPublishFrontiersCure1()
{
  // Assumes map_mutex_ is held
  int width = current_map_->info.width;
  int height = current_map_->info.height;

  // Copy map data
  std::vector<int8_t> map_data(current_map_->data.begin(), current_map_->data.end());

  // Compute reachable free space mask (flood fill based filter)
  std::vector<bool> reachable_mask;
  if (use_reachability_filter_) {
    reachable_mask = computeReachableFreeSpace(map_data, width, height);
  }

  // 1. Convert to image format
  cv::Mat image = occupancyToImage(map_data, height, width);

  // 2. CURE1 multi-threshold processing
  cv::Mat gray1, gray2, gray3;
  cv::threshold(image, gray1, 200, 255, cv::THRESH_BINARY_INV);
  cv::threshold(image, gray2, 20, 255, cv::THRESH_BINARY);
  cv::threshold(image, gray3, 120, 255, cv::THRESH_BINARY);

  // 3. Obstacle region: obs = gray1 AND gray2
  cv::Mat obs;
  cv::bitwise_and(gray1, gray2, obs);

  // 4. Canny edge detection
  cv::Mat edge;
  cv::Canny(gray3, edge, canny_low_, canny_high_);

  // 5. CURE1 dilation using circles
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (obs.at<uchar>(i, j) == 255) {
        cv::circle(obs, cv::Point(j, i), dilation_radius_, 200, -1, cv::LINE_AA);
      }
      if (edge.at<uchar>(i, j) == 255) {
        cv::circle(edge, cv::Point(j, i), dilation_radius_, 200, -1, cv::LINE_AA);
      }
    }
  }

  // 6. Invert dilated obstacles
  cv::Mat obs_inv;
  cv::threshold(obs, obs_inv, 20, 255, cv::THRESH_BINARY_INV);

  // 7. Final frontiers: obs_inv AND edge
  cv::Mat frontiers;
  cv::bitwise_and(obs_inv, edge, frontiers);

  // 8. Strict frontier validation: must have both free and unknown neighbors
  cv::Mat validated_frontiers = cv::Mat::zeros(frontiers.size(), CV_8UC1);
  int validated_count = 0;
  
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      if (frontiers.at<uchar>(y, x) == 0) {
        continue;  // Not a frontier candidate
      }
      
      // Check 8-connected neighbors
      bool has_free = false;
      bool has_unknown = false;
      
      for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
          if (dx == 0 && dy == 0) continue;
          
          int nx = x + dx;
          int ny = y + dy;
          int8_t neighbor_val = map_data[ny * width + nx];
          
          if (neighbor_val == 0) {
            has_free = true;
          } else if (neighbor_val == -1) {
            has_unknown = true;
          }
          // Note: neighbor_val can only be {-1, 0, 100} now
          // No need to check for transitional values (1-99)
        }
      }
      
      // Valid frontier point must be at the boundary of free and unknown
      if (has_free && has_unknown) {
        validated_frontiers.at<uchar>(y, x) = 255;
        validated_count++;
      }
    }
  }
  
  RCLCPP_DEBUG(this->get_logger(),
               "Frontier validation: %d/%d points validated as true free-unknown boundary",
               validated_count, cv::countNonZero(frontiers));

  // 9. Connected components analysis on validated frontiers
  cv::Mat labels, stats, centroids_cv;
  int num_labels = cv::connectedComponentsWithStats(validated_frontiers, labels, stats, centroids_cv, 8);

  std::vector<cure_explore_interfaces::msg::FrontierCluster> frontier_clusters;
  std::vector<geometry_msgs::msg::Point> cluster_centroids;
  int total_frontier_points = 0;
  std::vector<double> info_gains;
  int rejected_count = 0;

  // Process each cluster
  for (int i = 1; i < num_labels; i++) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);

    // Filter small regions
    if (area < min_frontier_size_) {
      continue;
    }

    // Get centroid
    double cx = centroids_cv.at<double>(i, 0);
    double cy = centroids_cv.at<double>(i, 1);

    // Validate centroid
    if (!isValidFrontierCentroid(cx, cy, map_data, width, height)) {
      rejected_count++;
      continue;
    }

    // Reachability filter: skip frontiers not adjacent to reachable free space
    if (use_reachability_filter_ &&
        !isFrontierAdjacentToReachable(cx, cy, reachable_mask, width, height)) {
      rejected_count++;
      RCLCPP_DEBUG(this->get_logger(),
                   "Frontier at (%.1f, %.1f) rejected: not reachable", cx, cy);
      continue;
    }

    auto [wx_c, wy_c] = gridToWorld(static_cast<int>(cx), static_cast<int>(cy));

    geometry_msgs::msg::Point centroid;
    centroid.x = wx_c;
    centroid.y = wy_c;
    centroid.z = 0.0;

    // Get all points in this cluster (filter boundary points)
    int margin = 10;
    std::vector<geometry_msgs::msg::Point> frontier_points;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        if (labels.at<int>(y, x) == i) {
          // Skip boundary points
          if (x < margin || x >= width - margin || y < margin || y >= height - margin) {
            continue;
          }
          auto [wx, wy] = gridToWorld(x, y);
          geometry_msgs::msg::Point pt;
          pt.x = wx;
          pt.y = wy;
          pt.z = 0.0;
          frontier_points.push_back(pt);
        }
      }
    }

    // Skip if no valid points
    if (frontier_points.empty()) {
      rejected_count++;
      continue;
    }

    // Calculate information gain
    double info_gain = calculateInformationGainAtCentroid(cx, cy, map_data, width, height);
    info_gains.push_back(info_gain);

    // Create cluster message
    cure_explore_interfaces::msg::FrontierCluster cluster;
    cluster.id = i;
    cluster.centroid = centroid;
    cluster.size = area;
    cluster.frontier_points = frontier_points;
    cluster.info_gain = static_cast<float>(info_gain);
    cluster.structure_coefficient = 0.5f;
    cluster.hazard_coefficient = 0.0f;
    cluster.objective_coefficient = 0.0f;
    cluster.associated_semantic_id = -1;
    cluster.associated_semantic_name = "";
    cluster.associated_semantic_category = "";

    frontier_clusters.push_back(cluster);
    cluster_centroids.push_back(centroid);
    total_frontier_points += frontier_points.size();
  }

  // Normalize info_gain to [0, 1]
  if (!info_gains.empty()) {
    double max_gain = *std::max_element(info_gains.begin(), info_gains.end());
    double min_gain = *std::min_element(info_gains.begin(), info_gains.end());

    for (size_t i = 0; i < frontier_clusters.size(); i++) {
      if (max_gain > min_gain) {
        frontier_clusters[i].info_gain = static_cast<float>(
          (info_gains[i] - min_gain) / (max_gain - min_gain));
      } else {
        frontier_clusters[i].info_gain = (max_gain > 0) ? 1.0f : 0.0f;
      }
    }
  }

  // Publish
  publishFrontiers(frontier_clusters, total_frontier_points);
  publishClusters(cluster_centroids);
  publishFrontierPointsVisualization(frontier_clusters);

  if (rejected_count > 0) {
    RCLCPP_DEBUG(this->get_logger(),
                 "Rejected %d invalid frontier clusters", rejected_count);
  }

  RCLCPP_INFO(this->get_logger(),
              "Detected %zu frontier clusters (%d points) using CURE1 Canny method",
              frontier_clusters.size(), total_frontier_points);
}

void FrontierDetector::publishFrontiers(
  const std::vector<cure_explore_interfaces::msg::FrontierCluster> & clusters,
  int total_points)
{
  cure_explore_interfaces::msg::FrontierArray frontier_array;
  frontier_array.header.frame_id = current_map_->header.frame_id;
  frontier_array.header.stamp = this->get_clock()->now();
  frontier_array.clusters = clusters;
  frontier_array.total_clusters = static_cast<int32_t>(clusters.size());
  frontier_array.total_frontier_points = total_points;

  frontier_pub_->publish(frontier_array);
}

void FrontierDetector::publishClusters(
  const std::vector<geometry_msgs::msg::Point> & centroids)
{
  visualization_msgs::msg::MarkerArray marker_array;

  // Delete old markers
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(delete_marker);

  for (size_t i = 0; i < centroids.size(); i++) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = current_map_->header.frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "frontier_clusters_vis";
    marker.id = static_cast<int>(i);
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position = centroids[i];
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;

    marker.color.r = 1.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 0.8f;

    marker_array.markers.push_back(marker);
  }

  cluster_vis_pub_->publish(marker_array);
}

void FrontierDetector::publishFrontierPointsVisualization(
  const std::vector<cure_explore_interfaces::msg::FrontierCluster> & clusters)
{
  visualization_msgs::msg::MarkerArray marker_array;

  // Delete old markers
  visualization_msgs::msg::Marker delete_marker;
  delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(delete_marker);

  // Assign colors to clusters
  auto cluster_colors = assignColorsToClusters(clusters);

  for (size_t idx = 0; idx < clusters.size(); idx++) {
    const auto & cluster = clusters[idx];

    // Points marker for cluster
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = current_map_->header.frame_id;
    marker.header.stamp = this->get_clock()->now();
    marker.ns = "frontier_points";
    marker.id = static_cast<int>(idx);
    marker.type = visualization_msgs::msg::Marker::POINTS;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.05;
    marker.scale.y = 0.05;

    auto [r, g, b] = cluster_colors[idx];
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = 0.8f;

    marker.points = cluster.frontier_points;
    marker_array.markers.push_back(marker);

    // Text label for cluster
    visualization_msgs::msg::Marker text_marker;
    text_marker.header.frame_id = current_map_->header.frame_id;
    text_marker.header.stamp = this->get_clock()->now();
    text_marker.ns = "frontier_labels";
    text_marker.id = static_cast<int>(idx + 1000);
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose.position = cluster.centroid;
    text_marker.pose.position.z = 0.3;
    text_marker.pose.orientation.w = 1.0;

    text_marker.scale.z = 0.2;
    text_marker.color.r = 1.0f;
    text_marker.color.g = 1.0f;
    text_marker.color.b = 1.0f;
    text_marker.color.a = 1.0f;

    char buffer[100];
    snprintf(buffer, sizeof(buffer), "ID:%d\nSize:%d\nGain:%.2f",
             cluster.id, cluster.size, cluster.info_gain);
    text_marker.text = buffer;

    marker_array.markers.push_back(text_marker);
  }

  frontier_points_vis_pub_->publish(marker_array);
}

std::vector<std::tuple<float, float, float>> FrontierDetector::assignColorsToClusters(
  const std::vector<cure_explore_interfaces::msg::FrontierCluster> & clusters)
{
  if (clusters.empty()) {
    return {};
  }

  size_t n_clusters = clusters.size();
  size_t n_colors = COLORS.size();

  // If few clusters, directly assign different colors
  if (n_clusters <= n_colors) {
    std::vector<std::tuple<float, float, float>> result;
    for (size_t i = 0; i < n_clusters; i++) {
      result.push_back(COLORS[i]);
    }
    return result;
  }

  // Distance function
  auto distance = [](const cure_explore_interfaces::msg::FrontierCluster & c1,
                     const cure_explore_interfaces::msg::FrontierCluster & c2) {
    double dx = c1.centroid.x - c2.centroid.x;
    double dy = c1.centroid.y - c2.centroid.y;
    return std::sqrt(dx * dx + dy * dy);
  };

  // Neighbor threshold (meters)
  double neighbor_threshold = 3.0;

  // Color assignments
  std::vector<int> color_assignments(n_clusters, -1);

  // Greedy algorithm
  for (size_t i = 0; i < n_clusters; i++) {
    std::set<int> neighbor_colors;

    for (size_t j = 0; j < i; j++) {
      if (color_assignments[j] != -1) {
        if (distance(clusters[i], clusters[j]) < neighbor_threshold) {
          neighbor_colors.insert(color_assignments[j]);
        }
      }
    }

    // Find first unused color
    bool assigned = false;
    for (size_t color_idx = 0; color_idx < n_colors && !assigned; color_idx++) {
      if (neighbor_colors.find(static_cast<int>(color_idx)) == neighbor_colors.end()) {
        color_assignments[i] = static_cast<int>(color_idx);
        assigned = true;
      }
    }

    // Fallback: use least used color among neighbors
    if (!assigned) {
      std::map<int, int> color_counts;
      for (size_t c = 0; c < n_colors; c++) {
        color_counts[static_cast<int>(c)] = 0;
      }

      for (size_t j = 0; j < i; j++) {
        if (color_assignments[j] != -1) {
          if (distance(clusters[i], clusters[j]) < neighbor_threshold) {
            color_counts[color_assignments[j]]++;
          }
        }
      }

      int min_color = 0;
      int min_count = color_counts[0];
      for (const auto & [color, count] : color_counts) {
        if (count < min_count) {
          min_count = count;
          min_color = color;
        }
      }
      color_assignments[i] = min_color;
    }
  }

  // Return color tuples
  std::vector<std::tuple<float, float, float>> result;
  for (size_t i = 0; i < n_clusters; i++) {
    result.push_back(COLORS[color_assignments[i]]);
  }

  return result;
}

}  // namespace real_cure_cpp
