/**
 * @file frontier_detector.hpp
 * @brief ROS 2 node for detecting frontiers using CURE1 Canny-based algorithm
 *
 * Implements the CURE1 frontier detection algorithm:
 * 1. Multi-threshold processing for obstacle/free/unknown classification
 * 2. Canny edge detection for boundary identification
 * 3. Connected component analysis for cluster extraction
 * 4. Information gain calculation for each cluster
 */

#ifndef CURE_EXPLORE_CPP__FRONTIER_DETECTOR_HPP_
#define CURE_EXPLORE_CPP__FRONTIER_DETECTOR_HPP_

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <cure_explore_interfaces/msg/frontier_cluster.hpp>
#include <cure_explore_interfaces/msg/frontier_array.hpp>

#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <tuple>
#include <set>
#include <queue>
#include <cmath>

namespace real_cure_cpp
{

class FrontierDetector : public rclcpp::Node
{
public:
  explicit FrontierDetector(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~FrontierDetector() override = default;

private:
  // Callbacks
  void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg);
  void resetCallback(const std_msgs::msg::Empty::SharedPtr msg);
  void timerCallback();

  // Core detection algorithm (CURE1)
  void detectAndPublishFrontiersCure1();

  // Image conversion
  cv::Mat occupancyToImage(const std::vector<int8_t> & map_data, int height, int width);

  // Validation
  bool isValidFrontierCentroid(double cx, double cy,
                               const std::vector<int8_t> & map_data,
                               int width, int height);

  // Reachability filter (flood fill based)
  std::vector<bool> computeReachableFreeSpace(
    const std::vector<int8_t> & map_data, int width, int height);
  bool isFrontierAdjacentToReachable(
    double cx, double cy,
    const std::vector<bool> & reachable_mask,
    int width, int height);

  // Information gain calculation
  double calculateInformationGainAtCentroid(double cx, double cy,
                                            const std::vector<int8_t> & map_data,
                                            int width, int height);

  // Coordinate transforms
  std::pair<double, double> gridToWorld(int gx, int gy);
  std::pair<int, int> worldToGrid(double wx, double wy);

  // Publishing
  void publishFrontiers(
    const std::vector<cure_explore_interfaces::msg::FrontierCluster> & clusters,
    int total_points);
  void publishClusters(const std::vector<geometry_msgs::msg::Point> & centroids);
  void publishFrontierPointsVisualization(
    const std::vector<cure_explore_interfaces::msg::FrontierCluster> & clusters);

  // Color assignment for visualization
  std::vector<std::tuple<float, float, float>> assignColorsToClusters(
    const std::vector<cure_explore_interfaces::msg::FrontierCluster> & clusters);

  // Parameters
  std::string global_map_topic_;
  std::string frontier_topic_;
  std::string cluster_vis_topic_;
  double update_rate_;
  int min_frontier_size_;
  double info_radius_;
  int canny_low_;
  int canny_high_;
  int dilation_radius_;
  bool use_reachability_filter_;  // Enable flood-fill based reachability filter

  // State
  std::mutex map_mutex_;
  nav_msgs::msg::OccupancyGrid::SharedPtr current_map_;
  bool map_received_;

  // Predefined colors for visualization
  static const std::vector<std::tuple<float, float, float>> COLORS;

  // ROS 2 interfaces
  rclcpp::Publisher<cure_explore_interfaces::msg::FrontierArray>::SharedPtr frontier_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_vis_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr frontier_points_vis_pub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace real_cure_cpp

#endif  // CURE_EXPLORE_CPP__FRONTIER_DETECTOR_HPP_
