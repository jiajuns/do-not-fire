/**
 * @file frontier_detector_node.cpp
 * @brief Main entry point for frontier detector node
 */

#include "real_cure_cpp/frontier_detector.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<real_cure_cpp::FrontierDetector>();

  RCLCPP_INFO(node->get_logger(), "Starting Frontier Detector (C++ CURE1)...");

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
