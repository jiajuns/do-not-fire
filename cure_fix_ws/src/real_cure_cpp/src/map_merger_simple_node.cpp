/**
 * @file map_merger_simple_node.cpp
 * @brief Standalone node for simplified map merger
 */

#include "real_cure_cpp/map_merger_simple.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  
  auto node = std::make_shared<real_cure_cpp::MapMergerSimple>();
  
  rclcpp::spin(node);
  rclcpp::shutdown();
  
  return 0;
}
