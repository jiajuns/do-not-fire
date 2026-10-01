#include <rclcpp/rclcpp.hpp>
#include "reliable_communication_server/reliable_server_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  auto node = std::make_shared<reliable_communication::ReliableServerNode>(options);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
