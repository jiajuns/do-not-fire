#pragma once

#include <rclcpp/rclcpp.hpp>
#include <communication_simulation/msg/communication_qo_s.hpp>
#include <map>
#include <mutex>
#include <chrono>
#include <deque>

namespace reliable_communication {

/**
 * @brief 通信质量监控器
 *
 * 监测延迟、丢包率、带宽并发布QoS消息
 */
class QoSMonitor {
public:
  QoSMonitor(rclcpp::Node* node, int robot_count);

  /**
   * @brief 记录发送的请求
   * @param robot_id 机器人ID
   * @param request_id 请求ID
   */
  void recordRequest(int robot_id, uint64_t request_id);

  /**
   * @brief 记录接收的响应
   * @param robot_id 机器人ID
   * @param request_id 请求ID
   * @param send_timestamp 发送时间戳
   */
  void recordResponse(int robot_id, uint64_t request_id, uint64_t send_timestamp);

  /**
   * @brief 记录数据传输
   * @param robot_id 机器人ID
   * @param bytes 字节数
   * @param is_send 是否为发送
   */
  void recordDataTransfer(int robot_id, size_t bytes, bool is_send);

  /**
   * @brief 更新连接状态
   * @param robot_id 机器人ID
   * @param connected 是否已连接
   */
  void updateConnectionStatus(int robot_id, bool connected);

private:
  void publishTimerCallback();

  struct QoSMetrics {
    // 延迟统计
    std::deque<double> latency_samples;  // 毫秒
    double avg_latency_ms = 0.0;

    // 丢包统计
    uint64_t total_requests = 0;
    uint64_t timeout_requests = 0;
    double packet_loss_percent = 0.0;

    // 带宽统计
    std::deque<std::pair<std::chrono::steady_clock::time_point, size_t>> bandwidth_history;
    double bandwidth_mbps = 0.0;

    // 连接状态
    uint8_t connection_status = 0;  // 0=断开, 2=已连接

    // 待响应的请求
    std::map<uint64_t, uint64_t> pending_requests;  // request_id -> send_timestamp
  };

  rclcpp::Node* node_;
  int robot_count_;

  std::map<int, QoSMetrics> metrics_;
  std::mutex metrics_mutex_;

  std::map<int, rclcpp::Publisher<communication_simulation::msg::CommunicationQoS>::SharedPtr> qos_publishers_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
};

}  // namespace reliable_communication
