#include "reliable_communication_server/qos_monitor.hpp"
#include <algorithm>
#include <numeric>

namespace reliable_communication {

constexpr size_t MAX_LATENCY_SAMPLES = 100;
constexpr size_t BANDWIDTH_WINDOW_SECONDS = 10;
constexpr double REQUEST_TIMEOUT_SECONDS = 5.0;

QoSMonitor::QoSMonitor(rclcpp::Node* node, int robot_count)
  : node_(node), robot_count_(robot_count) {

  // 创建发布器
  // Keep the canonical QoS topic aligned with SenseSimulation and the Lite semantic contract.
  for (int i = 1; i <= robot_count_; ++i) {
    std::string topic = "/robot_" + std::to_string(i) + "/communication_state";
    qos_publishers_[i] = node_->create_publisher<communication_simulation::msg::CommunicationQoS>(
      topic, 10
    );

    // 初始化metrics
    metrics_[i] = QoSMetrics();
  }

  // 创建定时发布器 (1 Hz)
  publish_timer_ = node_->create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&QoSMonitor::publishTimerCallback, this)
  );

  RCLCPP_INFO(node_->get_logger(), "QoSMonitor initialized for %d robots", robot_count_);
}

void QoSMonitor::recordRequest(int robot_id, uint64_t request_id) {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  auto& m = metrics_[robot_id];

  uint64_t now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
  m.pending_requests[request_id] = now_ns;
  m.total_requests++;
}

void QoSMonitor::recordResponse(int robot_id, uint64_t request_id, uint64_t send_timestamp) {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  auto& m = metrics_[robot_id];

  // 计算延迟
  uint64_t now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
  double latency_ms = (now_ns - send_timestamp) / 1e6;  // ns to ms

  m.latency_samples.push_back(latency_ms);
  if (m.latency_samples.size() > MAX_LATENCY_SAMPLES) {
    m.latency_samples.pop_front();
  }

  // 更新平均延迟
  m.avg_latency_ms = std::accumulate(
    m.latency_samples.begin(),
    m.latency_samples.end(),
    0.0
  ) / m.latency_samples.size();

  // 从待响应列表中移除
  m.pending_requests.erase(request_id);
}

void QoSMonitor::recordDataTransfer(int robot_id, size_t bytes, bool is_send) {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  auto& m = metrics_[robot_id];

  auto now = std::chrono::steady_clock::now();
  m.bandwidth_history.push_back({now, bytes});

  // 清理超过窗口时间的数据
  auto cutoff = now - std::chrono::seconds(BANDWIDTH_WINDOW_SECONDS);
  while (!m.bandwidth_history.empty() &&
         m.bandwidth_history.front().first < cutoff) {
    m.bandwidth_history.pop_front();
  }

  // 计算带宽 (Mbps)
  if (!m.bandwidth_history.empty()) {
    size_t total_bytes = 0;
    for (const auto& [time, size] : m.bandwidth_history) {
      total_bytes += size;
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      now - m.bandwidth_history.front().first
    ).count();

    if (duration > 0) {
      m.bandwidth_mbps = (total_bytes * 8.0) / (duration / 1000.0) / 1e6;
    }
  }
}

void QoSMonitor::updateConnectionStatus(int robot_id, bool connected) {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  auto& m = metrics_[robot_id];
  m.connection_status = connected ? 2 : 0;  // 0=断开, 2=已连接
}

void QoSMonitor::publishTimerCallback() {
  auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();

  std::lock_guard<std::mutex> lock(metrics_mutex_);

  for (int robot_id = 1; robot_id <= robot_count_; ++robot_id) {
    auto& m = metrics_[robot_id];

    // 检查超时的请求
    std::vector<uint64_t> timeout_requests;
    for (const auto& [request_id, send_time] : m.pending_requests) {
      double elapsed_s = (now_ns - send_time) / 1e9;
      if (elapsed_s > REQUEST_TIMEOUT_SECONDS) {
        timeout_requests.push_back(request_id);
      }
    }

    // 移除超时请求并更新统计
    for (auto request_id : timeout_requests) {
      m.pending_requests.erase(request_id);
      m.timeout_requests++;
    }

    // 计算丢包率
    if (m.total_requests > 0) {
      m.packet_loss_percent = 100.0 * m.timeout_requests / m.total_requests;
    }

    // 发布QoS消息
    communication_simulation::msg::CommunicationQoS qos_msg;
    qos_msg.latency_ms = static_cast<float>(m.avg_latency_ms);
    qos_msg.packet_loss_percent = static_cast<float>(m.packet_loss_percent);
    qos_msg.bandwidth_mbps = static_cast<float>(m.bandwidth_mbps);
    qos_msg.connection_status = m.connection_status;

    qos_publishers_[robot_id]->publish(qos_msg);
  }
}

}  // namespace reliable_communication
