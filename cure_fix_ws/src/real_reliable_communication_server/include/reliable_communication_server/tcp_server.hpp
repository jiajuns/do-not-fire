#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sys/socket.h>
#include <netinet/in.h>
#include <ifaddrs.h>
#include <string>
#include <map>
#include <memory>
#include <thread>
#include <mutex>
#include <functional>
#include <atomic>
#include <vector>

#include "reliable_communication_server/message_protocol.hpp"

namespace reliable_communication {

/**
 * @brief TCP服务器基础类
 *
 * 提供TCP连接管理、消息发送/接收、可扩展的消息处理框架
 */
class TCPServer {
public:
  using MessageCallback = std::function<void(
    int robot_id,
    uint64_t request_id,
    uint64_t timestamp,
    const std::vector<uint8_t>& payload
  )>;

  using ConnectionCallback = std::function<void(int robot_id, bool connected)>;
  using TelemetryCallback = std::function<void(
    int robot_id,
    MessageType msg_type,
    uint64_t request_id,
    uint64_t timestamp,
    size_t bytes,
    bool is_send
  )>;

  /**
   * @brief 构造函数
   * @param node ROS节点指针
   * @param port TCP端口
   * @param robot_count 机器人数量
   * @param listen_localhost 是否监听 127.0.0.1
   * @param listen_lan 是否监听 192.168.*.*
   */
  TCPServer(
    rclcpp::Node* node,
    int port,
    int robot_count,
    bool listen_localhost = true,
    bool listen_lan = true
  );

  ~TCPServer();

  /**
   * @brief 启动TCP服务器
   */
  void start();

  /**
   * @brief 停止TCP服务器
   */
  void stop();

  /**
   * @brief 注册消息处理回调
   * @param msg_type 消息类型
   * @param callback 回调函数
   */
  void registerMessageHandler(MessageType msg_type, MessageCallback callback);

  /**
   * @brief 注册连接状态变化回调
   * @param callback 回调函数
   */
  void registerConnectionCallback(ConnectionCallback callback);

  /**
   * @brief 注册 TCP 收发测量回调（供 server-side QoS/通信模拟使用）
   */
  void registerTelemetryCallback(TelemetryCallback callback);

  /**
   * @brief 发送消息给指定机器人
   * @param robot_id 机器人ID
   * @param msg_type 消息类型
   * @param request_id 请求ID
   * @param payload 负载数据
   * @return 是否发送成功
   */
  bool sendMessage(
    int robot_id,
    MessageType msg_type,
    uint64_t request_id,
    const std::vector<uint8_t>& payload
  );

  /**
   * @brief 检查机器人是否已连接
   * @param robot_id 机器人ID
   * @return 是否已连接
   */
  bool isConnected(int robot_id) const;

private:
  void acceptLoop();
  void receiveLoop(int robot_id, int client_fd);
  ssize_t recvExact(int fd, uint8_t* buffer, size_t size);
  void handleDisconnect(int robot_id);

  rclcpp::Node* node_;
  int port_;
  int robot_count_;

  // 多 socket 支持：监听 127.0.0.1 和 192.168.*.* 网络
  std::vector<int> server_fds_;
  std::vector<std::string> bound_ips_;
  bool listen_localhost_;  // 是否监听 127.0.0.1
  bool listen_lan_;        // 是否监听 192.168.*.*
  std::map<int, int> clients_;  // robot_id -> client_fd
  mutable std::mutex clients_mutex_;

  std::map<MessageType, MessageCallback> message_handlers_;
  std::vector<ConnectionCallback> connection_callbacks_;
  std::vector<TelemetryCallback> telemetry_callbacks_;

  std::atomic<bool> running_;
  std::unique_ptr<std::thread> accept_thread_;
  std::map<int, std::unique_ptr<std::thread>> recv_threads_;
};

}  // namespace reliable_communication
