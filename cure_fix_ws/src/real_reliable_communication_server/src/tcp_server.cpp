#include "reliable_communication_server/tcp_server.hpp"
#include <unistd.h>
#include <arpa/inet.h>
#include <cstring>
#include <chrono>
#include <sys/select.h>

namespace reliable_communication {

TCPServer::TCPServer(rclcpp::Node* node, int port, int robot_count, bool listen_localhost, bool listen_lan)
  : node_(node), port_(port), robot_count_(robot_count),
    listen_localhost_(listen_localhost), listen_lan_(listen_lan), running_(false) {
}

TCPServer::~TCPServer() {
  stop();
}

void TCPServer::start() {
  // 获取所有网络接口
  struct ifaddrs *ifaddr, *ifa;
  if (getifaddrs(&ifaddr) == -1) {
    RCLCPP_ERROR(node_->get_logger(), "Failed to get network interfaces!");
    return;
  }

  // 收集需要绑定的 IP 地址：根据配置决定是否监听 127.0.0.1 和 192.168.*.*
  std::vector<std::string> bind_ips;

  // 根据配置添加 127.0.0.1
  if (listen_localhost_) {
    bind_ips.push_back("127.0.0.1");
    RCLCPP_INFO(node_->get_logger(), "Will listen on localhost (127.0.0.1)");
  }

  // 遍历所有接口，找到 192.168.*.* 地址
  if (listen_lan_) {
    for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) continue;
      if (ifa->ifa_addr->sa_family != AF_INET) continue;  // 只处理 IPv4

      struct sockaddr_in *addr = (struct sockaddr_in *)ifa->ifa_addr;
      char ip_str[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, &addr->sin_addr, ip_str, INET_ADDRSTRLEN);

      // 检查是否为 192.168.*.* 地址
      if (strncmp(ip_str, "192.168.", 8) == 0) {
        bind_ips.push_back(ip_str);
        RCLCPP_INFO(node_->get_logger(), "Found 192.168.x.x interface: %s (%s)",
                    ifa->ifa_name, ip_str);
      }
    }
  }
  freeifaddrs(ifaddr);

  if (bind_ips.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No IP addresses to bind! (listen_localhost=%s, listen_lan=%s)",
                 listen_localhost_ ? "true" : "false", listen_lan_ ? "true" : "false");
    return;
  }

  // 为每个 IP 地址创建并绑定 socket
  for (const auto& bind_ip : bind_ips) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
      RCLCPP_ERROR(node_->get_logger(), "Socket creation failed for %s!", bind_ip.c_str());
      continue;
    }

    // 设置 SO_REUSEADDR
    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
      RCLCPP_WARN(node_->get_logger(), "Failed to set SO_REUSEADDR for %s", bind_ip.c_str());
    }

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    inet_pton(AF_INET, bind_ip.c_str(), &addr.sin_addr);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      RCLCPP_ERROR(node_->get_logger(), "Bind failed for %s:%d! errno=%d (%s)",
                   bind_ip.c_str(), port_, errno, strerror(errno));
      close(server_fd);
      continue;
    }

    if (listen(server_fd, robot_count_) < 0) {
      RCLCPP_ERROR(node_->get_logger(), "Listen failed for %s:%d!", bind_ip.c_str(), port_);
      close(server_fd);
      continue;
    }

    server_fds_.push_back(server_fd);
    bound_ips_.push_back(bind_ip);
    RCLCPP_INFO(node_->get_logger(), "✓ TCP listening on %s:%d", bind_ip.c_str(), port_);
  }

  if (server_fds_.empty()) {
    RCLCPP_ERROR(node_->get_logger(), "No socket bound successfully!");
    return;
  }

  running_ = true;
  accept_thread_ = std::make_unique<std::thread>(&TCPServer::acceptLoop, this);

  RCLCPP_INFO(node_->get_logger(), "TCP Server started, listening on %zu interfaces, port: %d",
              server_fds_.size(), port_);
}

void TCPServer::stop() {
  running_ = false;

  // 关闭所有客户端连接
  {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    for (auto& [robot_id, client_fd] : clients_) {
      close(client_fd);
    }
    clients_.clear();
  }

  // 关闭所有服务器 socket
  for (int fd : server_fds_) {
    if (fd >= 0) {
      close(fd);
    }
  }
  server_fds_.clear();
  bound_ips_.clear();

  // 等待线程结束
  if (accept_thread_ && accept_thread_->joinable()) {
    accept_thread_->join();
  }
  for (auto& [robot_id, thread] : recv_threads_) {
    if (thread && thread->joinable()) {
      thread->join();
    }
  }
  recv_threads_.clear();

  RCLCPP_INFO(node_->get_logger(), "TCP Server stopped");
}

void TCPServer::registerMessageHandler(MessageType msg_type, MessageCallback callback) {
  message_handlers_[msg_type] = callback;
  RCLCPP_INFO(node_->get_logger(), "Registered handler for message type 0x%02X",
              static_cast<uint8_t>(msg_type));
}

void TCPServer::registerConnectionCallback(ConnectionCallback callback) {
  connection_callbacks_.push_back(callback);
}

void TCPServer::registerTelemetryCallback(TelemetryCallback callback) {
  telemetry_callbacks_.push_back(callback);
}

bool TCPServer::sendMessage(
  int robot_id,
  MessageType msg_type,
  uint64_t request_id,
  const std::vector<uint8_t>& payload
) {
  int client_fd;
  {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = clients_.find(robot_id);
    if (it == clients_.end()) {
      RCLCPP_WARN(node_->get_logger(),
                  "⚠️⚠️⚠️ [TCP WARNING] Robot %d NOT CONNECTED - Cannot send message! ⚠️⚠️⚠️",
                  robot_id);
      return false;
    }
    client_fd = it->second;
  }

  // 构造消息
  Message msg;
  msg.header.magic = MESSAGE_MAGIC;
  msg.header.msg_type = static_cast<uint8_t>(msg_type);
  msg.header.request_id = request_id;
  msg.header.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  msg.header.payload_size = payload.size();
  msg.payload = payload;

  // 序列化并发送
  auto data = msg.serialize();
  ssize_t sent = send(client_fd, data.data(), data.size(), 0);
  if (sent < 0 || static_cast<size_t>(sent) != data.size()) {
    RCLCPP_ERROR(node_->get_logger(),
                 "🚨🚨🚨 [TCP ERROR] Failed to send to robot %d (sent=%zd, expected=%zu) - Disconnecting! 🚨🚨🚨",
                 robot_id, sent, data.size());
    handleDisconnect(robot_id);
    return false;
  }

  for (auto & callback : telemetry_callbacks_) {
    callback(robot_id, msg_type, request_id, msg.header.timestamp, data.size(), true);
  }

  return true;
}

bool TCPServer::isConnected(int robot_id) const {
  std::lock_guard<std::mutex> lock(clients_mutex_);
  return clients_.find(robot_id) != clients_.end();
}

void TCPServer::acceptLoop() {
  while (running_) {
    // 使用 select 监听所有服务器 socket
    fd_set read_fds;
    FD_ZERO(&read_fds);

    int max_fd = -1;
    for (int fd : server_fds_) {
      FD_SET(fd, &read_fds);
      if (fd > max_fd) max_fd = fd;
    }

    if (max_fd < 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    // 设置超时 100ms，避免阻塞太久
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 100000;  // 100ms

    int ready = select(max_fd + 1, &read_fds, nullptr, nullptr, &tv);
    if (ready < 0) {
      if (errno != EINTR) {
        RCLCPP_ERROR(node_->get_logger(), "select() failed: %s", strerror(errno));
      }
      continue;
    }
    if (ready == 0) {
      continue;  // 超时，没有连接
    }

    // 检查哪个 socket 有连接
    for (int server_fd : server_fds_) {
      if (!FD_ISSET(server_fd, &read_fds)) continue;

      struct sockaddr_in client_addr;
      socklen_t addr_len = sizeof(client_addr);

      int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &addr_len);
      if (client_fd < 0) {
        if (running_) {
          RCLCPP_ERROR(node_->get_logger(), "Accept failed");
        }
        continue;
      }

      // 接收机器人ID (第一个字节)
      uint8_t raw_id;
      ssize_t n = recv(client_fd, &raw_id, 1, 0);
      // Parse robot_ids from node parameter
      static std::map<int,int> tcp_real_to_slot;
      static bool mapping_initialized = false;
      if (!mapping_initialized) {
        std::string ids_str;
        node_->get_parameter("robot_ids", ids_str);
        if (!ids_str.empty()) {
          std::stringstream ss(ids_str); std::string token; int s = 1;
          while (std::getline(ss, token, ',')) {
            if (token.empty()) continue; int rid = std::stoi(token);
            if (rid > 0) { tcp_real_to_slot[rid] = s++;
              RCLCPP_INFO(node_->get_logger(), "TCP mapping: car %d -> slot %d", rid, tcp_real_to_slot[rid]); }
          }
        }
        mapping_initialized = true;
      }
      int slot = raw_id;
      auto it = tcp_real_to_slot.find(raw_id);
      if (it != tcp_real_to_slot.end()) { slot = it->second; }
      uint8_t robot_id = (uint8_t)slot;
      if (n != 1) {
        RCLCPP_WARN(node_->get_logger(), "Failed to receive robot ID");
        close(client_fd);
        continue;
      }

      // 处理重连：先关闭旧 fd 并 join 旧接收线程，避免销毁一个仍 joinable 的
      // std::thread 触发 std::terminate（"terminate without an active exception"）。
      // 先从 clients_ 移除旧条目，这样旧线程退出时的 handleDisconnect 不会误删新连接。
      {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        auto old_client = clients_.find(robot_id);
        if (old_client != clients_.end()) {
          RCLCPP_WARN(node_->get_logger(), "Robot %d reconnecting, closing old connection", robot_id);
          close(old_client->second);
          clients_.erase(old_client);
        }
      }
      {
        // 在 clients_mutex_ 之外 join（旧接收线程的 handleDisconnect 会加同一把锁，
        // 持锁 join 会死锁）。旧 fd 已关闭，旧线程会尽快从 recv 返回并退出。
        auto old_thread = recv_threads_.find(robot_id);
        if (old_thread != recv_threads_.end()) {
          if (old_thread->second && old_thread->second->joinable()) {
            old_thread->second->join();
          }
          recv_threads_.erase(old_thread);
        }
      }

      // 存储新连接
      {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        clients_[robot_id] = client_fd;
      }

      char addr_str[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, &client_addr.sin_addr, addr_str, INET_ADDRSTRLEN);
      RCLCPP_INFO(node_->get_logger(), "Robot %d connected from %s:%d",
                  robot_id, addr_str, ntohs(client_addr.sin_port));

      // 设置接收超时,减少CPU占用
      struct timeval recv_timeout;
      recv_timeout.tv_sec = 1;
      recv_timeout.tv_usec = 0;
      setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &recv_timeout, sizeof(recv_timeout));

      // 通知连接状态变化
      for (auto& callback : connection_callbacks_) {
        callback(robot_id, true);
      }

      // 启动接收线程
      recv_threads_[robot_id] = std::make_unique<std::thread>(
        &TCPServer::receiveLoop, this, robot_id, client_fd
      );
    }
  }
}

void TCPServer::receiveLoop(int robot_id, int client_fd) {
  uint8_t header_buffer[HEADER_SIZE];

  while (running_) {
    // 接收消息头 (带超时处理)
    ssize_t recv_result = recvExact(client_fd, header_buffer, HEADER_SIZE);
    if (recv_result < 0) {
      // 超时不算错误,继续等待
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        continue;
      }
      // 真正的错误,退出
      break;
    }
    if (recv_result == 0) {
      // 连接关闭
      break;
    }
    if (recv_result != HEADER_SIZE) {
      break;
    }

    // 解析消息头
    MessageHeader header = MessageHeader::deserialize(header_buffer);

    // 验证魔数
    if (header.magic != MESSAGE_MAGIC) {
      RCLCPP_WARN(node_->get_logger(), "Invalid magic from robot %d", robot_id);
      break;
    }

    // 接收负载
    std::vector<uint8_t> payload(header.payload_size);
    if (header.payload_size > 0) {
      if (recvExact(client_fd, payload.data(), header.payload_size) !=
          static_cast<ssize_t>(header.payload_size)) {
        break;
      }
    }

    // 调用处理回调
    MessageType msg_type = static_cast<MessageType>(header.msg_type);
    for (auto & callback : telemetry_callbacks_) {
      callback(robot_id, msg_type, header.request_id, header.timestamp, HEADER_SIZE + payload.size(), false);
    }
    auto it = message_handlers_.find(msg_type);
    if (it != message_handlers_.end()) {
      try {
        it->second(robot_id, header.request_id, header.timestamp, payload);
      } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Handler for 0x%02X failed: %s",
                     header.msg_type, e.what());
      }
    } else {
      RCLCPP_DEBUG(node_->get_logger(), "No handler for message type 0x%02X", header.msg_type);
    }
  }

  handleDisconnect(robot_id);
}

ssize_t TCPServer::recvExact(int fd, uint8_t* buffer, size_t size) {
  size_t received = 0;
  while (received < size) {
    ssize_t n = recv(fd, buffer + received, size - received, 0);
    if (n <= 0) {
      return n;
    }
    received += n;
  }
  return received;
}

void TCPServer::handleDisconnect(int robot_id) {
  {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = clients_.find(robot_id);
    if (it != clients_.end()) {
      close(it->second);
      clients_.erase(it);
    }
  }

  RCLCPP_WARN(node_->get_logger(), "Robot %d disconnected", robot_id);

  // 通知连接状态变化
  for (auto& callback : connection_callbacks_) {
    callback(robot_id, false);
  }
}

}  // namespace reliable_communication
