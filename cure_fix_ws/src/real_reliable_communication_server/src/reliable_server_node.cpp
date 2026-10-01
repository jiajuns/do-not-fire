#include "reliable_communication_server/reliable_server_node.hpp"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>

namespace reliable_communication {

ReliableServerNode::ReliableServerNode(const rclcpp::NodeOptions& options)
  : Node("reliable_communication_server", options) {

  // 声明参数
  this->declare_parameter<int>("robot_count", 2);
  this->declare_parameter<std::string>("robot_ids", "");
  this->declare_parameter<int>("tcp_port", 12140);
  this->declare_parameter<bool>("listen_localhost", true);  // 是否监听 127.0.0.1
  this->declare_parameter<bool>("listen_lan", false);         // 是否监听 192.168.*.*

  // 获取参数
  robot_count_ = this->get_parameter("robot_count").as_int();
  tcp_port_ = this->get_parameter("tcp_port").as_int();
  bool listen_localhost = this->get_parameter("listen_localhost").as_bool();
  bool listen_lan = this->get_parameter("listen_lan").as_bool();

  RCLCPP_INFO(this->get_logger(),
              "Initializing Reliable Communication Server: robot_count=%d, port=%d, localhost=%s, lan=%s",
              robot_count_, tcp_port_,
              listen_localhost ? "true" : "false",
              listen_lan ? "true" : "false");

  // 初始化TF2
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  // 注意：必须传入 this 节点指针，TransformListener 才能订阅 /tf 和 /tf_static 话题
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this);
  RCLCPP_INFO(this->get_logger(), "TF2 initialized for coordinate transformation");

  // 创建TCP服务器
  tcp_server_ = std::make_unique<TCPServer>(this, tcp_port_, robot_count_, listen_localhost, listen_lan);

  // 创建QoS监控器
  qos_monitor_ = std::make_unique<QoSMonitor>(this, robot_count_);

  // 创建Action管理器
  action_manager_ = std::make_unique<ActionManager>(this, tcp_server_.get());

  // 设置TF buffer
  action_manager_->setTfBuffer(tf_buffer_);

  // 注册连接状态回调
  tcp_server_->registerConnectionCallback(
    std::bind(&ReliableServerNode::onConnectionChange, this,
              std::placeholders::_1, std::placeholders::_2)
  );

  // 注册 TCP 级通信测量回调：server-side 统一统计带宽、连接状态和请求-响应延迟。
  tcp_server_->registerTelemetryCallback(
    [this](int robot_id, MessageType msg_type, uint64_t request_id, uint64_t timestamp, size_t bytes, bool is_send) {
      qos_monitor_->recordDataTransfer(robot_id, bytes, is_send);
      if (is_send) {
        switch (msg_type) {
          case MessageType::ACTION_GOAL:
          case MessageType::ACTION_CANCEL:
          case MessageType::PATH_PLANNING_RESPONSE:
            qos_monitor_->recordRequest(robot_id, request_id);
            break;
          default:
            break;
        }
      } else {
        switch (msg_type) {
          case MessageType::ACTION_FEEDBACK:
          case MessageType::ACTION_RESULT:
          case MessageType::PATH_PLANNING_REQUEST:
            qos_monitor_->recordResponse(robot_id, request_id, timestamp);
            break;
          default:
            break;
        }
      }
    }
  );

  // 设置TCP Action回调 - 转发到ROS Action
  action_manager_->setFeedbackCallback(
    std::bind(&ReliableServerNode::onGoalFeedback, this,
              std::placeholders::_1, std::placeholders::_2)
  );
  action_manager_->setResultCallback(
    std::bind(&ReliableServerNode::onGoalResult, this,
              std::placeholders::_1, std::placeholders::_2)
  );

  // 启动TCP服务器
  tcp_server_->start();

  // 为每个机器人创建ROS Action Server
  for (int i = 1; i <= robot_count_; ++i) {
    std::string action_name = "/robot_" + std::to_string(i) + "/navigate_to_goal";

    action_servers_[i] = rclcpp_action::create_server<NavigateToGoal>(
      this,
      action_name,
      [this, i](const rclcpp_action::GoalUUID& uuid,
                std::shared_ptr<const NavigateToGoal::Goal> goal) {
        return this->handleGoal(uuid, goal, i);
      },
      [this, i](const std::shared_ptr<GoalHandleNavigate> goal_handle) {
        return this->handleCancel(goal_handle, i);
      },
      [this, i](const std::shared_ptr<GoalHandleNavigate> goal_handle) {
        this->handleAccepted(goal_handle, i);
      }
    );

    RCLCPP_INFO(this->get_logger(), "✅ Created Action Server: %s", action_name.c_str());
  }

  // 创建目标点订阅器（用于测试/兼容旧接口）
  for (int i = 1; i <= robot_count_; ++i) {
    std::string topic = "/robot_" + std::to_string(i) + "/goal_pose";
    goal_subscribers_[i] = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      topic, 10,
      [this, i](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        this->onGoalPose(i, msg);
      }
    );
    RCLCPP_INFO(this->get_logger(), "Subscribed to %s for sending goals (legacy)", topic.c_str());
  }

  // 订阅全局地图
  global_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
    "/global_map", 10,
    std::bind(&ReliableServerNode::onGlobalMap, this, std::placeholders::_1)
  );

  // 订阅各机器人的局部地图
  for (int i = 1; i <= robot_count_; ++i) {
    std::string topic = "/robot_" + std::to_string(i) + "/map";
    local_map_subscribers_[i] = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      topic, rclcpp::QoS(1).transient_local(),
      [this, i](const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        this->onRobotLocalMap(i, msg);
      }
    );
  }

  // 创建系统重置订阅者和发布者
  reset_sub_ = this->create_subscription<cure_explore_interfaces::msg::ResetCommand>(
    "/cure_explore/reset_command", 10,
    std::bind(&ReliableServerNode::onResetCommand, this, std::placeholders::_1)
  );
  RCLCPP_INFO(this->get_logger(), "Subscribed to /cure_explore/reset_command");

  reset_ack_pub_ = this->create_publisher<cure_explore_interfaces::msg::ResetAck>(
    "/system_reset_ack", 10
  );
  RCLCPP_INFO(this->get_logger(), "Publishing to /system_reset_ack");

  // 创建目标点可视化发布器
  goal_vis_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/rl_goal_visualization", 10
  );
  RCLCPP_INFO(this->get_logger(), "Publishing goal visualization to /rl_goal_visualization");

  // 创建路径规划可视化发布器
  path_vis_pub_ = this->create_publisher<nav_msgs::msg::Path>(
    "/planned_path", 10
  );
  path_marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
    "/planned_path_visualization", 10
  );
  RCLCPP_INFO(this->get_logger(), "Publishing path visualization to /planned_path and /planned_path_visualization");

  // 创建目标状态发布器（用于通知RL立即cancel等情况）
  for (int i = 1; i <= robot_count_; ++i) {
    std::string topic = "/robot_" + std::to_string(i) + "/goal_immediate_status";
    goal_status_pubs_[i] = this->create_publisher<std_msgs::msg::Int8>(topic, 10);
    RCLCPP_INFO(this->get_logger(), "Publishing goal status to %s", topic.c_str());
  }

  // 设置路径可视化发布器到ActionManager
  action_manager_->setPathVisualizationPublishers(path_vis_pub_, path_marker_pub_);

  // 创建定时器，定期发布可视化（1Hz）
  vis_timer_ = this->create_wall_timer(
    std::chrono::seconds(1),
    [this]() {
      if (current_goals_.empty()) {
        return;
      }

      visualization_msgs::msg::MarkerArray marker_array;
      auto stamp = this->get_clock()->now();

      // 添加当前目标marker（ADD会自动替换同ID的旧marker，无需DELETE）
      for (const auto& [robot_id, goal] : current_goals_) {
        // Marker 1: 球体标记目标点
        visualization_msgs::msg::Marker sphere_marker;
        sphere_marker.header.stamp = stamp;
        sphere_marker.header.frame_id = goal.header.frame_id;
        sphere_marker.ns = "rl_goals";
        sphere_marker.id = robot_id;
        sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;
        sphere_marker.action = visualization_msgs::msg::Marker::ADD;
        sphere_marker.pose = goal.pose;
        sphere_marker.scale.x = 0.3;
        sphere_marker.scale.y = 0.3;
        sphere_marker.scale.z = 0.3;

        // 颜色：使用HSV生成不同颜色
        float h = (robot_id - 1) / static_cast<float>(robot_count_);
        float r, g, b;
        // HSV to RGB (简化版)
        int hi = static_cast<int>(h * 6.0f);
        float f = h * 6.0f - hi;
        float q = 1.0f - f;
        switch (hi % 6) {
          case 0: r = 1.0f; g = f; b = 0.0f; break;
          case 1: r = q; g = 1.0f; b = 0.0f; break;
          case 2: r = 0.0f; g = 1.0f; b = f; break;
          case 3: r = 0.0f; g = q; b = 1.0f; break;
          case 4: r = f; g = 0.0f; b = 1.0f; break;
          case 5: r = 1.0f; g = 0.0f; b = q; break;
          default: r = 1.0f; g = 1.0f; b = 1.0f; break;
        }

        sphere_marker.color.r = r;
        sphere_marker.color.g = g;
        sphere_marker.color.b = b;
        sphere_marker.color.a = 0.8;

        marker_array.markers.push_back(sphere_marker);

        // Marker 2: 文本标签显示robot ID
        visualization_msgs::msg::Marker text_marker;
        text_marker.header = sphere_marker.header;
        text_marker.ns = "rl_goals_text";
        text_marker.id = robot_id;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.pose = goal.pose;
        text_marker.pose.position.z += 0.5;  // 文本在球体上方
        text_marker.scale.z = 0.2;  // 文本高度
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        text_marker.text = "Robot " + std::to_string(robot_id);

        marker_array.markers.push_back(text_marker);
      }

      goal_vis_pub_->publish(marker_array);
    }
  );

  // 为每个机器人创建工作线程
  initRobotWorkers();

  // 创建健康检查定时器（5秒检查一次）
  health_check_timer_ = this->create_wall_timer(
    std::chrono::seconds(5),
    std::bind(&ReliableServerNode::checkWorkerHealth, this)
  );

  // 创建TF缓存更新定时器（50ms更新一次，20Hz）
  tf_update_timer_ = this->create_wall_timer(
    std::chrono::milliseconds(50),
    std::bind(&ReliableServerNode::updateTFCache, this)
  );
  RCLCPP_INFO(this->get_logger(), "TF cache update timer created (20Hz)");

  RCLCPP_INFO(this->get_logger(), "Reliable Communication Server initialized successfully");
}

void ReliableServerNode::initRobotWorkers() {
  // 先完成所有 worker 的注册，再启动线程。若边插入 map 边启动线程，
  // 新线程可能在对应元素写入前执行 robotWorkerLoop() 并解引用空指针。
  for (int i = 1; i <= robot_count_; ++i) {
    auto worker = std::make_unique<RobotWorker>();
    worker->running = true;
    robot_workers_[i] = std::move(worker);
  }

  for (int i = 1; i <= robot_count_; ++i) {
    int robot_id = i;
    robot_workers_.at(i)->thread = std::thread([this, robot_id]() {
      robotWorkerLoop(robot_id);
    });
    RCLCPP_INFO(this->get_logger(), "Created worker thread for robot %d", i);
  }
}

void ReliableServerNode::robotWorkerLoop(int robot_id) {
  auto& worker = robot_workers_[robot_id];

  while (true) {
    std::function<void()> task;

    {
      std::unique_lock<std::mutex> lock(worker->mutex);
      worker->cv.wait(lock, [&]() {
        return !worker->running || !worker->tasks.empty();
      });

      if (!worker->running && worker->tasks.empty()) {
        break;
      }

      if (!worker->tasks.empty()) {
        task = std::move(worker->tasks.front());
        worker->tasks.pop();
      }
    }

    if (task) {
      // 记录任务开始时间，用于超时检测
      worker->task_executing.store(true);
      worker->task_start_time = std::chrono::steady_clock::now();
      
      try {
        task();
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), 
                    "Robot %d worker thread caught exception: %s", 
                    robot_id, e.what());
      } catch (...) {
        RCLCPP_ERROR(this->get_logger(), 
                    "Robot %d worker thread caught unknown exception", 
                    robot_id);
      }
      
      worker->task_executing.store(false);
    }
  }

  RCLCPP_DEBUG(this->get_logger(), "Worker thread for robot %d exiting", robot_id);
}

void ReliableServerNode::submitTask(int robot_id, std::function<void()> task) {
  auto it = robot_workers_.find(robot_id);
  if (it == robot_workers_.end()) {
    RCLCPP_ERROR(this->get_logger(), "No worker thread for robot %d", robot_id);
    return;
  }

  auto& worker = it->second;
  {
    std::lock_guard<std::mutex> lock(worker->mutex);
    // 清空队列，只保留最新任务（机器人同一时刻只去一个目标）
    while (!worker->tasks.empty()) {
      worker->tasks.pop();
    }
    worker->tasks.push(std::move(task));
  }
  worker->cv.notify_one();
}
void ReliableServerNode::checkWorkerHealth() {
  const auto now = std::chrono::steady_clock::now();
  const auto TASK_TIMEOUT = std::chrono::seconds(10);  // 10秒超时

  for (auto& [robot_id, worker] : robot_workers_) {
    if (worker->task_executing.load()) {
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - worker->task_start_time
      );
      
      if (elapsed > TASK_TIMEOUT) {
        RCLCPP_ERROR(this->get_logger(),
                    "⚠️⚠️⚠️ Robot %d worker thread stuck for %ld seconds! "
                    "Clearing goal handle to allow recovery. ⚠️⚠️⚠️",
                    robot_id, elapsed.count());
        
        // 清除卡住的goal handle，允许新请求进入
        {
          std::lock_guard<std::mutex> lock(goal_handles_mutex_);
          auto it = active_goal_handles_.find(robot_id);
          if (it != active_goal_handles_.end() && it->second) {
            auto result = std::make_shared<NavigateToGoal::Result>();
            result->success = false;
            result->message = "Task timeout - worker thread stuck";
            result->final_distance = -1.0;
            result->elapsed_time_seconds = elapsed.count();
            
            try {
              it->second->abort(result);
            } catch (...) {
              // 忽略异常，确保清理继续
            }
            
            active_goal_handles_.erase(it);
            active_request_ids_.erase(robot_id);
            
            RCLCPP_WARN(this->get_logger(),
                       "🔧 Cleared stuck goal handle for robot %d, system can recover now",
                       robot_id);
          }
        }
      }
    }
  }
}
ReliableServerNode::~ReliableServerNode() {
  // 设置关闭标志
  shutdown_flag_.store(true);

  // 通知所有工作线程停止
  for (auto& [robot_id, worker] : robot_workers_) {
    {
      std::lock_guard<std::mutex> lock(worker->mutex);
      worker->running = false;
    }
    worker->cv.notify_one();
  }

  // 等待所有工作线程结束
  for (auto& [robot_id, worker] : robot_workers_) {
    if (worker->thread.joinable()) {
      worker->thread.join();
    }
  }
  robot_workers_.clear();

  if (tcp_server_) {
    tcp_server_->stop();
  }
  RCLCPP_INFO(this->get_logger(), "Reliable Communication Server shutdown");
}

// ===== 连接状态回调 =====
void ReliableServerNode::onConnectionChange(int robot_id, bool connected) {
  if (connected) {
    RCLCPP_INFO(this->get_logger(), "✅ Robot %d connected", robot_id);
  } else {
    RCLCPP_WARN(this->get_logger(),
                "⚠️⚠️⚠️ [DISCONNECT] Robot %d DISCONNECTED from TCP server! ⚠️⚠️⚠️",
                robot_id);

    // 如果有活跃的Goal，标记为失败
    std::lock_guard<std::mutex> lock(goal_handles_mutex_);
    auto it = active_goal_handles_.find(robot_id);
    if (it != active_goal_handles_.end() && it->second) {
      auto result = std::make_shared<NavigateToGoal::Result>();
      result->success = false;
      result->message = "Robot disconnected";
      result->final_distance = -1.0;
      result->elapsed_time_seconds = 0.0;

      it->second->abort(result);
      active_goal_handles_.erase(it);
      active_request_ids_.erase(robot_id);
      RCLCPP_WARN(this->get_logger(),
                  "⚠️ Aborted active goal for robot %d due to disconnect",
                  robot_id);
    }
  }

  // 更新QoS监控器
  qos_monitor_->updateConnectionStatus(robot_id, connected);
}

// ===== TCP反馈回调 - 转发到ROS Action =====
void ReliableServerNode::onGoalFeedback(
  int robot_id,
  const NavigateToGoal::Feedback& feedback
) {
  // 改用INFO级别,便于调试
  RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "📊 Robot %d feedback: distance=%.2fm, progress=%.1f%%, status=%d",
                        robot_id, feedback.distance_remaining, feedback.progress_percent,
                        feedback.nav_status);

  // 转发反馈到ROS Action
  std::lock_guard<std::mutex> lock(goal_handles_mutex_);
  auto it = active_goal_handles_.find(robot_id);
  if (it != active_goal_handles_.end() && it->second) {
    auto feedback_ptr = std::make_shared<NavigateToGoal::Feedback>(feedback);
    it->second->publish_feedback(feedback_ptr);
    RCLCPP_DEBUG(this->get_logger(), "✅ Forwarded feedback to Action Client for robot %d", robot_id);
  } else {
    // 正常情况：goal已完成，但反馈还在传输中（时序问题）
    RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Received feedback for robot %d but goal already completed (normal race condition)", robot_id);
  }
}

// ===== TCP结果回调 - 转发到ROS Action =====
void ReliableServerNode::onGoalResult(
  int robot_id,
  const NavigateToGoal::Result& result
) {
  // 检测立即cancel: 时间很短且失败（通常意味着目标不可达或规划失败）
  const float IMMEDIATE_CANCEL_THRESHOLD = 2.0;  // 2秒内失败认为是立即cancel
  bool is_immediate_cancel = !result.success && result.elapsed_time_seconds < IMMEDIATE_CANCEL_THRESHOLD;

  if (result.success) {
    RCLCPP_INFO(this->get_logger(),
                "✅ Robot %d completed goal: time=%.2fs, final_distance=%.2fm",
                robot_id, result.elapsed_time_seconds, result.final_distance);
  } else if (is_immediate_cancel) {
    RCLCPP_WARN(this->get_logger(),
                "⚡ Robot %d: IMMEDIATE CANCEL detected! Goal unreachable. "
                "time=%.2fs, msg='%s'",
                robot_id, result.elapsed_time_seconds, result.message.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(),
                "❌ Robot %d failed goal: %s",
                robot_id, result.message.c_str());
  }

  // 发布目标状态给RL
  // 状态值: -1=立即cancel(不可达), 0=失败, 1=成功
  auto status_it = goal_status_pubs_.find(robot_id);
  if (status_it != goal_status_pubs_.end() && status_it->second) {
    auto status_msg = std_msgs::msg::Int8();
    if (result.success) {
      status_msg.data = 1;  // 成功
    } else if (is_immediate_cancel) {
      status_msg.data = -1;  // 立即cancel（不可达）
    } else {
      status_msg.data = 0;  // 普通失败
    }
    status_it->second->publish(status_msg);
    RCLCPP_DEBUG(this->get_logger(),
                 "Published goal status %d for robot %d",
                 status_msg.data, robot_id);
  }

  // 转发结果到ROS Action
  std::lock_guard<std::mutex> lock(goal_handles_mutex_);
  auto it = active_goal_handles_.find(robot_id);
  if (it != active_goal_handles_.end() && it->second) {
    auto result_ptr = std::make_shared<NavigateToGoal::Result>(result);
    if (result.success) {
      it->second->succeed(result_ptr);
      RCLCPP_INFO(this->get_logger(), "✅ Action result forwarded to client for robot %d", robot_id);
    } else {
      it->second->abort(result_ptr);
      RCLCPP_INFO(this->get_logger(), "❌ Action aborted and forwarded to client for robot %d", robot_id);
    }
    active_goal_handles_.erase(it);
    active_request_ids_.erase(robot_id);
  } else {
    // 正常情况：旧goal已经被preempt并abort，这里收到的是机器人端的取消结果
    RCLCPP_DEBUG(this->get_logger(),
                 "Received result for robot %d but no active goal_handle "
                 "(likely already preempted by new goal or completed)",
                 robot_id);
  }
}

// ===== 话题订阅回调（用于测试/兼容） =====
void ReliableServerNode::onGoalPose(
  int robot_id,
  const geometry_msgs::msg::PoseStamped::SharedPtr msg
) {
  RCLCPP_INFO(this->get_logger(),
              "📍 [Legacy] Received goal for robot %d: (%.2f, %.2f, %.2f)",
              robot_id, msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);

  // 通过TCP发送目标给机器人（使用去重）
  float tolerance = 0.5;  // 默认容差
  uint64_t request_id = action_manager_->sendGoalWithDeduplication(robot_id, *msg, tolerance);

  if (request_id > 0) {
    RCLCPP_INFO(this->get_logger(),
                "🚀 Sent goal to robot %d via TCP [request_id=%lu]",
                robot_id, request_id);

    // 更新目标缓存用于可视化 - 添加robot_id前缀到frame_id
    geometry_msgs::msg::PoseStamped goal_with_prefix = *msg;
    std::string prefix = "robot_" + std::to_string(robot_id) + "/";
    if (goal_with_prefix.header.frame_id.rfind(prefix, 0) != 0) {
      goal_with_prefix.header.frame_id = prefix + goal_with_prefix.header.frame_id;
    }
    current_goals_[robot_id] = goal_with_prefix;
  } else {
    RCLCPP_ERROR(this->get_logger(),
                 "❌ Failed to send goal to robot %d (not connected or send failed)",
                 robot_id);
  }
}

// ===== ROS Action Server 回调 =====
rclcpp_action::GoalResponse ReliableServerNode::handleGoal(
  const rclcpp_action::GoalUUID& uuid,
  std::shared_ptr<const NavigateToGoal::Goal> goal,
  int robot_id
) {
  (void)uuid;  // 未使用

  auto start_time = std::chrono::steady_clock::now();

  RCLCPP_DEBUG(this->get_logger(),
               "🎯 [Action] handleGoal START for robot %d: (%.2f, %.2f)",
               robot_id, goal->target_pose.pose.position.x, goal->target_pose.pose.position.y);

  // 检查机器人是否已连接
  bool connected = tcp_server_->isConnected(robot_id);

  auto check_time = std::chrono::steady_clock::now();
  auto check_ms = std::chrono::duration_cast<std::chrono::milliseconds>(check_time - start_time).count();

  if (!connected) {
    RCLCPP_ERROR(this->get_logger(),
                 "🚨🚨🚨 [ACTION SERVER ERROR] Robot %d NOT CONNECTED via TCP! "
                 "Rejecting goal. Check if robot's reliable_client is running! (check took %ldms) 🚨🚨🚨",
                 robot_id, check_ms);
    return rclcpp_action::GoalResponse::REJECT;
  }

  RCLCPP_DEBUG(this->get_logger(),
               "🎯 [Action] handleGoal ACCEPT for robot %d (took %ldms)",
               robot_id, check_ms);

  // 接受目标
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse ReliableServerNode::handleCancel(
  const std::shared_ptr<GoalHandleNavigate> goal_handle,
  int robot_id
) {
  RCLCPP_INFO(this->get_logger(),
              "⚠️  [Action] Received cancel request for robot %d", robot_id);

  // 通过TCP发送取消请求
  std::lock_guard<std::mutex> lock(goal_handles_mutex_);
  auto it = active_request_ids_.find(robot_id);
  if (it != active_request_ids_.end()) {
    action_manager_->cancelGoal(robot_id, it->second);
  }

  return rclcpp_action::CancelResponse::ACCEPT;
}

void ReliableServerNode::handleAccepted(
  const std::shared_ptr<GoalHandleNavigate> goal_handle,
  int robot_id
) {
  // 检查是否正在关闭
  if (shutdown_flag_.load()) {
    RCLCPP_WARN(this->get_logger(), "Server shutting down, rejecting goal for robot %d", robot_id);
    auto result = std::make_shared<NavigateToGoal::Result>();
    result->success = false;
    result->message = "Server shutting down";
    result->final_distance = -1.0;
    result->elapsed_time_seconds = 0.0;
    goal_handle->abort(result);
    return;
  }

  // 提交任务到该机器人的工作线程（队列中只保留最新任务）
  submitTask(robot_id, [this, goal_handle, robot_id]() {
    auto task_start = std::chrono::steady_clock::now();
    
    // 检查关闭标志
    if (shutdown_flag_.load()) {
      return;
    }

    const auto goal = goal_handle->get_goal();
    const std::string& frame_id = goal->target_pose.header.frame_id;

    RCLCPP_INFO(this->get_logger(),
                "🚀 [Action] Executing goal for robot %d: (%.2f, %.2f), tolerance=%.2f, frame_id='%s'",
                robot_id, goal->target_pose.pose.position.x,
                goal->target_pose.pose.position.y, goal->tolerance, frame_id.c_str());

    // 根据 frame_id 决定处理方式
    geometry_msgs::msg::PoseStamped target_pose_to_send;

    if (frame_id == "map" || frame_id.empty()) {
      target_pose_to_send = goal->target_pose;
      RCLCPP_DEBUG(this->get_logger(),
                  "📍 Goal in robot-local frame '%s', forwarding directly", frame_id.c_str());
    } else if (frame_id == "global_map") {
      RCLCPP_DEBUG(this->get_logger(),
                  "🌐 Goal in global_map frame, transforming to robot_%d/map", robot_id);

      if (shutdown_flag_.load()) return;

      if (!transformGoalToRobotLocal(robot_id, goal->target_pose, target_pose_to_send)) {
        if (shutdown_flag_.load()) return;
        std::lock_guard<std::mutex> lock(goal_handles_mutex_);
        auto result = std::make_shared<NavigateToGoal::Result>();
        result->success = false;
        result->message = "Failed to transform goal from global_map to robot-local frame";
        result->final_distance = -1.0;
        result->elapsed_time_seconds = 0.0;
        goal_handle->abort(result);
        RCLCPP_ERROR(this->get_logger(),
                     "🚨🚨🚨 [TF ERROR] Failed to transform goal for robot %d! "
                     "Check TF tree: global_map -> robot_%d/map 🚨🚨🚨",
                     robot_id, robot_id);
        return;
      }

      RCLCPP_DEBUG(this->get_logger(),
                  "✅ Transformed goal: global_map(%.2f, %.2f) -> robot_%d/map(%.2f, %.2f)",
                  goal->target_pose.pose.position.x, goal->target_pose.pose.position.y,
                  robot_id,
                  target_pose_to_send.pose.position.x, target_pose_to_send.pose.position.y);
    } else {
      std::string robot_prefix = "robot_" + std::to_string(robot_id) + "/";
      if (frame_id.rfind(robot_prefix, 0) == 0) {
        target_pose_to_send = goal->target_pose;
        target_pose_to_send.header.frame_id = "map";
      } else {
        target_pose_to_send = goal->target_pose;
        target_pose_to_send.header.frame_id = "map";
        RCLCPP_WARN(this->get_logger(),
                    "⚠️ Unknown frame_id '%s', assuming robot-local", frame_id.c_str());
      }
    }

    if (shutdown_flag_.load()) return;

    // 先设置goal_handle再发送目标
    {
      std::lock_guard<std::mutex> lock(goal_handles_mutex_);
      auto old_it = active_goal_handles_.find(robot_id);
      if (old_it != active_goal_handles_.end() && old_it->second) {
        auto cancel_result = std::make_shared<NavigateToGoal::Result>();
        cancel_result->success = false;
        cancel_result->message = "Preempted by new goal";
        cancel_result->final_distance = -1.0;
        cancel_result->elapsed_time_seconds = 0.0;
        old_it->second->abort(cancel_result);
        RCLCPP_INFO(this->get_logger(), 
                    "⚠️ Robot %d: Old goal preempted and aborted (result sent to RL client)",
                    robot_id);
      }
      active_goal_handles_[robot_id] = goal_handle;
    }

    if (shutdown_flag_.load()) return;

    // 通过TCP发送目标
    auto send_start = std::chrono::steady_clock::now();
    uint64_t request_id = action_manager_->sendGoalWithDeduplication(
      robot_id, target_pose_to_send, goal->tolerance);
    auto send_end = std::chrono::steady_clock::now();
    auto send_ms = std::chrono::duration_cast<std::chrono::milliseconds>(send_end - send_start).count();
    
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(send_end - task_start).count();
    
    RCLCPP_DEBUG(this->get_logger(),
                 "⏱️ Robot %d goal processing time: send=%ldms, total=%ldms",
                 robot_id, send_ms, total_ms);

    if (request_id == 0) {
      if (shutdown_flag_.load()) return;
      std::lock_guard<std::mutex> lock(goal_handles_mutex_);
      active_goal_handles_.erase(robot_id);

      auto result = std::make_shared<NavigateToGoal::Result>();
      result->success = false;
      result->message = "Failed to send goal via TCP";
      result->final_distance = -1.0;
      result->elapsed_time_seconds = 0.0;
      goal_handle->abort(result);
      RCLCPP_ERROR(this->get_logger(),
                   "🚨🚨🚨 [TCP SEND FAILED] Cannot send goal to robot %d! 🚨🚨🚨",
                   robot_id);
      return;
    }

    // 设置request_id
    {
      std::lock_guard<std::mutex> lock(goal_handles_mutex_);
      active_request_ids_[robot_id] = request_id;
    }

    // 更新目标缓存用于可视化
    geometry_msgs::msg::PoseStamped goal_with_prefix = target_pose_to_send;
    std::string prefix = "robot_" + std::to_string(robot_id) + "/";
    if (goal_with_prefix.header.frame_id.rfind(prefix, 0) != 0) {
      goal_with_prefix.header.frame_id = prefix + goal_with_prefix.header.frame_id;
    }
    current_goals_[robot_id] = goal_with_prefix;

    RCLCPP_INFO(this->get_logger(),
                "✅ Goal sent to robot %d via TCP [request_id=%lu]",
                robot_id, request_id);
  });
}

// ===== 系统重置回调 =====
void ReliableServerNode::onResetCommand(
  const cure_explore_interfaces::msg::ResetCommand::SharedPtr msg
) {
  RCLCPP_INFO(this->get_logger(),
              "📋 Received reset command: type=%d", msg->command);

  // ResetCommand command values: 0=SHUTDOWN, 1=STARTUP, 2=RESTART
  switch (msg->command) {
    case 0:  // SHUTDOWN
      handleShutdown();
      break;
    case 1:  // STARTUP
      handleStartup();
      break;
    case 2:  // RESTART
      handleShutdown();
      // 短暂延迟后重启
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      handleStartup();
      break;
    default:
      RCLCPP_WARN(this->get_logger(),
                  "Unknown reset command: %d", msg->command);
      return;
  }
}

void ReliableServerNode::handleShutdown() {
  RCLCPP_INFO(this->get_logger(), "🔄 Resetting TCP communication server state...");

  // 1. 取消所有活跃的Action
  {
    std::lock_guard<std::mutex> lock(goal_handles_mutex_);
    for (auto& [robot_id, goal_handle] : active_goal_handles_) {
      if (goal_handle) {
        auto result = std::make_shared<NavigateToGoal::Result>();
        result->success = false;
        result->message = "System reset - clearing state";
        result->final_distance = -1.0;
        result->elapsed_time_seconds = 0.0;
        goal_handle->abort(result);
        RCLCPP_INFO(this->get_logger(),
                    "Aborted active goal for robot %d due to reset", robot_id);
      }
    }
    active_goal_handles_.clear();
    active_request_ids_.clear();
  }

  // 2. 清理Action Manager状态（重置请求ID计数器）
  if (action_manager_) {
    action_manager_->reset();
    RCLCPP_INFO(this->get_logger(), "Action manager state reset");
  }

  // 注意: TCP服务器保持运行,机器人重启后会自动重连

  // 3. 发送确认
  auto ack = cure_explore_interfaces::msg::ResetAck();
  ack.source = "reliable_communication_server";
  ack.success = true;
  ack.message = "Server state reset complete, TCP connections maintained";
  reset_ack_pub_->publish(ack);

  RCLCPP_INFO(this->get_logger(), "✅ State reset complete, TCP server still running, ACK sent");
}

void ReliableServerNode::handleStartup() {
  RCLCPP_INFO(this->get_logger(), "🚀 Server startup - no action needed (server always running)");

  // 服务器端不需要做任何事情:
  // - TCP服务器一直在运行
  // - ActionManager一直有效
  // - 机器人重启后会自动重连
  
  // 发送确认
  auto ack = cure_explore_interfaces::msg::ResetAck();
  ack.source = "reliable_communication_server";
  ack.success = true;
  ack.message = "Server ready (always running)";
  reset_ack_pub_->publish(ack);

  RCLCPP_INFO(this->get_logger(), "✅ Startup ACK sent");
}

void ReliableServerNode::onGlobalMap(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  if (action_manager_) {
    action_manager_->setGlobalMap(msg);
  }
}

void ReliableServerNode::onRobotLocalMap(int robot_id, const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
  RCLCPP_DEBUG(this->get_logger(),
               "🗺️ Received local map for robot %d: %dx%d, resolution=%.2f",
               robot_id, msg->info.width, msg->info.height, msg->info.resolution);

  if (action_manager_) {
    action_manager_->setRobotLocalMap(robot_id, msg);
  }
}

void ReliableServerNode::updateTFCache() {
  std::lock_guard<std::mutex> lock(tf_cache_mutex_);
  auto now = this->get_clock()->now();

  // 更新每个机器人的TF缓存
  for (int robot_id = 1; robot_id <= robot_count_; ++robot_id) {
    std::string robot_map_frame = "robot_" + std::to_string(robot_id) + "/map";
    std::string robot_base_frame = "robot_" + std::to_string(robot_id) + "/base_link";
    std::string global_frame = "global_map";

    bool all_valid = true;
    geometry_msgs::msg::TransformStamped robot_to_global, global_to_robot, robot_base_to_global;

    // 1. robot_X/map -> global_map
    try {
      auto transform = tf_buffer_->lookupTransform(
        global_frame, robot_map_frame, rclcpp::Time(0), tf2::durationFromSec(0.01)
      );
      robot_to_global_tf_[robot_id].transform = transform;
      robot_to_global_tf_[robot_id].last_update = now;
      robot_to_global_tf_[robot_id].valid = true;
      robot_to_global = transform;
    } catch (const tf2::TransformException& ex) {
      if (robot_to_global_tf_[robot_id].valid) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to update TF cache %s->%s: %s",
                             robot_map_frame.c_str(), global_frame.c_str(), ex.what());
      }
      robot_to_global_tf_[robot_id].valid = false;
      all_valid = false;
    }

    // 2. global_map -> robot_X/map
    try {
      auto transform = tf_buffer_->lookupTransform(
        robot_map_frame, global_frame, rclcpp::Time(0), tf2::durationFromSec(0.01)
      );
      global_to_robot_tf_[robot_id].transform = transform;
      global_to_robot_tf_[robot_id].last_update = now;
      global_to_robot_tf_[robot_id].valid = true;
      global_to_robot = transform;
    } catch (const tf2::TransformException& ex) {
      if (global_to_robot_tf_[robot_id].valid) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to update TF cache %s->%s: %s",
                             global_frame.c_str(), robot_map_frame.c_str(), ex.what());
      }
      global_to_robot_tf_[robot_id].valid = false;
      all_valid = false;
    }

    // 3. robot_X/base_link -> global_map (用于获取机器人当前位置)
    try {
      auto transform = tf_buffer_->lookupTransform(
        global_frame, robot_base_frame, rclcpp::Time(0), tf2::durationFromSec(0.01)
      );
      robot_base_to_global_tf_[robot_id].transform = transform;
      robot_base_to_global_tf_[robot_id].last_update = now;
      robot_base_to_global_tf_[robot_id].valid = true;
      robot_base_to_global = transform;
    } catch (const tf2::TransformException& ex) {
      if (robot_base_to_global_tf_[robot_id].valid) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Failed to update TF cache %s->%s: %s",
                             robot_base_frame.c_str(), global_frame.c_str(), ex.what());
      }
      robot_base_to_global_tf_[robot_id].valid = false;
      all_valid = false;
    }

    // 如果所有变换都有效，推送给ActionManager
    if (all_valid && action_manager_) {
      action_manager_->updateTFCache(robot_id, robot_to_global, global_to_robot, robot_base_to_global);
    }
  }
}

bool ReliableServerNode::transformGoalToRobotLocal(
  int robot_id,
  const geometry_msgs::msg::PoseStamped& goal_in_global,
  geometry_msgs::msg::PoseStamped& goal_out_local
) {
  // 使用缓存的TF进行转换
  std::lock_guard<std::mutex> lock(tf_cache_mutex_);

  if (!global_to_robot_tf_[robot_id].valid) {
    RCLCPP_WARN(this->get_logger(),
                "TF cache not available for robot %d, falling back to direct query",
                robot_id);
    // 回退到直接查询
    std::string target_frame = "robot_" + std::to_string(robot_id) + "/map";
    try {
      goal_out_local = tf_buffer_->transform(
        goal_in_global, target_frame, tf2::durationFromSec(0.05)
      );
      goal_out_local.header.frame_id = "map";
      return true;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_ERROR(this->get_logger(), "TF transform failed: %s", ex.what());
      return false;
    }
  }

  // 使用缓存的变换
  geometry_msgs::msg::PoseStamped pose_stamped;
  pose_stamped.header = goal_in_global.header;
  pose_stamped.pose = goal_in_global.pose;

  try {
    tf2::doTransform(pose_stamped, goal_out_local, global_to_robot_tf_[robot_id].transform);
    goal_out_local.header.frame_id = "map";

    RCLCPP_DEBUG(this->get_logger(),
                 "TF transform (cached) succeeded: global_map(%.2f, %.2f) -> robot_%d/map(%.2f, %.2f)",
                 goal_in_global.pose.position.x, goal_in_global.pose.position.y,
                 robot_id,
                 goal_out_local.pose.position.x, goal_out_local.pose.position.y);
    return true;
  } catch (const std::exception& ex) {
    RCLCPP_ERROR(this->get_logger(), "Failed to apply cached transform: %s", ex.what());
    return false;
  }
}

}  // namespace reliable_communication

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(reliable_communication::ReliableServerNode)
