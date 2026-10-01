#include "real_cure_cpp/simple_greedy_explorer.h"

SimpleGreedyExplorer::SimpleGreedyExplorer()
    : Node("simple_greedy_explorer",
           rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true))
{
    // 获取参数
    num_robots_ = this->get_parameter_or("robot_count", 2);
    decision_frequency_ = this->get_parameter_or("decision_frequency", 0.333);
    stuck_distance_threshold_ = this->get_parameter_or("stuck_distance_threshold", 0.3);
    stuck_time_threshold_ = this->get_parameter_or("stuck_time_threshold", 5.0);  // 默认5秒
    progress_stuck_time_threshold_ = this->get_parameter_or("progress_stuck_time_threshold", 10.0);  // 默认10秒
    progress_distance_threshold_ = this->get_parameter_or("progress_distance_threshold", 0.1);  // 默认0.1米
    min_frontier_separation_ = this->get_parameter_or("min_frontier_separation", 2.0);  // 默认2.0米
    direction_penalty_weight_ = this->get_parameter_or("direction_penalty_weight", 0.5);  // 默认0.5

    RCLCPP_INFO(this->get_logger(),
        "Simple Greedy Explorer starting with %d robots, decision frequency: %.2f Hz",
        num_robots_, decision_frequency_);
    RCLCPP_INFO(this->get_logger(),
        "Stuck detection: position < %.2f m for > %.1f s, progress < %.2f m for > %.1f s (NO FALLBACK, NO TIMEOUT)",
        stuck_distance_threshold_, stuck_time_threshold_, 
        progress_distance_threshold_, progress_stuck_time_threshold_);
    RCLCPP_INFO(this->get_logger(),
        "Dispersion strategy: min separation %.2f m, direction penalty weight %.2f",
        min_frontier_separation_, direction_penalty_weight_);

    // 初始化机器人名称 (robot_1, robot_2, ...)
    robot_names_.resize(num_robots_);
    for (int i = 0; i < num_robots_; ++i)
    {
        robot_names_[i] = "robot_" + std::to_string(i + 1);
    }

    // 订阅前沿信息
    frontiers_subscriber_ = this->create_subscription<cure_explore_interfaces::msg::FrontierArray>(
        "/frontiers", 10,
        std::bind(&SimpleGreedyExplorer::frontiersCallback, this, std::placeholders::_1));

    // 初始化机器人状态向量
    robot_is_busy_.resize(num_robots_, false);
    assigned_frontier_ids_.resize(num_robots_, -1);
    last_robot_positions_.resize(num_robots_);
    last_position_update_times_.resize(num_robots_);
    robot_nav2_status_.resize(num_robots_, 0);  // 0 = UNKNOWN
    last_distance_remaining_.resize(num_robots_, std::numeric_limits<float>::infinity());
    last_progress_update_times_.resize(num_robots_);
    action_clients_.resize(num_robots_);
    nav2_status_subscribers_.resize(num_robots_);
    goal_handles_.resize(num_robots_, nullptr);

    // 初始化 TF2
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 为每个机器人创建 action client 和 nav2 状态订阅器
    for (int i = 0; i < num_robots_; ++i)
    {
        // Action client
        std::string action_name = "/" + robot_names_[i] + "/navigate_to_goal";
        action_clients_[i] = rclcpp_action::create_client<NavigateToGoal>(this, action_name);

        RCLCPP_INFO(this->get_logger(), "Waiting for action server: %s", action_name.c_str());

        if (!action_clients_[i]->wait_for_action_server(std::chrono::seconds(5)))
        {
            RCLCPP_WARN(this->get_logger(),
                "Action server for robot %d not available after 5s: %s",
                i + 1, action_name.c_str());
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "Action server found for robot %d", i + 1);
        }

        // Nav2 状态订阅器
        std::string status_topic = "/" + robot_names_[i] + "/nav2_status";
        nav2_status_subscribers_[i] = this->create_subscription<std_msgs::msg::Int8>(
            status_topic, 10,
            [this, i](const std_msgs::msg::Int8::SharedPtr msg) {
                this->nav2StatusCallback(msg, i);
            });

        // 初始化位置跟踪
        last_robot_positions_[i].x = 0.0;
        last_robot_positions_[i].y = 0.0;
        last_robot_positions_[i].z = 0.0;
        last_position_update_times_[i] = std::chrono::steady_clock::now();
        last_progress_update_times_[i] = std::chrono::steady_clock::now();
    }

    // 创建目标分配定时器
    int timer_ms = static_cast<int>(1000.0 / decision_frequency_);
    assign_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(timer_ms),
        std::bind(&SimpleGreedyExplorer::assignGoals, this));

    // 创建卡死检测定时器（每1秒检查一次）
    stuck_check_timer_ = this->create_wall_timer(
        std::chrono::seconds(1),
        std::bind(&SimpleGreedyExplorer::checkStuckRobots, this));

    // 创建无效目标检查定时器（每2秒检查一次）
    invalid_goal_check_timer_ = this->create_wall_timer(
        std::chrono::seconds(2),
        std::bind(&SimpleGreedyExplorer::checkInvalidGoals, this));

    RCLCPP_INFO(this->get_logger(), "Simple Greedy Explorer initialized successfully");
}

void SimpleGreedyExplorer::frontiersCallback(
    const cure_explore_interfaces::msg::FrontierArray::SharedPtr msg)
{
    current_frontiers_ = msg->clusters;
    frontiers_received_ = true;

    RCLCPP_DEBUG(this->get_logger(),
        "Received %d frontier clusters", static_cast<int>(msg->clusters.size()));
    
    // 清理过期的黑名单项
    cleanupBlacklist();
}

void SimpleGreedyExplorer::nav2StatusCallback(const std_msgs::msg::Int8::SharedPtr msg, int robot_index)
{
    robot_nav2_status_[robot_index] = msg->data;

    RCLCPP_DEBUG(this->get_logger(),
        "Robot %d nav2 status: %d", robot_index + 1, msg->data);
}

geometry_msgs::msg::Point SimpleGreedyExplorer::getRobotPosition(int robot_index)
{
    geometry_msgs::msg::Point position;

    try
    {
        std::string target_frame = "global_map";
        std::string source_frame = robot_names_[robot_index] + "/base_link";

        geometry_msgs::msg::TransformStamped transform_stamped =
            tf_buffer_->lookupTransform(
                target_frame, source_frame,
                tf2::TimePointZero);

        position.x = transform_stamped.transform.translation.x;
        position.y = transform_stamped.transform.translation.y;
        position.z = 0.0;
    }
    catch (tf2::TransformException &ex)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Could not get transform for robot %d: %s", robot_index + 1, ex.what());
        position.x = std::numeric_limits<double>::infinity();
        position.y = std::numeric_limits<double>::infinity();
        position.z = 0.0;
    }

    return position;
}

bool SimpleGreedyExplorer::isRobotStuck(int robot_index)
{
    if (!robot_is_busy_[robot_index])
        return false;

    // 获取当前位置
    geometry_msgs::msg::Point current_pos = getRobotPosition(robot_index);
    if (!std::isfinite(current_pos.x))
        return false;

    // 计算移动距离
    double distance_moved = calculateDistance(current_pos, last_robot_positions_[robot_index]);

    // 计算经过的时间
    auto current_time = std::chrono::steady_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
        current_time - last_position_update_times_[robot_index]).count();

    // 机器人卡死判断：在指定时间内移动距离小于阈值
    if (time_elapsed >= stuck_time_threshold_ && distance_moved < stuck_distance_threshold_)
    {
        RCLCPP_WARN(this->get_logger(),
            "Robot %d is STUCK! Moved %.2f m in %.1f s (threshold: %.2f m)",
            robot_index + 1, distance_moved, time_elapsed, stuck_distance_threshold_);
        return true;
    }

    return false;
}

bool SimpleGreedyExplorer::isProgressStuck(int robot_index)
{
    if (!robot_is_busy_[robot_index])
        return false;

    // 计算进度未更新的时间
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last_progress = std::chrono::duration_cast<std::chrono::duration<double>>(
        current_time - last_progress_update_times_[robot_index]).count();

    // 如果导航进度长时间没有变化（distance_remaining几乎不变）
    if (time_since_last_progress >= progress_stuck_time_threshold_ &&
        std::isfinite(last_distance_remaining_[robot_index]))
    {
        RCLCPP_WARN(this->get_logger(),
            "Robot %d navigation progress STUCK! Distance remaining unchanged (%.2f m) for %.1f s (threshold: %.1f s)",
            robot_index + 1, last_distance_remaining_[robot_index],
            time_since_last_progress, progress_stuck_time_threshold_);
        return true;
    }

    return false;
}

void SimpleGreedyExplorer::checkStuckRobots()
{
    for (int i = 0; i < num_robots_; ++i)
    {
        bool position_stuck = isRobotStuck(i);
        bool progress_stuck = isProgressStuck(i);
        
        if (position_stuck || progress_stuck)
        {
            int stuck_frontier_id = assigned_frontier_ids_[i];
            
            // 查找被分配前沿的位置
            geometry_msgs::msg::Point stuck_frontier_pos;
            bool found_frontier = false;
            if (stuck_frontier_id >= 0)
            {
                for (const auto& frontier : current_frontiers_)
                {
                    if (frontier.id == stuck_frontier_id)
                    {
                        stuck_frontier_pos = frontier.centroid;
                        found_frontier = true;
                        break;
                    }
                }
            }
            
            if (position_stuck)
            {
                RCLCPP_WARN(this->get_logger(),
                    "Robot %d detected as POSITION STUCK", i + 1);
            }
            if (progress_stuck)
            {
                RCLCPP_WARN(this->get_logger(),
                    "Robot %d detected as PROGRESS STUCK", i + 1);
            }
            
            // 将卡住的前沿点位置加入黑名单
            if (found_frontier)
            {
                RCLCPP_WARN(this->get_logger(),
                    "Adding stuck frontier [%.2f, %.2f] to blacklist",
                    stuck_frontier_pos.x, stuck_frontier_pos.y);
                addToBlacklist(stuck_frontier_pos);
            }

            // 取消当前目标并标记为空闲
            cancelRobotGoal(i);
        }
    }
}

void SimpleGreedyExplorer::checkInvalidGoals()
{
    if (!frontiers_received_ || current_frontiers_.empty())
    {
        return;
    }

    for (int i = 0; i < num_robots_; ++i)
    {
        // 只检查正在执行任务的机器人
        if (robot_is_busy_[i] && assigned_frontier_ids_[i] >= 0)
        {
            // 检查该前沿点是否还在前沿列表中
            if (!isFrontierStillValid(assigned_frontier_ids_[i]))
            {
                RCLCPP_WARN(this->get_logger(),
                    "Robot %d's target frontier %d is no longer valid (already explored), canceling goal",
                    i + 1, assigned_frontier_ids_[i]);

                // 取消当前目标
                cancelRobotGoal(i);
            }
        }
    }
}

bool SimpleGreedyExplorer::isFrontierStillValid(int frontier_id)
{
    // 在当前前沿列表中查找该前沿ID
    for (const auto& frontier : current_frontiers_)
    {
        if (frontier.id == frontier_id)
        {
            return true;
        }
    }
    return false;
}

void SimpleGreedyExplorer::cancelRobotGoal(int robot_index)
{
    // 取消当前的导航目标
    if (goal_handles_[robot_index])
    {
        auto cancel_future = action_clients_[robot_index]->async_cancel_goal(goal_handles_[robot_index]);
        RCLCPP_INFO(this->get_logger(),
            "Canceling goal for robot %d", robot_index + 1);
    }

    // 立即标记为空闲，以便下次分配新目标
    robot_is_busy_[robot_index] = false;
    assigned_frontier_ids_[robot_index] = -1;
    goal_handles_[robot_index] = nullptr;
}

void SimpleGreedyExplorer::assignGoals()
{
    if (!frontiers_received_ || current_frontiers_.empty())
    {
        return;
    }

    // 为每个空闲的机器人分配目标
    for (int i = 0; i < num_robots_; ++i)
    {
        if (!robot_is_busy_[i])
        {
            int best_frontier_idx = selectBestFrontier(i);

            if (best_frontier_idx >= 0)
            {
                const auto& frontier = current_frontiers_[best_frontier_idx];
                sendGoalToRobot(i, frontier);

                RCLCPP_INFO(this->get_logger(),
                    "Assigned frontier %d to robot %d: [%.2f, %.2f], info_gain: %.2f",
                    frontier.id, i + 1,
                    frontier.centroid.x, frontier.centroid.y,
                    frontier.info_gain);

                robot_is_busy_[i] = true;
                assigned_frontier_ids_[i] = frontier.id;

                // 更新位置跟踪的起始点
                last_robot_positions_[i] = getRobotPosition(i);
                last_position_update_times_[i] = std::chrono::steady_clock::now();
                
                // 重置进度跟踪
                last_distance_remaining_[i] = std::numeric_limits<float>::infinity();
                last_progress_update_times_[i] = std::chrono::steady_clock::now();
            }
        }
    }
}

int SimpleGreedyExplorer::selectBestFrontier(int robot_index)
{
    if (current_frontiers_.empty())
        return -1;

    geometry_msgs::msg::Point robot_pos = getRobotPosition(robot_index);
    if (!std::isfinite(robot_pos.x))
        return -1;

    int best_idx = -1;
    double best_score = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < current_frontiers_.size(); ++i)
    {
        const auto& frontier = current_frontiers_[i];

        // 检查该前沿是否在黑名单中（通过位置判断）
        if (isPositionBlacklisted(frontier.centroid))
        {
            RCLCPP_DEBUG(this->get_logger(),
                "Skipping blacklisted frontier at [%.2f, %.2f]",
                frontier.centroid.x, frontier.centroid.y);
            continue;
        }

        // 检查该前沿是否已被其他机器人占用
        bool already_assigned = false;
        for (int j = 0; j < num_robots_; ++j)
        {
            if (j != robot_index && robot_is_busy_[j] && 
                assigned_frontier_ids_[j] == frontier.id)
            {
                already_assigned = true;
                break;
            }
        }

        if (already_assigned)
            continue;

        // 计算到前沿的距离
        double distance = calculateDistance(robot_pos, frontier.centroid);

        // 计算分散度惩罚（考虑与其他机器人目标的距离和方向）
        double dispersion_penalty = calculateDispersionPenalty(robot_index, frontier.centroid);

        // 贪心策略：优先选择信息增益高且距离近的前沿，同时考虑分散度
        // Score = info_gain / (distance + 1.0) * (1.0 - dispersion_penalty)
        double score = frontier.info_gain / (distance + 1.0) * (1.0 - dispersion_penalty);
        
        RCLCPP_DEBUG(this->get_logger(),
            "Robot %d evaluating frontier %d: distance=%.2f, info_gain=%.2f, dispersion_penalty=%.2f, score=%.4f",
            robot_index + 1, frontier.id, distance, frontier.info_gain, dispersion_penalty, score);

        if (score > best_score)
        {
            best_score = score;
            best_idx = static_cast<int>(i);
        }
    }

    return best_idx;
}

void SimpleGreedyExplorer::sendGoalToRobot(
    int robot_index,
    const cure_explore_interfaces::msg::FrontierCluster& frontier)
{
    if (!action_clients_[robot_index]->wait_for_action_server(std::chrono::seconds(0)))
    {
        RCLCPP_WARN(this->get_logger(),
            "Action server not available for robot %d", robot_index + 1);
        return;
    }

    auto goal_msg = NavigateToGoal::Goal();
    goal_msg.target_pose.header.frame_id = "global_map";
    // Zero stamp = "latest available TF". A now() stamp ages past the cars' 10s TF buffer
    // while the goal is active, so Nav2's controller stops being able to transform the goal
    // pose and the robot stalls until the next reassignment. Time(0) transforms forever.
    goal_msg.target_pose.header.stamp = rclcpp::Time(0);
    goal_msg.target_pose.pose.position = frontier.centroid;
    goal_msg.target_pose.pose.orientation.w = 1.0;
    goal_msg.tolerance = 0.5;  // 0.5米到达容差

    auto send_goal_options = rclcpp_action::Client<NavigateToGoal>::SendGoalOptions();

    send_goal_options.goal_response_callback =
        [this, robot_index, frontier_pos = frontier.centroid](const GoalHandleNavigateToGoal::SharedPtr& goal_handle) {
            if (goal_handle) {
                goal_handles_[robot_index] = goal_handle;
                RCLCPP_DEBUG(this->get_logger(), "Goal accepted by robot %d", robot_index + 1);
            } else {
                RCLCPP_WARN(this->get_logger(),
                    "Goal rejected by robot %d (likely planning failed), adding frontier [%.2f, %.2f] to blacklist",
                    robot_index + 1, frontier_pos.x, frontier_pos.y);
                // 目标被拒绝（通常是规划失败），加入黑名单
                addToBlacklist(frontier_pos);
                robot_is_busy_[robot_index] = false;
                assigned_frontier_ids_[robot_index] = -1;
            }
        };

    send_goal_options.feedback_callback =
        [this, robot_index](
            GoalHandleNavigateToGoal::SharedPtr goal_handle,
            const std::shared_ptr<const NavigateToGoal::Feedback> feedback) {
            this->feedbackCallback(goal_handle, feedback, robot_index);
        };

    send_goal_options.result_callback =
        [this, robot_index](const GoalHandleNavigateToGoal::WrappedResult& result) {
            this->resultCallback(result, robot_index);
        };

    action_clients_[robot_index]->async_send_goal(goal_msg, send_goal_options);
}

void SimpleGreedyExplorer::feedbackCallback(
    GoalHandleNavigateToGoal::SharedPtr,
    const std::shared_ptr<const NavigateToGoal::Feedback> feedback,
    int robot_index)
{
    float distance_remaining = feedback->distance_remaining;
    
    RCLCPP_DEBUG(this->get_logger(),
        "Robot %d feedback: distance remaining = %.2f m", 
        robot_index + 1, distance_remaining);

    // 检查进度是否有变化（距离减少超过阈值）
    if (std::isfinite(last_distance_remaining_[robot_index]) &&
        std::isfinite(distance_remaining))
    {
        float distance_change = last_distance_remaining_[robot_index] - distance_remaining;
        
        // 如果距离减少超过阈值，说明有进展
        if (distance_change > progress_distance_threshold_)
        {
            last_distance_remaining_[robot_index] = distance_remaining;
            last_progress_update_times_[robot_index] = std::chrono::steady_clock::now();
            
            RCLCPP_DEBUG(this->get_logger(),
                "Robot %d making progress: distance reduced by %.2f m",
                robot_index + 1, distance_change);
        }
    }
    else
    {
        // 首次接收feedback或上次值无效
        last_distance_remaining_[robot_index] = distance_remaining;
        last_progress_update_times_[robot_index] = std::chrono::steady_clock::now();
    }
}

void SimpleGreedyExplorer::resultCallback(
    const GoalHandleNavigateToGoal::WrappedResult& result,
    int robot_index)
{
    // 保存失败的前沿点位置
    geometry_msgs::msg::Point failed_frontier_pos;
    bool has_failed_frontier = false;
    
    int failed_frontier_id = assigned_frontier_ids_[robot_index];
    if (failed_frontier_id >= 0)
    {
        for (const auto& frontier : current_frontiers_)
        {
            if (frontier.id == failed_frontier_id)
            {
                failed_frontier_pos = frontier.centroid;
                has_failed_frontier = true;
                break;
            }
        }
    }
    
    robot_is_busy_[robot_index] = false;
    assigned_frontier_ids_[robot_index] = -1;
    goal_handles_[robot_index] = nullptr;

    switch (result.code)
    {
        case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(),
                "Robot %d reached goal successfully", robot_index + 1);
            break;
        case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_WARN(this->get_logger(),
                "Robot %d goal was aborted (planning failed or timeout)",
                robot_index + 1);
            // 规划失败或超时，将该前沿点位置加入黑名单
            if (has_failed_frontier)
            {
                RCLCPP_WARN(this->get_logger(),
                    "Adding frontier [%.2f, %.2f] to blacklist",
                    failed_frontier_pos.x, failed_frontier_pos.y);
                addToBlacklist(failed_frontier_pos);
            }
            break;
        case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_INFO(this->get_logger(),
                "Robot %d goal was canceled", robot_index + 1);
            break;
        default:
            RCLCPP_ERROR(this->get_logger(),
                "Robot %d goal ended with unknown result code", robot_index + 1);
            break;
    }
}

double SimpleGreedyExplorer::calculateDistance(
    const geometry_msgs::msg::Point& p1,
    const geometry_msgs::msg::Point& p2)
{
    return std::hypot(p2.x - p1.x, p2.y - p1.y);
}

double SimpleGreedyExplorer::calculateAngle(
    const geometry_msgs::msg::Point& from,
    const geometry_msgs::msg::Point& to)
{
    return std::atan2(to.y - from.y, to.x - from.x);
}

// 检查位置是否在黑名单中
bool SimpleGreedyExplorer::isPositionBlacklisted(const geometry_msgs::msg::Point& pos)
{
    for (const auto& blacklisted : blacklisted_frontiers_)
    {
        double dist = calculateDistance(pos, blacklisted.position);
        if (dist < blacklist_position_threshold_)
        {
            return true;
        }
    }
    return false;
}

// 将位置加入黑名单
void SimpleGreedyExplorer::addToBlacklist(const geometry_msgs::msg::Point& pos)
{
    // 检查是否已经在黑名单中
    if (isPositionBlacklisted(pos))
    {
        return;
    }
    
    BlacklistedFrontier blacklisted;
    blacklisted.position = pos;
    blacklisted.blacklisted_time = std::chrono::steady_clock::now();
    blacklisted_frontiers_.push_back(blacklisted);
    
    RCLCPP_INFO(this->get_logger(),
        "Added position [%.2f, %.2f] to blacklist (total: %zu)",
        pos.x, pos.y, blacklisted_frontiers_.size());
}

// 清理黑名单（移除过期的项）
void SimpleGreedyExplorer::cleanupBlacklist()
{
    auto current_time = std::chrono::steady_clock::now();
    
    blacklisted_frontiers_.erase(
        std::remove_if(blacklisted_frontiers_.begin(), blacklisted_frontiers_.end(),
            [this, current_time](const BlacklistedFrontier& blacklisted) {
                auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(
                    current_time - blacklisted.blacklisted_time).count();
                return elapsed > blacklist_timeout_;
            }),
        blacklisted_frontiers_.end());
}

// 计算向量叉积 (O->A) x (O->B)
double SimpleGreedyExplorer::crossProduct(
    const geometry_msgs::msg::Point& O,
    const geometry_msgs::msg::Point& A,
    const geometry_msgs::msg::Point& B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

// 检测线段AB和线段CD是否相交
bool SimpleGreedyExplorer::doSegmentsIntersect(
    const geometry_msgs::msg::Point& A,
    const geometry_msgs::msg::Point& B,
    const geometry_msgs::msg::Point& C,
    const geometry_msgs::msg::Point& D)
{
    // 使用叉积判断两条线段是否相交
    // 线段AB和CD相交当且仅当：
    // 1. C和D在直线AB的两侧（或在AB上）
    // 2. A和B在直线CD的两侧（或在CD上）
    
    double d1 = crossProduct(C, D, A);
    double d2 = crossProduct(C, D, B);
    double d3 = crossProduct(A, B, C);
    double d4 = crossProduct(A, B, D);
    
    // 真相交：点在两侧
    if (((d1 > 1e-6 && d2 < -1e-6) || (d1 < -1e-6 && d2 > 1e-6)) &&
        ((d3 > 1e-6 && d4 < -1e-6) || (d3 < -1e-6 && d4 > 1e-6)))
    {
        return true;
    }
    
    return false;
}

double SimpleGreedyExplorer::calculateDispersionPenalty(
    int robot_index,
    const geometry_msgs::msg::Point& frontier_pos)
{
    double max_penalty = 0.0;
    
    // 获取当前机器人位置
    geometry_msgs::msg::Point current_robot_pos = getRobotPosition(robot_index);
    if (!std::isfinite(current_robot_pos.x))
        return 0.0;
    
    // 计算当前机器人到候选前沿的方向
    double candidate_angle = calculateAngle(current_robot_pos, frontier_pos);
    
    // 检查所有其他忙碌中的机器人
    for (int j = 0; j < num_robots_; ++j)
    {
        if (j == robot_index || !robot_is_busy_[j] || assigned_frontier_ids_[j] < 0)
            continue;
        
        // 找到其他机器人的目标前沿
        geometry_msgs::msg::Point other_frontier_pos;
        bool found = false;
        for (const auto& frontier : current_frontiers_)
        {
            if (frontier.id == assigned_frontier_ids_[j])
            {
                other_frontier_pos = frontier.centroid;
                found = true;
                break;
            }
        }
        
        if (!found)
            continue;
        
        // 1. 计算距离惩罚：如果两个目标距离太近，增加惩罚
        double distance_to_other = calculateDistance(frontier_pos, other_frontier_pos);
        double distance_penalty = 0.0;
        if (distance_to_other < min_frontier_separation_)
        {
            // 距离越近，惩罚越大
            distance_penalty = (1.0 - distance_to_other / min_frontier_separation_) * 0.7;
        }
        
        // 2. 计算方向惩罚：如果两个目标方向相同，增加惩罚
        geometry_msgs::msg::Point other_robot_pos = getRobotPosition(j);
        double direction_penalty = 0.0;
        double angle_diff = 0.0;  // 声明在外层作用域
        if (std::isfinite(other_robot_pos.x))
        {
            double other_angle = calculateAngle(other_robot_pos, other_frontier_pos);
            angle_diff = std::abs(candidate_angle - other_angle);
            
            // 将角度差标准化到 [0, pi]
            if (angle_diff > M_PI)
                angle_diff = 2.0 * M_PI - angle_diff;
            
            // 角度差越小（方向越相同），惩罚越大
            // 角度差 < 30° (pi/6) 时开始惩罚
            if (angle_diff < M_PI / 6.0)
            {
                direction_penalty = (1.0 - angle_diff / (M_PI / 6.0)) * direction_penalty_weight_;
            }
        }
        
        // 3. 计算路径交叉惩罚：检测两条路径是否相交
        double path_crossing_penalty = 0.0;
        if (std::isfinite(other_robot_pos.x))
        {
            // 检测线段 (current_robot_pos → frontier_pos) 与 (other_robot_pos → other_frontier_pos) 是否相交
            if (doSegmentsIntersect(current_robot_pos, frontier_pos,
                                   other_robot_pos, other_frontier_pos))
            {
                path_crossing_penalty = 0.8;  // 路径交叉，严重惩罚
                RCLCPP_DEBUG(this->get_logger(),
                    "Path crossing detected: Robot %d→[%.2f,%.2f] crosses Robot %d→[%.2f,%.2f]",
                    robot_index + 1, frontier_pos.x, frontier_pos.y,
                    j + 1, other_frontier_pos.x, other_frontier_pos.y);
            }
        }
        
        // 综合惩罚：取最大值
        double total_penalty = std::max({distance_penalty, direction_penalty, path_crossing_penalty});
        max_penalty = std::max(max_penalty, total_penalty);
        
        RCLCPP_DEBUG(this->get_logger(),
            "Robot %d to frontier [%.2f, %.2f] vs Robot %d to [%.2f, %.2f]: "
            "dist=%.2f (penalty=%.2f), angle_diff=%.2f° (penalty=%.2f), path_cross=%.2f, total=%.2f",
            robot_index + 1, frontier_pos.x, frontier_pos.y,
            j + 1, other_frontier_pos.x, other_frontier_pos.y,
            distance_to_other, distance_penalty,
            angle_diff * 180.0 / M_PI, direction_penalty,
            path_crossing_penalty, total_penalty);
    }
    
    // 限制惩罚在 [0, 0.9] 范围，避免完全排除某些前沿
    return std::min(max_penalty, 0.9);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleGreedyExplorer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
