#ifndef SIMPLE_GREEDY_EXPLORER_H
#define SIMPLE_GREEDY_EXPLORER_H

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/int8.hpp>
#include <cure_explore_interfaces/msg/frontier_array.hpp>
#include <cure_explore_interfaces/msg/frontier_cluster.hpp>
#include <reliable_communication_interfaces/action/navigate_to_goal.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>

/**
 * @brief 简化版贪心探索器 - 专为实车设计
 * 
 * 特性：
 * - 无回退策略
 * - 通过检测移动距离判定卡死
 * - 卡死检测时间：5秒
 * - 检测到卡死立即切换下一个目标点
 * - 无超时限定
 */
class SimpleGreedyExplorer : public rclcpp::Node
{
public:
    using NavigateToGoal = reliable_communication_interfaces::action::NavigateToGoal;
    using GoalHandleNavigateToGoal = rclcpp_action::ClientGoalHandle<NavigateToGoal>;

    SimpleGreedyExplorer();

private:
    // === Callbacks ===
    void frontiersCallback(const cure_explore_interfaces::msg::FrontierArray::SharedPtr msg);
    void nav2StatusCallback(const std_msgs::msg::Int8::SharedPtr msg, int robot_index);

    // === Core Functions ===
    void assignGoals();
    void checkStuckRobots();
    void checkInvalidGoals();
    geometry_msgs::msg::Point getRobotPosition(int robot_index);
    bool isRobotStuck(int robot_index);
    bool isProgressStuck(int robot_index);
    bool isFrontierStillValid(int frontier_id);
    void cancelRobotGoal(int robot_index);
    void sendGoalToRobot(int robot_index, const cure_explore_interfaces::msg::FrontierCluster& frontier);
    int selectBestFrontier(int robot_index);

    // === Action Response Callbacks ===
    void feedbackCallback(
        GoalHandleNavigateToGoal::SharedPtr,
        const std::shared_ptr<const NavigateToGoal::Feedback> feedback,
        int robot_index);
    void resultCallback(
        const GoalHandleNavigateToGoal::WrappedResult& result,
        int robot_index);

    // === Parameters ===
    int num_robots_;
    double decision_frequency_;           // 决策频率（Hz）
    double stuck_distance_threshold_;     // 卡死检测距离阈值（米）
    double stuck_time_threshold_;         // 卡死检测时间阈值（秒），默认5秒
    double progress_stuck_time_threshold_;    // 导航进度停滞检测时间阈值（秒），默认10秒
    double progress_distance_threshold_;      // 进度距离变化阈值（米），默认0.1米
    double min_frontier_separation_;          // 前沿点最小分离距离（米），默认2.0米
    double direction_penalty_weight_;         // 方向相同惩罚权重，默认0.5

    // === State ===
    std::vector<std::string> robot_names_;
    std::vector<bool> robot_is_busy_;
    std::vector<int> assigned_frontier_ids_;
    std::vector<geometry_msgs::msg::Point> last_robot_positions_;
    std::vector<std::chrono::steady_clock::time_point> last_position_update_times_;
    std::vector<int8_t> robot_nav2_status_;
    std::vector<float> last_distance_remaining_;
    std::vector<std::chrono::steady_clock::time_point> last_progress_update_times_;
    std::vector<GoalHandleNavigateToGoal::SharedPtr> goal_handles_;

    // === Frontiers ===
    std::vector<cure_explore_interfaces::msg::FrontierCluster> current_frontiers_;
    bool frontiers_received_ = false;
    
    // 黑名单：规划失败或超时的前沿点位置（前沿点ID会变，只能用位置判断）
    struct BlacklistedFrontier {
        geometry_msgs::msg::Point position;
        std::chrono::steady_clock::time_point blacklisted_time;
    };
    std::vector<BlacklistedFrontier> blacklisted_frontiers_;
    double blacklist_position_threshold_ = 0.5;  // 判定为同一前沿点的距离阈值（米）
    double blacklist_timeout_ = 60.0;  // 黑名单过期时间（秒），避免永久黑名单
    
    bool isPositionBlacklisted(const geometry_msgs::msg::Point& pos);
    void addToBlacklist(const geometry_msgs::msg::Point& pos);
    void cleanupBlacklist();

    // === ROS 2 Interfaces ===
    rclcpp::Subscription<cure_explore_interfaces::msg::FrontierArray>::SharedPtr frontiers_subscriber_;
    std::vector<rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr> nav2_status_subscribers_;
    std::vector<rclcpp_action::Client<NavigateToGoal>::SharedPtr> action_clients_;
    rclcpp::TimerBase::SharedPtr assign_timer_;
    rclcpp::TimerBase::SharedPtr stuck_check_timer_;
    rclcpp::TimerBase::SharedPtr invalid_goal_check_timer_;

    // === TF2 ===
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // === Utility ===
    double calculateDistance(const geometry_msgs::msg::Point& p1, 
                            const geometry_msgs::msg::Point& p2);
    double calculateAngle(const geometry_msgs::msg::Point& from,
                         const geometry_msgs::msg::Point& to);
    double calculateDispersionPenalty(int robot_index, 
                                      const geometry_msgs::msg::Point& frontier_pos);
    
    // 路径交叉检测辅助函数
    double crossProduct(const geometry_msgs::msg::Point& O,
                       const geometry_msgs::msg::Point& A,
                       const geometry_msgs::msg::Point& B);
    bool doSegmentsIntersect(const geometry_msgs::msg::Point& A,
                            const geometry_msgs::msg::Point& B,
                            const geometry_msgs::msg::Point& C,
                            const geometry_msgs::msg::Point& D);
};

#endif // SIMPLE_GREEDY_EXPLORER_H
