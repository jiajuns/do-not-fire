#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace reliable_communication {

// 消息魔数
constexpr uint32_t MESSAGE_MAGIC = 0x52434D4D;  // 'RCMM'

// 消息头大小 (固定25字节)
constexpr size_t HEADER_SIZE = 25;  // 4 + 1 + 8 + 8 + 4

// 消息类型 (可扩展)
enum class MessageType : uint8_t {
  ACTION_GOAL = 0x01,      // Action目标
  ACTION_FEEDBACK = 0x02,  // Action反馈
  ACTION_RESULT = 0x03,    // Action结果
  ACTION_CANCEL = 0x04,    // 取消Action
  QOS_REPORT = 0x05,       // 通信质量报告
  HEARTBEAT = 0x06,        // 心跳包
  PATH_PLANNING_REQUEST = 0x10,  // 路径规划请求
  PATH_PLANNING_RESPONSE = 0x11, // 路径规划响应
  // 在此添加新的消息类型
};

// 消息头结构
struct MessageHeader {
  uint32_t magic;        // 魔数
  uint8_t msg_type;      // 消息类型
  uint64_t request_id;   // 请求ID
  uint64_t timestamp;    // 时间戳(纳秒)
  uint32_t payload_size; // 负载大小

  // 序列化到字节数组
  std::vector<uint8_t> serialize() const;

  // 从字节数组反序列化
  static MessageHeader deserialize(const uint8_t* data);
};

// 完整消息结构
struct Message {
  MessageHeader header;
  std::vector<uint8_t> payload;

  // 序列化完整消息
  std::vector<uint8_t> serialize() const;
};

}  // namespace reliable_communication
