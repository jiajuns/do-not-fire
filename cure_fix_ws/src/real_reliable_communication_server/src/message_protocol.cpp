#include "reliable_communication_server/message_protocol.hpp"
#include <cstring>
#include <arpa/inet.h>

namespace reliable_communication {

std::vector<uint8_t> MessageHeader::serialize() const {
  std::vector<uint8_t> data(HEADER_SIZE);
  size_t offset = 0;

  // Magic (4 bytes, little-endian)
  uint32_t magic_le = htole32(magic);
  std::memcpy(data.data() + offset, &magic_le, sizeof(uint32_t));
  offset += sizeof(uint32_t);

  // Message type (1 byte)
  data[offset] = msg_type;
  offset += sizeof(uint8_t);

  // Request ID (8 bytes, little-endian)
  uint64_t request_id_le = htole64(request_id);
  std::memcpy(data.data() + offset, &request_id_le, sizeof(uint64_t));
  offset += sizeof(uint64_t);

  // Timestamp (8 bytes, little-endian)
  uint64_t timestamp_le = htole64(timestamp);
  std::memcpy(data.data() + offset, &timestamp_le, sizeof(uint64_t));
  offset += sizeof(uint64_t);

  // Payload size (4 bytes, little-endian)
  uint32_t payload_size_le = htole32(payload_size);
  std::memcpy(data.data() + offset, &payload_size_le, sizeof(uint32_t));

  return data;
}

MessageHeader MessageHeader::deserialize(const uint8_t* data) {
  MessageHeader header;
  size_t offset = 0;

  // Magic
  uint32_t magic_le;
  std::memcpy(&magic_le, data + offset, sizeof(uint32_t));
  header.magic = le32toh(magic_le);
  offset += sizeof(uint32_t);

  // Message type
  header.msg_type = data[offset];
  offset += sizeof(uint8_t);

  // Request ID
  uint64_t request_id_le;
  std::memcpy(&request_id_le, data + offset, sizeof(uint64_t));
  header.request_id = le64toh(request_id_le);
  offset += sizeof(uint64_t);

  // Timestamp
  uint64_t timestamp_le;
  std::memcpy(&timestamp_le, data + offset, sizeof(uint64_t));
  header.timestamp = le64toh(timestamp_le);
  offset += sizeof(uint64_t);

  // Payload size
  uint32_t payload_size_le;
  std::memcpy(&payload_size_le, data + offset, sizeof(uint32_t));
  header.payload_size = le32toh(payload_size_le);

  return header;
}

std::vector<uint8_t> Message::serialize() const {
  auto header_data = header.serialize();
  std::vector<uint8_t> data;
  data.reserve(header_data.size() + payload.size());
  data.insert(data.end(), header_data.begin(), header_data.end());
  data.insert(data.end(), payload.begin(), payload.end());
  return data;
}

}  // namespace reliable_communication
