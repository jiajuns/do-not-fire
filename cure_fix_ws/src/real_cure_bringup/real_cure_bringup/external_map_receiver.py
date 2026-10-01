#!/usr/bin/env python3
"""
external_map_receiver.py
独立 TCP 接收 bingda 小车地图 -> 发布 /external_robot/map (OccupancyGrid)
完全不碰已有三车通信系统
"""

import socket
import struct
import zlib
import threading
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose


def deserialize_grid(raw: bytes) -> OccupancyGrid:
    """binary -> OccupancyGrid, reverse of serialize_grid"""
    msg = OccupancyGrid()
    # header
    msg.header.stamp.sec, msg.header.stamp.nanosec = struct.unpack_from("!Ii", raw, 0)
    off = 12
    # info basic
    res, w, h = struct.unpack_from("!fii", raw, off)
    off += 12
    ox, oy, oz = struct.unpack_from("!ddd", raw, off)
    off += 24
    qz, qw = struct.unpack_from("!dd", raw, off)
    off += 16
    # data
    data_len = w * h
    msg.data = list(raw[off:off + data_len])
    # populate
    msg.info = MapMetaData()
    msg.info.resolution = res
    msg.info.width = w
    msg.info.height = h
    msg.info.origin = Pose()
    msg.info.origin.position.x = ox
    msg.info.origin.position.y = oy
    msg.info.origin.position.z = oz
    msg.info.origin.orientation.z = qz
    msg.info.origin.orientation.w = qw
    return msg


class ExternalMapReceiver(Node):
    def __init__(self):
        super().__init__('external_map_receiver',
                         automatically_declare_parameters_from_overrides=True)

        self.tcp_port = self.get_parameter('external_map_port').value
        self.map_frame = self.get_parameter('external_robot_frame').value

        self.map_pub = self.create_publisher(OccupancyGrid, '/external_robot/map', 10)

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(('0.0.0.0', self.tcp_port))
        self._server.listen(2)
        self._server.settimeout(1.0)

        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(f'ExternalMapReceiver listening on TCP:{self.tcp_port}')

    def _accept_loop(self):
        while self._running:
            try:
                conn, addr = self._server.accept()
                self.get_logger().info(f'Map client connected: {addr}')
                t = threading.Thread(target=self._recv_loop, args=(conn,), daemon=True)
                t.start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    self.get_logger().error(f'Accept error: {e}')

    def _recv_loop(self, conn):
        conn.settimeout(5.0)
        buf = b''
        while self._running:
            try:
                # read 4-byte length prefix
                while len(buf) < 4:
                    chunk = conn.recv(4 - len(buf))
                    if not chunk:
                        return
                    buf += chunk
                data_len = struct.unpack("!I", buf[:4])[0]
                buf = buf[4:]
                # read compressed data
                while len(buf) < data_len:
                    chunk = conn.recv(data_len - len(buf))
                    if not chunk:
                        return
                    buf += chunk
                compressed = buf[:data_len]
                buf = buf[data_len:]

                raw = zlib.decompress(compressed)
                msg = deserialize_grid(raw)
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.map_frame
                self.map_pub.publish(msg)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f'Recv error: {e}')
                break
        conn.close()

    def destroy_node(self):
        self._running = False
        self._server.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExternalMapReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
