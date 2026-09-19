#!/usr/bin/env python3

import json
import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import String

CAMERA_TRANSLATION = np.array([
    0.346,   # x
    0.0335,  # y
    0.53     # z
], dtype=float)

R_BASE_CAMERA = np.array([
    [ 0.0, -1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0]
], dtype=float)


class TransformNode(Node):

    def __init__(self):
        super().__init__('transform_node')

        # CONTROL -> TRANSFORM
        self.create_subscription(String, '/raw_pick_pose', self.pick_callback, 10)
        self.create_subscription(String, '/raw_keep_pick_pose', self.keep_pick_callback, 10)

        # TRANSFORM -> CONTROL
        self.pick_pub = self.create_publisher(String, '/pick_pose', 10)
        self.keep_pick_pub = self.create_publisher(String, '/keep_pick', 10)

        self.get_logger().info('TRANSFORM 준비 완료')

    # ========================================================
    # PICK
    # ========================================================

    def pick_callback(self, msg):
        self.transform_and_publish(
            msg,
            self.pick_pub,
            '/raw_pick_pose'
        )

    # ========================================================
    # KEEP PICK
    # ========================================================

    def keep_pick_callback(self, msg):
        self.transform_and_publish(
            msg,
            self.keep_pick_pub,
            '/raw_keep_pick_pose'
        )

    # ========================================================
    # Transform
    # ========================================================

    def transform_and_publish(self, msg, publisher, topic_name):
        data = json.loads(msg.data)

        # Camera 기준 좌표
        position_camera = self.read_position(data)

        # Camera 기준 yaw
        yaw_camera = float(data['angle'])

        # 위치 변환
        position_rotated, position_base = self.transform_position(position_camera)

        # yaw 변환
        yaw_base = self.transform_yaw(yaw_camera)

        self.get_logger().info(
            f'\n'
            f'[{topic_name}]\n'
            f'Camera 좌표 : '
            f'[{position_camera[0]:.4f}, '
            f'{position_camera[1]:.4f}, '
            f'{position_camera[2]:.4f}]\n'
            f'Base 좌표   : '
            f'[{position_base[0]:.4f}, '
            f'{position_base[1]:.4f}, '
            f'{position_base[2]:.4f}]\n'
            f'Yaw         : '
            f'{yaw_camera:.2f}° -> {yaw_base:.2f}°'
        )

        output = {
            'x': float(position_base[0])+0.01,
            'y': float(position_base[1]),
            'z': float(position_base[2]),
            'angle': yaw_base
        }

        output_msg = String()
        output_msg.data = json.dumps(output)
        publisher.publish(output_msg)


    @staticmethod
    def transform_position(position_camera):

        # 1. Rotation
        position_rotated = R_BASE_CAMERA @ position_camera

        # 2. Translation
        position_base = position_rotated + CAMERA_TRANSLATION

        return position_rotated, position_base
    

    @staticmethod
    def transform_yaw(yaw_deg):
        yaw_base = -yaw_deg ## 비전 각도 +
        return (yaw_base + 180.0) % 360.0 - 180.0


    @staticmethod
    def read_position(data):
        position = np.array([
            float(data['x']),
            float(data['y']),
            float(data['z'])
        ], dtype=float)

        return position

def main(args=None):

    rclpy.init(args=args)

    node = TransformNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()