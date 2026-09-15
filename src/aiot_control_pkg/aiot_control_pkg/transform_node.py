#!/usr/bin/env python3

import json
import math
import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import String


# Base 좌표계 기준 카메라 원점 좌표
tx = 0.346  # TODO: camera x position
ty = 0.0335  # TODO: camera y position
tz = 0.53  # TODO: camera z position


# Base 기준 -> 카메라 기준:
#   Z -90 deg
#   X 180 deg
#
# p_base = T_BASE_CAMERA @ p_camera

T_BASE_CAMERA = np.array([
    [ 0.0, -1.0,  0.0, tx],
    [-1.0,  0.0,  0.0, ty],
    [ 0.0,  0.0, -1.0, tz],
    [ 0.0,  0.0,  0.0, 1.0]
], dtype=float)


class TransformNode(Node):

    def __init__(self):
        super().__init__('transform_node')

        # CONTROL -> TRANSFORM
        self.create_subscription(String, '/raw_pick_pose', self.pick_callback, 10)
        self.create_subscription(String, '/raw_keep_pick_pose', self.keep_pick_callback, 10)

        # TRANSFORM -> CONTROL
        self.pick_pub = self.create_publisher(String, '/pick_pose', 10)
        self.keep_pick_pub = self.create_publisher(String, '/keep_pick_pose', 10)

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
        try:
            data = json.loads(msg.data)

            position = self.read_position(data)
            yaw = float(data['yaw'])

            position_base = self.transform_position(position)
            yaw_base = self.transform_yaw(yaw)

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError
        ) as exc:
            self.get_logger().error(
                f'{topic_name} 형식 오류: {exc}'
            )
            return

        output = {
            'xyz': position_base.tolist(),
            'yaw': yaw_base
        }

        output_msg = String()
        output_msg.data = json.dumps(output)

        publisher.publish(output_msg)

    # ========================================================
    # Camera position -> Base position
    # ========================================================

    @staticmethod
    def transform_position(position):
        p_camera = np.array([
            position[0],
            position[1],
            position[2],
            1.0
        ], dtype=float)

        p_base = T_BASE_CAMERA @ p_camera

        return p_base[:3]

    # ========================================================
    # Camera yaw -> Base yaw
    # ========================================================

    @staticmethod
    def transform_yaw(yaw_deg):
        yaw_rad = math.radians(yaw_deg)

        direction_camera = np.array([
            math.cos(yaw_rad),
            math.sin(yaw_rad),
            0.0
        ], dtype=float)

        direction_base = (
            T_BASE_CAMERA[:3, :3] @ direction_camera
        )

        yaw_base = math.degrees(
            math.atan2(
                direction_base[1],
                direction_base[0]
            )
        )

        return (yaw_base + 180.0) % 360.0 - 180.0

    # ========================================================

    @staticmethod
    def read_position(data):
        if 'xyz' in data:
            position = np.asarray(
                data['xyz'], dtype=float
            )

        elif 'position' in data:
            position = np.asarray(
                data['position'], dtype=float
            )

        else:
            raise KeyError(
                'xyz 또는 position이 없습니다.'
            )

        if position.shape != (3,):
            raise ValueError(
                'position은 [x, y, z] 형식이어야 합니다.'
            )

        if not np.all(np.isfinite(position)):
            raise ValueError(
                'position에 유효하지 않은 값이 있습니다.'
            )

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