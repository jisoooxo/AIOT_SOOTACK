#!/usr/bin/env python3

import json
import math
import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import String


# ============================================================
# Camera -> Base Translation
# 카메라 원점의 Base 좌표
# ============================================================

CAMERA_TRANSLATION = np.array([
    0.346,   # x
    0.0335,  # y
    0.53     # z
], dtype=float)


# ============================================================
# Camera -> Base Rotation
#
# Camera 좌표계를 Base 좌표계 방향으로 변환
#
# 현재 관계:
#   Camera x -> -Base y
#   Camera y -> -Base x
#   Camera z -> -Base z
#
# p_base = R_BASE_CAMERA @ p_camera + CAMERA_TRANSLATION
# ============================================================

R_BASE_CAMERA = np.array([
    [ 0.0, -1.0,  0.0],
    [-1.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0]
], dtype=float)


class TransformNode(Node):

    def __init__(self):
        super().__init__('transform_node')

        # CONTROL -> TRANSFORM
        self.create_subscription(
            String,
            '/raw_pick_pose',
            self.pick_callback,
            10
        )

        self.create_subscription(
            String,
            '/raw_keep_pick_pose',
            self.keep_pick_callback,
            10
        )

        # TRANSFORM -> CONTROL
        self.pick_pub = self.create_publisher(
            String,
            '/pick_pose',
            10
        )

        self.keep_pick_pub = self.create_publisher(
            String,
            '/keep_pick_pose',
            10
        )

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

    def transform_and_publish(
        self,
        msg,
        publisher,
        topic_name
    ):
        try:
            data = json.loads(msg.data)

            # Camera 기준 좌표
            position_camera = self.read_position(data)

            # Camera 기준 yaw
            yaw_camera = float(data['yaw'])

            # 위치 변환
            position_rotated, position_base = (
                self.transform_position(position_camera)
            )

            # yaw 변환
            yaw_base = self.transform_yaw(yaw_camera)

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

        # ====================================================
        # 변환 결과 출력
        # ====================================================

        self.get_logger().info(
            f'\n'
            f'[{topic_name}]\n'
            f'Camera 좌표 : '
            f'[{position_camera[0]:.4f}, '
            f'{position_camera[1]:.4f}, '
            f'{position_camera[2]:.4f}]\n'
            f'Rotation 후 : '
            f'[{position_rotated[0]:.4f}, '
            f'{position_rotated[1]:.4f}, '
            f'{position_rotated[2]:.4f}]\n'
            f'Base 좌표   : '
            f'[{position_base[0]:.4f}, '
            f'{position_base[1]:.4f}, '
            f'{position_base[2]:.4f}]\n'
            f'Yaw         : '
            f'{yaw_camera:.2f}° -> {yaw_base:.2f}°'
        )

        # ====================================================
        # Publish
        # ====================================================

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
    def transform_position(position_camera):

        # 1. Rotation
        position_rotated = (
            R_BASE_CAMERA @ position_camera
        )

        # 2. Translation
        position_base = (
            position_rotated
            + CAMERA_TRANSLATION
        )

        return position_rotated, position_base

    # ========================================================
    # Camera yaw -> Base yaw
    # ========================================================

    @staticmethod
    def transform_yaw(yaw_deg):

        yaw_rad = math.radians(yaw_deg)

        # Camera xy 평면에서 yaw 방향 벡터
        direction_camera = np.array([
            math.cos(yaw_rad),
            math.sin(yaw_rad),
            0.0
        ], dtype=float)

        # Base 좌표계로 회전
        direction_base = (
            R_BASE_CAMERA @ direction_camera
        )

        yaw_base = math.degrees(
            math.atan2(
                direction_base[1],
                direction_base[0]
            )
        )

        # -180 ~ 180 범위
        return (
            (yaw_base + 180.0)
            % 360.0
            - 180.0
        )

    # ========================================================
    # Position 읽기
    # ========================================================

    @staticmethod
    def read_position(data):

        if 'xyz' in data:
            position = np.asarray(
                data['xyz'],
                dtype=float
            )

        elif 'position' in data:
            position = np.asarray(
                data['position'],
                dtype=float
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