#!/usr/bin/env python3

import json
import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import String

## 베이스로 부터 떨어진 거리
tx = 0.0
ty = 0.0
tz = 0.0

T_BASE_BOX = np.array([
    [1.0, 0.0, 0.0, tx],
    [0.0, 1.0, 0.0, ty],
    [0.0, 0.0, 1.0, tz],
    [0.0, 0.0, 0.0, 1.0]
], dtype=float)


class PackParamNode(Node):

    def __init__(self):
        super().__init__('pack_param_node')

        # Heuristic -> Pack Param
        self.create_subscription(String, '/heuristic/plan_place', self.plan_place_callback, 10)

        # Pack Param -> Control
        self.plan_place_pub = self.create_publisher(String, '/control/plan_place', 10)

        self.get_logger().info('PACK PARAM 준비 완료')

    def plan_place_callback(self, msg):

        data = json.loads(msg.data)

        position_box = np.array([
            float(data['x']),
            float(data['y']),
            float(data['z']),
            1.0
        ])

        # BOX 원점 기준 -> Robot Base 원점 기준
        position_base = T_BASE_BOX @ position_box

        result = {
            'x': float(position_base[0]),
            'y': float(position_base[1]),
            'z': float(position_base[2])
        }

        out = String()
        out.data = json.dumps(result)
        self.plan_place_pub.publish(out)

def main(args=None):

    rclpy.init(args=args)

    node = PackParamNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()