#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Empty

from dynamixel_sdk import PortHandler, PacketHandler


# ============================================================
# Dynamixel 설정
# ============================================================

DEVICE_NAME = '/dev/ttyUSB3'
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0

DXL_ID_7 = 7
DXL_ID_8 = 8

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1

EXTENDED_POSITION_MODE = 4

RESET_POSITION = {
    DXL_ID_7: 540.0, DXL_ID_8: 60.0,
}

ALIGN_POSITION = {
    DXL_ID_7: 60.0, DXL_ID_8: 540.0,
}


class PackAlign(Node):

    def __init__(self):
        super().__init__('pack_align')

        self.port_handler = PortHandler(DEVICE_NAME)
        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise RuntimeError(f'Dynamixel 포트 열기 실패: {DEVICE_NAME}')

        if not self.port_handler.setBaudRate(BAUDRATE):
            raise RuntimeError(f'Baudrate 설정 실패: {BAUDRATE}')

        self.setup_motors()

        self.create_subscription(Bool, '/ui/select_pack', self.select_pack_callback, 10)
        self.create_subscription(Bool, '/reset', self.reset_callback, 10)

        self.get_logger().info('PACK ALIGN 준비 완료')

    def setup_motors(self):
        for dxl_id in (DXL_ID_7, DXL_ID_8):

            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                dxl_id,
                ADDR_TORQUE_ENABLE,
                TORQUE_DISABLE
            )

            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                dxl_id,
                ADDR_OPERATING_MODE,
                EXTENDED_POSITION_MODE
            )

            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                dxl_id,
                ADDR_TORQUE_ENABLE,
                TORQUE_ENABLE
            )

    def degree_to_raw(self, degree):
        return int(degree / 360.0 * 4096)


    def move(self, positions):
        for dxl_id, degree in positions.items():

            goal_raw = self.degree_to_raw(degree)

            dxl_comm_result, dxl_error = \
                self.packet_handler.write4ByteTxRx(
                    self.port_handler,
                    dxl_id,
                    ADDR_GOAL_POSITION,
                    goal_raw
                )

            if dxl_comm_result != 0:
                self.get_logger().error(
                    f'ID {dxl_id} 통신 실패: '
                    f'{self.packet_handler.getTxRxResult(dxl_comm_result)}'
                )
                continue

            if dxl_error != 0:
                self.get_logger().error(
                    f'ID {dxl_id} 오류: '
                    f'{self.packet_handler.getRxPacketError(dxl_error)}'
                )


    def select_pack_callback(self, msg):

        if not msg.data:
            return

        self.get_logger().info('PACK 정렬 자세 이동')

        self.move(ALIGN_POSITION)


    def reset_callback(self, msg):

        self.get_logger().info('PACK ALIGN 리셋')

        self.move(RESET_POSITION)


    def destroy_node(self):

        for dxl_id in (DXL_ID_7, DXL_ID_8):
            self.packet_handler.write1ByteTxRx(
                self.port_handler,
                dxl_id,
                ADDR_TORQUE_ENABLE,
                TORQUE_DISABLE
            )

        self.port_handler.closePort()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = PackAlign()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()