#!/usr/bin/env python3

import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray, String

from alot_config import *

try:
    import serial
except ImportError:
    serial = None

try:
    from dynamixel_sdk import COMM_SUCCESS, GroupSyncWrite, PacketHandler, PortHandler
except ImportError:
    COMM_SUCCESS = 0
    GroupSyncWrite = PacketHandler = PortHandler = None


class KeepMotorControlNode(Node):
    def __init__(self):
        super().__init__("keep_motor_control_node")

        self.trajectory = None
        self.index = 0
        self.motion_timer = None

        self.port_handler = None
        self.packet_handler = None
        self.group_sync_write = None
        self.pneumatic_serial = None

        self._last_present_joint_warn_time = 0.0

        self.create_subscription(
            Float64MultiArray, TRAJECTORY_TOPIC, self.trajectory_callback, 10
        )
        self.create_subscription(
            String, PNEUMATIC_CMD_TOPIC, self.pneumatic_callback, 10
        )
        self.done_pub = self.create_publisher(Bool, CONTROL_DONE_TOPIC, 10)
        self.present_joint_pub = self.create_publisher(
            Float64MultiArray, PRESENT_JOINT_TOPIC, 10
        )

        try:
            self._open_pneumatic_serial()

            if DYNAMIXEL_ENABLED:
                self._open_dynamixel()

                if STARTUP_HOME_ENABLED:
                    self._move_to_startup_home()

                self.create_timer(
                    PRESENT_JOINT_PUBLISH_PERIOD_SEC,
                    self._publish_present_joint_state,
                )
        except BaseException:
            self._shutdown_hardware()
            raise

    def _check_result(self, dxl_id, comm_result, error, label):
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(
                f"DXL ID {dxl_id} {label}: "
                f"{self.packet_handler.getTxRxResult(comm_result)}"
            )
        if error:
            raise RuntimeError(
                f"DXL ID {dxl_id} {label}: "
                f"{self.packet_handler.getRxPacketError(error)}"
            )

    def _write_register(self, dxl_id, address, value, size):
        writer = (
            self.packet_handler.write1ByteTxRx
            if size == 1
            else self.packet_handler.write4ByteTxRx
        )
        comm_result, error = writer(
            self.port_handler,
            dxl_id,
            address,
            int(value),
        )
        self._check_result(dxl_id, comm_result, error, f"write{size} @{address}")

    def _open_dynamixel(self):
        if PortHandler is None:
            raise RuntimeError("dynamixel_sdk가 없습니다. 설치 후 다시 실행하세요.")

        self.port_handler = PortHandler(DYNAMIXEL_PORT)
        self.packet_handler = PacketHandler(DYNAMIXEL_PROTOCOL)

        if not self.port_handler.openPort():
            raise RuntimeError(f"DYNAMIXEL port open failed: {DYNAMIXEL_PORT}")
        if not self.port_handler.setBaudRate(DYNAMIXEL_BAUDRATE):
            raise RuntimeError(
                f"DYNAMIXEL baudrate set failed: {DYNAMIXEL_BAUDRATE}"
            )

        self.group_sync_write = GroupSyncWrite(
            self.port_handler,
            self.packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        for dxl_id in DXL_IDS:
            self._write_register(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, 1)
            self._write_register(
                dxl_id, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE, 1
            )
            self._write_register(
                dxl_id, ADDR_PROFILE_ACCELERATION, PROFILE_ACCELERATION, 4
            )
            self._write_register(
                dxl_id, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY, 4
            )
            self._write_register(dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, 1)

        self.get_logger().info(
            f"DYNAMIXEL 준비 완료 | IDs={DXL_IDS} | "
            f"{DYNAMIXEL_PORT} @ {DYNAMIXEL_BAUDRATE}"
        )

    def _read_present_ticks(self):
        ticks = []
        for dxl_id in DXL_IDS:
            value, comm_result, error = self.packet_handler.read4ByteTxRx(
                self.port_handler,
                dxl_id,
                ADDR_PRESENT_POSITION,
            )
            self._check_result(
                dxl_id,
                comm_result,
                error,
                f"read4 @{ADDR_PRESENT_POSITION}",
            )
            ticks.append(int(value))
        return np.asarray(ticks, dtype=int)

    @staticmethod
    def _joint_rad_to_ticks(q_rad):
        q_rad = np.asarray(q_rad, dtype=float)
        return ZERO_TICKS + np.rint(
            MOTOR_DIRECTION * q_rad * TICKS_PER_RAD
        ).astype(int)

    def _sync_write_ticks(self, ticks):
        ticks = np.asarray(ticks, dtype=int)

        if np.any(ticks < MOTOR_MIN_TICKS) or np.any(ticks > MOTOR_MAX_TICKS):
            raise RuntimeError(
                "Goal tick limit violation | "
                f"goal={ticks.tolist()} | "
                f"min={MOTOR_MIN_TICKS.tolist()} | "
                f"max={MOTOR_MAX_TICKS.tolist()}"
            )

        self.group_sync_write.clearParam()

        for dxl_id, tick in zip(DXL_IDS, ticks):
            value = int(tick) & 0xFFFFFFFF
            param = [
                value & 0xFF,
                (value >> 8) & 0xFF,
                (value >> 16) & 0xFF,
                (value >> 24) & 0xFF,
            ]
            if not self.group_sync_write.addParam(dxl_id, param):
                self.group_sync_write.clearParam()
                raise RuntimeError(f"GroupSyncWrite addParam failed: ID={dxl_id}")

        comm_result = self.group_sync_write.txPacket()
        self.group_sync_write.clearParam()

        if comm_result != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm_result))

    def _publish_present_joint_state(self):
        if (
            not DYNAMIXEL_ENABLED
            or self.motion_timer is not None
            or self.trajectory is not None
        ):
            return

        try:
            ticks = self._read_present_ticks()
            msg = Float64MultiArray()
            msg.data = (
                (ticks - ZERO_TICKS) / (MOTOR_DIRECTION * TICKS_PER_RAD)
            ).tolist()
            self.present_joint_pub.publish(msg)
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_present_joint_warn_time >= 1.0:
                self.get_logger().warning(f"Present joint publish failed: {exc}")
                self._last_present_joint_warn_time = now

    def _move_to_startup_home(self):
        self._sync_write_ticks(STARTUP_HOME_TICKS)
        start = time.monotonic()

        while True:
            if not rclpy.ok():
                raise KeyboardInterrupt

            present = self._read_present_ticks()
            error = np.abs(STARTUP_HOME_TICKS - present)

            if np.all(error <= POSITION_TOLERANCE_TICKS):
                self.get_logger().info(
                    f"Startup HOME 도착 | present={present.tolist()}"
                )
                return

            if time.monotonic() - start > STARTUP_HOME_TIMEOUT_SEC:
                raise RuntimeError(
                    "Startup HOME timeout | "
                    f"target={STARTUP_HOME_TICKS.tolist()} | "
                    f"present={present.tolist()} | "
                    f"error={error.tolist()}"
                )

            time.sleep(STARTUP_HOME_POLL_SEC)

    def trajectory_callback(self, msg):
        if self.motion_timer is not None:
            self.get_logger().warning("Motor busy | 새 trajectory 무시")
            return
        if not msg.data:
            return

        point_count = int(msg.data[0])
        flat = np.asarray(msg.data[1:], dtype=float)
        expected = point_count * DOF

        if point_count <= 0 or len(flat) != expected:
            self.get_logger().error(
                f"Invalid trajectory | got={len(flat)} | expected={expected}"
            )
            return

        self.trajectory = flat.reshape(point_count, DOF)
        self.index = 0
        self.motion_timer = self.create_timer(CONTROL_DT, self._stream_next_point)

    def _stream_next_point(self):
        if self.trajectory is None:
            return

        if self.index >= len(self.trajectory):
            final_ticks = self._joint_rad_to_ticks(self.trajectory[-1])

            if DYNAMIXEL_ENABLED:
                present = self._read_present_ticks()
                error = np.abs(final_ticks - present)

                if not np.all(error <= POSITION_TOLERANCE_TICKS):
                    return

            self.destroy_timer(self.motion_timer)
            self.motion_timer = None
            self.trajectory = None

            done = Bool()
            done.data = True
            self.done_pub.publish(done)
            return

        try:
            ticks = self._joint_rad_to_ticks(self.trajectory[self.index])
            if DYNAMIXEL_ENABLED:
                self._sync_write_ticks(ticks)
            self.index += 1
        except Exception as exc:
            self.get_logger().error(f"Motor write failed: {exc}")
            if self.motion_timer is not None:
                self.destroy_timer(self.motion_timer)
                self.motion_timer = None
            self.trajectory = None

    def _open_pneumatic_serial(self):
        if not PNEUMATIC_SERIAL_ENABLED:
            return
        if serial is None:
            self.get_logger().error("pyserial이 없습니다. pip3 install pyserial")
            return

        try:
            self.pneumatic_serial = serial.Serial(
                PNEUMATIC_SERIAL_PORT,
                PNEUMATIC_BAUDRATE,
                timeout=0.2,
            )
            time.sleep(2.0)
            self.get_logger().info(
                f"Pneumatic Arduino 준비 완료 | "
                f"{PNEUMATIC_SERIAL_PORT} @ {PNEUMATIC_BAUDRATE}"
            )
        except Exception as exc:
            self.pneumatic_serial = None
            self.get_logger().error(f"Pneumatic serial open failed: {exc}")

    def pneumatic_callback(self, msg):
        command = msg.data.strip().lower()
        serial_cmd = {
            "on": PNEUMATIC_ON_SERIAL_CMD,
            "off": PNEUMATIC_OFF_SERIAL_CMD,
        }.get(command)

        if serial_cmd is None:
            self.get_logger().warning(f"Unknown pneumatic command: {msg.data}")
            return
        if self.pneumatic_serial is None:
            self.get_logger().error(
                f"PNEUMATIC {command.upper()} failed: serial unavailable"
            )
            return

        try:
            self.pneumatic_serial.write((serial_cmd + "\n").encode("utf-8"))
            self.pneumatic_serial.flush()
            self.get_logger().info(
                f"PNEUMATIC {command.upper()} | Arduino cmd={serial_cmd}"
            )
        except Exception as exc:
            self.get_logger().error(f"Pneumatic write failed: {exc}")

    def _shutdown_hardware(self):
        if self.motion_timer is not None:
            try:
                self.destroy_timer(self.motion_timer)
            except Exception:
                pass
            self.motion_timer = None
        self.trajectory = None

        if self.packet_handler is not None and self.port_handler is not None:
            for dxl_id in DXL_IDS:
                try:
                    self.packet_handler.write1ByteTxRx(
                        self.port_handler,
                        dxl_id,
                        ADDR_TORQUE_ENABLE,
                        TORQUE_DISABLE,
                    )
                except Exception as exc:
                    try:
                        self.get_logger().error(
                            f"Torque OFF failed | ID={dxl_id} | {exc}"
                        )
                    except Exception:
                        pass

            try:
                self.get_logger().warning("DYNAMIXEL TORQUE OFF | all joints")
            except Exception:
                pass

            try:
                self.port_handler.closePort()
            except Exception:
                pass

            self.port_handler = None
            self.packet_handler = None
            self.group_sync_write = None

        if self.pneumatic_serial is not None:
            try:
                self.pneumatic_serial.close()
            except Exception:
                pass
            self.pneumatic_serial = None

    def destroy_node(self):
        self._shutdown_hardware()
        super().destroy_node()


def main(args=None):
    node = None
    rclpy.init(args=args)

    try:
        node = KeepMotorControlNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node is not None:
            node.get_logger().warning("종료 요청 감지 | torque off")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()