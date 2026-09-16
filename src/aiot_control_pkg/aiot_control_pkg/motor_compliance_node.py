#!/usr/bin/env python3

import math
import time
import serial
import numpy as np
import rclpy
from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64MultiArray

PORT_XH = '/dev/dynamixel_0'
PORT_XM = '/dev/dynamixel_1'
ARDUINO_PORT = '/dev/ttyUSB0'

XH_IDS = {1, 2, 3, 4}
ARM_IDS = [1, 2, 3, 4, 5, 6]
DOF = 6

BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0
ARDUINO_BAUDRATE = 115200

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

POSITION_MODE = 3
PROFILE_VELOCITY = 30
PROFILE_ACCELERATION = 20

ADDR_GOAL_CURRENT = 102

CURRENT_BASED_POSITION_MODE = 5
JOINT6_ID = 6

JOINT6_ADAPTIVE_CURRENT_RAW = 70

HOME_RAW = np.full(DOF, 2048, dtype=int)
CONTROL_READY = np.deg2rad([0.0, -90.0, 0.0, 113.0, 67.0, 0.0])
JOINT_MIN = np.deg2rad([-170.0, -120.0, -170.0, -150.0, -120.0, -360.0])
JOINT_MAX = np.deg2rad([170.0, 120.0, 170.0, 150.0, 120.0, 360.0])

CONTROL_DT = 0.02
JOINT_STATE_DT = 0.10
MIN_MOVE_TIME = 2.0
PATH_MAX_SPEED = math.radians(15.0)
MAX_Q_STEP = math.radians(40.0) + 0.02
FINISH_TOLERANCE = math.radians(2.5)


def wrapped_q_delta(q_goal, q_start):
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi


def signed_delta_tick(raw_now, raw_home):
    return (int(raw_now) - int(raw_home) + 2048) % 4096 - 2048


class MotorControlNode(Node):

    def __init__(self):
        super().__init__('motor_control_node')

        self.home_active = False
        self.home_done = False

        self.packet = PacketHandler(PROTOCOL_VERSION)
        self.port_xh = PortHandler(PORT_XH)
        self.port_xm = PortHandler(PORT_XM)
        self.xh_open = False
        self.xm_open = False
        self.arduino = None
        self.closed = False

        self.motion_active = False
        self.joint6_compliant = False
        self.start_time = None
        self.duration = 0.0
        self.q_start = np.zeros(DOF, dtype=float)
        self.q_goal = np.zeros(DOF, dtype=float)
        self.q_cmd_prev = np.zeros(DOF, dtype=float)
        self.goal_sent = False

        self.path_points = None
        self.path_cumulative = None
        self.path_total = 0.0
        self.path_active = False

        self.setup_motors()
        self.hold_current_positions()
        self.set_torque(True)
        self.setup_arduino()

        self.motion_done_pub = self.create_publisher(Empty, '/arm/motion_done', 10)
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint6_compliance_done_pub = self.create_publisher(Bool, '/arm/joint6_compliance_done', 10)

        self.create_subscription(Float64MultiArray, '/arm/joint_target', self.joint_target_callback, 10)
        self.create_subscription(Float64MultiArray, '/arm/joint_waypoints', self.joint_waypoints_callback, 10)
        self.create_subscription(Empty, '/arm/request_joint_state', self.joint_state_request_callback, 10)
        self.create_subscription(Bool, '/pneumatic/command', self.pneumatic_callback, 10)
        self.create_subscription(Bool, '/arm/joint6_compliance', self.joint6_compliance_callback, 10)

        self.control_timer = self.create_timer(CONTROL_DT, self.control_loop)
        self.joint_state_timer = self.create_timer(JOINT_STATE_DT, self.publish_current_joint_state)

        self.get_logger().info('MOTOR 준비 완료')
        self.start_home()

    def start_home(self):
        self.q_goal = CONTROL_READY
        self.q_start = self.read_current_q()
        self.q_cmd_prev = self.q_start.copy()
        self.duration = self.move_duration(self.q_start, self.q_goal)
        self.start_time = time.monotonic()
        self.motion_active = True
        self.home_active = True
        self.goal_sent = False

    def joint_target_callback(self, msg):
        if not self.home_done:
            return

        if self.motion_active:
            return

        if len(msg.data) != DOF:
            return

        q_goal = np.asarray(msg.data, dtype=float)
        if not np.all(np.isfinite(q_goal)):
            return

        self.start_motion(q_goal)

    def joint_waypoints_callback(self, msg):
        if not self.home_done:
            return

        if self.motion_active:
            return

        if len(msg.data) == 0 or len(msg.data) % DOF != 0:
            return

        waypoints = np.asarray(
            msg.data,
            dtype=float
        ).reshape(-1, DOF)

        if not np.all(np.isfinite(waypoints)):
            return

        self.start_waypoint_motion(waypoints)

    def joint6_compliance_callback(self, msg):
        enabled = bool(msg.data)

        if self.motion_active:
            result = Bool()
            result.data = self.joint6_compliant
            self.joint6_compliance_done_pub.publish(result)
            return

        self.set_joint6_compliance(enabled)

        result = Bool()
        result.data = self.joint6_compliant
        self.joint6_compliance_done_pub.publish(result)

    def set_joint6_compliance(self, enabled):
        if self.joint6_compliant == enabled:
            return

        # Operating Mode는 EEPROM 영역이므로 Torque OFF 필요
        self.write1(
            JOINT6_ID,
            ADDR_TORQUE_ENABLE,
            0
        )

        if enabled:
            self.write1(
                JOINT6_ID,
                ADDR_OPERATING_MODE,
                CURRENT_BASED_POSITION_MODE
            )
        else:
            self.write1(
                JOINT6_ID,
                ADDR_OPERATING_MODE,
                POSITION_MODE
            )

        # 모드 변경 시 profile 값이 초기화되므로 다시 설정
        self.write4(
            JOINT6_ID,
            ADDR_PROFILE_ACCELERATION,
            PROFILE_ACCELERATION
        )

        self.write4(
            JOINT6_ID,
            ADDR_PROFILE_VELOCITY,
            PROFILE_VELOCITY
        )

        if enabled:
            self.write2(
                JOINT6_ID,
                ADDR_GOAL_CURRENT,
                JOINT6_ADAPTIVE_CURRENT_RAW
            )

        # Mode 전환 중 q6가 움직였을 수도 있으므로 현재 위치를 새 goal로 잡은 뒤 Torque ON
        raw_now = self.read4(
            JOINT6_ID,
            ADDR_PRESENT_POSITION
        ) % 4096

        self.write4(
            JOINT6_ID,
            ADDR_GOAL_POSITION,
            raw_now
        )

        self.write1(
            JOINT6_ID,
            ADDR_TORQUE_ENABLE,
            1
        )

        self.joint6_compliant = enabled

    def start_motion(self, q_goal):
        self.q_goal = np.clip(np.asarray(q_goal, dtype=float), JOINT_MIN, JOINT_MAX)
        self.q_start = self.read_current_q()
        self.q_cmd_prev = self.q_start.copy()
        self.duration = self.move_duration(self.q_start, self.q_goal)
        self.start_time = time.monotonic()
        self.motion_active = True
        self.goal_sent = False

        self.get_logger().info(
            f'이동 시작: q_deg={np.rad2deg(self.q_goal).round(2).tolist()}'
        )

    def start_waypoint_motion(self, waypoints):
        q_current = self.read_current_q()

        waypoints = np.clip(
            np.asarray(waypoints, dtype=float), JOINT_MIN, JOINT_MAX
        )

        points = [q_current.copy()]

        for q in waypoints:
            delta = wrapped_q_delta(q, points[-1])

            if np.max(np.abs(delta)) > 1.0e-6:
                points.append(q.copy())

        if len(points) < 2:
            self.motion_done_pub.publish(Empty())
            return

        self.path_points = np.asarray(points, dtype=float)

        segment_lengths = []

        for i in range(len(self.path_points) - 1):
            delta = wrapped_q_delta(
                self.path_points[i + 1],
                self.path_points[i]
            )

            segment_lengths.append(float(np.max(np.abs(delta))))

        segment_lengths = np.asarray(segment_lengths, dtype=float)

        self.path_cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])

        self.path_total = float(self.path_cumulative[-1])

        self.q_start = q_current.copy()
        self.q_goal = self.path_points[-1].copy()
        self.q_cmd_prev = q_current.copy()

        self.duration = max(MIN_MOVE_TIME, 1.875 * self.path_total / PATH_MAX_SPEED)

        self.start_time = time.monotonic()

        self.motion_active = True
        self.path_active = True
        self.goal_sent = False

        self.get_logger().info(
            f'연속 경로 시작: '
            f'waypoints={len(waypoints)}, '
            f'duration={self.duration:.2f}s'
        )

    def control_loop(self):
        if not self.motion_active:
            return

        if self.path_active:
            self.control_waypoint_path()
            return

        elapsed = time.monotonic() - self.start_time
        ratio = min(elapsed / self.duration, 1.0)

        if ratio < 1.0:
            s = 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5

            q_ref = (self.q_start + wrapped_q_delta(self.q_goal, self.q_start) * s)

            q_ref = np.clip(q_ref, JOINT_MIN, JOINT_MAX)

            q_step = np.clip(
                wrapped_q_delta(q_ref, self.q_cmd_prev),
                -MAX_Q_STEP, MAX_Q_STEP
            )

            q_cmd = np.clip(
                self.q_cmd_prev + q_step,
                JOINT_MIN, JOINT_MAX
            )

            self.q_cmd_prev = q_cmd

            self.write_arm_positions(self.q_to_raw(q_cmd))

            return

        if not self.goal_sent:
            self.write_arm_positions(
                self.q_to_raw(self.q_goal)
            )

            self.q_cmd_prev = self.q_goal.copy()
            self.goal_sent = True

        q_actual = self.read_current_q()

        error_each = self.motion_error_each(
            self.q_goal,
            q_actual
        )

        error = np.max(error_each)

        if error > FINISH_TOLERANCE:
            return

        self.motion_active = False
        self.start_time = None

        if self.home_active:
            self.home_active = False
            self.home_done = True
            self.get_logger().info('HOME 정렬 완료')
            return

        self.motion_done_pub.publish(Empty())

    def motion_error_each(self, q_goal, q_actual):
        error_each = np.abs(
            wrapped_q_delta(
                q_goal,
                q_actual
            )
        )

        if self.joint6_compliant:
            error_each[5] = 0.0

        return error_each

    def control_waypoint_path(self):
        elapsed = time.monotonic() - self.start_time
        ratio = min(elapsed / self.duration, 1.0)

        if ratio < 1.0:
            s = 10.0 * ratio**3 - 15.0 * ratio**4 + 6.0 * ratio**5

            path_position = self.path_total * s

            index = np.searchsorted(
                self.path_cumulative,
                path_position,
                side='right'
            ) - 1

            index = min(
                max(index, 0),
                len(self.path_points) - 2
            )

            segment_start = self.path_cumulative[index]
            segment_end = self.path_cumulative[index + 1]

            segment_length = segment_end - segment_start

            if segment_length <= 1.0e-9:
                local_ratio = 1.0
            else:
                local_ratio = (path_position - segment_start) / segment_length

            q0 = self.path_points[index]
            q1 = self.path_points[index + 1]

            q_ref = q0 + wrapped_q_delta(q1, q0) * local_ratio

            q_ref = np.clip(q_ref, JOINT_MIN, JOINT_MAX)

            q_step = np.clip(
                wrapped_q_delta(
                    q_ref,
                    self.q_cmd_prev
                ),
                -MAX_Q_STEP, MAX_Q_STEP
            )

            q_cmd = np.clip(
                self.q_cmd_prev + q_step,
                JOINT_MIN, JOINT_MAX
            )

            self.q_cmd_prev = q_cmd

            self.write_arm_positions(self.q_to_raw(q_cmd))

            return

        if not self.goal_sent:
            self.write_arm_positions(self.q_to_raw(self.q_goal))

            self.q_cmd_prev = self.q_goal.copy()
            self.goal_sent = True

        q_actual = self.read_current_q()

        error_each = self.motion_error_each(self.q_goal, q_actual)

        error = np.max(error_each)

        if error > FINISH_TOLERANCE:
            return

        self.motion_active = False
        self.path_active = False
        self.start_time = None

        self.get_logger().info(
            f'연속 경로 완료: '
            f'error={math.degrees(error):.2f}deg'
        )

        self.motion_done_pub.publish(Empty())

    def move_duration(self, q_start, q_goal):
        max_delta = float(np.max(np.abs(wrapped_q_delta(q_goal, q_start))))
        max_speed = MAX_Q_STEP / CONTROL_DT
        required_time = 1.875 * max_delta / max_speed
        return max(MIN_MOVE_TIME, required_time * 1.2)

    def pneumatic_callback(self, msg):
        self.command_pneumatic(bool(msg.data))

    def setup_arduino(self):
            
        self.arduino = serial.Serial(
            ARDUINO_PORT,
            ARDUINO_BAUDRATE,
            timeout=0.2,
            write_timeout=1.0
        )
        time.sleep(2.0)
        self.arduino.reset_input_buffer()
        self.arduino.reset_output_buffer()
        self.command_pneumatic(False)

    def command_pneumatic(self, enabled):
        if self.arduino is None or not self.arduino.is_open:
            return

        self.arduino.write(b'ON\n' if enabled else b'OFF\n')
        self.arduino.flush()

    def joint_state_request_callback(self, _msg):
        self.publish_current_joint_state()

    def publish_current_joint_state(self):
        q = self.read_current_q()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base'
        msg.name = [f'joint{i}' for i in range(1, DOF + 1)]
        msg.position = q.tolist()

        self.joint_state_pub.publish(msg)

    def setup_motors(self):
        if not self.port_xh.openPort():
            raise RuntimeError
        self.xh_open = True

        if not self.port_xm.openPort():
            raise RuntimeError
        self.xm_open = True

        if not self.port_xh.setBaudRate(BAUDRATE):
            raise RuntimeError

        if not self.port_xm.setBaudRate(BAUDRATE):
            raise RuntimeError

        for dxl_id in ARM_IDS:
            _, comm, error = self.packet.ping(
                self.port(dxl_id),
                dxl_id
            )
            self.check_result(dxl_id, comm, error, 'ping')

        self.set_torque(False)
        time.sleep(0.05)

        for dxl_id in ARM_IDS:
            self.write1(dxl_id, ADDR_OPERATING_MODE, POSITION_MODE)
            self.write4(dxl_id, ADDR_PROFILE_ACCELERATION, PROFILE_ACCELERATION)
            self.write4(dxl_id, ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY)

    def hold_current_positions(self):
        raw = self.read_arm_positions()
        self.q_cmd_prev = self.raw_to_q(raw)
        self.write_arm_positions(raw)

    def set_torque(self, enabled):
        for dxl_id in ARM_IDS:
            self.write1(dxl_id, ADDR_TORQUE_ENABLE, int(enabled))

    def read_current_q(self):
        return self.raw_to_q(self.read_arm_positions())

    def read_arm_positions(self):
        return [
            self.read4(dxl_id, ADDR_PRESENT_POSITION) % 4096
            for dxl_id in ARM_IDS
        ]

    def write_arm_positions(self, raw_list):
        for dxl_id, raw in zip(ARM_IDS, raw_list):
            self.write4(dxl_id, ADDR_GOAL_POSITION, int(raw) % 4096)

    def raw_to_q(self, raw_list):
        return np.array([
            math.radians(
                signed_delta_tick(raw, HOME_RAW[i]) * 360.0 / 4096.0
            )
            for i, raw in enumerate(raw_list)
        ], dtype=float)

    def q_to_raw(self, q):
        q = np.clip(q, JOINT_MIN, JOINT_MAX)

        return [
            (int(HOME_RAW[i]) + round(math.degrees(q[i]) * 4096.0 / 360.0)) % 4096
            for i in range(DOF)
        ]

    def port(self, dxl_id):
        return self.port_xh if dxl_id in XH_IDS else self.port_xm

    def write1(self, dxl_id, address, value):
        comm, error = self.packet.write1ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address,
            int(value)
        )
        self.check_result(dxl_id, comm, error, 'write1')

    def write2(self, dxl_id, address, value):
        comm, error = self.packet.write2ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address,
            int(value) & 0xFFFF
        )
        self.check_result(dxl_id, comm, error, 'write2')

    def write4(self, dxl_id, address, value):
        comm, error = self.packet.write4ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address,
            int(value)
        )
        self.check_result(dxl_id, comm, error, 'write4')

    def read4(self, dxl_id, address):
        value, comm, error = self.packet.read4ByteTxRx(
            self.port(dxl_id),
            dxl_id,
            address
        )
        self.check_result(dxl_id, comm, error, 'read4')
        return int(value)

    def check_result(self, dxl_id, comm, error, action):
        if comm != COMM_SUCCESS:
            raise RuntimeError(
                f'ID {dxl_id} {action}: {self.packet.getTxRxResult(comm)}'
            )

        if error:
            raise RuntimeError(
                f'ID {dxl_id} {action}: {self.packet.getRxPacketError(error)}'
            )

    def shutdown(self):
        if self.closed:
            return

        self.closed = True

        if hasattr(self, 'control_timer'):
            self.control_timer.cancel()

        if hasattr(self, 'joint_state_timer'):
            self.joint_state_timer.cancel()

        self.command_pneumatic(False)

        if self.xh_open and self.xm_open:
            self.hold_current_positions()
            time.sleep(0.05)
            self.set_torque(False)

        if self.arduino is not None and self.arduino.is_open:
            self.arduino.close()

        if self.xh_open:
            self.port_xh.closePort()
            self.xh_open = False

        if self.xm_open:
            self.port_xm.closePort()
            self.xm_open = False


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = MotorControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is None:
            print(f'[FATAL] {exc}')
        else:
            node.get_logger().fatal(str(exc))
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()