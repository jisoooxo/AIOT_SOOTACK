#!/usr/bin/env python3

import json
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64MultiArray, String, Int8

from aiot_control_pkg.kinematics_aiot import AIOTKinematics, wrap_to_pi
# topdown max xy = 0.415
# X = 0.30 -> y = -0.275 ~ 0.275

# 박스 잡기 범위
# pick_Z_MIN = 0.035
# pick_Z_MAX = 0.155
# height -> 0.02 ~ 0.08

DOF = 6

PNEUMATIC_ON_HOLD = 0.5
PNEUMATIC_OFF_HOLD = 1.0

FLIP_PLACE_X_OFFSET = 0.03
FLIP_SAFE_Z = 0.25
FLIP_DOWN_STEPS = 6
FLIP_RETREAT_DISTANCE = 0.02
FLIP_RETREAT_STEPS = 3
FLIP_RISE_Z = 0.20
FLIP_RISE_STEPS = 6

HOME_Q = np.zeros(DOF, dtype=float)
CONTROL_READY = np.deg2rad([0.0, -90.0, 0.0, 113.0, 67.0, 0.0])

KEEP_PLACE_POSITIONS = [
    np.array([ 0.08, 0.30, 0.16], dtype=float),
    np.array([-0.08, 0.30, 0.16], dtype=float),
]

STATE_IDLE = 'IDLE'
STATE_WAIT_PICK_POSE = 'WAIT_PICK_POSE'
STATE_WAIT_KEEP_POSE = 'WAIT_KEEP_POSE'
STATE_WAIT_JOINT_STATE = 'WAIT_JOINT_STATE'

STATE_WAIT_APPROACH = 'WAIT_APPROACH'
STATE_WAIT_TARGET = 'WAIT_TARGET'
STATE_WAIT_HOLD = 'WAIT_HOLD'
STATE_WAIT_LIFT_1 = 'WAIT_LIFT_1'
STATE_WAIT_LIFT_2 = 'WAIT_LIFT_2'

STATE_WAIT_FLIP2_PARALLEL = 'WAIT_FLIP2_PARALLEL'
STATE_WAIT_FLIP2_DOWN = 'WAIT_FLIP2_DOWN'
STATE_WAIT_FLIP2_ESCAPE = 'WAIT_FLIP2_ESCAPE'

STATE_WAIT_FLIP2_COMPLIANCE_ON = 'WAIT_FLIP2_COMPLIANCE_ON'
STATE_WAIT_FLIP2_ADAPTIVE = 'WAIT_FLIP2_ADAPTIVE'
STATE_WAIT_FLIP2_COMPLIANCE_OFF = 'WAIT_FLIP2_COMPLIANCE_OFF'

STATE_WAIT_FLIP13_COMPLIANCE_ON = 'WAIT_FLIP13_COMPLIANCE_ON'
STATE_WAIT_FLIP13_ADAPTIVE = 'WAIT_FLIP13_ADAPTIVE'
STATE_WAIT_FLIP13_COMPLIANCE_OFF = 'WAIT_FLIP13_COMPLIANCE_OFF'

STATE_WAIT_PRE_PLACE_HOME = 'WAIT_PRE_PLACE_HOME'
STATE_WAIT_HOME = 'WAIT_HOME'


class AIOTControlNode(Node):

    def __init__(self):
        super().__init__('control_node')

        self.kinematics = AIOTKinematics(self.get_logger())

        self.joint_target_pub = self.create_publisher(Float64MultiArray, '/arm/joint_target', 10)
        self.joint_state_request_pub = self.create_publisher(Empty, '/arm/request_joint_state', 10)
        self.joint_waypoints_pub = self.create_publisher(Float64MultiArray, '/arm/joint_waypoints', 10)
        self.pneumatic_pub = self.create_publisher(Bool, '/pneumatic/command', 10)
        self.joint6_compliance_pub = self.create_publisher(Bool, '/arm/joint6_compliance', 10)

        self.flip_done_pub = self.create_publisher(Int8, '/control/flip_done', 10)
        self.pick_done_pub = self.create_publisher(Bool, '/control/pick_done', 10)
        self.place_done_pub = self.create_publisher(Bool, '/control/place_done', 10)
        self.keep_done_pub = self.create_publisher(Bool, '/control/keep_done', 10)
        self.raw_pick_pose_pub = self.create_publisher(String, '/raw_pick_pose', 10)
        self.raw_keep_pick_pose_pub = self.create_publisher(String, '/raw_keep_pick_pose', 10)

        self.create_subscription(String, '/vision/pick_target', self.pick_target_callback, 10) ## vision(raw) -> control
        self.create_subscription(String, '/pick_pose', self.pick_pose_callback, 10) ## transform -> control
        self.create_subscription(String, '/control/plan_place', self.plan_place_callback, 10)
        self.create_subscription(String, '/vision/keep_pick_pose', self.keep_pick_pose_callback, 10) ## vision(raw) -> control
        self.create_subscription(String, '/keep_pick', self.keep_pick_callback, 10) ## transform -> control
        self.create_subscription(Empty, '/arm/motion_done', self.motion_done_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.create_subscription(Bool, '/arm/joint6_compliance_done', self.joint6_compliance_done_callback, 10)

        self.state = STATE_IDLE
        self.task = None

        self.current_q = np.zeros(DOF, dtype=float)
        self.command_q = None

        self.index = None
        self.height = None
        self.need_flip = False

        self.pick_position = None
        self.pick_yaw = None

        self.place_position = None
        self.place_yaw = None

        self.keep_position = None
        self.keep_yaw = None
        self.keep_place_index = 0
        
        self.approach_q = None
        self.target_q = None
        self.lift_q_1 = None
        self.lift_q_2 = None

        self.flip_down_q = None

        self.action_deadline = None
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info('CONTROL 준비 완료')

    def pick_target_callback(self, msg):  ## vision(raw) -> control
        if self.state != STATE_IDLE:
            return
    
        result = self.vision_pick_target(msg)
        if result is None:
            return

        position, yaw, self.index, self.height, self.need_flip = result

        self.height = max(self.height, 0.02)

        raw_msg = String()
        raw_msg.data = json.dumps({
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
            'angle': yaw
        })
        self.state = STATE_WAIT_PICK_POSE
        self.raw_pick_pose_pub.publish(raw_msg)

    def pick_pose_callback(self, msg):  ## transform -> control
        if self.state != STATE_WAIT_PICK_POSE:
            return

        data = json.loads(msg.data)
        self.pick_position = self.read_position(data)
        self.pick_position[2] = np.clip(self.pick_position[2], 0.035, 0.155)
        self.pick_yaw = math.radians(float(data['angle']))

        self.task = 'pick'
        self.state = STATE_WAIT_JOINT_STATE
        self.joint_state_request_pub.publish(Empty())

    def plan_place_callback(self, msg):
        if self.state != STATE_IDLE:
            return

        data = json.loads(msg.data)
        self.place_position = self.read_position(data)
        self.place_yaw = math.radians(180.0)

        self.task = 'place'
        self.state = STATE_WAIT_JOINT_STATE
        self.joint_state_request_pub.publish(Empty())

    def keep_pick_pose_callback(self, msg):  ## vision(raw) -> control
        if self.state != STATE_IDLE:
            return

        data = json.loads(msg.data)

        position = np.array([
            float(data['x']),
            float(data['y']),
            float(data['z'])
        ], dtype=float)

        angle = float(data['angle'])

        raw_msg = String()
        raw_msg.data = json.dumps({
            'x': float(position[0]),
            'y': float(position[1]),
            'z': float(position[2]),
            'angle': angle
        })

        self.state = STATE_WAIT_KEEP_POSE
        self.raw_keep_pick_pose_pub.publish(raw_msg)

    def keep_pick_callback(self, msg):  ## transform -> control
        if self.state != STATE_WAIT_KEEP_POSE:
            return

        data = json.loads(msg.data)

        self.keep_position = self.read_position(data)
        self.keep_yaw = math.radians(
            float(data['angle'])
        )

        self.task = 'keep_pick'
        self.state = STATE_WAIT_JOINT_STATE

        self.joint_state_request_pub.publish(Empty())

    def joint_state_callback(self, msg):
        if len(msg.position) < DOF:
            return

        if self.state != STATE_WAIT_JOINT_STATE:
            return

        self.current_q = np.asarray(msg.position[:DOF], dtype=float)

        if self.task == 'pick':
            self.start_pick()
        elif self.task == 'place':
            self.state = STATE_WAIT_PRE_PLACE_HOME
            self.publish_joint_target(CONTROL_READY)
        elif self.task == 'keep_pick':
            self.start_keep_pick()

    def joint6_compliance_done_callback(self, msg):
        enabled = bool(msg.data)

        if self.state == STATE_WAIT_FLIP13_COMPLIANCE_ON and enabled:
            self.start_flip13_adaptive()
            return

        if self.state == STATE_WAIT_FLIP13_COMPLIANCE_OFF and not enabled:
            self.state = STATE_WAIT_HOME
            self.publish_joint_target(CONTROL_READY)
            return

        if self.state == STATE_WAIT_FLIP2_COMPLIANCE_ON and enabled:
            self.start_flip2_adaptive()
            return

        if self.state == STATE_WAIT_FLIP2_COMPLIANCE_OFF and not enabled:
            self.start_flip2_escape()
            return

    def vision_pick_target(self, msg):
        data = json.loads(msg.data)

        idx = int(data['idx'])

        position = np.array([
            float(data['x']),
            float(data['y']),
            float(data['z'])
        ], dtype=float)

        height = float(data['height'])
        angle = float(data['angle'])
        need_flip = bool(int(data['need_flip']))

        return (position, angle, idx, height, need_flip)

    def start_pick(self):
        pick_yaw = self.pick_yaw

        self.publish_pneumatic(False)
        self.start_topdown(self.pick_position, pick_yaw, 'PICK')

    def finish_pick(self):
        if self.need_flip:
            self.start_flip_place()
            return

        self.publish_done(self.pick_done_pub)
        self.reset_task()

    def start_place(self):
        self.start_topdown(self.place_position, self.place_yaw, 'PLACE')

    def start_keep_pick(self):
        self.publish_pneumatic(False)
        self.start_topdown(self.keep_position, self.keep_yaw, 'KEEP PICK')

    def start_keep_place(self):
        position = KEEP_PLACE_POSITIONS[self.keep_place_index]
        self.keep_place_index = (self.keep_place_index + 1) % len(KEEP_PLACE_POSITIONS)

        self.task = 'keep_place'
        self.get_logger().info(f'KEEP PLACE: position={position.tolist()}')
        self.start_topdown(position, self.keep_yaw, 'KEEP PLACE')

    def start_topdown(self, position, yaw, name):
        self.approach_q, self.target_q, self.lift_q_1, self.lift_q_2 = self.solve_topdown_path(position, self.current_q, yaw, name)
        self.state = STATE_WAIT_APPROACH
        self.publish_joint_target(self.approach_q)

    def solve_topdown_path(self, position, previous_q, yaw, name):
        approach = position.copy()
        approach[2] += 0.07

        try:
            target_yaw = yaw

            # 일반 PLACE: q6 = base - 180도
            if self.task == 'place':
                target_yaw = math.radians(180.0)

            # PICK
            # 기본: vision - base
            # flip 1: base - vision + 90
            # flip 2: base - vision     
            # flip 3: base - vision - 90
            elif self.task == 'pick' and self.need_flip:

                if self.index == 1:
                    target_yaw = yaw + math.radians(90.0)

                elif self.index == 2:
                    target_yaw = yaw

                elif self.index == 3:
                    target_yaw = yaw - math.radians(90.0)

            target_yaw = wrap_to_pi(target_yaw)

            q_approach = self.kinematics.solve_topdown_pose(
                approach,
                previous_q,
                target_yaw
            )

            q_target = self.kinematics.solve_topdown_pose(
                position,
                q_approach,
                target_yaw
            )

            q_lift_1 = q_approach.copy()

            q_lift_2 = q_lift_1.copy()

            if (
                self.task == 'pick'
                and self.need_flip
                and self.index in (1, 2, 3)
            ):
                q_lift_2[5] = math.radians(0.0)

            return q_approach, q_target, q_lift_1, q_lift_2

        except RuntimeError:
            raise RuntimeError(
                f'{name} 탑다운 경로 생성 실패: position={position.tolist()}'
            )

    def start_flip_place(self):
        self.task = 'flip'

        q2_offset = self.flip_q2_offset_deg(self.height)
        q3_offset = self.flip_q3_offset_deg(self.height)

        if self.index == 1:
            self.target_q = np.deg2rad([58.0, -90.0-q2_offset, -90.0-q3_offset, 110.0, 25.0, 0])
        elif self.index == 2:
            self.start_flip2()
            return
        elif self.index == 3:
            self.target_q = np.deg2rad([-58.0, -90.0-q2_offset, 90.0+q3_offset, 110.0, 25.0, 0])
        else:
            return

        self.state = STATE_WAIT_TARGET
        self.publish_joint_target(self.target_q)

    def flip_q2_offset_deg(self, height):
        height = float(np.clip(height, 0.02, 0.08))
        return 5.0 if height <= 0.076 else 3.0

    def flip_q3_offset_deg(self, height):
        height = float(np.clip(height, 0.02, 0.08))

        if height <= 0.03:
            return 13.0
        if height <= 0.04:
            return 12.0
        if height <= 0.05:
            return 10.0
        if height <= 0.06:
            return 8.0
        if height <= 0.076:
            return 5.0

        return 4.0

    def start_flip13_adaptive(self):
        q_adaptive = self.target_q.copy()

        q_adaptive[1] = math.radians(
            -90.0 - 10.0  ##### adaptive offset deg #####
        )
        if self.index == 1:
            q_adaptive[2] -= math.radians(2.0)

        elif self.index == 3:
            q_adaptive[2] += math.radians(2.0)

        self.state = STATE_WAIT_FLIP13_ADAPTIVE
        self.publish_joint_target(q_adaptive)

        self.get_logger().info(
            f'FLIP {self.index} adaptive place: q2={math.degrees(q_adaptive[1]):.1f}deg'
        )

    def start_flip2(self):
        place_x = self.pick_position[0] + FLIP_PLACE_X_OFFSET

        safe_position = np.array([
            place_x,
            self.pick_position[1],
            FLIP_SAFE_Z
        ], dtype=float)

        # 현재 LIFT 완료 자세에서 SAFE 위치 + 평행 자세를 동시에 만족하도록 이동
        q_safe = self.kinematics.solve_parallel_pose(
            safe_position, self.current_q
        )

        q_safe[5] = math.radians(0.0)

        self.state = STATE_WAIT_FLIP2_PARALLEL
        self.publish_joint_target(q_safe)

    def start_flip2_down(self):
        place_x = self.pick_position[0] + FLIP_PLACE_X_OFFSET

        pre_place_z = self.height + 0.01

        waypoints = []
        q_prev = self.current_q.copy()

        for i in range(1, FLIP_DOWN_STEPS + 1):
            ratio = i / FLIP_DOWN_STEPS

            position = np.array([
                place_x,
                self.pick_position[1],
                FLIP_SAFE_Z + (pre_place_z - FLIP_SAFE_Z) * ratio
            ], dtype=float)

            q = self.kinematics.solve_parallel_pose(position, q_prev)

            q[5] = math.radians(0.0)

            waypoints.append(q)
            q_prev = q

        self.flip_down_q = waypoints[-1]

        self.state = STATE_WAIT_FLIP2_DOWN
        self.publish_joint_waypoints(waypoints)
        
    def start_flip2_adaptive(self):
        place_x = self.pick_position[0] + FLIP_PLACE_X_OFFSET

        adaptive_position = np.array([
            place_x,
            self.pick_position[1],
            self.height - 0.002 ##### compliance 제어를 하면서 내려가는 높이 #####
        ], dtype=float)

        q_adaptive = self.kinematics.solve_parallel_pose(adaptive_position, self.current_q)

        q_adaptive[5] = self.current_q[5]

        self.flip_down_q = q_adaptive.copy()

        self.state = STATE_WAIT_FLIP2_ADAPTIVE
        self.publish_joint_target(q_adaptive)

    def start_flip2_escape(self):
        place_position = np.array([
            self.pick_position[0] + FLIP_PLACE_X_OFFSET,
            self.pick_position[1],
            self.height
        ], dtype=float)

        axis = self.kinematics.fk_matrix(self.flip_down_q)[:3, 2]
        retreat_dx = -FLIP_RETREAT_DISTANCE if axis[0] >= 0.0 else FLIP_RETREAT_DISTANCE

        waypoints = []
        q_prev = self.flip_down_q.copy()

        for i in range(1, FLIP_RETREAT_STEPS + 1):
            position = place_position.copy()
            position[0] += retreat_dx * (i / FLIP_RETREAT_STEPS)

            q = self.kinematics.solve_parallel_pose(position, q_prev)
            waypoints.append(q)
            q_prev = q

        retreat_position = place_position.copy()
        retreat_position[0] += retreat_dx

        for i in range(1, FLIP_RISE_STEPS + 1):
            position = retreat_position.copy()
            position[2] = self.height + (FLIP_RISE_Z - self.height) * (i / FLIP_RISE_STEPS)

            q = self.kinematics.solve_parallel_pose(position, q_prev)
            waypoints.append(q)
            q_prev = q

        self.state = STATE_WAIT_FLIP2_ESCAPE
        self.publish_joint_waypoints(waypoints)

    def command_joint6_compliance(self, enabled):
        msg = Bool()
        msg.data = enabled
        self.joint6_compliance_pub.publish(msg)

    def motion_done_callback(self, _msg):
        if self.command_q is not None:
            self.current_q = self.command_q.copy()
            self.command_q = None

        if self.state == STATE_WAIT_PRE_PLACE_HOME:
            self.start_place()
            return

        if self.state == STATE_WAIT_APPROACH:
            self.state = STATE_WAIT_TARGET
            self.publish_joint_target(self.target_q)
            return

        if self.state == STATE_WAIT_TARGET:
            if self.task == 'flip' and self.index in (1, 3):
                self.state = STATE_WAIT_FLIP13_COMPLIANCE_ON
                self.command_joint6_compliance(True)
                return

            if self.task in ('pick', 'keep_pick'):
                self.publish_pneumatic(True)
                hold_time = PNEUMATIC_ON_HOLD
            else:
                self.publish_pneumatic(False)
                hold_time = PNEUMATIC_OFF_HOLD

            self.action_deadline = time.monotonic() + hold_time
            self.state = STATE_WAIT_HOLD
            return

        if self.state == STATE_WAIT_LIFT_1:

            if (
                self.task == 'pick'
                and self.need_flip
                and self.index in (1, 2, 3)
            ):
                self.state = STATE_WAIT_LIFT_2
                self.publish_joint_target(self.lift_q_2)
                return

            if self.task == 'pick':
                self.finish_pick()

            elif self.task in ('place', 'keep_place'):
                self.state = STATE_WAIT_HOME
                self.publish_joint_target(CONTROL_READY)

            elif self.task == 'keep_pick':
                self.start_keep_place()

            return

        if self.state == STATE_WAIT_LIFT_2:
            if self.task == 'pick':
                self.finish_pick()

            return

        if self.state == STATE_WAIT_FLIP13_ADAPTIVE:
            self.publish_pneumatic(False)

            self.action_deadline = (
                time.monotonic() + PNEUMATIC_OFF_HOLD
            )

            self.state = STATE_WAIT_HOLD
            return

        if self.state == STATE_WAIT_FLIP2_PARALLEL:
            self.start_flip2_down()
            return

        if self.state == STATE_WAIT_FLIP2_DOWN:
            self.state = STATE_WAIT_FLIP2_COMPLIANCE_ON
            self.command_joint6_compliance(True)
            return

        if self.state == STATE_WAIT_FLIP2_ADAPTIVE:
            self.publish_pneumatic(False)

            self.action_deadline = (
                time.monotonic() + PNEUMATIC_OFF_HOLD
            )

            self.state = STATE_WAIT_HOLD
            return

        if self.state == STATE_WAIT_FLIP2_ESCAPE:
            self.state = STATE_WAIT_HOME
            self.publish_joint_target(CONTROL_READY)
            return

        if self.state == STATE_WAIT_HOME:
            if self.task == 'flip':
                msg = Int8()
                msg.data = self.index
                self.flip_done_pub.publish(msg)

            elif self.task == 'place':
                self.publish_done(self.place_done_pub)

            elif self.task == 'keep_place':
                self.publish_done(self.keep_done_pub)

            self.reset_task()
            return

    def timer_callback(self):
        if self.action_deadline is None or time.monotonic() < self.action_deadline:
            return

        self.action_deadline = None

        if self.state != STATE_WAIT_HOLD:
            return

        if self.task == 'flip':
            if self.index == 2:
                self.state = STATE_WAIT_FLIP2_COMPLIANCE_OFF
                self.command_joint6_compliance(False)

            elif self.index in (1, 3):
                self.state = STATE_WAIT_FLIP13_COMPLIANCE_OFF
                self.command_joint6_compliance(False)

            else:
                self.state = STATE_WAIT_HOME
                self.publish_joint_target(CONTROL_READY)

            return

        self.state = STATE_WAIT_LIFT_1
        self.publish_joint_target(self.lift_q_1)

    def publish_joint_target(self, q):
        msg = Float64MultiArray()
        msg.data = np.asarray(q, dtype=float).tolist()
        self.command_q = np.asarray(q, dtype=float).copy()
        self.joint_target_pub.publish(msg)

    def publish_joint_waypoints(self, waypoints):
        msg = Float64MultiArray()
        msg.data = np.concatenate(waypoints).tolist()
        self.command_q = np.asarray(waypoints[-1], dtype=float).copy()
        self.joint_waypoints_pub.publish(msg)

    def publish_pneumatic(self, enabled):
        msg = Bool()
        msg.data = bool(enabled)
        self.pneumatic_pub.publish(msg)

    def publish_done(self, publisher):
        msg = Bool()
        msg.data = True
        publisher.publish(msg)

    @staticmethod
    def read_position(data):
        return np.array([
            float(data['x']),
            float(data['y']),
            float(data['z'])
        ], dtype=float)

    def reset_task(self):
        self.state = STATE_IDLE
        self.task = None

        self.command_q = None
        self.action_deadline = None

        self.index = None
        self.height = None
        self.need_flip = False

        self.pick_position = None
        self.pick_yaw = None

        self.place_position = None
        self.place_yaw = None

        self.keep_position = None
        self.keep_yaw = None

        self.approach_q = None
        self.target_q = None
        self.lift_q_1 = None
        self.lift_q_2 = None

        self.flip_down_q = None


def main(args=None):
    rclpy.init(args=args)
    node = AIOTControlNode()

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