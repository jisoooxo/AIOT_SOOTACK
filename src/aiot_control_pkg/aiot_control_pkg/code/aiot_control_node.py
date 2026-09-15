#!/usr/bin/env python3

import json
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64MultiArray, String, Int8

from aiot_control_pkg.kinematics_aiot import AIOTKinematics, JOINT_MIN, JOINT_MAX, wrap_to_pi
from scipy.optimize import least_squares

# X_최적 = 0.315

#### 박스 잡기 범위
# pick_Z_MIN = 0.035
# pick_Z_MAX = 0.155
# height -> 0.02 ~ 0.08
####

DOF = 6

PNEUMATIC_ON_HOLD = 0.5
PNEUMATIC_OFF_HOLD = 1.0

FLIP_PLACE_X_OFFSET = 0.03
FLIP_SAFE_Z = 0.30
FLIP_DOWN_STEPS = 3
FLIP_RETREAT_DISTANCE = 0.02
FLIP_RETREAT_STEPS = 3
FLIP_RISE_Z = 0.20
FLIP_RISE_STEPS = 6

HOME_Q = np.zeros(DOF, dtype=float)

KEEP_PLACE_POSITIONS = [
    np.array([ 0.08, -0.30, 0.16], dtype=float),
    np.array([-0.08, -0.30, 0.16], dtype=float),
]

STATE_IDLE = 'IDLE'
STATE_WAIT_PICK_POSE = 'WAIT_PICK_POSE'
STATE_WAIT_KEEP_POSE = 'WAIT_KEEP_POSE'
STATE_WAIT_JOINT_STATE = 'WAIT_JOINT_STATE'

STATE_WAIT_APPROACH = 'WAIT_APPROACH'
STATE_WAIT_TARGET = 'WAIT_TARGET'
STATE_WAIT_HOLD = 'WAIT_HOLD'
STATE_WAIT_LIFT = 'WAIT_LIFT'

STATE_WAIT_FLIP2_PARALLEL = 'WAIT_FLIP2_PARALLEL'
STATE_WAIT_FLIP2_DOWN = 'WAIT_FLIP2_DOWN'
STATE_WAIT_FLIP2_ESCAPE = 'WAIT_FLIP2_ESCAPE'
STATE_WAIT_HOME = 'WAIT_HOME'


class AIOTControlNode(Node):

    def __init__(self):
        super().__init__('control_node')

        self.kinematics = AIOTKinematics(self.get_logger())

        self.joint_target_pub = self.create_publisher(Float64MultiArray, '/arm/joint_target', 10)
        self.joint_state_request_pub = self.create_publisher(Empty, '/arm/request_joint_state', 10)
        self.joint_waypoints_pub = self.create_publisher(Float64MultiArray, '/arm/joint_waypoints', 10)
        self.pneumatic_pub = self.create_publisher(Bool, '/pneumatic/command', 10)

        self.flip_done_pub = self.create_publisher(Int8, '/control/flip_done', 10)
        self.pick_done_pub = self.create_publisher(Bool, '/control/pick_done', 10)
        self.place_done_pub = self.create_publisher(Bool, '/control/place_done', 10)
        self.keep_done_pub = self.create_publisher(Bool, '/control/keep_done', 10)
        self.raw_pick_pose_pub = self.create_publisher(String, '/raw_pick_pose', 10)
        self.raw_keep_pick_pose_pub = self.create_publisher(String, '/raw_keep_pick_pose', 10)

        self.create_subscription(String, '/vision/pick_target', self.pick_target_callback, 10)
        self.create_subscription(String, '/pick_pose', self.pick_pose_callback, 10)
        self.create_subscription(String, '/control/plan_place', self.plan_place_callback, 10)
        self.create_subscription(String, '/vision/keep_pick_pose', self.keep_pick_target_callback, 10)
        self.create_subscription(String, '/keep_pick_pose', self.keep_pick_pose_callback, 10)
        self.create_subscription(Empty, '/arm/motion_done', self.motion_done_callback, 10)
        self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)

        self.state = STATE_IDLE
        self.task = None

        self.current_q = np.zeros(DOF, dtype=float)
        self.command_q = None

        self.index = None
        self.height = None
        self.need_flip = 0

        self.pick_position = None
        self.pick_yaw = None

        self.place_position = None
        self.place_yaw = None

        self.keep_position = None
        self.keep_yaw = None
        self.keep_place_index = 0
        
        self.approach_q = None
        self.target_q = None
        self.lift_q = None

        self.flip_down_q = None

        self.action_deadline = None
        self.timer = self.create_timer(0.05, self.timer_callback)

        self.get_logger().info('CONTROL 준비 완료')

    def pick_target_callback(self, msg):
        if self.state != STATE_IDLE:
            return

        result = self.vision_pick_target(msg)
        if result is None:
            return

        position, yaw, self.index, self.height, self.need_flip = result

        self.height = max(self.height, 0.02)

        raw_msg = String()
        raw_msg.data = json.dumps({
            'xyz': position.tolist(),
            'yaw': yaw
        })

        self.state = STATE_WAIT_PICK_POSE
        self.raw_pick_pose_pub.publish(raw_msg)

    def pick_pose_callback(self, msg):
        if self.state != STATE_WAIT_PICK_POSE:
            return

        data = json.loads(msg.data)
        self.pick_position = self.read_position(data)
        self.pick_position[2] = np.clip(self.pick_position[2], 0.035, 0.155)
        self.pick_yaw = math.radians(float(data['yaw']))

        self.task = 'pick'
        self.state = STATE_WAIT_JOINT_STATE
        self.joint_state_request_pub.publish(Empty())

    def plan_place_callback(self, msg):
        if self.state != STATE_IDLE:
            return

        data = json.loads(msg.data)
        self.place_position = self.read_position(data)
        self.place_yaw = math.radians(float(data['yaw']))

        self.task = 'place'
        self.state = STATE_WAIT_JOINT_STATE
        self.joint_state_request_pub.publish(Empty())

    def keep_pick_target_callback(self, msg):
        if self.state != STATE_IDLE:
            return

        result = self.vision_keep_pick_target(msg)
        if result is None:
            return

        position, yaw = result

        raw_msg = String()
        raw_msg.data = json.dumps({
            'xyz': position.tolist(),
            'yaw': yaw
        })

        self.state = STATE_WAIT_KEEP_POSE
        self.raw_keep_pick_pose_pub.publish(raw_msg)

    def keep_pick_pose_callback(self, msg):
        if self.state != STATE_WAIT_KEEP_POSE:
            return

        data = json.loads(msg.data)
        self.keep_position = self.read_position(data)
        self.keep_yaw = math.radians(float(data['yaw']))

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
            self.start_place()
        elif self.task == 'keep_pick':
            self.start_keep_pick()

    def vision_pick_target(self, msg):
        parts = msg.data.strip().split(',')

        if len(parts) != 7:
            self.get_logger().error(
                f"/vision/pick_target 파싱 실패: {msg.data!r}"
            )
            return None

        idx = int(parts[0])
        cx, cy, cz, height, angle1 = (float(p) for p in parts[1:6])
        need_flip = bool(int(parts[6]))

        position = np.array([cx, cy, cz], dtype=float)

        return position, angle1, idx, height, need_flip

    def vision_keep_pick_target(self, msg):
        parts = msg.data.strip().split(',')

        if len(parts) != 4:
            self.get_logger().error(
                f"/vision/keep_pick_target 파싱 실패: {msg.data!r}"
            )
            return None

        x, y, z, yaw = (float(p) for p in parts)

        position = np.array([x, y, z], dtype=float)

        return position, yaw

    def start_pick(self):
        pick_yaw = self.pick_yaw

        if self.need_flip == 1:
            if self.index == 1:
                pick_yaw += math.radians(90.0)
            elif self.index == 3:
                pick_yaw -= math.radians(90.0)

        self.publish_pneumatic(False)
        self.start_topdown(self.pick_position, pick_yaw, 'PICK')

    def finish_pick(self):
        if self.need_flip == 1:
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
        self.approach_q, self.target_q, self.lift_q = self.solve_topdown_path(position, self.current_q, yaw, name)
        self.state = STATE_WAIT_APPROACH
        self.publish_joint_target(self.approach_q)

    def solve_topdown_path(self, position, previous_q, yaw, name):
        for clearance in (0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04):
            approach = position.copy()
            approach[2] += clearance

            try:
                q_approach = self.solve_topdown_pose(approach, previous_q, yaw)
                q_target = self.solve_topdown_pose(position, q_approach, yaw)
                return q_approach, q_target, q_approach.copy()
            except RuntimeError:
                continue

        raise RuntimeError(f'{name} 탑다운 경로 생성 실패: position={position.tolist()}')

    def solve_topdown_pose(self, position, previous_q, yaw):
        position = np.asarray(position, dtype=float)
        previous_q = np.asarray(previous_q, dtype=float)
        target_axis = np.array([0.0, 0.0, -1.0])

        seeds = [previous_q[:5].copy()]

        for q3_deg in (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0):
            seed = previous_q[:5].copy()
            seed[2] = math.radians(q3_deg)
            seeds.append(np.clip(seed, JOINT_MIN[:5], JOINT_MAX[:5]))

        candidates = []

        for seed in seeds:
            def build_q(active_q):
                q = previous_q.copy()
                q[:5] = active_q
                q[5] = previous_q[5]
                return q

            def residual(active_q):
                q = build_q(active_q)
                transform = self.kinematics.fk_matrix(q)
                position_error = transform[:3, 3] - position
                axis_error = transform[:3, 2] - target_axis
                joint_delta = (q[:5] - previous_q[:5] + np.pi) % (2.0 * np.pi) - np.pi
                return np.concatenate([100.0 * position_error, 20.0 * axis_error, 0.02 * joint_delta])

            result = least_squares(residual, seed, bounds=(JOINT_MIN[:5], JOINT_MAX[:5]), max_nfev=500)

            q = build_q(result.x)
            transform = self.kinematics.fk_matrix(q)
            position_error = np.linalg.norm(transform[:3, 3] - position)
            axis_error = np.linalg.norm(transform[:3, 2] - target_axis)

            candidates.append((position_error + axis_error, position_error, axis_error, q))

        candidates.sort(key=lambda x: x[0])
        _, position_error, axis_error, q = candidates[0]

        if position_error > 0.005:
            raise RuntimeError(f'탑다운 IK 위치 오차 초과: {position_error:.4f}m')

        if axis_error > 0.02:
            raise RuntimeError(f'탑다운 IK 자세 오차 초과: {axis_error:.4f}')

        q[5] = self.target_q6(yaw, q, previous_q[5])

        self.get_logger().info(
            f'탑다운 IK: target={position.round(4).tolist()}, '
            f'pos_error={position_error:.4f}m, axis_error={axis_error:.4f}, '
            f'q={np.rad2deg(q).round(1).tolist()}'
        )

        return q

    def target_q6(self, target_yaw, q, previous_q6):
        q_zero = q.copy()
        q_zero[5] = 0.0

        base_yaw = self.kinematics.tool_yaw(q_zero)
        desired = wrap_to_pi(float(target_yaw) - base_yaw)

        candidates = np.array([desired - 2.0 * math.pi, desired, desired + 2.0 * math.pi])
        valid = candidates[(candidates >= JOINT_MIN[5]) & (candidates <= JOINT_MAX[5])]

        if valid.size == 0:
            return float(np.clip(desired, JOINT_MIN[5], JOINT_MAX[5]))

        return float(valid[np.argmin(np.abs(valid - previous_q6))])


    def start_flip_place(self):
        self.task = 'flip'

        q2_offset = self.flip_q2_offset_deg(self.height)
        q3_offset = self.flip_q3_offset_deg(self.height)

        if self.index == 1:
            self.target_q = np.deg2rad([58.0, -90.0-q2_offset, -90.0-q3_offset, 110.0, 30.0, 0.0])
        elif self.index == 2:
            self.start_flip2()
            return
        elif self.index == 3:
            self.target_q = np.deg2rad([-58.0, -90.0-q2_offset, 90.0+q3_offset, 110.0, 30.0, 0.0])
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
            return 10.0
        if height <= 0.05:
            return 8.0
        if height <= 0.06:
            return 6.0
        if height <= 0.07:
            return 5.0

        return 4.0

    def start_flip2(self):
        q_parallel = self.solve_parallel_orientation(self.current_q)
        self.state = STATE_WAIT_FLIP2_PARALLEL
        self.publish_joint_target(q_parallel)

    def start_flip2_down(self):
        place_x = self.pick_position[0] + FLIP_PLACE_X_OFFSET
        safe_position = np.array([place_x, self.pick_position[1], FLIP_SAFE_Z], dtype=float)

        waypoints = []
        q_prev = self.current_q.copy()

        q_safe = self.solve_parallel_pose(safe_position, q_prev)
        waypoints.append(q_safe)
        q_prev = q_safe

        for i in range(1, FLIP_DOWN_STEPS + 1):
            ratio = i / FLIP_DOWN_STEPS
            position = safe_position.copy()
            position[2] = FLIP_SAFE_Z + (self.height - FLIP_SAFE_Z) * ratio

            q = self.solve_parallel_pose(position, q_prev)
            waypoints.append(q)
            q_prev = q

        self.flip_down_q = waypoints[-1]
        self.state = STATE_WAIT_FLIP2_DOWN
        self.publish_joint_waypoints(waypoints)

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

            q = self.solve_parallel_pose(position, q_prev)
            waypoints.append(q)
            q_prev = q

        retreat_position = place_position.copy()
        retreat_position[0] += retreat_dx

        for i in range(1, FLIP_RISE_STEPS + 1):
            position = retreat_position.copy()
            position[2] = self.height + (FLIP_RISE_Z - self.height) * (i / FLIP_RISE_STEPS)

            q = self.solve_parallel_pose(position, q_prev)
            waypoints.append(q)
            q_prev = q

        self.state = STATE_WAIT_FLIP2_ESCAPE
        self.publish_joint_waypoints(waypoints)


    def solve_parallel_orientation(self, previous_q):
        previous_q = np.asarray(previous_q, dtype=float)
        seeds = [previous_q[:5].copy()]

        for q3_deg in (-90.0, -60.0, -30.0, 30.0, 60.0, 90.0):
            seed = previous_q[:5].copy()
            seed[2] = math.radians(q3_deg)
            seeds.append(np.clip(seed, JOINT_MIN[:5], JOINT_MAX[:5]))

        candidates = []

        for target_axis in (np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])):
            for seed in seeds:
                def build_q(active_q):
                    q = previous_q.copy()
                    q[:5] = active_q
                    q[5] = previous_q[5]
                    return q

                def residual(active_q):
                    q = build_q(active_q)
                    axis_error = self.kinematics.fk_matrix(q)[:3, 2] - target_axis
                    joint_delta = (q[:5] - previous_q[:5] + np.pi) % (2.0 * np.pi) - np.pi
                    return np.concatenate([20.0 * axis_error, 0.02 * joint_delta])

                result = least_squares(residual, seed, bounds=(JOINT_MIN[:5], JOINT_MAX[:5]), max_nfev=500)

                q = build_q(result.x)
                axis_error = np.linalg.norm(self.kinematics.fk_matrix(q)[:3, 2] - target_axis)
                joint_motion = np.linalg.norm((q[:5] - previous_q[:5] + np.pi) % (2.0 * np.pi) - np.pi)

                candidates.append((axis_error, joint_motion, q))

        candidates.sort(key=lambda x: (x[0], x[1]))
        parallel_error, _, q = candidates[0]

        if parallel_error > 0.05:
            raise RuntimeError(f'평행 자세 오차 초과: {parallel_error:.4f}')

        self.get_logger().info(
            f'평행 자세 정렬: parallel_error={parallel_error:.4f}, '
            f'q={np.rad2deg(q).round(1).tolist()}'
        )

        return q

    def solve_parallel_pose(self, position, previous_q):
        position = np.asarray(position, dtype=float)
        previous_q = np.asarray(previous_q, dtype=float)

        seeds = [previous_q[:5].copy()]

        for q3_deg in (-90.0, -60.0, -30.0, 30.0, 60.0, 90.0):
            seed = previous_q[:5].copy()
            seed[2] = math.radians(q3_deg)
            seeds.append(np.clip(seed, JOINT_MIN[:5], JOINT_MAX[:5]))

        previous_axis = self.kinematics.fk_matrix(previous_q)[:3, 2]
        target_axis = np.array([1.0, 0.0, 0.0]) if previous_axis[0] >= 0.0 else np.array([-1.0, 0.0, 0.0])

        candidates = []

        for seed in seeds:
            def build_q(active_q):
                q = previous_q.copy()
                q[:5] = active_q
                q[5] = previous_q[5]
                return q

            def residual(active_q):
                q = build_q(active_q)
                transform = self.kinematics.fk_matrix(q)
                position_error = transform[:3, 3] - position
                axis_error = transform[:3, 2] - target_axis
                joint_delta = (q[:5] - previous_q[:5] + np.pi) % (2.0 * np.pi) - np.pi
                return np.concatenate([100.0 * position_error, 20.0 * axis_error, 0.02 * joint_delta])

            result = least_squares(residual, seed, bounds=(JOINT_MIN[:5], JOINT_MAX[:5]), max_nfev=500)

            q = build_q(result.x)
            transform = self.kinematics.fk_matrix(q)
            position_error = np.linalg.norm(transform[:3, 3] - position)
            parallel_error = np.linalg.norm(transform[:3, 2] - target_axis)

            candidates.append((position_error + parallel_error, position_error, parallel_error, q))

        candidates.sort(key=lambda x: x[0])
        _, position_error, parallel_error, q = candidates[0]

        if position_error > 0.005:
            raise RuntimeError(f'평행 IK 위치 오차 초과: {position_error:.4f}m')

        if parallel_error > 0.05:
            raise RuntimeError(f'평행 IK 자세 오차 초과: {parallel_error:.4f}')

        self.get_logger().info(
            f'평행 IK: target={position.round(4).tolist()}, '
            f'pos_error={position_error:.4f}m, parallel_error={parallel_error:.4f}, '
            f'q={np.rad2deg(q).round(1).tolist()}'
        )

        return q


    def motion_done_callback(self, _msg):
        if self.command_q is not None:
            self.current_q = self.command_q.copy()
            self.command_q = None

        if self.state == STATE_WAIT_APPROACH:
            self.state = STATE_WAIT_TARGET
            self.publish_joint_target(self.target_q)
            return

        if self.state == STATE_WAIT_TARGET:
            if self.task in ('pick', 'keep_pick'):
                self.publish_pneumatic(True)
                hold_time = PNEUMATIC_ON_HOLD
            else:
                self.publish_pneumatic(False)
                hold_time = PNEUMATIC_OFF_HOLD

            self.action_deadline = time.monotonic() + hold_time
            self.state = STATE_WAIT_HOLD
            return

        if self.state == STATE_WAIT_LIFT:
            if self.task == 'pick':
                self.finish_pick()

            elif self.task in ('place', 'keep_place'):
                self.state = STATE_WAIT_HOME
                self.publish_joint_target(HOME_Q)

            elif self.task == 'keep_pick':
                self.start_keep_place()

            return

        if self.state == STATE_WAIT_FLIP2_PARALLEL:
            self.start_flip2_down()
            return

        if self.state == STATE_WAIT_FLIP2_DOWN:
            self.publish_pneumatic(False)
            self.action_deadline = time.monotonic() + PNEUMATIC_OFF_HOLD
            self.state = STATE_WAIT_HOLD
            return

        if self.state == STATE_WAIT_FLIP2_ESCAPE:
            self.state = STATE_WAIT_HOME
            self.publish_joint_target(HOME_Q)
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
                self.start_flip2_escape()
            else:
                self.state = STATE_WAIT_HOME
                self.publish_joint_target(HOME_Q)
            return

        self.state = STATE_WAIT_LIFT
        self.publish_joint_target(self.lift_q)

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
    def read_bool(value):
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            value = value.strip().lower()
            if value == 'true':
                return True
            if value == 'false':
                return False

        return

    def reset_task(self):
        self.state = STATE_IDLE
        self.task = None

        self.command_q = None
        self.action_deadline = None

        self.index = None
        self.height = None
        self.need_flip = 0

        self.pick_position = None
        self.pick_yaw = None

        self.place_position = None
        self.place_yaw = None

        self.keep_position = None
        self.keep_yaw = None

        self.approach_q = None
        self.target_q = None
        self.lift_q = None

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