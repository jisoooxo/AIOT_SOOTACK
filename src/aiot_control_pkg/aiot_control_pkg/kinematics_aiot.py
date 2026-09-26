#!/usr/bin/env python3

import math
import numpy as np
from scipy.optimize import least_squares

DOF = 6

DH_D = np.array([0.1271, 0.0, 0.2200, 0.0, 0.0], dtype=float)
DH_A = np.array([0.0, 0.0, 0.0, 0.2200, 0.0], dtype=float)
DH_ALPHA = np.deg2rad([-90.0, 90.0, -90.0, 0.0, 90.0])
DH_THETA_OFFSET = np.deg2rad([0.0, 0.0, 0.0, -90.0, 90.0])

# 공압 그리퍼 끝은 6번 회전축 방향으로 이동
Z_OFFSET = 0.16

JOINT_MIN = np.deg2rad([-170.0, -120.0, -170.0, -150.0, -120.0, -360.0])
JOINT_MAX = np.deg2rad([170.0, 120.0, 170.0, 150.0, 120.0, 360.0])

IK_POSITION_WEIGHT = 100.0
IK_CONTINUITY_WEIGHT = 0.02
IK_TOLERANCE = 0.005
IK_MAX_NFEV = 300


def wrap_to_pi(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def wrapped_q_delta(q_goal, q_start):
    return (q_goal - q_start + np.pi) % (2.0 * np.pi) - np.pi


def dh_matrix(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def rz_matrix(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def tz_matrix(distance):
    transform = np.eye(4, dtype=float)
    transform[2, 3] = float(distance)
    return transform


class AIOTKinematics:

    def __init__(self, logger=None):
        self.logger = logger

    def fk_matrix(self, q):
        q = np.asarray(q, dtype=float)

        transform = np.eye(4, dtype=float)

        for i in range(5):
            transform = transform @ dh_matrix(
                q[i] + DH_THETA_OFFSET[i],
                DH_D[i],
                DH_A[i],
                DH_ALPHA[i]
            )

        # 6번 모터 yaw
        transform = transform @ rz_matrix(q[5])

        # 공압 그리퍼 길이 적용
        transform = transform @ tz_matrix(Z_OFFSET)

        return transform

    def fk_position(self, q):
        return self.fk_matrix(q)[:3, 3].copy()

    def tool_axis(self, q):
        return self.fk_matrix(q)[:3, 2].copy()

    def tool_yaw(self, q):
        rotation = self.fk_matrix(q)[:3, :3]

        return math.atan2(
            float(rotation[1, 0]),
            float(rotation[0, 0])
        )

    def target_q1(self, target, previous_q1):
        xy = np.asarray(target, dtype=float)[:2]

        if np.linalg.norm(xy) < 1.0e-8:
            return float(previous_q1)

        desired = wrap_to_pi(math.atan2(float(xy[1]), float(xy[0])))
        candidates = np.array([
            desired - 2.0 * math.pi,
            desired,
            desired + 2.0 * math.pi
        ])
        valid = candidates[
            (candidates >= JOINT_MIN[0]) & (candidates <= JOINT_MAX[0])
        ]

        if valid.size == 0:
            return float(np.clip(desired, JOINT_MIN[0], JOINT_MAX[0]))

        return float(valid[np.argmin(np.abs(valid - previous_q1))])

    def _seeds(self, previous_q, q1_target):
        result = []

        seed = previous_q.copy()
        result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))

        seed = previous_q.copy()
        seed[0] = q1_target
        result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))

        seed = np.zeros(DOF, dtype=float)
        seed[0] = q1_target
        seed[5] = previous_q[5]
        result.append(np.clip(seed, JOINT_MIN, JOINT_MAX))

        return result

    def solve_point(self, target, previous_q):
        target = np.asarray(target, dtype=float)
        previous_q = np.asarray(previous_q, dtype=float)

        if target.shape != (3,):
            raise ValueError('target은 [x, y, z] 형태여야 합니다.')
        if previous_q.shape != (DOF,):
            raise ValueError(f'previous_q는 {DOF}개 관절값이어야 합니다.')

        # 위치 IK에서는 q1~q5만 계산. q6는 공압 끝점 위치에 영향이 없으므로 유지.
        active_indices = np.arange(5, dtype=int)
        q1_target = self.target_q1(target, previous_q[0])
        candidates = []

        for seed in self._seeds(previous_q, q1_target):

            def build_q(active_q):
                q = previous_q.copy()
                q[active_indices] = active_q
                q[5] = previous_q[5]
                return q

            def residual(active_q):
                q = build_q(active_q)
                position_error = self.fk_position(q) - target
                continuity = wrapped_q_delta(
                    q[active_indices],
                    previous_q[active_indices]
                )
                return np.concatenate([
                    IK_POSITION_WEIGHT * position_error,
                    IK_CONTINUITY_WEIGHT * continuity
                ])

            result = least_squares(
                residual,
                seed[active_indices],
                bounds=(
                    JOINT_MIN[active_indices], JOINT_MAX[active_indices]
                ),
                max_nfev=IK_MAX_NFEV
            )

            q = build_q(result.x)
            position_error = float(
                np.linalg.norm(self.fk_position(q) - target)
            )
            joint_motion = float(np.linalg.norm(
                wrapped_q_delta(
                    q[active_indices],
                    previous_q[active_indices]
                )
            ))
            candidates.append((position_error, joint_motion, q))

        selected = min(candidates, key=lambda item: (item[0], item[1]))

        if selected[0] > IK_TOLERANCE:
            raise RuntimeError(
                f'IK 위치가 닿을 수 없는 곳에 있음: {selected[0]:.4f} m'
            )

        if self.logger:
            self.logger.info(
                f'공압 IK: target={target.round(4).tolist()}, '
                f'error={selected[0]:.4f}m, '
                f'q={np.rad2deg(selected[2]).round(1).tolist()}'
            )

        return selected[2]

    def solve_topdown_pose(self, position, previous_q, yaw):
        position = np.asarray(position, dtype=float)
        previous_q = np.asarray(previous_q, dtype=float)
        target_axis = np.array([0.0, 0.0, -1.0])

        q1_target = math.atan2(position[1], position[0])

        seeds = [previous_q[:5].copy()]

        for q1 in (previous_q[0], q1_target):
            for q3_deg in (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0):
                seed = previous_q[:5].copy()

                seed[0] = q1
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
                transform = self.fk_matrix(q)
                position_error = transform[:3, 3] - position
                axis_error = transform[:3, 2] - target_axis
                joint_delta = (q[:5] - previous_q[:5] + np.pi) % (2.0 * np.pi) - np.pi
                return np.concatenate([100.0 * position_error, 0.6 * axis_error, 0.01 * joint_delta])

            result = least_squares(residual, seed, bounds=(JOINT_MIN[:5], JOINT_MAX[:5]), max_nfev=500)

            q = build_q(result.x)
            transform = self.fk_matrix(q)
            position_error = np.linalg.norm(transform[:3, 3] - position)
            axis_error = np.linalg.norm(transform[:3, 2] - target_axis)
            score = (position_error / 0.005 + axis_error / 0.02)
            candidates.append((score, position_error, axis_error, q))

        candidates.sort(key=lambda x: x[0])
        _, position_error, axis_error, q = candidates[0]

        if position_error > 0.005:
            raise RuntimeError(f'탑다운 IK 위치 오차 초과: {position_error:.4f}m')

        if axis_error > 0.02:
            raise RuntimeError(f'탑다운 IK 자세 오차 초과: {axis_error:.4f}')

        q[5] = self.target_q6(yaw, q, previous_q[5], position)

        if self.logger:
            self.logger.info(
                f'탑다운 IK: target={position.round(4).tolist()}, '
                f'pos_error={position_error:.4f}m, axis_error={axis_error:.4f}, '
                f'q={np.rad2deg(q).round(1).tolist()}'
            )

        return q

    def target_q6(self, target_yaw, q, previous_q6, position):
    
        base_yaw = math.atan2(position[1], position[0])

        desired_raw = base_yaw - float(target_yaw)

        desired = wrap_to_pi(desired_raw)

        candidates = np.array([
            desired - 2.0 * math.pi,
            desired,
            desired + 2.0 * math.pi
        ])

        valid = candidates[
            (candidates >= JOINT_MIN[5]) & (candidates <= JOINT_MAX[5])
        ]

        if valid.size == 0:
            selected = float(np.clip(desired, JOINT_MIN[5], JOINT_MAX[5]))

            return selected

        selected = float(valid[np.argmin(np.abs(valid - previous_q6))])

        return selected

    def solve_parallel_pose(self, position, previous_q):
        position = np.asarray(position, dtype=float)
        previous_q = np.asarray(previous_q, dtype=float)

        q1_target = math.atan2(position[1], position[0])

        active_indices = np.array([1, 3, 4], dtype=int)

        seed = previous_q[active_indices].copy()

        radial_axis = np.array([
            math.cos(q1_target),
            math.sin(q1_target),
            0.0
        ])

        previous_axis = self.fk_matrix(previous_q)[:3, 2]

        target_axis = (
            radial_axis
            if np.dot(previous_axis, radial_axis) >= 0.0
            else -radial_axis
        )

        def build_q(active_q):
            q = previous_q.copy()

            q[0] = q1_target
            q[2] = previous_q[2]
            q[active_indices] = active_q

            return q

        def residual(active_q):
            q = build_q(active_q)

            transform = self.fk_matrix(q)

            position_error = transform[:3, 3] - position

            axis_error = transform[:3, 2] - target_axis

            joint_delta = (
                q[active_indices] - previous_q[active_indices] + np.pi
            ) % (2.0 * np.pi) - np.pi

            return np.concatenate([
                60.0 * position_error,
                10.0 * axis_error,
                0.05 * joint_delta
            ])

        result = least_squares(
            residual,
            seed,
            bounds=(JOINT_MIN[active_indices], JOINT_MAX[active_indices]),
            max_nfev=500
        )

        q = build_q(result.x)

        transform = self.fk_matrix(q)

        position_error = np.linalg.norm(transform[:3, 3] - position)

        parallel_error = np.linalg.norm(transform[:3, 2] - target_axis)

        if position_error > 0.005:
            raise RuntimeError(
                f'평행 IK 위치 오차 초과: {position_error:.4f}m'
            )

        if parallel_error > 0.05:
            raise RuntimeError(
                f'평행 IK 자세 오차 초과: {parallel_error:.4f}'
            )

        if self.logger:
            self.logger.info(
                f'평행 IK: target={position.round(4).tolist()}'
            )

        return q