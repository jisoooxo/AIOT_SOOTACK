#!/usr/bin/env python3
"""
SOOMAC AIOT 4-DOF Arm 공통 설정.

좌표계: +X 전방, +Y 좌측, +Z 위
관절: J1 yaw, J2~J4 pitch
"""

import numpy as np

# ============================================================
# ROS topics
# ============================================================
TRAJECTORY_TOPIC = "/keep_joint_trajectory"
CONTROL_DONE_TOPIC = "/keep_control_done"
PNEUMATIC_CMD_TOPIC = "/keep_pneumatic_cmd"

# Motor Control -> Arm Control.
# 실제 joint rad를 Vision Camera -> Base 변환에 사용한다.
PRESENT_JOINT_TOPIC = "/keep_present_joint_rad"
PRESENT_JOINT_PUBLISH_PERIOD_SEC = 0.10
PRESENT_JOINT_MAX_AGE_SEC = 0.50

# box_detector의 pick point topic.
VISION_PICK_TOPIC = "/box_detector/pick_point"
VISION_CAMERA_FRAME = "camera_color_optical_frame"

# Main2 Manager <-> Arm Control (JSON carried in std_msgs/String)
MAIN2_ARM_COMMAND_TOPIC = "/main2/arm_command"
MAIN2_ARM_RESULT_TOPIC = "/main2/arm_result"
VISION_PICK_BASE_TOPIC = "/main2/vision_pick_base"

# Main <-> Main2 setting interface.
MAIN2_KEEP_SET_TOPIC = "/main/keep_set"
MAIN2_SETTING_START_TOPIC = "/main/setting_start"
MAIN2_KEEP_SET_DONE_TOPIC = "/main2/keep_set_done"
MAIN2_SETTING_DONE_TOPIC = "/main2/setting_done"
MAIN2_STATUS_TOPIC = "/main2/status"


# ============================================================
# Frame convention
# ============================================================
# Base frame: +X 전방, +Y 좌측, +Z 위.
# R_DH_PHYSICAL_TOOL: DH terminal frame <- Physical tool frame.
# HOME에서 tool +X/+Y/+Z -> base +Z/+X/+Y.
R_DH_PHYSICAL_TOOL = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
], dtype=float)

# Camera -> Physical Tool:
# camera +X -> tool -Z, +Y -> +Y, +Z -> +X.
R_PHYSICAL_TOOL_CAMERA = np.array([
    [ 0.0, 0.0, 1.0],
    [ 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=float)

# Camera origin -> Tool tip origin [camera frame, m].
CAMERA_TO_TOOL_ORIGIN_CAMERA_M = np.array([
    0.034,
    0.027,
    0.070,
], dtype=float)

# Tool tip origin -> Camera origin [physical tool frame].
T_PHYSICAL_TOOL_CAMERA_M = (
    -R_PHYSICAL_TOOL_CAMERA @ CAMERA_TO_TOOL_ORIGIN_CAMERA_M
)

# ============================================================
# Robot geometry [m]
# ============================================================
DOF = 4

GROUND_TO_J1_Z = 0.04595
J1_TO_J2_Z = 0.06745
J2_TO_J3_Z = 0.20000
J3_TO_J4_Z = 0.20000
J4_TO_TOOL_Z = 0.12610


# +joint angle -> +tick
# J2~J4는 현재 장착방향 기준 확정.
# J1은 horn 장착방향이 동일하다는 가정으로 +1.
MOTOR_DIRECTION = np.array([+1, +1, +1, +1], dtype=int)


# ============================================================
# Standard DH parameters
# ============================================================
# Standard DH:
# A_i = RotZ(theta_i) * TransZ(d_i) * TransX(a_i) * RotX(alpha_i)
#
# Standard DH:
# A_i = RotZ(theta_i) * TransZ(d_i) * TransX(a_i) * RotX(alpha_i)
#
#      theta          d [m]        a [m]   alpha
# J1 : q1             0.11340      0       -90 deg
# J2 : q2 - 90 deg    0            0.200     0 deg
# J3 : q3             0            0.200     0 deg
# J4 : q4 + 90 deg    0            0        +90 deg
# Tool: fixed         0.12610      0          0 deg

DH_D1 = GROUND_TO_J1_Z + J1_TO_J2_Z
DH_A2 = J2_TO_J3_Z
DH_A3 = J3_TO_J4_Z
DH_D_TOOL = J4_TO_TOOL_Z

DH_THETA_OFFSETS_RAD = np.deg2rad(np.array([
    0.0,    # J1
    -90.0,  # J2
    0.0,    # J3
    +90.0,  # J4
]))

DH_ALPHA_RAD = np.deg2rad(np.array([
    -90.0,  # J1
    0.0,    # J2
    0.0,    # J3
    +90.0,  # J4
]))

DH_D_M = np.array([
    DH_D1,
    0.0,
    0.0,
    0.0,
])

DH_A_M = np.array([
    0.0,
    DH_A2,
    DH_A3,
    0.0,
])

# Physical/model joint q -> Standard-DH theta sign.
# 이전 5-DOF 최종 코드와 동일하게 q를 그대로 IKPy에 전달한다.
DH_JOINT_SIGN = np.array([
    +1.0,   # J1
    +1.0,   # J2
    +1.0,   # J3
    +1.0,   # J4
])

# ============================================================
# Joint zero / limits
# ============================================================
DXL_IDS = [1, 2, 3, 4]

ZERO_TICKS = np.array([2048, 2048, 1850, 2048], dtype=int)

# XM430 / XL430 Position resolution: 4096 pulse/rev
TICKS_PER_REV = 4096
TICKS_PER_RAD = TICKS_PER_REV / (2.0 * np.pi)

HOME_JOINT_RAD = np.zeros(DOF, dtype=float)

# Joint limits [deg].
JOINT_LIMIT_LOWER_DEG = np.array([-178.0, -135.0, -135.0, -130.0])
JOINT_LIMIT_UPPER_DEG = np.array([+178.0, +135.0, +135.0, +130.0])

JOINT_LIMIT_LOWER_RAD = np.deg2rad(JOINT_LIMIT_LOWER_DEG)
JOINT_LIMIT_UPPER_RAD = np.deg2rad(JOINT_LIMIT_UPPER_DEG)

# Joint limit -> motor tick limit.
JOINT_LIMIT_LOWER_TICKS = np.rint(
    ZERO_TICKS + MOTOR_DIRECTION * JOINT_LIMIT_LOWER_RAD * TICKS_PER_RAD
).astype(int)

JOINT_LIMIT_UPPER_TICKS = np.rint(
    ZERO_TICKS + MOTOR_DIRECTION * JOINT_LIMIT_UPPER_RAD * TICKS_PER_RAD
).astype(int)

# MOTOR_DIRECTION 부호를 고려해 실제 min/max 정규화.
MOTOR_MIN_TICKS = np.minimum(JOINT_LIMIT_LOWER_TICKS, JOINT_LIMIT_UPPER_TICKS)
MOTOR_MAX_TICKS = np.maximum(JOINT_LIMIT_LOWER_TICKS, JOINT_LIMIT_UPPER_TICKS)

# ============================================================
# Pick & Place
# ============================================================

# Pick/Place 후 절대 운반 높이 [m].
# Base-frame Z=170 mm까지 수직 상승한 뒤 이동한다.
TRAVEL_HEIGHT_Z_M = 0.170

# Vision 좌표는 Main2가 샘플링한 뒤 Arm command에 사용한다.


# ============================================================
# Main2 shooting scenario
# ============================================================
# Mode 1에서 사용하는 박스 종류.
MAIN2_BOX_ORDER = ("A", "C", "F", "H")
MAIN2_MAX_COUNT_PER_TYPE = 3

# ------------------------------------------------------------
# Common named poses [tick]
# ------------------------------------------------------------
# OCR view.
OCR_VIEW_TICKS = np.array(
    [2048, 1790, 2865, 2976],
    dtype=int,
)

# AB=EF, CD=GH view pose를 재사용한다.
DEPTH_AB_EF_TICKS = np.array(
    [2175, 2305, 2250, 3326],
    dtype=int,
)
DEPTH_CD_GH_TICKS = np.array(
    [1700, 2305, 2250, 3326],
    dtype=int,
)

DEPTH_VIEW_TICKS = {
    "AB": DEPTH_AB_EF_TICKS,
    "CD": DEPTH_CD_GH_TICKS,
    "EF": DEPTH_AB_EF_TICKS,
    "GH": DEPTH_CD_GH_TICKS,
}

# Scout 이동 전 충돌 회피 자세.
ARM_SAFE_TICKS = np.array(
    [2048, 1570, 3350, 3100],
    dtype=int,
)

# Mode 2 Basket view. J1=60 tick은 현재 -178 deg 제한 안쪽이다.
BASKET_VIEW_TICKS = np.array(
    [60, 2048, 2572, 3372],
    dtype=int,
)

# Mode 2 Keep view.
KEEP_VIEW_TICKS = np.array(
    [2048, 2048, 2572, 3372],
    dtype=int,
)

# Named pose를 Arm Control에서 공통으로 사용한다.
ARM_NAMED_POSE_TICKS = {
    "OCR_VIEW": OCR_VIEW_TICKS,
    "DEPTH_VIEW_AB": DEPTH_VIEW_TICKS["AB"],
    "DEPTH_VIEW_CD": DEPTH_VIEW_TICKS["CD"],
    "DEPTH_VIEW_EF": DEPTH_VIEW_TICKS["EF"],
    "DEPTH_VIEW_GH": DEPTH_VIEW_TICKS["GH"],
    "ARM_SAFE": ARM_SAFE_TICKS,
    "BASKET_VIEW": BASKET_VIEW_TICKS,
    "KEEP_VIEW": KEEP_VIEW_TICKS,
    "HOME": ZERO_TICKS.copy(),
}

ARM_NAMED_POSE_JOINT_RAD = {
    name: (ticks - ZERO_TICKS) / (MOTOR_DIRECTION * TICKS_PER_RAD)
    for name, ticks in ARM_NAMED_POSE_TICKS.items()
}

# ------------------------------------------------------------
# Mode 1: Shelf -> Basket
# ------------------------------------------------------------
# 선반 상판 Base-frame Z [m].
SHELF_Z_M = {
    "A": -0.02,
    "C": -0.02,
    "F": -0.02,
    "H": -0.02,
}

# Mode 1 Basket drop 좌표: [-150, 15, 80] mm.
BASKET_DROP_XYZ_M = {
    "A": np.array([-0.150, 0.015, 0.080], dtype=float),
    "C": np.array([-0.150, 0.015, 0.080], dtype=float),
    "F": np.array([-0.150, 0.015, 0.080], dtype=float),
    "H": np.array([-0.150, 0.015, 0.080], dtype=float),
}

# ------------------------------------------------------------
# Mode 2 / 3 common Rail placement
# ------------------------------------------------------------
# 첫 Rail place 좌표 [300, 110, 75] mm.
RAIL_FIRST_PLACE_XYZ_M = np.array(
    [0.300, 0.110, 0.075],
    dtype=float,
)

# 두 번째/세 번째 Rail box는 55 mm 간격으로 배치한다.
RAIL_PLACE_OFFSET_M = -0.055

RAIL_PLACE_OFFSET_AXIS = "Y"

# Rail 최대 박스 수.
RAIL_MAX_BOX_COUNT = 3

# Main2 한 사이클의 Rail 총 박스 수.
MAIN2_SETTING_TOTAL_COUNT = 3

# ------------------------------------------------------------
# Timing / vision sampling
# ------------------------------------------------------------
OCR_SHOW_WAIT_SEC = 2.5
SCOUT_MOVE_WAIT_SEC = 5.0
STACK_SAMPLE_COUNT = 15
STACK_SAMPLE_TIMEOUT_SEC = 6.0
STACK_SAMPLE_MAX_SPREAD_M = 0.025

MAIN2_CALIBRATION_COMPLETE = True

# View pose 복귀 직후 Vision 재수용 대기시간.
VISION_REARM_DELAY_SEC = 1.0

# ============================================================
# Vision coordinate calibration
# ============================================================
# Camera -> Base 변환 후 적용하는 Vision Z 고정 offset.
# 아래 Robot Z compensation과는 별도 보정이다.
VISION_BASE_Z_OFFSET_M = -0.035

PNEUMATIC_ON_WAIT_SEC = 2.50
PNEUMATIC_OFF_WAIT_SEC = 1.50

# ============================================================
# Cartesian trajectory
# ============================================================
CONTROL_DT = 0.040          # 25 Hz
VERTICAL_SPEED = 0.060      # m/s
HORIZONTAL_SPEED = 0.12    # m/s
MIN_SEGMENT_TIME = 0.60
MIN_TRAJECTORY_POINTS = 15

IK_POSITION_TOLERANCE_M = 0.005
IK_ORIENTATION_TOLERANCE_DEG = 3.0

# ============================================================
# Cartesian Z compensation
# ============================================================
# r = sqrt(x^2 + y^2) 기반 Robot Z 보정.
# z_error_mm = Z_COMP_A * r_mm + Z_COMP_B
Z_COMPENSATION_ENABLED = True
Z_COMP_A = -0.1230
Z_COMP_B = -1.164

# HOME -> 첫 PICK_ABOVE는 joint-space transition.
STARTUP_TRANSITION_TIME_SEC = 2.5
RETURN_HOME_TIME_SEC = 3.0
TOOL_DOWN_VECTOR = np.array([0.0, 0.0, -1.0])

# ============================================================
# Pneumatic gripper - Arduino serial
# ============================================================
PNEUMATIC_SERIAL_ENABLED = True
PNEUMATIC_SERIAL_PORT = "/dev/ttyACM0"
PNEUMATIC_BAUDRATE = 115200

PNEUMATIC_ON_SERIAL_CMD = "1"
PNEUMATIC_OFF_SERIAL_CMD = "2"

# ============================================================
# DYNAMIXEL
# ============================================================
DYNAMIXEL_ENABLED = True
DYNAMIXEL_PORT = "/dev/ttyUSB0"
DYNAMIXEL_BAUDRATE = 1_000_000
DYNAMIXEL_PROTOCOL = 2.0

# XM430 / XL430 공통 X-series control table
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1
POSITION_CONTROL_MODE = 3

# Dynamixel 내부 profile.
PROFILE_ACCELERATION = 15
PROFILE_VELOCITY = 30

# Goal Position 도달 허용오차 [tick].
POSITION_TOLERANCE_TICKS = np.array(
    [15, 32, 27, 25],
    dtype=int,
)

# Startup homing
STARTUP_HOME_ENABLED = True
STARTUP_HOME_TICKS = np.array([2048, 2048, 1850, 2048], dtype=int)
STARTUP_HOME_TIMEOUT_SEC = 8.0
STARTUP_HOME_POLL_SEC = 0.05
