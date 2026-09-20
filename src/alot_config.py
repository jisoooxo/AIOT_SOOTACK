#!/usr/bin/env python3
"""
alot_config.py

4축 Top-down Pick & Place 실기용 설정.

좌표계
+x : 로봇 정면
+y : 로봇 왼쪽
+z : 위

관절
J1 : +Z yaw
J2 : +Y pitch
J3 : +Y pitch
J4 : +Y pitch

모터
J1, J2 : XM430 계열
J3, J4 : XL430 계열

초기 자세
모든 링크가 +Z 방향으로 뻗은 자세 = 각 모터 2048 tick = 수학적 0 rad
"""

import numpy as np
from typing import Dict

# ============================================================
# ROS topics
# ============================================================
TRAJECTORY_TOPIC = "/keep_joint_trajectory"
CONTROL_DONE_TOPIC = "/keep_control_done"
PNEUMATIC_CMD_TOPIC = "/keep_pneumatic_cmd"

# Motor Control -> Arm Control
# 실제 Dynamixel Present Position을 joint rad로 변환해 publish한다.
# Vision Camera -> Base 변환에서 명령 q가 아니라 실제 q를 사용하기 위함.
PRESENT_JOINT_TOPIC = "/keep_present_joint_rad"
PRESENT_JOINT_PUBLISH_PERIOD_SEC = 0.10
PRESENT_JOINT_MAX_AGE_SEC = 0.50

# box_detect_node.py uses "~/pick_point" with node name "box_detector",
# so the resolved ROS2 topic is /box_detector/pick_point.
VISION_PICK_TOPIC = "/box_detector/pick_point"
VISION_CAMERA_FRAME = "camera_color_optical_frame"

# Main2 Manager <-> Arm Control (JSON carried in std_msgs/String)
MAIN2_ARM_COMMAND_TOPIC = "/main2/arm_command"
MAIN2_ARM_RESULT_TOPIC = "/main2/arm_result"
VISION_PICK_BASE_TOPIC = "/main2/vision_pick_base"

# Main <-> Main2 setting interface
# /main/keep_set만 std_msgs/String(JSON)을 사용한다.
# 나머지 start/done 신호는 std_msgs/Bool의 True를 단발성으로 사용한다.
MAIN2_KEEP_SET_TOPIC = "/main/keep_set"
MAIN2_SETTING_START_TOPIC = "/main/setting_start"
MAIN2_KEEP_SET_DONE_TOPIC = "/main2/keep_set_done"
MAIN2_SETTING_DONE_TOPIC = "/main2/setting_done"
MAIN2_STATUS_TOPIC = "/main2/status"


# ============================================================
# Frame convention
# ============================================================
# Base frame (right-hand rule)
#   +X : robot front
#   +Y : robot left
#   +Z : up
#
# Kinematics uses the Standard-DH convention defined below.
#
# IMPORTANT: two orientations are used in one codebase.
# 1) DH terminal frame:
#    - IKPy FK/IK internal frame.
#    - At HOME it is aligned with the Base frame.
#
# 2) Physical tool frame:
#    - Real J4 / pneumatic gripper / camera mounting frame.
#    - Same origin as the pneumatic tool tip.
#    - At HOME:
#        physical tool +X -> base +Z
#        physical tool +Y -> base +X
#        physical tool +Z -> base +Y
#
# Fixed rotation: DH terminal frame <- Physical tool frame.
R_DH_PHYSICAL_TOOL = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
], dtype=float)

# Camera optical axes relative to Physical Tool:
#
# Center-tick / straight HOME pose에서 실제 확인:
#   camera +X -> base -Y
#   camera +Y -> base +X
#   camera +Z -> base +Z
#
# Physical Tool @ HOME:
#   tool +X -> base +Z
#   tool +Y -> base +X
#   tool +Z -> base +Y
#
# 따라서 Camera -> Physical Tool:
#   camera +X -> tool -Z
#   camera +Y -> tool +Y
#   camera +Z -> tool +X
R_PHYSICAL_TOOL_CAMERA = np.array([
    [ 0.0, 0.0, 1.0],
    [ 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=float)

# Physical measurement:
# Camera origin -> Tool tip origin, expressed in CAMERA frame:
#   [+34, +27, +70] mm
CAMERA_TO_TOOL_ORIGIN_CAMERA_M = np.array([
    0.034,
    0.027,
    0.070,
], dtype=float)

# What T_physical_tool_camera needs:
# Tool tip origin -> Camera origin, expressed in PHYSICAL TOOL frame.
# Result = [-70, -27, +34] mm.
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

J1_AXIS = np.array([0.0, 0.0, 1.0])
J2_AXIS = np.array([0.0, 1.0, 0.0])
J3_AXIS = np.array([0.0, 1.0, 0.0])
J4_AXIS = np.array([0.0, 1.0, 0.0])

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
# Previous 5-DOF frame convention을 유지하되 J5 yaw를 제거한 4-DOF 구성.
# 현재 실측 치수를 반영하면 HOME(q=[0,0,0,0])에서 tool tip은
# base +Z 방향으로 총 639.5 mm 위치하고 tool frame은 base frame과 정렬된다.
#
#      theta                  d [m]                        a [m]   alpha
# J1 : q1                     45.95+67.45 mm              0       -90 deg
# J2 : q2 - 90 deg            0                            0.200   0
# J3 : q3                     0                            0.200   0
# J4 : q4 + 90 deg            0                            0       +90 deg
# Tool: fixed                 0.1261                       0       0
#
# 이 부호/오프셋은 사용자가 제공한 이전 5-DOF 최종 파라미터 파일의
# Standard-DH frame convention을 그대로 유지하고 J5 yaw만 제거한 것이다.
# q1~q4는 별도 부호 반전 없이 model joint angle 그대로 IKPy에 전달한다.

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
DXL_MODELS = ["XM430", "XM430", "XL430", "XL430"]

ZERO_TICKS = np.array([2048, 2048, 1850, 2048], dtype=int)

# XM430 / XL430 Position resolution: 4096 pulse/rev
TICKS_PER_REV = 4096
TICKS_PER_RAD = TICKS_PER_REV / (2.0 * np.pi)

HOME_JOINT_RAD = np.zeros(DOF, dtype=float)

# 아직 실제 기구 간섭 제한은 미확정.
# 우선 안전하게 넓지 않게 사용하고, 실기 확인 후 조정 권장.
JOINT_LIMIT_LOWER_DEG = np.array([-178.0, -135.0, -135.0, -130.0])
JOINT_LIMIT_UPPER_DEG = np.array([+178.0, +135.0, +135.0, +130.0])

JOINT_LIMIT_LOWER_RAD = np.deg2rad(JOINT_LIMIT_LOWER_DEG)
JOINT_LIMIT_UPPER_RAD = np.deg2rad(JOINT_LIMIT_UPPER_DEG)

# 수학적 joint limit을 2048 zero 기준 motor tick limit으로 변환.
JOINT_LIMIT_LOWER_TICKS = np.rint(
    ZERO_TICKS + MOTOR_DIRECTION * JOINT_LIMIT_LOWER_RAD * TICKS_PER_RAD
).astype(int)

JOINT_LIMIT_UPPER_TICKS = np.rint(
    ZERO_TICKS + MOTOR_DIRECTION * JOINT_LIMIT_UPPER_RAD * TICKS_PER_RAD
).astype(int)

# 방향 부호 때문에 lower/upper가 뒤집히는 경우까지 안전하게 정규화.
MOTOR_MIN_TICKS = np.minimum(JOINT_LIMIT_LOWER_TICKS, JOINT_LIMIT_UPPER_TICKS)
MOTOR_MAX_TICKS = np.maximum(JOINT_LIMIT_LOWER_TICKS, JOINT_LIMIT_UPPER_TICKS)

# ============================================================
# Pick & Place
# ============================================================
TRAVEL_EXTRA_Z = 0.020

# Pick/Place 후 주변 박스와의 충돌을 피하기 위한 절대 운반 높이 [m].
# 상대 clearance가 아니라 Base-frame Z=160 mm까지 수직 상승한 뒤 이동한다.
TRAVEL_HEIGHT_Z_M = 0.170

# Vision이 직접 Pick & Place를 시작하지 않는다.
# Main2가 VISION_PICK_BASE_TOPIC을 샘플링한 뒤 Arm command를 보낸다.
VISION_AUTO_ENABLED = False

# Initial detection pose supplied from real robot [J1, J2, J3, J4] ticks.
DETECTION_POSE_TICKS = np.array(
    [2048, 2048, 2572, 3372],
    dtype=int,
)

# Convert the detection pose to the same model-joint convention used by IKPy.
DETECTION_JOINT_RAD = (
    (DETECTION_POSE_TICKS - ZERO_TICKS)
    / (MOTOR_DIRECTION * TICKS_PER_RAD)
)

# ============================================================
# Main2 shooting scenario
# ============================================================
# 촬영용 고정 선택 종류. UI에서는 각 종류의 개수(1~3)만 바꾼다.
MAIN2_BOX_ORDER = ("A", "C", "F", "H")
MAIN2_DEFAULT_COUNTS: Dict[str, int] = {
    "A": 1,
    "C": 1,
    "F": 1,
    "H": 1,
}
MAIN2_MAX_COUNT_PER_TYPE = 3

# ------------------------------------------------------------
# Common named poses [tick]
# ------------------------------------------------------------
# OCR 연출 자세. Scout 이동 전/후 같은 자세를 재사용한다.
OCR_VIEW_TICKS = np.array(
    [2048, 1790, 2865, 2976],
    dtype=int,
)

# Scout가 이동하므로 AB=EF, CD=GH 자세를 재사용한다.
DEPTH_AB_EF_TICKS = np.array(
    [2175, 2305, 2250, 3326],
    dtype=int,
)
DEPTH_CD_GH_TICKS = np.array(
    [1900, 2305, 2250, 3326],
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

# Mode 2: Basket -> Rail 인식 자세.
# J1=60 tick은 현재 J1 -175 deg 제한 안쪽에 들어오도록 조정한 값이다.
BASKET_VIEW_TICKS = np.array(
    [60, 2048, 2572, 3372],
    dtype=int,
)

# Mode 3: Keep -> Rail 인식 자세.
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
# 선반 상판의 Base-frame Z 높이 [m]. 현재 임시값 5 mm 사용.
SHELF_Z_M = {
    "A": -0.02,
    "C": -0.02,
    "F": -0.02,
    "H": -0.02,
}

# Mode 1 공통 Basket drop 좌표.
# [-150, 15, 50] mm = [-0.150, 0.015, 0.050] m.
BASKET_DROP_XYZ_M = {
    "A": np.array([-0.150, 0.015, 0.080], dtype=float),
    "C": np.array([-0.150, 0.015, 0.080], dtype=float),
    "F": np.array([-0.150, 0.015, 0.080], dtype=float),
    "H": np.array([-0.150, 0.015, 0.080], dtype=float),
}

# ------------------------------------------------------------
# Mode 2 / 3 common Rail placement
# ------------------------------------------------------------
# 첫 Rail place 좌표 [300, 200, 150] mm.
RAIL_FIRST_PLACE_XYZ_M = np.array(
    [0.300, 0.110, 0.075],
    dtype=float,
)

# 두 번째/세 번째 박스는 Y 음의 방향으로 55 mm 간격 배치한다.
RAIL_PLACE_OFFSET_M = -0.055

# Rail offset axis.
RAIL_PLACE_OFFSET_AXIS = "Y"

# 한 번의 Rail 배치에서 최대 3개를 사용한다.
RAIL_MAX_BOX_COUNT = 3

# Main2 한 사이클에서 Rail에 올리는 총 박스 수.
MAIN2_SETTING_TOTAL_COUNT = 3

# ------------------------------------------------------------
# Timing / vision sampling
# ------------------------------------------------------------
OCR_SHOW_WAIT_SEC = 2.5
SCOUT_MOVE_WAIT_SEC = 5.0
STACK_SAMPLE_COUNT = 15
STACK_SAMPLE_TIMEOUT_SEC = 6.0
STACK_SAMPLE_MAX_SPREAD_M = 0.025

# Mode 1 파라미터 입력 완료.
MAIN2_CALIBRATION_COMPLETE = True

# Ignore vision while moving, and briefly after returning to a view pose.
VISION_REARM_DELAY_SEC = 1.0

# ============================================================
# Vision coordinate calibration
# ============================================================
# Camera -> Base 변환 후 나타나는 "비전 좌표 자체의 Z 오차" 보정.
#
# 주의:
# - 이것은 아래의 Z_COMPENSATION과 다른 보정이다.
# - VISION_BASE_Z_OFFSET_M:
#       Vision으로 계산된 Base Z 좌표에 직접 더하는 값.
#
# 예)
#   Vision Z가 실제보다 항상 약 +40 mm 높게 나오면:
#       VISION_BASE_Z_OFFSET_M = -0.040
#
# 카메라 calibration 샘플 기준 Z가 실제보다 약 +28 mm 높게 측정되어
# Base-frame vision Z에 -28 mm 고정 offset을 적용한다.
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
# 실측 calibration 결과 기반 r = sqrt(x^2 + y^2) 선형 Z 보정.
#
# 오차 정의:
#   z_error_mm = z_actual_mm - z_target_mm
#
# 보정식:
#   z_error_mm = Z_COMP_A * r_mm + Z_COMP_B
#
# 실제 IK에는 반대 방향 보정을 적용:
#   z_ik_mm = z_target_mm - z_error_mm
#
# 현재 계수는 2차 calibration 결과를 바탕으로 residual까지 반영한 값.
# 필요 시 이 두 값만 수정하면 Arm Control 전체 Cartesian IK에 반영된다.
Z_COMPENSATION_ENABLED = True
Z_COMP_A = -0.1230
Z_COMP_B = -1.164

# HOME(2048,2048,2048,2048)은 tool이 +Z를 보는 수직 자세다.
# 따라서 HOME -> 첫 PICK_ABOVE만 joint-space 준비 이동으로 수행하고,
# 이후 Pick & Place 구간은 Cartesian 수직/수평 경로를 유지한다.
STARTUP_TRANSITION_TIME_SEC = 2.5
RETURN_HOME_TIME_SEC = 3.0
TOOL_DOWN_VECTOR = np.array([0.0, 0.0, -1.0])

# ============================================================
# Pneumatic gripper - Arduino serial
# ============================================================
PNEUMATIC_SERIAL_ENABLED = True
PNEUMATIC_SERIAL_PORT = "/dev/ttyACM0"
PNEUMATIC_BAUDRATE = 115200

# 이전 실기 확인값
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
LEN_PRESENT_POSITION = 4

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1
POSITION_CONTROL_MODE = 3

# 초기 실기에서는 과격한 움직임을 피하기 위해 보수적으로 설정.
# trajectory 자체에도 quintic 가감속이 있으므로 모터 내부 profile은 너무 공격적으로 두지 않음.
PROFILE_ACCELERATION = 15
PROFILE_VELOCITY = 30

# Goal position 도달 허용오차 [tick]
# 관절별 실제 추종 오차 특성에 맞춰 개별 허용오차를 사용한다.
# J1은 비교적 정확하게 추종하고,
# J2~J4는 하중/기구 영향으로 오차가 더 크게 발생할 수 있다.
#
# 로그 기준 정상 구간에서 관찰된 최대 오차:
#   J1: 약 1~3 tick
#   J2: 약 26~27 tick
#   J3: 약 27 tick
#   J4: 약 31~32 tick
#
# 단, J3가 80~100 tick 이상 벗어나는 경우는 정상 오차로 보지 않고
# 별도의 추종/하중 문제로 판단하기 위해 tolerance를 과도하게 넓히지 않는다.
POSITION_TOLERANCE_TICKS = np.array(
    [15, 32, 27, 25],
    dtype=int,
)

# Startup homing
STARTUP_HOME_ENABLED = True
STARTUP_HOME_TICKS = np.array([2048, 2048, 1850, 2048], dtype=int)
STARTUP_HOME_TIMEOUT_SEC = 8.0
STARTUP_HOME_POLL_SEC = 0.05
