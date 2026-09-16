"""
PickPlan (Main -> vision, /main/plan_pick)  - 지금 집을 박스 1개에 대한 계획, JSON
    {"idx": int, "face": str, "axis": str}
    예: {"idx": 2, "face": "yz", "axis": "z"}
    이 토픽은 pick 전용이라 별도 상태/구분 필드 없음 (keep은 /main/keep_ready로 분리돼 있음)

PickTarget (vision -> 제어, /vision/pick_target)  - JSON, 7개 필드
    {"idx": int, "x": m, "y": m, "z": m, "height": m, "angle": deg, "need_flip": bool}
    예: {"idx": 2, "x": 0.123, "y": -0.045, "z": 0.382, "height": 0.071, "angle": 15.2, "need_flip": true}

FlipDone (제어 -> vision, /control/flip_done)  - 뒤집기 완료 신호 (need_flip=True였던 idx에 대해서만 옴)
    std_msgs/Int8, msg.data = idx (문자열 아님, 파싱 불필요)
    예: 2

PlaceDone (제어 -> vision, /place_done)  - 동작 완료(박스가 프레임 밖으로 집혀 나감) 신호
    "idx"
    예: "2"

SecondPickPlan (vision -> 제어, /vision/pick_target)  - /flip_done 받은 후 재측정한 2차 정보.
    need_flip=True였던 idx에 대해서만 발행됨. PickTarget과 동일한 JSON 스키마.

KeepReady (Main -> vision, /main/keep_ready)  - keep할 박스 idx
    std_msgs/Int8, msg.data = idx (문자열/JSON 아님, 파싱 불필요)
    예: 2

KeepPickPose (vision -> 제어, /vision/keep_pick_pose)  - JSON, 4개 필드
    {"x": m, "y": m, "z": m, "angle": deg}
    예: {"x": 0.123, "y": -0.045, "z": 0.382, "angle": 15.2}

BoxSizes (vision -> Main, /vision/box_sizes)  - 박스 3개 정보를 한 번에, JSON 배열
    [{"id": int, "x": m, "y": m, "z": m}, ...]
    예: [{"id": 1, "x": 0.102, "y": 0.201, "z": 0.053}, {"id": 2, ...}, {"id": 3, ...}]
---------------------------------------------------------------------------
"""

import json

belt_height = 0 # cm 기준, robot base <-> belt까지 높이

def FROM_JSON_main_plan_pick(msg_str):
    """
    '/main/plan_pick' String 메시지(msg.data) 받음
    {"idx": int, "face": str, "axis": str} -> dict
    """
    try:
        data = json.loads(msg_str)
        idx = int(data['idx'])
        goal_face = str(data['face']).strip()
        vertical_axis = str(data['axis']).strip()
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"PickPlan JSON 형식 오류: {msg_str!r} ({exc})")
  
    return {
        'idx': idx,
        'goal_face': goal_face,
        'vertical_axis': vertical_axis,
    }


def TO_JSON_vision_pick_target(pick):
    """
    compute_pick_target()이 반환한 dict -> '/vision/pick_target' String 메시지(msg.data)로 보낼 JSON 문자열.
    """
    return json.dumps({
        'idx': pick['idx'],
        'x': round(pick['cx'] * 0.01, 3),
        'y': round(pick['cy'] * 0.01, 3),
        'z': round(pick['cz'] * 0.01, 3),
        'height': round(pick['height'] * 0.01, 3),
        'angle': round(pick['angle'], 2),
        'need_flip': int(pick['need_flip']),
    })


def parse_idx_message(msg_str):
    """
    idx 하나만 담긴 String 메시지 파싱. /flip_done, /control_done 용.
    "2" -> 2
    """
    return int(msg_str.strip())


def TO_JSON_vision_keep_pick_pose(cx, cy, cz, angle):
    """
    keep_pick_pose_callback에서 계산한 평균 위치/각도(cm) -> '/vision/keep_pick_pose'로
    """
    return json.dumps({
        'x': round(cx * 0.01, 3), # 미터변환
        'y': round(cy * 0.01, 3),
        'z': round(cz * 0.01, 3),
        'angle': round(angle, 2),
    })


def TO_JSON_vision_box_sizes(entries):
    """
    entries: [(box_id, avg_w, avg_h, avg_z), ...] (cm 단위)
    -> '/vision/box_sizes'로 보낼 JSON 배열 문자열. x/y/z는 m 단위로 변환해서 보냄.
    """
    return json.dumps([
        {'idx': box_id, 'x': round(x * 0.01, 3), 'y': round(y * 0.01, 3), 'z': round(z * 0.01, 3)}
        for box_id, x, y, z in entries
    ])


# ----------------------------------------------------------------------
#   ('align', 'short'|'long')  : short-axis 또는 long-axis 기준 orientation으로 정렬
#   ('flip_3d',)               : 3D로 90도 뒤집기 (좌/우 or 앞쪽, position에 따라 방향 결정)
#   ('rotate_inplace', deg)    : 면 유지한 채 그 자리에서 자전
# ----------------------------------------------------------------------
MIDDLE_PLAN = {
    ('yz', 'y'): [('align', 'long'), ('flip_3d',), ('rotate_inplace', 90)],
    ('yz', 'z'): [('align', 'long'), ('flip_3d',)],
    ('xz', 'x'): [('align', 'short'), ('flip_3d',), ('rotate_inplace', 90)],
    ('xz', 'z'): [('align', 'short'), ('flip_3d',)],
    ('xy', 'x'): [('align', 'long')],
    ('xy', 'y'): [('align', 'short')],
}


def get_flip_plan(goal_face, vertical_axis):
    """
    goal_face: 'xy' | 'yz' | 'xz'
    vertical_axis: 'x' | 'y' | 'z'
    반환: 동작 시퀀스(list of tuples). 정의 안 된 조합이면 ValueError.
    """
    key = (goal_face, vertical_axis)
    if key not in MIDDLE_PLAN:
        raise ValueError(f"정의되지 않은 조합: {key}")
    return MIDDLE_PLAN[key]


def needs_flip(plan_steps):
    """플랜 시퀀스에 flip_3d가 포함되어 있으면 True (xy 조합만 False, yz/xz는 전부 True)."""
    return any(step[0] == 'flip_3d' for step in plan_steps)


def needs_rotate_inplace(plan_steps):
    """플랜 시퀀스에 rotate_inplace가 포함되어 있으면 True.
    (뒤집은 후 2차 orientation 계산 시, 정렬된 축에서 90도 더 돌려야 하는지 판단하는 데 씀)"""
    return any(step[0] == 'rotate_inplace' for step in plan_steps)


def get_align_axis_mode(plan_steps):
    """플랜의 첫 align 단계가 'short'인지 'long'인지 반환."""
    for step in plan_steps:
        if step[0] == 'align':
            return step[1]
    raise ValueError(f"플랜에 align 단계가 없음: {plan_steps}")


def wrap_angle_90(angle):
    """각도를 (-90, 90] 범위로 보정. +90/-90 양쪽 다 넘어가는 경우 모두 처리.
    (기존 convert_angle_axis의 wrap 로직은 빼는 방향만 처리했었는데,
    2차 orientation 계산에서 +90 하는 경우도 나와서 양방향 처리하는 공용 함수로 뺌)"""
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def convert_angle_axis(angle_short_deg, axis_mode):
    """
    box_detect_node.py의 accum_result는 orientation_axis='short' 고정값 하나만 저장하므로,
    long-axis 각도가 필요한 조합이면 여기서 수학적으로 유도함 (재계산 없이).

    angle_short_deg: short-axis 기준 각도 (-90~90), accum_result[idx]의 avg_angle
    axis_mode: 'short' | 'long'
    반환: axis_mode 기준 각도 (-90~90)
    """
    if axis_mode == 'short':
        return angle_short_deg
    # long = short - 90, 범위(-90~90) 밖으로 나가면 보정
    return wrap_angle_90(angle_short_deg - 90.0)


def compute_second_angle(avg_angle_short, rotate_inplace):
    """
    뒤집은 후 재측정한 short-axis 각도로부터 2차 orientation 각도(angle2)를 계산.
    -> 단순히 더 적게 돌려도 되는 쪽으로

    avg_angle_short: 재측정된 short-axis 기준 각도 (-90~90)
    rotate_inplace: pick_comm.needs_rotate_inplace(plan_steps) 결과
    반환: angle2 (-90~90)
    """
    angle_long = wrap_angle_90(avg_angle_short - 90.0)
    angle2_base = avg_angle_short if abs(avg_angle_short) <= abs(angle_long) else angle_long
    if rotate_inplace:
        return wrap_angle_90(angle2_base + 90.0)
    return angle2_base


def compute_pick_target(idx, goal_face, vertical_axis, accum_result):
    """
    /main/plan_pick 콜백에서 호출. accum_result[idx]의 평균값을 사용해 /vision/pick_target으로 보낼 payload 생성.
    반환: dict (GripTarget payload)
    """
    if idx not in accum_result:
        raise ValueError(f"accum_result에 idx {idx}가 없음 (아직 누적 안 됐거나 놓친 박스)")

    raw_w, raw_h, size_z, avg_angle, avg_cx, avg_cy, avg_cz, _ = accum_result[idx] # 계산해둔 누적 평균값

    # 처음 물체가 들어왔을 때 윗면 기준: 긴 변=size_x, 짧은 변=size_y, 깊이=size_z
    size_x = max(raw_w, raw_h)
    size_y = min(raw_w, raw_h)

    plan_steps = get_flip_plan(goal_face, vertical_axis)
    need_flip = needs_flip(plan_steps)
    axis_mode = get_align_axis_mode(plan_steps)
    angle = convert_angle_axis(avg_angle, axis_mode)

    # control 플래이스 위치를 위한 height
    if goal_face == 'xy':
        height_dim = 0
    elif goal_face == 'yz':
        height_dim = size_x
    else:  # goal_face == 'xz'
        height_dim = size_y

    return {
        'idx': idx,
        'cx': avg_cx, 'cy': avg_cy, 'cz': avg_cz,
        'height': height_dim / 2 + belt_height,  
        'angle': angle,
        'need_flip': int(need_flip),
    }