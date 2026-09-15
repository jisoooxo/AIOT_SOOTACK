"""
PickPlan (휴리스틱 -> vision, /main/plan_pick)  - 지금 집을 박스 1개에 대한 계획
    "idx,goal_face,vertical_axis,pick"
    예: "2,yz,z,pick"   (idx=2, goal_face='yz', vertical_axis='z')

PickTarget (vision -> 제어, /vision/pick_target)
    "idx,cx,cy,cz,height,angle1,need_flip"
    예: "2,12.30,-4.50,38.20,7.10,15.20,1"
    need_flip은 '1'/'0'으로 표기

FlipDone (제어 -> vision, /control/flip_done)  - 뒤집기 완료 신호 (need_flip=True였던 idx에 대해서만 옴)
    std_msgs/Int8, msg.data = idx (문자열 아님, 파싱 불필요)
    예: 2

PlaceDone (제어 -> vision, /place_done)  - 동작 완료(박스가 프레임 밖으로 집혀 나감) 신호
    "idx"
    예: "2"

SecondPickPlan (vision -> 제어, /vision/pick_target)  - /flip_done 받은 후 재측정한 2차 정보.
    need_flip=True였던 idx에 대해서만 발행됨.
    "idx,cx,cy,cz,height,angle1,need_flip"
    예: "2,12.30,-4.50,38.20,0,15.20,0"
---------------------------------------------------------------------------
"""

belt_height = 0 # cm 기준, robot base <-> belt까지 높이

def parse_pick_plan(msg_str):
    """
    '/main/plan_pick' String 메시지(msg.data) 받음
    "idx,goal_face,vertical_axis,pick" -> dict
    4번째 필드는 항상 "pick" 
    """
    parts = msg_str.strip().split(',')
    if len(parts) != 4:
        raise ValueError(f"PickPlan 문자열 형식 오류(필드 4개 필요): {msg_str!r}")
    idx_s, goal_face, vertical_axis, tag = parts
    if tag.strip().lower() != 'pick':
        raise ValueError(f"PickPlan 4번째 필드는 'pick'이어야 함: {msg_str!r}")
    return {
        'idx': int(idx_s),
        'goal_face': goal_face.strip(), # 공백같은거 제거
        'vertical_axis': vertical_axis.strip(),
    }


def serialize_grip_target(payload):
    """
    build_grip_target()이 반환한 dict -> '/vision/pick_target' String 메시지(msg.data)로 보낼 문자열.
    """
    return (
        f"{payload['idx']},"
        f"{payload['cx']*0.01:.3f},{payload['cy']*0.01:.3f},{payload['cz']*0.01:.3f},"
        f"{payload['height']*0.01:.3f},{payload['angle1']:.2f},"
        f"{1 if payload['need_flip'] else 0}"
    )


def parse_idx_message(msg_str):
    """
    idx 하나만 담긴 String 메시지 파싱. /flip_done, /control_done 용.
    "2" -> 2
    """
    return int(msg_str.strip())


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


def build_grip_target(idx, goal_face, vertical_axis, accum_result):
    """
    /main/plan_pick 콜백에서 호출. accum_result[idx]의 평균값을 사용해 /vision/pick_target으로 보낼 payload 생성.

    idx: 1,2,3 (PickPlan에서 옴)
    goal_face, vertical_axis: PickPlan에서 옴
    accum_result: box_detect_node.py의 accum_result 딕셔너리
                  {box_id: (avg_w, avg_h, avg_z, avg_angle, avg_cx, avg_cy, avg_cz, top_entries)}

    반환: dict (GripTarget payload)
    """
    if idx not in accum_result: 
        raise ValueError(f"accum_result에 idx {idx}가 없음 (아직 누적 안 됐거나 놓친 박스)")

    avg_w, avg_h, avg_z, avg_angle, avg_cx, avg_cy, avg_cz, _ = accum_result[idx] # 계산해둔 누적 평균값

    plan_steps = get_flip_plan(goal_face, vertical_axis)
    need_flip = needs_flip(plan_steps)
    axis_mode = get_align_axis_mode(plan_steps)
    angle1 = convert_angle_axis(avg_angle, axis_mode)

    # 배치 후 바닥에서 위로 올라오는 높이는 top면에 따라 어느 축 치수가 수직이 되는지가 다름.
    # top면=xy -> z, yz -> x, xz -> y 가 배치 후 수직 축.
    # z는 현재(뒤집기 전) 이미 바닥 기준 수직 치수(avg_z)로 측정돼 있고,
    # x/y는 MIDDLE_PLAN에서 yz는 항상 align='long', xz는 항상 align='short'로 고정돼 있으므로
    # 현재 OBB 치수(avg_w, avg_h) 중 axis_mode에 맞는 쪽(긴 쪽/짧은 쪽)이 곧 그 축의 실측값이 됨.
    if goal_face == 'xy':
        height_dim = avg_z
    elif axis_mode == 'long':
        height_dim = max(avg_w, avg_h)
    else:
        height_dim = min(avg_w, avg_h)

    return {
        'idx': idx,
        'cx': avg_cx, 'cy': avg_cy, 'cz': avg_cz,
        'height': height_dim / 2 + belt_height,  # 바닥 기준 높이(cm). avg_cz는 카메라 광축 방향 depth로 별개 값.
        'angle1': angle1,
        'need_flip': need_flip,
    }


def build_second_pick_plan(idx, rotate_inplace, avg_w, avg_h, avg_z, avg_angle_short, avg_cx, avg_cy, avg_cz):
    """
    /flip_done 콜백에서 호출. 뒤집은 후 새로 10프레임 재측정한 평균값으로 /second_pick_plan payload 생성.
    need_flip=True였던 idx에 대해서만 호출됨.

    idx: PickPlan에서 왔던 그 idx
    rotate_inplace: /pick_plan 처리 시점에 pick_comm.needs_rotate_inplace(plan_steps)로 저장해둔 값
    avg_w, avg_h, avg_z, avg_angle_short, avg_cx, avg_cy, avg_cz:
        /flip_done 이후 새로 쌓은 재측정 프레임들의 평균값 (box_detect_node.py에서 계산해서 넘김,
        기존 accum_result 계산 방식과 동일한 방식 - fill_ratio 상위 TOP_K_FRAMES 평균)

    반환: dict (SecondPickPlan payload)
    """
    angle2 = compute_second_angle(avg_angle_short, rotate_inplace)
    return {
        'idx': idx,
        'cx': avg_cx, 'cy': avg_cy, 'cz': avg_cz,
        'height2': 0, # 무조건 0
        'angle2': angle2,
        'need_flip': 0, # 무조건 0
    }


def serialize_second_pick_plan(payload):
    """
    build_second_pick_plan()이 반환한 dict -> '/second_pick_plan' String 메시지(msg.data)로 보낼 문자열.
    """
    return (
        f"{payload['idx']},"
        f"{payload['cx']:.2f},{payload['cy']:.2f},{payload['cz']:.2f},"
        f"{payload['height2']:.2f},{payload['angle2']:.2f}"
    )
