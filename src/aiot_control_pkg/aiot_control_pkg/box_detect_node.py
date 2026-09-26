'''
AIOT용 박스 검출 ROS2 노드.

캡처(RealSense+SAM2)는 box_capture.BoxCapture
"SAM2 마스크 + depth로 박스 하나 크기 계산"은 box_size_plane.compute_box_size
box_detect_new는 ID 트래킹/프레임 누적/belt_stop/ROS 토픽만 처리
'''

from collections import defaultdict, deque

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int8, String

from . import pick_comm
from . import box_capture
from .box_capture import W, H
from .box_size_plane2 import compute_box_size2 as compute_box_size


# ---- 프레임 누적 파라미터 ----
ACCUM_FRAMES = 8             # 누적할 프레임 수
TOP_K_FRAMES = 2              # 상위 몇 프레임을 평균낼지 (fill_ratio 기준)
REFLIP_FRAMES = 8            # /flip_done 이후 2차 재측정에 모을 프레임 수
# ---- 컨베이어 정지 판단 ----
STABLE_FRAMES    = 4          # 4frame 기준 판단
STABLE_TOL_CM    = 1.0        # OBB 실측 w, h 각각의 허용 변화량 (cm)
MAX_BOXES        = 3          # 안정된 박스가 이 개수가 되면 belt_stop 발행

# ----------------------------------------------------------------------
# ID 트래킹 함수
# ----------------------------------------------------------------------
def assign_ids_ordered(frame_dets_by_x, tracked_ids):
    """
    단순 순서 매칭 (개수가 정확히 같을 때) + 노이즈 등으로 더 많이 검출됐을 때는 위치 기반 매칭.
    tracked_ids: {id: {'cx':px, 'cy':px, 'dims':(d1,d2,d3)}} - 이 함수는 이 딕셔너리를 갱신하지 않음.
                 (위치/크기는 seed 시점 값으로 고정 - 노이즈로 흔들리지 않게 하기 위함)
    frame_dets_by_x: 이번 프레임 검출값을 cx 기준 오른쪽부터 정렬한 리스트

    반환: {id: frame_dets_by_x의 인덱스} - 이번 프레임에 매칭된 것만 포함
          (검출 개수가 tracked_ids보다 적으면 매칭 안 함 - 빈 dict)
    """
    alive_ids_sorted = sorted(tracked_ids.keys())
    n_ids = len(alive_ids_sorted)
    if n_ids == 0 or len(frame_dets_by_x) < n_ids:
        return {}

    if len(frame_dets_by_x) == n_ids:
        return {tid: i for i, tid in enumerate(alive_ids_sorted)}

    # 검출 개수가 더 많은 경우 -> 저장된 위치(cx, cy) 기반, tracked id마다 하나씩 그리디하게 매칭
    used = set()
    result = {}
    for tid in alive_ids_sorted:
        tcx, tcy = tracked_ids[tid]['cx'], tracked_ids[tid]['cy']
        best_j, best_dist = None, None
        for j, det in enumerate(frame_dets_by_x):
            if j in used:
                continue
            dist = (det['cx'] - tcx) ** 2 + (det['cy'] - tcy) ** 2
            if best_dist is None or dist < best_dist:
                best_dist, best_j = dist, j
        if best_j is not None:
            result[tid] = best_j
            used.add(best_j)
    return result


def format_plane_fit(pf):
    """평면 피팅(r1 윗면 마스크) 진단값을 로그용 문자열로."""
    if not pf:
        return "plane=n/a"
    if pf['fallback']:
        return f"plane=마스크 부족(점 {pf['n_pts']}<30 -> 3구역 폴백)"
    s = (f"plane 점={pf['n_pts']} 제외={pf['n_pts'] - pf['n_keep']} "
         f"잔차 std={pf['resid_std_mm']:.2f}mm 범위=[{pf['resid_min_mm']:+.2f},{pf['resid_max_mm']:+.2f}]mm")
    if pf['note']:
        s += f" [{pf['note']}]"
    return s


def release_id(tracked_ids, box_id):
    """vision 쪽에서 해당 idx에 대한 pick_target 발행이 끝났을 때 -> 트래커에서 제거."""
    tracked_ids.pop(box_id, None)


def make_pick_target_vis(target_cx, target_cy, target_cz, fx, fy, ppx, ppy):
    """
    /vision/pick_target으로 실제 나간 중앙점(target_cx, target_cy, target_cz, cm 단위, 카메라 좌표계)을
    카메라 내부파라미터로 역투영해서 화면 표시용 점을 만듦.
    ppx, ppy: 카메라 주점(principal point, self.capture.cx/cy) - box_size_plane2.py의
    X=(xs-cx)*z/fx 투영식의 역변환에 필요.
    """
    if target_cz <= 0:
        return None
    target_px = int(target_cx * fx / target_cz + ppx)
    target_py = int(target_cy * fy / target_cz + ppy)
    return {'target_point': (target_px, target_py)}


class BoxDetectNode(Node):
    def __init__(self):
        super().__init__('box_detect')

        # ---- pub/sub ----
        self.create_subscription(Bool, '/main/vision_start', self.start_callback, 10) # 이거 오면 벨트스탑 보내기.
        self.start = False  # /main/vision_start 로 True 오기 전엔 검출 로직 안 돌림
        self.belt_stop_pub = self.create_publisher(Bool, '/belt_stop', 10)  # 한번만 발행.
        self.create_subscription(Bool, '/main/box_ready', self.box_ready_callback, 10) # 이거 오면 누적시작 -> 무조건 다 reset 해야함
        self.box_sizes_pub = self.create_publisher(String, '/vision/box_sizes', 10) 

        # pick인 경우
        self.create_subscription(String, '/main/plan_pick', self.pick_plan_callback, 10) # idx, face, axis
        self.pick_target_pub = self.create_publisher(String, '/vision/pick_target', 10) # 
        self.create_subscription(Int8, '/control/flip_done', self.flip_done_callback, 10) # idx
        # keep인 경우
        self.create_subscription(Int8, '/main/keep_ready', self.keep_plan_callback, 10) # idx -> keep
        self.keep_pose_pub = self.create_publisher(String, '/vision/keep_pick_pose', 10) # 추가 -> keep_pose


        # ---- box_ready -> 크기 누적/발행 상태 ----
        self.box_ready = False
        self.box_sizes_sent = False # box_ready 받고 나서 box_sizes 발행했는지 여부

        # ---- 캡처(RealSense + SAM2 + 대충 위치 잡기) ----
        self.capture = box_capture.BoxCapture()

        # ---- 프레임 누적 버퍼 ----
        # accum_buf[box_id] = list of (frame_no, real_w, real_h, z_cm, fill_ratio, angle_deg, cx_cm, cy_cm, cz_cm, plane_fit)
        self.accum_buf = defaultdict(list)
        self.frame_in_window = 0          # 현재 윈도우에서 처리한 프레임 수
        self.global_frame = 0             # 전체 프레임 번호
        self.accum_result = {}            # {box_id: (avg_w, avg_h, avg_z, avg_angle, avg_cx, avg_cy, avg_cz, [...])}

        # ---- 안정성 트래킹 버퍼 ----
        # stable_buf[rank_idx] = deque of (rw_cm, rh_cm), 최근 STABLE_FRAMES개
        self.stable_buf  = defaultdict(lambda: deque(maxlen=STABLE_FRAMES))
        self.stable_flag = {}    # {rank_idx: bool} - 현재 안정 여부
        self.belt_stop   = False # 컨베이어 정지 상태
        self.belt_stop_sent = False  # vision_start 한 번 켜질 때마다 /belt_stop은 딱 한 번만 발행

        # ---- ID 부여 ----
        # tracked_ids[box_id] = {'cx':px, 'cy':px, 'dims':(d1,d2,d3)} box_ready가 들어오면 오른쪽부터 1,2,3 seed.
        self.tracked_ids = {}

        # ---- pick_target 발행 시각화 상태 ----
        # pick_target_vis[idx] = make_pick_target_vis
        self.pick_target_vis = {}

        # ---- 뒤집기(2차 orientation) 관련 ----
        # rotate_inplace_needed[idx] = /plan_pick 처리 시점에 저장해둔 bool.
        # /flip_done 왔을 때 angle2 계산(pick_comm.compute_second_angle)에 재사용 - need_flip=True였던 idx만 값이 있음.
        self.rotate_inplace_needed = {}
        # reflip_pending[idx] = /flip_done 받은 이후 새로 쌓는 재측정 프레임 리스트.
        #  (frame_no, real_w, real_h, z_cm, fill_ratio, angle_deg, cx_cm, cy_cm, cz_cm)
        self.reflip_pending = {}

        self.display_scale = 0.5
        cv2.namedWindow("box detect", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("box detect", int(W * self.display_scale), int(H * self.display_scale))

    # --------------------------------------------------------------
    # 휴리스틱 -> 비전 (집을 것)
    # --------------------------------------------------------------
    def pick_plan_callback(self, msg): 
        try: 
            plan = pick_comm.FROM_JSON_main_plan_pick(msg.data) # idx, face, axis, "pick" 
        except ValueError as e:
            self.get_logger().error(f"/plan_pick 파싱 실패: {e}")
            return

        try:
            target_box = pick_comm.compute_pick_target( # 픽 플랜 기반으로 누적값 가져와서 vision_pick_target 발행
                idx=plan['idx'],
                goal_face=plan['goal_face'],
                vertical_axis=plan['vertical_axis'],
                accum_result=self.accum_result,
            )
        except ValueError as e:
            self.get_logger().error(f"/pick_target 생성 실패: {e}")
            return

        # need_flip=True인 경우, /flip_done 왔을 때 2차 각도 계산에 쓸 rotate_inplace 여부를 미리 저장해둠.
        if target_box['need_flip']:
            plan_steps = pick_comm.get_flip_plan(plan['goal_face'], plan['vertical_axis'])
            self.rotate_inplace_needed[plan['idx']] = pick_comm.needs_rotate_inplace(plan_steps)

        out = String()
        out.data = pick_comm.TO_JSON_vision_pick_target(target_box)
        self.pick_target_pub.publish(out)
        self.get_logger().info(f"/pick_target 발행: {out.data}")

        # 화면 표시용 - pick_target 발행할 때마다 그 idx 중앙점을 새로 표시
        vis = make_pick_target_vis(target_box['cx'], target_box['cy'], target_box['cz'],
                                    self.capture.fx, self.capture.fy,
                                    self.capture.cx, self.capture.cy)
        if vis is not None:
            self.pick_target_vis[plan['idx']] = vis

        # id 해제 -> need_flip=False 기준
        if not target_box['need_flip']:
            release_id(self.tracked_ids, plan['idx'])
            self.rotate_inplace_needed.pop(plan['idx'], None)
            self.reflip_pending.pop(plan['idx'], None)
            self.get_logger().info(f"idx={plan['idx']} need_flip=0 - id 즉시 해제")

    # --------------------------------------------------------------
    # /flip_done 콜백
    # --------------------------------------------------------------
    def flip_done_callback(self, msg):
        idx = msg.data
        # 프레임 새로 모으기
        self.reflip_pending[idx] = []
        self.get_logger().info(f"/flip_done idx={idx} - 재측정 시작")

    # --------------------------------------------------------------
    # /main/keep_ready 콜백
    # --------------------------------------------------------------
    def keep_plan_callback(self, msg):
        idx = msg.data

        if idx not in self.accum_result:
            self.get_logger().error(f"/vision/keep_pick_pose 생성 실패: accum_result에 idx {idx}가 없음"
                                     f" (아직 누적 안 됐거나 놓친 박스)")
            return

        _, _, _, avg_angle, avg_cx, avg_cy, avg_cz, _ = self.accum_result[idx]

        out = String()
        out.data = pick_comm.TO_JSON_vision_keep_pick_pose(avg_cx, avg_cy, avg_cz, avg_angle)
        self.keep_pose_pub.publish(out)
        self.get_logger().info(f"/vision/keep_pick_pose 발행: {out.data}")

        # 화면 표시용 - keep_pick_pose 발행할 때마다 그 idx의 중앙점만 표시
        vis = make_pick_target_vis(avg_cx, avg_cy, avg_cz,
                                    self.capture.fx, self.capture.fy,
                                    self.capture.cx, self.capture.cy)
        if vis is not None:
            self.pick_target_vis[idx] = vis

        # 이 keep_pick_pose 발행이 이 idx에 대한 vision 쪽 마지막 개입이므로 바로 id 해제
        release_id(self.tracked_ids, idx)
        self.rotate_inplace_needed.pop(idx, None)
        self.reflip_pending.pop(idx, None)
        self.get_logger().info(f"idx={idx} keep 발행 완료 - id 즉시 해제")

    def _log_box_ready_summary(self, box_id, entries, top_entries,
                               avg_w, avg_h, avg_z, avg_angle, avg_cx, avg_cy, avg_cz):
        """box_ready 이후 box_sizes 발행 시점에 박스별로: 평균에 쓴 프레임값, 평면피팅 잔차, 발행값을 로그로."""

        n_short = sum(1 for e in entries if e[9] and e[9]['fallback'])
        resid_rng = [(round(e[9]['resid_min_mm'], 2), round(e[9]['resid_max_mm'], 2))
                     if e[9] and e[9]['resid_max_mm'] is not None else None
                     for e in entries]
        lines = [f"Box{box_id}:\n 누적 {len(entries)}프레임 중 fill_ratio 상위 "
                 f"{len(top_entries)}프레임 평균 사용"]
        for e in top_entries:
            lines.append(f" 각 xyz=({e[6]:.1f},{e[7]:.1f},{e[8]:.1f})cm angle={e[5]:.1f}deg | "
                         f"{format_plane_fit(e[9])}")
        lines.append(f"  전체 프레임 평면 잔차 범위(mm, -=카메라 쪽 +=먼 쪽): {resid_rng}")
        if n_short:
            lines.append(f"  [경고] 평면피팅 마스크 부족 프레임 {n_short}/{len(entries)}")
            
        lines.append(f"  평균: w/h/z={avg_w:.1f}/{avg_h:.1f}/{avg_z:.1f}cm "
                     f" 발행 xyz=({avg_cx:.1f},{avg_cy:.1f},{avg_cz:.1f})cm angle={avg_angle:.1f}deg\n")
        self.get_logger().info("\n".join(lines))
        

    def start_callback(self, msg):
        start = self.start
        self.start = bool(msg.data)
        if self.start:
            self.belt_stop_sent = False # belt_stop 아직 안보냄, 발행하고 나면 True로 바뀜.
            self.belt_stop = False
            if not start:
                self.get_logger().info("start 신호 받음 - 검출 시작")
        elif not self.start and start:
            self.get_logger().info("start 해제 - 검출 중지")

    # --------------------------------------------------------------
    # /main/box_ready 콜백: 박스 크기 누적 시작 트리거 -> True 올 때마다 누적 재시작
    # --------------------------------------------------------------
    def box_ready_callback(self, msg):
        self.box_ready = True
        self.accum_buf = defaultdict(list)
        self.frame_in_window = 0 # 현재 윈도우에서 처리한 프레임 수
        self.box_sizes_sent = False # ready 받으면 리셋, 누적 리셋
        self.accum_result = {}  # 이전 턴 누적 결과(화면 표시용 포함) 정리
        self.pick_target_vis = {}  # 이전 턴 pick_target 시각화 정리
        # box_ready 올 때마다 트래킹도 전부 리셋.
        self.tracked_ids = {}
        self.stable_buf = defaultdict(lambda: deque(maxlen=STABLE_FRAMES))
        self.stable_flag = {}
        self.rotate_inplace_needed = {}
        self.reflip_pending = {}
        self.get_logger().info("box_ready 신호 받음 - 박스 크기 누적 재시작")

    def process_frame(self):
        frame = self.capture.get_frame(detect=self.start) # 박스 대충위치 + sam2 마스크
        if frame is None:
            return False  # 스트림 끝(bag 재생 종료 등)

        color_img, depth_m, valid, candidates = frame
        if color_img is None:
            return True  # 깨진 프레임 - 계속 진행

        if valid is None or not np.any(valid):
            self._show(color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
            return True

        floor_m = self.capture.floor_m

        # 디버그 시각화용 누적 마스크(경계용)
        sil_used_vis = np.zeros(depth_m.shape, dtype=bool)
        sil_candidate_vis = np.zeros(depth_m.shape, dtype=bool)
        jump_used_area_vis = np.zeros(depth_m.shape, dtype=bool)
        drop_vis_accum = np.zeros(depth_m.shape, dtype=np.float32)

        count = 0
        # 이 프레임에서 나온 박스들: id 배정 전 임시 dict로 저장
        frame_dets = []

        if candidates: # box_capture 결과
            overlay = color_img.copy()
            dbg = {'jump': jump_used_area_vis, 'sil_cand': sil_candidate_vis,
                   'drop': drop_vis_accum, 'sil_used': sil_used_vis}
            for idx, cand in enumerate(candidates):
                det = compute_box_size( # 정밀 사이즈 체크
                    cand['seg_mask'], depth_m, floor_m,
                    self.capture.fx, self.capture.fy, self.capture.cx, self.capture.cy,
                    cand['bcx'], cand['bcy'],
                    overlay=overlay, color_img=color_img, dbg=dbg, label=f"Box {idx+1}")
                if det is not None:
                    count += 1
                    frame_dets.append(det) # 하나씩 저장
            color_img = cv2.addWeighted(overlay, 0.5, color_img, 0.5, 0)

        # ---- belt_stop 판단 ----
        frame_dets_by_x = sorted(frame_dets, key=lambda d: d['cx'], reverse=True)  # 오른쪽부터 정렬

        detected_ranks = set()
        for box_rank, det in enumerate(frame_dets_by_x):
            rank_idx = box_rank + 1
            detected_ranks.add(rank_idx)
            rw, rh = det['real_w'], det['real_h']
            self.stable_buf[rank_idx].append((max(rw, rh), min(rw, rh)))

            # STABLE_FRAMES 프레임 쌓였는지 확인
            if len(self.stable_buf[rank_idx]) == STABLE_FRAMES:
                ws = [e[0] for e in self.stable_buf[rank_idx]]
                hs = [e[1] for e in self.stable_buf[rank_idx]]
                w_stable = (max(ws) - min(ws)) <= STABLE_TOL_CM
                h_stable = (max(hs) - min(hs)) <= STABLE_TOL_CM
                self.stable_flag[rank_idx] = w_stable and h_stable
            else:
                self.stable_flag[rank_idx] = False

        # 이 프레임에서 안 보인 순번 → 안정 버퍼 초기화
        for rank_idx in list(self.stable_buf.keys()):
            if rank_idx not in detected_ranks:
                self.stable_buf[rank_idx].clear()
                self.stable_flag[rank_idx] = False

        # 안정된 박스 수 집계 → belt_stop 판단
        stable_count = sum(1 for v in self.stable_flag.values() if v)
        new_belt_stop = stable_count >= MAX_BOXES

        # 상태 변화(3개됨)가 있을 때만 publish (이미 stop 상태이면 발행X)
        if new_belt_stop != self.belt_stop:
            self.belt_stop = new_belt_stop # 갱신
            if not self.belt_stop_sent: # belt_stop 아직 안 보낸 상태(발행 한번만 하려고)
                msg = Bool()
                msg.data = self.belt_stop
                self.belt_stop_pub.publish(msg)
                self.belt_stop_sent = True
                print(f"[belt_stop] → {self.belt_stop}  (stable_count={stable_count})")
            # else:
            #     print(f"[belt_stop] 내부 상태만 {self.belt_stop}로 갱신 (이미 발행함 - 재발행 안 함)")

        # ---- ID 부여 -> box_ready 이후 ----
        # (belt_stop=True인 상태에서 box_ready가 True가 되면, 그 다음 프레임의 검출값으로 시딩)
        if self.belt_stop and self.box_ready and len(self.tracked_ids) == 0:
            for new_id, det in enumerate(frame_dets_by_x, start=1):
                self.tracked_ids[new_id] = {
                    'cx': det['cx'], 'cy': det['cy'],
                    'dims': tuple(sorted([det['real_w'], det['real_h'], det['z_cm']])),
                }

        # ---- ID 배정 ---- -> box_ready 후 시딩된 tracked_ids 기준
        id_to_rank = assign_ids_ordered(frame_dets_by_x, self.tracked_ids)
        # id_to_rank: {box_id: frame_dets_by_x의 인덱스} - 검출 개수가 살아있는 id 개수와 같을 때만 채워짐

        # ---- 화면에 현재 트래킹 중인 id가 이번 프레임에 보였는지 표시용 ----
        detected_ids = set(id_to_rank.keys())

        # ---- 누적 버퍼 (id 기준) ----
        # box_ready가 True이고 아직 이번 사이클 결과를 안 보냈을 때만 채움
        if self.box_ready and not self.box_sizes_sent:
            for box_id, det_j in id_to_rank.items():
                det = frame_dets_by_x[det_j]
                self.accum_buf[box_id].append((
                    self.global_frame, det['real_w'], det['real_h'], det['z_cm'], det['fill_ratio'],
                    det['angle_deg'], det['center_x_cm'], det['center_y_cm'], det['center_z_cm'],
                    det.get('plane_fit')
                ))
            self.frame_in_window += 1

        # ---- 뒤집기 후 2차 재측정 버퍼 (id 기준, ACCUM_FRAMES 윈도우와 독립적) ----
        # /flip_done 콜백에서 self.reflip_pending[idx] = [] 로 시작된 idx에 대해서만,
        # 그 시점 이후 들어오는 프레임을 별도로 REFLIP_FRAMES개 모음.
        for box_id, det_j in id_to_rank.items():
            if box_id not in self.reflip_pending:
                continue
            det = frame_dets_by_x[det_j]
            self.reflip_pending[box_id].append((
                self.global_frame, det['real_w'], det['real_h'], det['z_cm'], det['fill_ratio'],
                det['angle_deg'], det['center_x_cm'], det['center_y_cm'], det['center_z_cm'],
                det.get('plane_fit')
            ))

            if len(self.reflip_pending[box_id]) >= REFLIP_FRAMES:
                entries = self.reflip_pending[box_id]
                # fill_ratio 기준 상위 TOP_K_FRAMES 평균 - 메인 accum_result 계산 방식과 동일
                sorted_entries = sorted(entries, key=lambda e: e[4], reverse=True)
                top_entries = sorted_entries[:TOP_K_FRAMES]
                r_avg_angle = float(np.mean([e[5] for e in top_entries]))
                r_avg_cx = float(np.mean([e[6] for e in top_entries]))
                r_avg_cy = float(np.mean([e[7] for e in top_entries]))
                r_avg_cz = float(np.mean([e[8] for e in top_entries]))

                rotate_inplace = self.rotate_inplace_needed.get(box_id, False)
                angle2 = pick_comm.compute_second_angle(r_avg_angle, rotate_inplace)
                second_target_box = {
                    'idx': box_id,
                    'cx': r_avg_cx, 'cy': r_avg_cy, 'cz': r_avg_cz,
                    'height': 0,
                    'angle': angle2,
                    'need_flip': False,
                }
                out = String()
                out.data = pick_comm.TO_JSON_vision_pick_target(second_target_box)
                self.pick_target_pub.publish(out)
                self.get_logger().info(f"/vision/pick_target 발행(2차, 뒤집기 후): {out.data}")

                # 화면 표시용 - 2차 pick_target도 발행할 때마다 중앙점 새로 표시
                vis = make_pick_target_vis(r_avg_cx, r_avg_cy, r_avg_cz,
                                            self.capture.fx, self.capture.fy,
                                            self.capture.cx, self.capture.cy)
                if vis is not None:
                    self.pick_target_vis[box_id] = vis

                del self.reflip_pending[box_id]  # 재측정 끝 - 이 idx는 더 이상 reflip 버퍼에 안 쌓음

                # 2차 pick_target 발행이 이 idx에 대한 vision 쪽 마지막 개입이므로
                # /control/place_done을 기다리지 않고 바로 id 해제.
                release_id(self.tracked_ids, box_id)
                self.rotate_inplace_needed.pop(box_id, None)
                self.get_logger().info(f"idx={box_id} 2차 발행 완료 - id 즉시 해제")

        self.global_frame += 1

        # ---- ACCUM_FRAMES마다 상위 TOP_K_FRAMES 평균 계산 ----
        if self.frame_in_window >= ACCUM_FRAMES:
            # 지금 이 사이클이 box_ready로 시작된, 아직 발행 전인 사이클이면 -> 이번에 발행까지 함.
            should_send = self.box_ready and not self.box_sizes_sent
            self.accum_result = {}
            box_size_entries = []
            for box_id, entries in self.accum_buf.items():
                if len(entries) == 0:
                    continue
                # fill_ratio 기준 내림차순 정렬 → 상위 TOP_K_FRAMES 선택
                # fill_ratio가 높을수록 OBB 안에 윗면 픽셀이 빽빽하게 차있는 프레임
                # → 옆면 포함 프레임은 OBB가 커지면서 fill_ratio가 낮아져 자동 배제
                sorted_entries = sorted(entries, key=lambda e: e[4], reverse=True)
                top_entries = sorted_entries[:TOP_K_FRAMES]
                avg_w = float(np.mean([e[1] for e in top_entries]))
                avg_h = float(np.mean([e[2] for e in top_entries]))
                avg_z = float(np.mean([e[3] for e in top_entries]))
                avg_angle = float(np.mean([e[5] for e in top_entries]))
                avg_cx = float(np.mean([e[6] for e in top_entries]))
                avg_cy = float(np.mean([e[7] for e in top_entries]))
                avg_cz = float(np.mean([e[8] for e in top_entries]))
                self.accum_result[box_id] = (avg_w, avg_h, avg_z, avg_angle, avg_cx, avg_cy, avg_cz, top_entries)
                # print(f"[ACCUM] Box{box_id}: w={avg_w:.1f} h={avg_h:.1f} z={avg_z:.1f} cm "
                #       f"angle={avg_angle:.1f} deg "
                #       f"center(cam)=({avg_cx:.1f}, {avg_cy:.1f}, {avg_cz:.1f}) cm "
                #       f"(from frames {[e[0] for e in top_entries]})")

                # ---- box_ready로 시작된 사이클이면 이 박스 크기를 /vision/box_sizes 발행에 포함 ----
                if should_send:
                    box_size_entries.append((box_id, avg_w, avg_h, avg_z))
                    self._log_box_ready_summary(box_id, entries, top_entries,
                                                avg_w, avg_h, avg_z, avg_angle, avg_cx, avg_cy, avg_cz)

            # ---- 박스 전체(최대 3개) 정보를 한 번에 JSON 배열로 발행 ----
            if should_send and box_size_entries:
                out = String()
                out.data = pick_comm.TO_JSON_vision_box_sizes(box_size_entries)
                self.box_sizes_pub.publish(out)
                self.get_logger().info(f"/vision/box_sizes 발행: {out.data}")
                self.box_sizes_sent = True  # 한 번 보냈으니 다음 box_ready rising edge까지 더 안 보냄

            # 버퍼 초기화
            self.accum_buf = defaultdict(list)
            self.frame_in_window = 0

        # ---- 좌측 하단에 누적 결과 표시 (id 기준) ----
        if self.accum_result:
            line_h = 20   # 줄 간격(px)
            total_lines = len(self.accum_result) * 2
            base_y = H - (total_lines * line_h) - 10

            for row_i, box_id in enumerate(sorted(self.accum_result.keys())):
                avg_w, avg_h, avg_z, avg_angle, avg_cx, avg_cy, avg_cz, top_entries = self.accum_result[box_id]
                y_pos = base_y + row_i * line_h * 2

                # 결과 줄
                cv2.putText(
                    color_img,
                    f"Box{box_id}: x={avg_w:.1f} y={avg_h:.1f} z={avg_z:.1f} cm angle={avg_angle:.1f} deg "
                    f"center=({avg_cx:.1f},{avg_cy:.1f},{avg_cz:.1f})cm",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

                # 디버그 줄: 어떤 프레임 데이터를 평균냈는지
                # debug_str = "  " + "  ".join(
                #     [f"[fr{e[0]}:{e[1]:.1f}/{e[2]:.1f}/{e[3]:.1f}/{e[5]:.1f}deg/"
                #      f"({e[6]:.1f},{e[7]:.1f},{e[8]:.1f})]"
                #      for e in top_entries])
                # cv2.putText(
                #     color_img,
                #     debug_str,
                #     (10, y_pos + line_h),
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # ---- 화면에 현재 트래킹 중인 id 표시 (디버깅용) ----
        # 위치는 seed 시점 고정값이라, 실제 박스가 살짝 움직였어도 표시 위치는 그대로임(의도된 동작).
        for box_id, pos in self.tracked_ids.items():
            seen = box_id in detected_ids
            color = (0, 255, 0) if seen else (0, 0, 255)  # 이번 프레임에 안 보이면 빨간색
            cv2.putText(color_img, f"id{box_id}", (int(pos['cx']) - 15, int(pos['cy']) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---- /vision/pick_target으로 실제 발행된 최종 중앙점 표시 (마젠타, 새로 발행될 때마다 갱신) ----
        for vis in self.pick_target_vis.values():
            cv2.circle(color_img, vis['target_point'], 6, (255, 0, 255), -1)

        # ---- STOP 표시 ----
        if self.belt_stop:
            cv2.putText(color_img, "STOP", (W - 400, H // 2 -320),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)

        # print(' ')

        cv2.putText(color_img,
                    f"Boxes: {count}  fr:{self.global_frame}  win:{self.frame_in_window}/{ACCUM_FRAMES}"
                    f"  stable:{stable_count}/{MAX_BOXES}  tracked:{list(self.tracked_ids.keys())}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.rectangle(color_img, (self.capture.x0, self.capture.y0),
                      (self.capture.x1 - 1, self.capture.y1 - 1), (255, 0, 0), 1)

        # 경계 인식 디버깅용
        # - 하늘색: SAM2 경계 근처에서 jump 테스트(진짜 실루엣)는 통과했지만 아직 clean_mask에
        #           채택은 안 된 후보 (newly_added 조건 등으로 걸러진 것들)
        # - 노란색: 실제로 clean_mask에 추가된 실루엣 픽셀
        silhouette_vis = np.zeros_like(color_img)
        silhouette_vis[sil_candidate_vis] = (0, 255, 0)  # 하늘색(BGR)
        silhouette_vis[sil_used_vis] = (0, 0, 0)  # 검정
        color_img = cv2.addWeighted(color_img, 1.0, silhouette_vis, 0.4, 0)  # 실루엣 시각화

        self._show(color_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True

    def _show(self, color_img):
        """전체 해상도로 그린 결과를 화면 표시용으로만 축소해서 보여줌 (INTER_AREA로 깔끔하게)."""
        disp = cv2.resize(color_img, None, fx=self.display_scale, fy=self.display_scale,
                           interpolation=cv2.INTER_AREA)
        cv2.imshow("box detect", disp)

    def shutdown(self):
        self.capture.shutdown()
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = BoxDetectNode()
    node.get_logger().info("START: box_detect node")

    try:
        running = True
        while running:
            running = node.process_frame()
            rclpy.spin_once(node, timeout_sec=0)
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()