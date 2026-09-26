"""
RealSense 캡처 + 필터 
뎁스맵 기반 대충 위치 잡고 sam2 mask 까지만
카메라 설정(laser_power, ROI, 해상도), SAM2 박스 프롬프트 확장 비율 바꿀때 여기서 바꾸기
수정: 박스 살짝 붙어있는 경우 한 박스로 되던거 수정 
"""

from curses.ascii import FF
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

pc = "JISU"
# pc = "JUNMI"

# ----------------------------------------------------------------------
# 카메라/스트림 파라미터
# ----------------------------------------------------------------------
W, H, FPS = 1280, 720, 30
ROI_X_MIN, ROI_X_MAX = 250, 1280
ROI_Y_MIN, ROI_Y_MAX = 250, 650
resize = 2 # 1/2배로 축소시켜서 imshow 띄움

# ROI_X_MIN, ROI_X_MAX = 250, 1280
# ROI_Y_MIN, ROI_Y_MAX = 150, 450

# 리얼센스 설정 리셋
HW_RESET_ON_START = False    # False로 하면 리셋 안하고 이전 설정 그대로. 리셋하면 노출/화이트밸런스 초기화됨. 

# ---- 실행 모드 ----
MODE = "real"  # "real" or "bag"
BAG_PATH = "/home/leejunmi/realsense_bag/0919(5).db3"

DEPTH_SENSOR_OPTIONS = {
    rs.option.enable_auto_exposure: 1,     # 1(켜기)
    # rs.option.exposure:             6000,   # us. 벨트 모션블러 줄이려면 낮게
    # rs.option.gain:                 16,
    # 노출 1이 자동 켜진것, 아니면 흰 부분이 날라가서 노출을 못 줄임, IR 영상이 재대로 안 만들어짐(무늬가 없어서 댑스 못찾음)->댑스 0으로 측정.
    # IR 패턴 프로잭터의 출력 정도(쏘는 무늬의 밝기), 높높으면 밝은 부분 날라갈수도
    rs.option.laser_power:  160,        #360.0,  # 최대 근처 (범위 밖이면 클램프)
    rs.option.emitter_enabled:      1,
    # emitter: IR 프로잭터를 켜나 끄냐(주변광 IR 즉 햇빛 의존, 실내:on필수)
    # rs.option.depth_units:        0.0001, # 필요 시. 코드가 get_depth_scale()로 자동 반영
}
COLOR_SENSOR_OPTIONS = {
    # RGB(SAM2 입력)는 자동 노출/화이트밸런스 유지. SAM2는 노출 변화에 둔감하고,
    # 수동값을 잘못 넣으면 화면 색만 이상해짐. 재현성은 depth만 고정하면 충분.
    rs.option.enable_auto_exposure:      1,
    rs.option.enable_auto_white_balance: 1,
}

FLOOR_M = 0.527

# ----------------------------------------------------------------------
# SAM2 이전
'''height = floor_m - depth_m → 바닥 기준 높이 계산
(height > H_MIN_M) & (height < H_MAX_M) & valid → 1cm~20cm 사이 높이인 픽셀만 1차 마스크
MORPH_OPEN (작은 노이즈 점 제거) → MORPH_CLOSE (작은 구멍 메꾸기), 커널은 둘 다 MORPH_KERNEL=10'''
# ----------------------------------------------------------------------
H_MIN_M      = 0.025    # 바닥으로부터 1cm 이상 올라온 물체만
H_MAX_M      = 0.2      # 20cm 이상은 무시
MIN_AREA_CM2 = 8.0      # 너무 작은 물체는 제거(높이 컴포넌트 기준)
MORPH_KERNEL = 10        # 모폴로지 커널
MORPH_OPEN_ENABLED  = True   # 침식->팽창(작은 노이즈 점 제거) 효과 테스트할 때 False로
MORPH_CLOSE_ENABLED = False   # 팽창->침식(작은 구멍 메꾸기) 효과 테스트할 때 False로

BOX_EXPAND_RATIO = 0.2  # SAM2 박스 프롬프트 확대 비율

# 수정
# ---- 붙어있는 박스 분리 (SAM2 point 프롬프트) ----
BOX_MASK_SKIP_SPLIT_FILL_RATIO = 0.8   # box mask 결과가 이 이하일때만 겹쳐있다고 판단
POINT_MASK_MIN_FILL_RATIO = 0.8   # point 프롬프트 후보 중 OBB fill_ratio이 이상만 채택
SPLIT_SAME_IOU = 0.5              # 좌/우 point 결과의 IoU가 이 이상이면 같은 박스를 잡은 것 -> 단일 박스로 합침
SPLIT_SAME_OVERLAP = 0.8          # 포함비율(inter/min(area))이 이 이상이면(크기 차이 큰 포함 관계) 단일 박스로 합침

DEBUG_ROUGH = False  # 대충 위치 잡기 시각화

# ----------------------------------------------------------------------
# SAM2 설정
# ----------------------------------------------------------------------

if pc == "JISU":
    SAM2_CKPT   = '/home/pc/sam2/checkpoints/sam2.1_hiera_tiny.pt'
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
elif pc == "JUNMI":
    SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"          # 가장 작은 모델로..
    SAM2_CKPT   = "/home/leejunmi/sam2/checkpoints/sam2.1_hiera_tiny.pt"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _obb_fill_ratio(binary_mask):
    """박스안 마스크 얼마나 차있는지 정도"""
    ys, xs = np.where(binary_mask)
    if len(xs) < 5:
        return 1.0  # 너무 작아서 판단 불가 - 정상으로 취급
    rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    obb_area = rect[1][0] * rect[1][1]
    if obb_area <= 0:
        return 1.0
    return len(xs) / obb_area


def _half_point(xs, ys, seg_mask):
    """반쪽 픽셀(xs, ys)의 평균 좌표를 point 프롬프트 위치로 씀. 마스크(mask) 밖이면 가장 가까운 픽셀로 옮김."""
    cx, cy = float(xs.mean()), float(ys.mean())
    px, py = int(round(cx)), int(round(cy))
    if seg_mask[py, px]:
        return px, py
    k = int(np.argmin((xs - cx) ** 2 + (ys - cy) ** 2))
    return int(xs[k]), int(ys[k])



class BoxCapture:
    """RealSense + SAM2 
    (color_img, depth_m, valid, candidates)"""

    def __init__(self, floor_m=FLOOR_M, mode=MODE):
        self.floor_m = floor_m
        self.mode = mode

        # ---- SAM2 ----
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
        self.predictor = SAM2ImagePredictor(sam2_model)

        # ---- RealSense 파이프라인 ----
        self.pipeline = rs.pipeline()
        config = rs.config()

        if mode == "real":
            if HW_RESET_ON_START:
                self._hardware_reset()

            config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
            profile = self.pipeline.start(config)

            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()

            # 뷰어 상태와 무관하게 매 실행 같은 값으로 시작하도록 명시적으로 설정 + 실제 적용값 출력
            self._apply_options(depth_sensor, DEPTH_SENSOR_OPTIONS, "depth")
            color_sensor = device.first_color_sensor()
            self._apply_options(color_sensor, COLOR_SENSOR_OPTIONS, "color")

        elif mode == "bag":
            config.enable_device_from_file(BAG_PATH, repeat_playback=True)
            profile = self.pipeline.start(config)

            device = profile.get_device()
            playback = device.as_playback()
            playback.set_real_time(False)  # bag 재생 속도 제한 해제(빠르게 재생)

            depth_sensor = device.first_depth_sensor()

        else:
            raise ValueError(f"알 수 없는 mode: {mode}")

        self.depth_scale = depth_sensor.get_depth_scale()

        # ---- 시작 시점 센서 온도 출력 (열 안정화 확인용) ----
        # 콜드 스타트 직후엔 이 값이 계속 오르고, 그동안 depth가 mm 단위로 드리프트함.
        try:
            self.pipeline.wait_for_frames()   # 온도 레지스터가 채워지도록 첫 프레임 한 장 받기
        except Exception:
            pass


        self.align = rs.align(rs.stream.color)

        ## 필터
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)

        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial.set_option(rs.option.holes_fill, 0)

        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)

        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)

        self.threshold_filter = rs.threshold_filter(min_dist=0.1, max_dist=0.55)

        color_intr = profile.get_stream(rs.stream.color) \
                            .as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = color_intr.fx, color_intr.fy
        self.cx, self.cy = color_intr.ppx, color_intr.ppy

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL))
        self.roi_mask = np.zeros((H, W), dtype=bool)
        self.x0, self.x1 = max(0, ROI_X_MIN), min(W, ROI_X_MAX)
        self.y0, self.y1 = max(0, ROI_Y_MIN), min(H, ROI_Y_MAX)
        self.roi_mask[self.y0:self.y1, self.x0:self.x1] = True

        # 처리 해상도(W, H)는 그대로 두고 화면에 보여지는 창 크기만 축소
        # cv2.namedWindow("rough detect", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("rough detect", W // resize, H // resize) 


    def _point_mask(self, pt, other_pt, region):
        """pt에 positive point 프롬프트를 줘서 나온 후보 3개 중 하나를 고름(region 박스 밖은 버림).
        반대쪽 점(other_pt)을 포함하지 않는 후보를 우선(=다른 박스를 삼키지 않은 것), 그중 사각형다운
        (fill_ratio 높은) 큰 것. 그런 후보가 없으면(=박스 하나) 전체 후보에서 같은 기준으로 고름."""
        pt_masks, _, _ = self.predictor.predict(
            point_coords=np.array([pt]), point_labels=np.array([1]), multimask_output=True)
        ex0, ey0, ex1, ey1 = region
        cands = []
        for m in pt_masks:
            c = np.zeros(m.shape, dtype=bool)
            c[ey0:ey1, ex0:ex1] = m[ey0:ey1, ex0:ex1].astype(bool)
            cands.append(c)
        ox, oy = other_pt
        pool = [c for c in cands if not c[oy, ox]] or cands
        ok = [c for c in pool if _obb_fill_ratio(c) >= POINT_MASK_MIN_FILL_RATIO]
        if ok:
            return max(ok, key=np.count_nonzero)
        return max(pool, key=_obb_fill_ratio)

    # --------------------------------------------------------------
    # 필터 적용 + 높이 기반 위치 잡기 + SAM2 mask 추출
    # --------------------------------------------------------------
    def get_frame(self, detect=True):
        """
        candidates 원소: {'seg_mask': bool(H,W) SAM2 마스크, 'bcx': int, 'bcy': int(컴포넌트 중심 픽셀)}
        """
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            if self.mode == "bag":
                print("bag 재생 끝 - 종료")
                return None
            raise

        frames = self.align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None, []

        filtered = self.threshold_filter.process(depth_frame)
        filtered = self.depth_to_disparity.process(filtered)
        filtered = self.spatial.process(filtered)
        filtered = self.temporal.process(filtered)
        filtered = self.disparity_to_depth.process(filtered)
        # filtered = self.hole_filling.process(filtered)  # hole을 매꿀때 부정확해질수 있음 -> 제거
        depth_frame = filtered.as_depth_frame()

        depth_raw = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self.depth_scale
        valid = (depth_raw > 0) & self.roi_mask

        if not np.any(valid):
            return color_img, depth_m, valid, []

        if not detect:
            return color_img, depth_m, valid, []

        # 카메라 높이 측정
        print(f'max:{depth_m[valid].max()}, median:{float(np.median(depth_m[valid]))}')
        height = self.floor_m - depth_m

        mask = ((height > H_MIN_M) & (height < H_MAX_M) & valid).astype(np.uint8) * 255
        if MORPH_OPEN_ENABLED:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        if MORPH_CLOSE_ENABLED:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        m_per_px_x = self.floor_m / self.fx
        m_per_px_y = self.floor_m / self.fy
        area_per_px_cm2 = (m_per_px_x * 100.0) * (m_per_px_y * 100.0)
        min_area_px = MIN_AREA_CM2 / area_per_px_cm2 if area_per_px_cm2 > 0 else 0

        # 평면 조건 만족하는 픽셀중 컴포넌트 묶기
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

        boxes_for_sam = []
        valid_indices = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area_px:
                continue
            x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            # sam 프롬프트 박스 용 박스 확대
            mx = w * BOX_EXPAND_RATIO / 2.0
            my = h * BOX_EXPAND_RATIO / 2.0
            ex0 = max(0, int(round(x - mx)))
            ey0 = max(0, int(round(y - my)))
            ex1 = min(W, int(round(x + w + mx)))
            ey1 = min(H, int(round(y + h + my)))

            boxes_for_sam.append([ex0, ey0, ex1, ey1])
            valid_indices.append(i)

        candidates = []
        if boxes_for_sam:
            self.predictor.set_image(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            boxes_np = np.array(boxes_for_sam)

            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,  # 박스 프롬프트로 넣음
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks[:, 0, :, :]

            for j, i in enumerate(valid_indices):
                seg_mask = masks[j].astype(bool)
                bcx, bcy = int(round(centroids[i][0])), int(round(centroids[i][1]))
                single = {'seg_mask': seg_mask, 'bcx': bcx, 'bcy': bcy,
                          'source': 'box', 'fill_ratio': _obb_fill_ratio(seg_mask)}

                # box 프롬프트 결과가 사각형답지 않을 때만 분리 시도. 좌/우 분할과 point 위치는
                # SAM2 마스크가 아니라 height 맵(depth)으로 잡은 컴포넌트 픽셀 기준.
                if not seg_mask.any() or single['fill_ratio'] >= BOX_MASK_SKIP_SPLIT_FILL_RATIO:
                    candidates.append(single)
                    continue
                comp = (labels == i)
                ys, xs = np.where(comp)
                mid = (xs.min() + xs.max()) / 2.0
                left, right = xs < mid, xs >= mid
                if np.count_nonzero(left) < min_area_px or np.count_nonzero(right) < min_area_px:
                    candidates.append(single)
                    continue

                ptL = _half_point(xs[left], ys[left], comp)
                ptR = _half_point(xs[right], ys[right], comp)
                ex0, ey0, ex1, ey1 = boxes_for_sam[j]
                mL = self._point_mask(ptL, ptR, (ex0, ey0, ex1, ey1))
                mR = self._point_mask(ptR, ptL, (ex0, ey0, ex1, ey1))

                inter = np.count_nonzero(mL & mR)
                union = np.count_nonzero(mL | mR)
                if not mL.any() or not mR.any():
                    # 한쪽 point 프롬프트가 빈 마스크를 냄 -> 분리 불가, box 프롬프트 결과 그대로 사용.
                    candidates.append(single)
                    continue
                overlap = inter / min(np.count_nonzero(mL), np.count_nonzero(mR))
                if union == 0 or inter / union >= SPLIT_SAME_IOU or overlap >= SPLIT_SAME_OVERLAP:
                    # 좌/우 점이 같은 박스를 잡음 -> 박스 하나. box 프롬프트 결과 그대로 사용.
                    candidates.append(single)
                    continue

                for name, m, pt in (('ptL', mL, ptL), ('ptR', mR, ptR)):
                    ys2, xs2 = np.where(m)
                    candidates.append({
                        'seg_mask': m,
                        'bcx': int(round(xs2.mean())), 'bcy': int(round(ys2.mean())),
                        'source': name, 'fill_ratio': _obb_fill_ratio(m),
                        'peel_point': pt,
                    })

        if DEBUG_ROUGH:
            self._debug_rough(color_img, height, valid, mask,
                                n, stats, centroids, min_area_px, area_per_px_cm2,
                                boxes_for_sam, candidates)

        return color_img, depth_m, valid, candidates

# ----------------------------------------------------------------------------------------------------------------------------# --------------------------------------------------------------


    # --------------------------------------------------------------
    # RealSense 센서 설정 헬퍼
    # --------------------------------------------------------------
    @staticmethod
    def _hardware_reset():
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            print("[hw_reset] 장치 없음 - 건너뜀")
            return
        for d in devs:
            try:
                d.hardware_reset()
            except Exception as e:
                print(f"[hw_reset] 실패: {e}")
        # 재열거 대기
        t0 = time.time()
        while time.time() - t0 < 10.0:
            if len(rs.context().query_devices()) > 0:
                time.sleep(1.5)   # 안정화 여유
                print("[hw_reset] 완료")
                return
            time.sleep(0.5)
        print("[hw_reset] 재열거 타임아웃 - 그대로 진행")

    @staticmethod
    def _apply_options(sensor, options, tag):
        """options({rs.option: value})를 sensor에 씀(범위 밖이면 클램프). 그 후 실제 적용값을 출력.
        매 실행 이 출력이 동일한지 보면 설정이 재현되는지 바로 확인 가능."""
        for opt, val in options.items():
            if not sensor.supports(opt):
                print(f"[opt/{tag}] {opt} 미지원 - 건너뜀")
                continue
            try:
                rng = sensor.get_option_range(opt)
                clamped = min(max(float(val), rng.min), rng.max)
                sensor.set_option(opt, clamped)
            except Exception as e:
                print(f"[opt/{tag}] set {opt} = {val} 실패: {e}")
        print(f"---- {tag} sensor options (실제 적용값) ----")
        for opt in options:
            if sensor.supports(opt):
                print(f"  {str(opt):32s} = {sensor.get_option(opt)}")
        try:
            print(f"  {'depth_scale':32s} = {sensor.get_depth_scale()}")
        except Exception:
            pass
        print("-" * 44)

    # --------------------------------------------------------------
    # SAM2 이전 '대충 위치 잡기' 단계 시각화 
    # --------------------------------------------------------------
    def _debug_rough(self, color_img, height, valid, mask,
                     n, stats, centroids, min_area_px, area_per_px_cm2, boxes_for_sam,
                     candidates=()):
        Hh, Ww = mask.shape[0], mask.shape[1]

        h_range  = ((height > H_MIN_M) & (height < H_MAX_M) & valid).astype(np.uint8) * 255

        p1 = cv2.cvtColor(h_range,  cv2.COLOR_GRAY2BGR)   # 1) 높이 범위 + valid
        p3 = cv2.cvtColor(mask,     cv2.COLOR_GRAY2BGR)   # 3) 최종 mask (AND + morph)
        p4 = color_img.copy()                             # 4) 컴포넌트 -> SAM2 프롬프트
        p5 = color_img.copy()                             # 5) SAM2/peel 최종 candidates

        labels = ["1) height in-range & valid",
                  "3) final mask (AND+morph)", "4) components -> SAM2 box",
                  "5) final candidates (box=단일, ptL/ptR=좌우 point 분리)"]
        for p, t in zip((p1, p3, p4, p5), labels):
            cv2.putText(p, t, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.rectangle(p, (self.x0, self.y0), (self.x1 - 1, self.y1 - 1), (255, 0, 0), 2)

        for i in range(1, n):
            x = int(stats[i, cv2.CC_STAT_LEFT]);  y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
            a_px = int(stats[i, cv2.CC_STAT_AREA])
            passed = a_px >= min_area_px
            col = (0, 255, 0) if passed else (0, 0, 255)
            cv2.rectangle(p4, (x, y), (x + w, y + h), col, 2)
            cv2.putText(p4, f"{a_px * area_per_px_cm2:.0f}cm2 {'OK' if passed else 'small'}",
                        (x, max(12, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cxi, cyi = int(round(centroids[i][0])), int(round(centroids[i][1]))
            cv2.circle(p4, (cxi, cyi), 4, col, -1)

        for (ex0, ey0, ex1, ey1) in boxes_for_sam:
            cv2.rectangle(p4, (ex0, ey0), (ex1, ey1), (0, 200, 255), 2)

        cv2.putText(p4, f"min_area={min_area_px:.0f}px({MIN_AREA_CM2:.0f}cm2)  "
                        f"H_MIN/MAX={H_MIN_M}/{H_MAX_M}m  comp={n - 1}  ->SAM2={len(boxes_for_sam)}",
                    (10, Hh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ---- 5) candidate별로 색 다르게 마스크 칠하고, source(box/ptL/ptR)+fill_ratio+면적 라벨 ----
        palette = [(255, 0, 255), (0, 255, 0), (255, 255, 0), (0, 165, 255), (255, 0, 0), (0, 0, 255)]
        for ci, cand in enumerate(candidates):
            col = palette[ci % len(palette)]
            seg = cand['seg_mask']
            p5[seg] = (0.5 * p5[seg] + 0.5 * np.array(col)).astype(np.uint8)
            ys_obb, xs_obb = np.where(seg)
            if len(xs_obb) >= 5:
                obb = cv2.minAreaRect(np.column_stack((xs_obb, ys_obb)).astype(np.float32))
                obb_pts = np.int32(cv2.boxPoints(obb))
                cv2.polylines(p5, [obb_pts], isClosed=True, color=col, thickness=2)
            source = cand.get('source', '?')
            fr = cand.get('fill_ratio')
            area_cm2 = int(np.count_nonzero(seg)) * area_per_px_cm2
            fr_str = f"fr={fr:.2f}" if fr is not None else ""
            cv2.putText(p5, f"#{ci} {source} {area_cm2:.0f}cm2 {fr_str}",
                        (cand['bcx'] - 20, cand['bcy'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            cv2.circle(p5, (cand['bcx'], cand['bcy']), 4, col, -1)
            if 'peel_point' in cand:
                cv2.drawMarker(p5, cand['peel_point'], col, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(p5, f"candidates={len(candidates)}", (10, Hh - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        montage = np.vstack([np.hstack([p1, p3]), np.hstack([p4, p5])])
        montage = cv2.resize(montage, (montage.shape[1] // resize, montage.shape[0] // resize))
        cv2.imshow("rough detect", montage) # 시각화 4분할화면 


    def shutdown(self):
        self.pipeline.stop()