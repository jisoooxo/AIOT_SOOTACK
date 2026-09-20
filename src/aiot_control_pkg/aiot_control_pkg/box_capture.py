"""
RealSense 캡처 + 필터 
뎁스맵 기반 대충 위치 잡고 sam2 mask 까지만
카메라 설정(laser_power, ROI, 해상도), SAM2 박스 프롬프트 확장 비율 바꿀때 여기서 바꾸기
"""

from curses.ascii import FF
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# pc = "JISU"
pc = "JISU"

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
BAG_PATH = "/home/leejunmi/realsense_bag/0909(1).db3"

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

FLOOR_M = 0.530 

# ----------------------------------------------------------------------
# SAM2 이전
'''height = floor_m - depth_m → 바닥 기준 높이 계산
(height > H_MIN_M) & (height < H_MAX_M) & valid → 1cm~20cm 사이 높이인 픽셀만 1차 마스크
MORPH_OPEN (작은 노이즈 점 제거) → MORPH_CLOSE (작은 구멍 메꾸기), 커널은 둘 다 MORPH_KERNEL=10'''
# ----------------------------------------------------------------------
H_MIN_M      = 0.025    # 바닥으로부터 1cm 이상 올라온 물체만
H_MAX_M      = 0.2      # 20cm 이상은 무시
MIN_AREA_CM2 = 10.0      # 너무 작은 물체는 제거(높이 컴포넌트 기준)
MORPH_KERNEL = 10        # 모폴로지 커널
MORPH_OPEN_ENABLED  = True   # 침식->팽창(작은 노이즈 점 제거) 효과 테스트할 때 False로
MORPH_CLOSE_ENABLED = True   # 팽창->침식(작은 구멍 메꾸기) 효과 테스트할 때 False로

FLAT_MASK_ENABLED = False     # 평면(윗면) 조건 필터 사용 여부 -> 하니까 마스크가 끊겨서 제외
FLAT_STD_THRESH_M = 0.005    # 깊이 표준편차 5mm 이하만 평면으로 인정
FLAT_WIN = 10                # 로컬 윈도우 크기 (px)

BOX_EXPAND_RATIO = 0.2  # SAM2 박스 프롬프트 확대 비율

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
        for opt in (rs.option.asic_temperature, rs.option.projector_temperature):
            if depth_sensor.supports(opt):
                print(f"[temp] {str(opt):26s} = {depth_sensor.get_option(opt):.1f} C  (start)")

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

        print(f'max:{depth_m[valid].max()}, median:{float(np.median(depth_m[valid]))}')
        height = self.floor_m - depth_m

        # 윈도우로 전체 프레임 나눠서 평면 조건 계산(옆면 거르기용, 효과 테스트는 FLAT_MASK_ENABLED로)
        if FLAT_MASK_ENABLED:
            mean = cv2.blur(depth_m, (FLAT_WIN, FLAT_WIN))
            mean_sq = cv2.blur(depth_m * depth_m, (FLAT_WIN, FLAT_WIN))
            local_std = np.sqrt(np.maximum(mean_sq - mean * mean, 0))
            flat_mask = local_std < FLAT_STD_THRESH_M
        else:
            flat_mask = np.ones_like(valid, dtype=bool)

        mask = ((height > H_MIN_M) & (height < H_MAX_M) & valid).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, (flat_mask.astype(np.uint8) * 255))
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

        if DEBUG_ROUGH:
            self._debug_rough(color_img, height, valid, flat_mask, mask,
                                n, stats, centroids, min_area_px, area_per_px_cm2,
                                boxes_for_sam)

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
                bcx, bcy = int(round(centroids[i][0])), int(round(centroids[i][1]))
                candidates.append({'seg_mask': masks[j].astype(bool), 'bcx': bcx, 'bcy': bcy})

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
    def _debug_rough(self, color_img, height, valid, flat_mask, mask,
                     n, stats, centroids, min_area_px, area_per_px_cm2, boxes_for_sam):
        Hh = mask.shape[0]

        h_range  = ((height > H_MIN_M) & (height < H_MAX_M) & valid).astype(np.uint8) * 255
        flat_vis = ((flat_mask & self.roi_mask).astype(np.uint8)) * 255

        p1 = cv2.cvtColor(h_range,  cv2.COLOR_GRAY2BGR)   # 1) 높이 범위 + valid
        p2 = cv2.cvtColor(flat_vis, cv2.COLOR_GRAY2BGR)   # 2) 평탄도 마스크
        p3 = cv2.cvtColor(mask,     cv2.COLOR_GRAY2BGR)   # 3) 최종 mask (AND + morph)
        p4 = color_img.copy()                             # 4) 컴포넌트 -> SAM2 프롬프트

        labels = ["1) height in-range & valid", "2) flat_mask (in ROI)",
                  "3) final mask (AND+morph)", "4) components -> SAM2 box"]
        for p, t in zip((p1, p2, p3, p4), labels):
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

        montage = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
        montage = cv2.resize(montage, (montage.shape[1] // resize, montage.shape[0] // resize))
        cv2.imshow("rough detect", montage)


    def shutdown(self):
        self.pipeline.stop()