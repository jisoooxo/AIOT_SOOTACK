"""
SAM2 seg_mask + depth_m 받아서 박스 하나의 실측 크기/각도/중심좌표를 계산하는 모듈.

1차 z_center 한 개 기준 윗면 좁개 뽑음, 평면 피팅함 -> 경걔판단까지만
TOL은 1차 판정용 빡세게 따로 + 2차 널널하게 따로
"""


"""
박스 사이즈 
폼클랜징 16 6.6 4.7
솜 13 6.5 5.4
초록색 15 5.1 3.5

"""


import cv2
import numpy as np

# ----------------------------------------------------------------------
# 윗면 판정 TOL: 박스 높이(z_cm) 구간별로 다르게
# ----------------------------------------------------------------------
TOL_LOW_Z  = 0.0025     # z <= 3
TOL_Z_THRESH_CM = 3
TOL_LOW_Z0  = 0.0023     # 3 < z <= 5 # 선풍기, 넓은 면은 0.003, 좁은 면은 0.0025
TOL_Z_THRESH_CM0 = 5
TOL_LOW_Z1  = 0.00225     # 5 < z <= 10
TOL_Z_THRESH_CM1 = 10
TOL_LOW_Z2  = 0.002     # 10 < z <= 15
TOL_Z_THRESH_CM2 = 15
TOL_LOW_Z3  = 0.0015     # 15 < z <= 18
TOL_Z_THRESH_CM3 = 18
TOL_HIGH_Z = 0.001     # z >  18

# 1차(r1, 평면 피팅용 재료 수집)는 이만큼 더 타이트하게(기존 tol - TOL_R1_TIGHTEN_M)
TOL_R1_TIGHTEN_M = 0.0015  # 0.15cm
TOL_R1_MIN_M = 0.0005   # 위에서 빼도 이 밑으로는 안 내려가게(0/음수 방지)


# 카메라 쪽은 기존 TOL보다 이 배수만큼 널널하게 허용.
TOL_CAM_SIDE_MULT = 1.3

BOX_CENTER_WIN_PX = 10  # (1차 z)박스 중앙 +-px 윈도우에서 z_center 산출
REFINE_WIN_PX = 5      # (1차 z)OBB 3등분점, z_center 재측정 윈도우(px)

# ---- SAM2 경계 스냅 파라미터 (2차에서만 씀)
# -> 일단 안하도록 함 ,.. ----
DILATE_SNAP_PX = 40            # top_mask2의 실제 경계에서부터 이정도 px 안에 들어와야댐, 안되면 노란색으로 뜬다

# ---- 바닥-윗면 경계(실루엣) 판정 (2차에서만 씀) ----
EDGE_COMPARE_PX = 3            # 경계 바로 안쪽/바깥쪽 비교 폭(px)
SILHOUETTE_HEIGHT_TOL_RATIO = 0.000000000000001 #0.15     # 박스 높이의 이 비율만큼을 허용 오차로 씀
SILHOUETTE_HEIGHT_TOL_MIN_M = 0.000000000000001 #0.0015  # 허용 오차 하한(1.5mm)

SILHOUETTE_Z_TOL_MULT = 5   # 실루엣 후보도 z_ref(윗면 기준 depth)에서 이 배수(*tol)까지 허용

# ---- OBB 실측 크기 필터 ----
MIN_AREA_OBB_CM2 = 10.0       # OBB 실측 면적 기준

orientation_axis = 'long' 

_SNAP_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * DILATE_SNAP_PX + 1, 2 * DILATE_SNAP_PX + 1))
_EDGE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * EDGE_COMPARE_PX + 1, 2 * EDGE_COMPARE_PX + 1))


def _select_tol(z_cm):
    if z_cm <= TOL_Z_THRESH_CM:            # 3 이하
        return TOL_LOW_Z
    if z_cm <= TOL_Z_THRESH_CM0:           # 5 이하
        return TOL_LOW_Z0
    if z_cm <= TOL_Z_THRESH_CM1:           # 10 이하
        return TOL_LOW_Z1
    if z_cm <= TOL_Z_THRESH_CM2:           # 15 이하
        return TOL_LOW_Z2
    if z_cm <= TOL_Z_THRESH_CM3:           # 18 이하
        return TOL_LOW_Z3
    return TOL_HIGH_Z


# --------------------------------------------------------------
# 1차 (평면 피팅용) 윗면 추출: |depth - z_ref| <= tol 인 픽셀을 윗면으로 잡아
# 조각 잇기로 정리한 clean_mask와 그 OBB, 그리고 OBB 긴 축을 3구역으로 나눈 뒤
# 각 구역 중심(t=1/6, 1/2, 5/6)에서 잰 depth를 돌려준다.
# --------------------------------------------------------------
def _extract_top_face_r1(seg_mask, depth_m, z_center, tol, overlay, color_img):
    """반환: dict 또는 None(검출 실패 -> 호출부에서 continue)
      xs, ys, end_a, end_b, refine_z_by_zone([z0,z1,z2], 없는 점은 유효값 평균으로 채움), z_center(3점 평균)
    """
    H, W = depth_m.shape

    valid = seg_mask & (depth_m > 0)
    if not np.any(valid):
        return None

    vals = depth_m[valid]
    # 비대칭 밴드: dz>0(바닥 쪽, 더 멀고 낮음)은 tol로 빡빡하게, dz<0(카메라 쪽, 더 높음)은 널널하게
    dz = vals - z_center
    top_mask = np.zeros_like(seg_mask, dtype=bool)
    top_mask[valid] = (dz <= tol) & (dz >= -tol * TOL_CAM_SIDE_MULT)

    if overlay is not None:
        overlay[seg_mask] = (
            0.5 * overlay[seg_mask] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)
        overlay[top_mask] = (
            0.5 * overlay[top_mask] + 0.5 * np.array([0, 0, 255])
        ).astype(np.uint8)

    top_mask_u8 = (top_mask.astype(np.uint8) * 255)

    # top_mask 조각 끊긴 거 이어붙이기
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    closed = cv2.morphologyEx(top_mask_u8, cv2.MORPH_CLOSE, close_kernel)
    n_cc, _, _, _ = cv2.connectedComponentsWithStats(closed, 8)
    if n_cc <= 1:
        return None
    # 옆 방향으로 크게 퍼지지 않도록 침식만
    top_boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    top_expanded = cv2.dilate(top_mask_u8, top_boundary_kernel)
    clean_mask = (closed > 0) & (top_expanded > 0)

    ys, xs = np.where(clean_mask)
    if len(xs) <= 5:
        return None

    rect_px = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    box_px = np.int32(cv2.boxPoints(rect_px))

    # OBB 긴 축 양 끝점 (짧은 변들의 중점)
    box_pts_f = box_px.astype(np.float32)
    edge01 = np.linalg.norm(box_pts_f[0] - box_pts_f[1])
    edge12 = np.linalg.norm(box_pts_f[1] - box_pts_f[2])
    if edge01 < edge12:
        end_a = (box_pts_f[0] + box_pts_f[1]) / 2.0
        end_b = (box_pts_f[2] + box_pts_f[3]) / 2.0
    else:
        end_a = (box_pts_f[1] + box_pts_f[2]) / 2.0
        end_b = (box_pts_f[3] + box_pts_f[0]) / 2.0

    # 긴 축을 3구역([0,1/3],[1/3,2/3],[2/3,1])으로 나눈 뒤 각 구역 중심에서 depth 재측정 -> 구역별 기준 z
    refine_z_by_zone = [None, None, None]
    for zi, t in enumerate((1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0)):
        pt = end_a + t * (end_b - end_a)
        pcx, pcy = int(round(pt[0])), int(round(pt[1]))
        ry0, ry1 = max(0, pcy - REFINE_WIN_PX), min(H, pcy + REFINE_WIN_PX + 1)
        rx0, rx1 = max(0, pcx - REFINE_WIN_PX), min(W, pcx + REFINE_WIN_PX + 1)
        r_patch = depth_m[ry0:ry1, rx0:rx1]
        r_valid = r_patch[r_patch > 0]
        if r_valid.size > 0:
            refine_z_by_zone[zi] = float(np.median(r_valid))

    present = [z for z in refine_z_by_zone if z is not None]
    z_center_refined = float(np.mean(present)) if present else float(z_center)
    refine_z_by_zone = [z if z is not None else z_center_refined for z in refine_z_by_zone]

    return {
        'xs': xs, 'ys': ys,
        'end_a': end_a, 'end_b': end_b,
        'refine_z_by_zone': refine_z_by_zone, 'z_center': z_center_refined,
    }


def compute_box_size2(seg_mask, depth_m, floor_m, fx, fy, cx, cy, bcx, bcy,
                       overlay=None, overlay1=None, color_img=None, dbg=None, label=""):
    """SAM2 seg_mask 하나 + 이 프레임의 depth_m으로 박스 하나의 실측 크기를 계산.

    box_size_plane.py와 다르게, 평면 피팅(z_ref_map) 후의 dz<=tol 테스트 + 실루엣 스냅까지만 하고
    closing/dilate-open/침식은 생략한 채로 그대로 최종 결과로 씀.

    seg_mask   : bool (H,W), SAM2가 뽑은 이 박스의 마스크
    depth_m    : float32 (H,W), 미터 단위 depth
    floor_m    : 바닥까지 거리(m)
    fx,fy,cx,cy: 컬러 카메라 내부 파라미터 (핀홀 back-projection용)
    bcx,bcy    : 대충 잡은 컴포넌트 중심 픽셀좌표 (1차 z_center 산출 시작점)
    overlay/color_img: 디버그 시각화용(선택). None이면 그림 안 그림.
    overlay1   : 1차(r1, 단일 z_center 기준) 결과만 따로 보고 싶을 때 넘기는 별도 캔버스.
    dbg        : {'jump','sil_cand','drop','sil_used'} 누적 마스크(선택). 2차 실루엣 스냅 후보/채택
                 결과를 여기에 누적해줌 - box_detect_new.py의 하늘색/노란색 시각화용.
    label      : 로그에 붙일 표시자, 예: "Box 1"

    반환: dict(cx, cy, real_w, real_h, z_cm, fill_ratio, angle_deg,
               center_x_cm, center_y_cm, center_z_cm) 또는 None(검출 실패)
    """
    H, W = depth_m.shape

    # 박스 중앙 윈도우에서 median depth = z_center
    wy0, wy1 = max(0, bcy - BOX_CENTER_WIN_PX), min(H, bcy + BOX_CENTER_WIN_PX + 1)
    wx0, wx1 = max(0, bcx - BOX_CENTER_WIN_PX), min(W, bcx + BOX_CENTER_WIN_PX + 1)
    center_patch = depth_m[wy0:wy1, wx0:wx1]
    center_valid = center_patch[center_patch > 0]
    if center_valid.size == 0:
        return None   # 박스 중앙에 depth 자체가 없으면 skip
    z_center = float(np.median(center_valid))

    if color_img is not None:
        cv2.rectangle(color_img, (wx0, wy0), (wx1 - 1, wy1 - 1), (0, 255, 255), 1)

    # ---- 윗면 판정 TOL: 박스 높이(z_cm) 구간별로 다르게 ----
    z_cm = (floor_m - z_center) * 100
    tol = _select_tol(z_cm)
    tol1 = max(TOL_R1_MIN_M, tol - TOL_R1_TIGHTEN_M)  # 1차는 더 타이트하게(평면 피팅 재료 오염 방지)

    # ---- 1차: 박스 전체를 단일 기준 z(z_center)로 윗면/OBB 추출 (평면 피팅 재료 수집용) ----
    r1 = _extract_top_face_r1(seg_mask, depth_m, z_center, tol1,
                               overlay=overlay1, color_img=color_img)
    if r1 is None:
        return None

    # ---- 1차 top 픽셀에 평면(depth ~= c0 + c1*x + c2*y)을 맞춰 픽셀별 기준 z 맵 생성 ----
    z_ref_map = None
    p1x, p1y = r1['xs'].astype(np.float32), r1['ys'].astype(np.float32)
    p1d = depth_m[r1['ys'], r1['xs']]
    fit_ok = p1d > 0
    p1x, p1y, p1d = p1x[fit_ok], p1y[fit_ok], p1d[fit_ok]
    # 로그용 평면 피팅 진단값 (계산에는 영향 없음)
    plane_fit = {'n_pts': int(p1x.size), 'fallback': True, 'n_keep': 0,
                 'resid_std_mm': None, 'resid_min_mm': None, 'resid_max_mm': None, 'note': ''}
    if p1x.size >= 30:
        A = np.column_stack((np.ones_like(p1d), p1x, p1y))
        coef, *_ = np.linalg.lstsq(A, p1d, rcond=None)
        resid = p1d - A @ coef
        # 옆면 등 평면에서 벗어난 픽셀 1회 걸러내고 재적합
        keep = np.abs(resid) <= max(0.003, 3.0 * float(np.median(np.abs(resid))))
        if 30 <= int(keep.sum()) < p1x.size:
            coef, *_ = np.linalg.lstsq(A[keep], p1d[keep], rcond=None)
        resid_final = p1d - A @ coef
        plane_fit.update(
            fallback=False, n_keep=int(keep.sum()),
            resid_std_mm=float(np.std(resid_final[keep])) * 1000.0,
            resid_min_mm=float(np.min(resid_final)) * 1000.0,   # 부호 있음: 음수=평면보다 카메라 쪽, 양수=평면보다 먼 쪽
            resid_max_mm=float(np.max(resid_final)) * 1000.0)
        if int(keep.sum()) < 30:
            plane_fit['note'] = f"inlier {int(keep.sum())}<30 -> 이상치 제거 재적합 생략"
        sy, sx = np.where(seg_mask)
        z_ref_map = np.full(depth_m.shape, r1['z_center'], dtype=np.float32)
        z_ref_map[sy, sx] = (coef[0] + coef[1] * sx + coef[2] * sy).astype(np.float32)

    if z_ref_map is None:
        # 평면 적합 실패(포인트 부족 등) 시: 긴 축 3구역 계단 기준으로 fallback
        zone_z = np.asarray(r1['refine_z_by_zone'], dtype=np.float32)  # [z0, z1, z2]
        end_a, end_b = r1['end_a'], r1['end_b']
        axis = end_b - end_a
        axis_len2 = float(axis[0] ** 2 + axis[1] ** 2)
        z_ref_map = np.full(depth_m.shape, r1['z_center'], dtype=np.float32)
        if axis_len2 > 1e-6:
            sy, sx = np.where(seg_mask)
            tt = ((sx - end_a[0]) * axis[0] + (sy - end_a[1]) * axis[1]) / axis_len2
            zone_idx = np.clip((tt * 3.0).astype(np.int32), 0, 2)
            z_ref_map[sy, sx] = zone_z[zone_idx]

    # ---- 2차: 픽셀별 z_ref_map(평면)으로 dz<=tol 테스트 + 실루엣 스냅만 ----
    valid2 = seg_mask & (depth_m > 0)
    vals2 = depth_m[valid2]
    ref2 = z_ref_map[valid2]
    dz2 = vals2 - ref2
    top_mask2 = np.zeros_like(seg_mask, dtype=bool)
    top_mask2[valid2] = (dz2 <= tol) & (dz2 >= -tol * TOL_CAM_SIDE_MULT)

    # 바닥-윗면 경계(실루엣) 스냅: SAM2 경계 바로 안쪽 띠와 바로 바깥(배경)을 짝지어 비교
    seg_u8 = seg_mask.astype(np.uint8) * 255
    seg_border = seg_u8 - cv2.erode(seg_u8, _EDGE_KERNEL)   # 안쪽 가장자리 띠
    seg_border_mask = seg_border > 0

    d_outside_prop = np.where((seg_u8 == 0) & (depth_m > 0), depth_m, 0.0).astype(np.float32)
    local_outside_depth = cv2.dilate(d_outside_prop, _EDGE_KERNEL)

    box_height_m = floor_m - r1['z_center']
    silhouette_tol_m = max(SILHOUETTE_HEIGHT_TOL_MIN_M,
                           box_height_m * SILHOUETTE_HEIGHT_TOL_RATIO)
    drop = local_outside_depth - depth_m
    # 배경과의 낙차만 보면 옆면 중간 픽셀도 우연히 박스 높이만큼 낙차가 나서 통과할 수 있음 ->
    # 그 픽셀 자체 depth도 z_ref_map(윗면 기준 평면) 근처여야 한다는 조건을 추가
    z_ref_ok = np.abs(depth_m - z_ref_map) <= tol * SILHOUETTE_Z_TOL_MULT
    is_true_silhouette = (local_outside_depth > 0) & \
        (np.abs(drop - box_height_m) <= silhouette_tol_m) & \
        z_ref_ok

    top_mask2_u8 = (top_mask2.astype(np.uint8) * 255)
    dilated2 = cv2.dilate(top_mask2_u8, _SNAP_KERNEL)
    newly_added2 = (dilated2 > 0) & ~top_mask2 & seg_mask
    near_silhouette2 = newly_added2 & seg_border_mask & is_true_silhouette
    top_mask2 = top_mask2 | near_silhouette2

    if dbg is not None:
        dbg['jump'] |= seg_border_mask
        dbg['sil_cand'] |= seg_border_mask & is_true_silhouette
        dbg['drop'][seg_border_mask] = drop[seg_border_mask]
        dbg['sil_used'] |= near_silhouette2

    if overlay is not None:
        overlay[seg_mask] = (
            0.5 * overlay[seg_mask] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)
        overlay[top_mask2] = (
            0.5 * overlay[top_mask2] + 0.5 * np.array([0, 0, 255])
        ).astype(np.uint8)

    ys, xs = np.where(top_mask2)
    if len(xs) <= 5:
        return None

    rect_px = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))

    # 후처리 없이 곧바로 median depth로 최종 z_center 산출
    z_center = float(np.median(depth_m[top_mask2]))

    # ---- 실측 크기 계산  ----
    X = (xs - cx) * z_center / fx
    Y = (ys - cy) * z_center / fy
    pts_m = np.column_stack((X, Y)).astype(np.float32)
    final_rect_m = cv2.minAreaRect(pts_m)
    real_w, real_h = final_rect_m[1]

    # 오리엔테이션 계산 (최종 크기 박스 기준, y축=0도, 시계방향 +, 반시계방향 -)
    final_box_m = cv2.boxPoints(final_rect_m).astype(np.float32)
    fe01 = np.linalg.norm(final_box_m[0] - final_box_m[1])
    fe12 = np.linalg.norm(final_box_m[1] - final_box_m[2])
    if fe01 < fe12:
        fa = (final_box_m[0] + final_box_m[1]) / 2.0
        fb = (final_box_m[2] + final_box_m[3]) / 2.0
    else:
        fa = (final_box_m[1] + final_box_m[2]) / 2.0
        fb = (final_box_m[3] + final_box_m[0]) / 2.0
    fdx, fdy = float(fb[0] - fa[0]), float(fb[1] - fa[1])
    if fdy < 0:
        fdx, fdy = -fdx, -fdy

    angle_deg = float(np.degrees(np.arctan2(-fdx, fdy)))
    if orientation_axis == 'long':
        angle_deg = angle_deg + 90.0
        if angle_deg > 90.0:
            angle_deg -= 180.0

    # ---- 박스 중앙점 카메라 좌표(X, Y, Z) ----
    center_x_cm = float(final_rect_m[0][0]) * 100.0
    center_y_cm = float(final_rect_m[0][1]) * 100.0
    center_z_cm = z_center * 100.0

    # print(f"{label} -> {real_w*100:.1f}, {real_h*100:.1f}, "
    #       f"z:{(floor_m - z_center)*100:.1f} cm, angle:{angle_deg:.1f} deg, "
    #       f"center(cam)=({center_x_cm:.1f}, {center_y_cm:.1f}, {center_z_cm:.1f}) cm")

    # ---- OBB 실측 면적 필터 ----
    if (real_w * 100) * (real_h * 100) < MIN_AREA_OBB_CM2:
        return None

    if color_img is not None:
        box_px = np.int32(cv2.boxPoints(rect_px))
        cv2.drawContours(color_img, [box_px], 0, (0, 0, 255), 2)
        cv2.putText(
            color_img,
            f"{real_w*100:.1f}x{real_h*100:.1f} cm  {angle_deg:.1f} deg",
            (int(rect_px[0][0]), int(rect_px[0][1]) - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
        )

    obb_cx_px = float(rect_px[0][0])
    obb_cy_px = float(rect_px[0][1])
    pixel_count = int(np.count_nonzero(top_mask2))
    obb_area_px = rect_px[1][0] * rect_px[1][1]
    # fill_ratio: OBB 안에 실제로 채워진 픽셀 비율
    fill_ratio = pixel_count / obb_area_px if obb_area_px > 0 else 0
    z_cm_val = (floor_m - z_center) * 100

    return {
        'cx': obb_cx_px, 'cy': obb_cy_px,
        'real_w': real_w * 100, 'real_h': real_h * 100,
        'z_cm': z_cm_val, 'fill_ratio': fill_ratio,
        'angle_deg': angle_deg,
        'center_x_cm': center_x_cm, 'center_y_cm': center_y_cm,
        'center_z_cm': center_z_cm,
        'plane_fit': plane_fit,
    }


if __name__ == "__main__":
    import box_capture

    cap = box_capture.BoxCapture()
    try:
        while True:
            frame = cap.get_frame()
            if frame is None:
                break
            color_img, depth_m, valid, candidates = frame
            if color_img is None:
                continue
            if valid is None or not np.any(valid):
                cv2.imshow("box_size_plane2 debug", color_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            overlay = color_img.copy()
            overlay1 = color_img.copy()          # 1차(r1) 전용 캔버스 - 최종이랑 겹쳐 보이지 않게 분리
            color_img_orig = color_img.copy()    # 1차 창 블렌드용 원본 (color_img는 아래서 계속 그려짐)

            count = 0
            for i, cand in enumerate(candidates):
                det = compute_box_size2(
                    cand['seg_mask'], depth_m, cap.floor_m,
                    cap.fx, cap.fy, cap.cx, cap.cy, cand['bcx'], cand['bcy'],
                    overlay=overlay, overlay1=overlay1, color_img=color_img, label=f"Box {i+1}")
                if det is not None:
                    count += 1

            color_img1 = cv2.addWeighted(overlay1, 0.5, color_img_orig, 0.5, 0)
            cv2.putText(color_img1, f"boxes: {count}  [1cha(r1) - danil z_center]", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("box_size_plane2 debug - 1cha(r1)", color_img1)

            color_img = cv2.addWeighted(overlay, 0.5, color_img, 0.5, 0)
            cv2.putText(color_img, f"boxes: {count}  [final - no post-processing]", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("box_size_plane2 debug - final(no post-processing)", color_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.shutdown()
        cv2.destroyAllWindows()
