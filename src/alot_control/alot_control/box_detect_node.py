#!/usr/bin/env python3
"""
box_detect_node_ver_5.py — top-down 박스 인식 + 흡착점 발행 (ROS2 노드).

ver_4 대비 변경
    1. 파지점 = '평면 위' 무게중심 (plane_centroid_px).
       기존 모서리(능선)에서 거리 변환 최댓값 측정해서 중심점을 정하는 알고리즘이
       박스가 회전하며 대각선이 되면 능선을 타고 거리 측정이 불안정해짐을 인식.
       (직사각형의 최댓값은 긴 축을 따라 거의 평평한 능선이라, 박스가 돌면
       경계 레스터화 패턴이 바뀌며 파지점이 능선을 미끄러짐)
       기존 로직에서 벗어나서 무게중심 계산으로 알고리즘을 수정하였다.
       그런데 카메라가 또 많이 기울게 되면 직사각형이 사다리꼴로 보여
       단순 넓이 적분인 기존 방식에 문제가 생길 것으로 판단.
       역투영 후 적분하여 평면 위 무게중김 계산으로 해결하였다.

    2. 최상층 시드 밴드의 ±1cm 여유 제거.
       시드는 한쪽만 막힌 밴드(depth <= top + layer_m)라 표면보다 가까운 노이즈는
       여유 없이도 들어온다. ±1cm 는 밴드를 깎기만 해서, layer_m 을 0.6cm 로
       줄이면 시드가 비어 검출이 0 이 됐다.

    3. 시각화를 위해 result / binary 윈도우 창을 추가하였다.



ver_4 에서 그대로 넘어온 것
    최상층 평면 반복 재적합, 박스별 옆면 제거,
    up_axis, ROI 단위 박스 처리.

    python3 box_detect_node_ver_5.py

창
    result  빨강 원 = 지금 발행 중인 흡착점, 초록 원 = 나머지 후보,
            원 반지름 = 경계까지 여유, 연분홍 = 필터 전 원본 윤곽.
            상단 줄에 fps / raw(검출) / out(추적) / drop(합쳐져 버려진 박스).
    binary  박스 분리에 쓴 이진 마스크.
    q 또는 ESC 로 종료. 모니터 없는 PC(SSH 등)면 show_window 를 false 로.

    ## 그런데 지금 드는 생각이 지금 흡착점 발행하는 방식을 바꿔야 할듯?
    # 아닌가 어차피 바스켓에서 꺼내게 된다면 의미 없을지도

발행
    ~/pick_point  geometry_msgs/PointStamped  다음에 집을 흡착점 하나.
                  frame_id = camera_frame (카메라 optical frame)
                  왜곡계수가 반영된 3D 좌표. 핸드아이 변환은 안 한다.
                  공압 그리퍼라 자세는 안 보낸다. 수직으로 내려오면 된다.

파라미터 (전부 ros2 param set 으로 런타임 변경 가능)
    camera_frame 발행 헤더의 frame_id
    pad_radius_m 흡착패드 반경. 경계 여유가 이보다 작으면 안 내보낸다.
    max_tilt_deg 윗면이 이보다 기울면 수직으로 내려와도 진공이 안 붙는다.
                 up_axis 가 없으면 '광축 대비' 각도라서, 카메라 자체가 기운 만큼 더해진다.
    up_axis      camera_frame 에서 본 위쪽(중력 반대) 단위벡터. [0.0, 0.0, 0.0] 이면 안 씀.
                 정확히 내려다보는 카메라면 [0.0, 0.0, -1.0].
                 반드시 실수로 쓸 것. [0,0,-1] 처럼 정수로 주면 Humble 의 정적 타입
                 검사에 걸려 거부된다.
    range_m      이보다 먼 건 배경. 팔레트 바닥 바로 앞에 맞춘다.
                 # 이건 바스켓 완성되면 측정 pose에서 카메라와 바닥까지의 거리 측정해서 변경해야 함.
    layer_m      최상층 후보 두께. 박스 높이보다 작게.
                 기울기에는 강하지만 깊이 노이즈보다는 충분히 커야 한다 (0.8cm 이상에서 검증).
    min_area_px  이보다 작은 덩어리 무시. 해상도와 화각에 딸린 값이라
                 스트림 설정을 바꾸면 다시 잡아야 한다.
    seam         박스 사이 어두운 틈 감도. 0 으로 두지 말 것 (위 2번).
    blur         가우시안 블러. 0 이면 노이즈가 전부 틈으로 잡힌다.
    show_window  result / binary 창 표시. 기본 True.
"""

import cv2
import numpy as np

W, H, FPS = 640, 480, 30   # D435 컬러는 848x480 조합이 없는 유닛이 있다
PLANE_TOL = 0.008          # RANSAC 인라이어 허용 오차 (m)
SAFE_FRAC = 0.8            # 파지점 여유가 최대의 이 비율 미만이면 안전한 곳으로 옮긴다
K3, K5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
_rays = {}


def deproject(uu, vv, zz, K):
    """픽셀+깊이 -> 카메라 optical frame 3D."""
    return np.stack([(uu - K["cx"]) * zz / K["fx"],
                     (vv - K["cy"]) * zz / K["fy"], zz], -1)


def sample_points(mask, depth_m, K, n=4000):
    # mask 픽셀 중 최대 n 개만 뽑아 역투영. 전부 역투영한 뒤 버리면 느리다.
    vv, uu = np.nonzero(mask)
    if len(vv) > n:
        i = np.random.randint(0, len(vv), n)
        vv, uu = vv[i], uu[i]
    return deproject(uu, vv, depth_m[vv, uu], K)


def plane_distance(depth_m, nrm, d, K, roi=None):
    """
    평면까지의 부호있는 거리. 양수 = 평면보다 카메라 쪽(위), 음수 = 아래.
    p = Z*ray 이므로 n·p - d = Z*(n·ray) - d. (H,W,3) 배열을 만들 필요가 없다.
    roi=(y0, y1, x0, x1) 이면 depth_m 은 그 영역만 잘라서 넘긴 것.
    """
    if roi is None:
        key = (depth_m.shape, K["fx"], K["fy"], K["cx"], K["cy"])
        if key not in _rays:
            v, u = np.mgrid[0:depth_m.shape[0], 0:depth_m.shape[1]].astype(np.float32)
            _rays[key] = ((u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"])
        rx, ry = _rays[key]
    else:
        y0, y1, x0, x1 = roi
        rx = ((np.arange(x0, x1, dtype=np.float32) - K["cx"]) / K["fx"])[None, :]
        ry = ((np.arange(y0, y1, dtype=np.float32) - K["cy"]) / K["fy"])[:, None]
    return depth_m * (nrm[0] * rx + nrm[1] * ry + nrm[2]) - d


def fit_plane(pts, iters=40):
    # RANSAC 평면. pts (N,3) -> (normal, d) 또는 None. n·p=d, |n|=1, nz<0.
    if len(pts) < 50:
        return None
    if len(pts) > 4000:                        # 30fps 유지용 서브샘플
        pts = pts[np.random.randint(0, len(pts), 4000)]

    best_n, best_inl = 0, None
    for i0, i1, i2 in np.random.randint(0, len(pts), (iters, 3)):
        v = np.cross(pts[i1] - pts[i0], pts[i2] - pts[i0])
        L = np.linalg.norm(v)
        if L < 1e-9:
            continue
        v /= L
        inl = np.abs(pts @ v - v @ pts[i0]) < PLANE_TOL
        if inl.sum() > best_n:
            best_n, best_inl = inl.sum(), inl
    if best_inl is None:
        return None

    P = pts[best_inl]                          # 인라이어로 최소자승 재적합
    c = P.mean(0)
    nrm = np.linalg.svd(P - c, full_matrices=False)[2][2]
    if nrm[2] > 0:
        nrm = -nrm
    return nrm, float(nrm @ c)


def tilt_deg(nrm, up=None):
    """
    윗면 법선의 기울기(도). up 이 없으면 광축 대비(0 = 카메라를 정면으로 마주봄),
    있으면 실제 수직 대비. 수직 하강 그리퍼에게 의미 있는 건 후자다.
    """
    ref = np.array([0.0, 0.0, -1.0]) if up is None else up
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, float(nrm @ ref))))))


def fit_top_plane(depth_m, valid, K, layer_m, up=None, iters=3):
    """
    최상층 평면을 카메라 기울기와 무관하게 추정한다.

    초기값: up 이 있으면 그 축 방향 높이의 상위층, 없으면 Z 밴드.
    Z 밴드는 카메라가 기울면 가까운 쪽 띠만 덮는다. 좁은 띠에서 뽑은 법선은
    깊이 계통오차(가장자리 휨, 박스 높이 편차)에 몇 도씩 흔들리고, 그 오차가
    화면 반대편으로 갈수록 선형으로 커진다.
    그래서 '현재 평면 기준' ±layer_m 밴드로 후보를 다시 뽑아 재적합한다.
    이 밴드는 평면을 따라 기울어 있으므로 윗면 전체를 덮고, 보통 1~2 회에 수렴한다.
    layer_m < 박스 높이 이므로 아래층은 들어오지 않는다.

    시드 밴드는 한쪽만 막혀 있다(Z 는 위쪽, up 은 아래쪽 경계만 있다).
    표면 너머 노이즈는 여유 없이도 들어오므로 기준점에 여유를 더하지 않는다.
    """
    if up is not None:
        h = plane_distance(depth_m, up, 0.0, K)          # up 축 방향 높이 (클수록 위)
        top_h = float(np.percentile(h[valid], 98))
        cand = valid & (h >= top_h - layer_m)
    else:
        top = float(np.percentile(depth_m[valid], 2))
        cand = valid & (depth_m <= top + layer_m)
    if cand.sum() < 500:
        return None
    fit = fit_plane(sample_points(cand, depth_m, K))
    if fit is None:
        return None

    for _ in range(iters):
        cand = valid & (np.abs(plane_distance(depth_m, *fit, K)) < layer_m)
        if cand.sum() < 500:
            break
        new = fit_plane(sample_points(cand, depth_m, K))
        if new is None:
            break
        moved = np.degrees(np.arccos(min(1.0, abs(float(new[0] @ fit[0])))))
        fit = new
        if moved < 0.2:                                 # 수렴
            break
    return fit


def plane_centroid_px(contour, nrm, d, K):
    """
    윤곽을 박스 평면에 역투영해 평면 위 다각형 무게중심을 구하고 다시 픽셀로.

    이미지에서 바로 무게중심을 재면 안 된다. 카메라가 기울면 직사각형 윗면이
    사다리꼴로 찍히고, 이미지 무게중심은 넓게 보이는 쪽으로 밀린다
    (250x90mm 박스, 기울기 25도에서 3.5mm). 평면 위에서 재면 원근이 안 들어온다.
    평면을 아니까 이미지<->평면 관계가 호모그래피라, 영상을 펴는 대신
    윤곽 점만 평면으로 보내 신발끈 공식으로 푼다.
    """
    p = contour.reshape(-1, 2).astype(np.float64)
    r = np.stack([(p[:, 0] - K["cx"]) / K["fx"],
                  (p[:, 1] - K["cy"]) / K["fy"], np.ones(len(p))], -1)
    P = r * (d / (r @ nrm))[:, None]                 # 평면 위 3D 점

    e1 = np.cross(nrm, [0.0, 0.0, 1.0])              # 평면 위 직교 기저
    L = np.linalg.norm(e1)
    e1 = e1 / L if L > 1e-9 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(nrm, e1)
    O = P.mean(0)
    a, b = (P - O) @ e1, (P - O) @ e2
    cr = a * np.roll(b, -1) - np.roll(a, -1) * b
    A = cr.sum() / 2.0
    if abs(A) < 1e-12:
        return None
    C = (O + ((a + np.roll(a, -1)) * cr).sum() / (6 * A) * e1
           + ((b + np.roll(b, -1)) * cr).sum() / (6 * A) * e2)
    if C[2] <= 1e-6:
        return None
    return K["fx"] * C[0] / C[2] + K["cx"], K["fy"] * C[1] / C[2] + K["cy"]


def grasp_point(seg, seed):
    # 이 알고리즘이 굳이 필요할까 싶네. 박스 작은 것도 있긴 할텐데, 쓸 거면 최적화해야할 듯
    """
    seed(무게중심)의 경계 여유를 재고, 여유가 너무 작으면 안전한 곳으로 옮긴다.
    윗면 일부가 파여 있으면(이음새 컷이 인쇄나 주름을 반쯤 자름) 무게중심이 홈 안에
    떨어질 수 있다. 정상 박스에서는 이 분기를 타지 않는다.
    """
    dt = cv2.distanceTransform(seg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    hh, ww = seg.shape
    u = min(max(int(round(seed[0])), 0), ww - 1)
    v = min(max(int(round(seed[1])), 0), hh - 1)
    if dt[v, u] >= dt.max() * SAFE_FRAC:
        return float(seed[0]), float(seed[1]), float(dt[v, u])
    sv, su = np.nonzero(dt >= dt.max() * SAFE_FRAC)
    k = int(np.argmin((su - seed[0]) ** 2 + (sv - seed[1]) ** 2))
    return float(su[k]), float(sv[k]), float(dt[sv[k], su[k]])


def regrow(mask, seeds, depth_m, valid, min_area):
    # grasp_point와 이하동문
    """
    이음새로 잘라낸 조각(seeds)을 mask 안에서 다시 키워 원래 면적을 복원한다.
    안내 영상은 color 가 아니라 depth. 인쇄물은 depth 에 없으므로 거기서
    성장이 멈추지 않고 진짜 경계에서만 멈춘다.
    """
    n, cc, stats, _ = cv2.connectedComponentsWithStats(seeds, 8)
    keep = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area * 0.3, 1, 0)
    keep[0] = 0
    if keep.sum() == 0:
        return np.zeros(mask.shape, np.int32)

    remap = np.zeros(n, np.int32)
    remap[keep == 1] = np.arange(1, keep.sum() + 1)
    markers = remap[cc] + 1
    markers[mask == 0] = 1                                  # 배경
    markers[(mask > 0) & (remap[cc] == 0)] = 0              # 파먹힌 영역 = 미정

    dn = cv2.normalize(np.where(valid, depth_m, 0), None, 0, 255,
                       cv2.NORM_MINMAX).astype(np.uint8)
    guide = cv2.cvtColor(cv2.GaussianBlur(dn, (3, 3), 0), cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(guide, markers)
    markers[markers <= 1] = 1
    return (markers - 1).astype(np.int32)


def find_boxes(color, depth_m, K, range_m, layer_m, min_area, seam=8, blur_k=5, up=None):
    """
    반환 (boxes, binary, dropped).
    boxes 는 가까운(=높은) 순. dropped 는 한 라벨에 박스 크기 조각이 여럿 있어서
    가장 큰 것만 남기고 버린 개수 (= seam 컷이 못 가른 박스).
    """
    empty = np.zeros(depth_m.shape, np.uint8)

    # 0) depth 구멍 메우기. 안 하면 0 픽셀이 전부 경계로 잡혀 마스크가 벌집이 된다.
    holes = depth_m <= 0
    if holes.any():
        depth_m = np.where(holes, cv2.dilate(depth_m, K3), depth_m)   # max=보수적
    depth_m = cv2.medianBlur(depth_m, 3)

    # 1) 배경 컷 -> 2) 최상층 평면 (기울기 보정 반복 재적합).
    valid = (depth_m > 0.2) & (depth_m < range_m)
    if valid.sum() < 500:
        return [], empty, 0
    fit = fit_top_plane(depth_m, valid, K, layer_m, up)
    if fit is None:
        return [], empty, 0

    sd = plane_distance(depth_m, *fit, K)
    mask = (valid & (sd > -layer_m) & (sd < PLANE_TOL * 2)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K5)

    # 3) 딱 붙은 박스는 깊이로 안 갈라진다. 사이의 어두운 틈으로 자른다.
    # 여기서 문제 생기기 딱 좋긴 함.
    binary = mask
    if seam > 0:
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if blur_k >= 3:
            gray = cv2.GaussianBlur(gray, (blur_k | 1, blur_k | 1), 0)
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((13, 13), np.uint8))
        cut = cv2.dilate((bh > seam).astype(np.uint8) * 255, K3)
        binary = cv2.morphologyEx(cv2.bitwise_and(mask, cv2.bitwise_not(cut)),
                                  cv2.MORPH_OPEN, K3)

    labels = regrow(mask, binary, depth_m, valid, min_area)

    # 4) 박스별 처리. 전부 박스 bounding box(ROI) 안에서만 계산한다.
    boxes, dropped = [], 0
    Hh, Ww = depth_m.shape
    for li in range(1, labels.max() + 1):
        region = (labels == li).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(region)
        if w * h < min_area:
            continue
        x0, y0, x1, y1 = max(x - 1, 0), max(y - 1, 0), min(x + w + 1, Ww), min(y + h + 1, Hh)
        r = (slice(y0, y1), slice(x0, x1))
        dr = depth_m[r]
        sel = (region[r] > 0) & valid[r]
        if sel.sum() < 50:
            continue

        sv, su = np.nonzero(sel)
        if len(sv) > 4000:
            i = np.random.randint(0, len(sv), 4000)
            sv, su = sv[i], su[i]
        f = fit_plane(deproject(su + x0, sv + y0, dr[sv, su], K), iters=20)
        if f is None:
            continue

        # 자기 평면에 붙은 픽셀만 윗면으로 남긴다. 카메라가 기울면 옆면 윗부분이
        # 슬랩(평면 아래 layer_m 까지) 안에 들어와 윗면 라벨에 붙는다.
        on = np.abs(plane_distance(dr, *f, K, roi=(y0, y1, x0, x1))) < PLANE_TOL * 2
        seg = cv2.morphologyEx((sel & on).astype(np.uint8) * 255, cv2.MORPH_OPEN, K3)
        n_cc, cc, st, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
        if n_cc < 2:
            continue
        areas = st[1:, cv2.CC_STAT_AREA]
        dropped += max(0, int((areas >= min_area).sum()) - 1)   # 가장 큰 것 빼고 버려지는 박스
        seg = (cc == 1 + int(np.argmax(areas))).astype(np.uint8) * 255

        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                   offset=(x0, y0))
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        (rw, rh) = cv2.minAreaRect(c)[1]
        if area < min_area or rw * rh <= 0 or area / (rw * rh) < 0.70:
            continue                                    # 사각형답지 않으면 버림
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > 6.0:
            continue

        # 흡착점 = 평면 위 무게중심. 1px 0 테두리를 붙여 화면 경계도 경계로 본다.
        cen = plane_centroid_px(c, *f, K)
        if cen is None:
            continue
        pad = cv2.copyMakeBorder(seg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        gu, gv, clear = grasp_point(pad, (cen[0] - x0 + 1, cen[1] - y0 + 1))
        gu, gv = gu - 1 + x0, gv - 1 + y0
        ray = np.array([(gu - K["cx"]) / K["fx"], (gv - K["cy"]) / K["fy"], 1.0])
        den = float(f[0] @ ray)
        if abs(den) < 1e-6:
            continue

        boxes.append({"center": (float(gu), float(gv)), "depth": float(f[1] / den),
                      "normal": f[0], "clear_px": float(clear), "contour": c})

    boxes.sort(key=lambda b: b["depth"])
    return boxes, binary, dropped


def track_boxes(tracks, dets, alpha=0.20, max_jump=80.0, max_lost=8, min_hits=3):
    """
    검출을 프레임 간에 이어붙여 떨림을 잡는다. tracks 는 제자리에서 갱신된다.
    칼만은 안 쓴다. 박스가 제자리에 있어 속도 예측이 무의미함
    EMA만 사용.
    """
    vec = lambda d: np.array([*d["center"], d["depth"], *d["normal"], d["clear_px"]])

    for t in tracks:
        t["lost"] += 1
    pairs = sorted((np.hypot(t["v"][0] - d["center"][0], t["v"][1] - d["center"][1]), ti, di)
                   for ti, t in enumerate(tracks) for di, d in enumerate(dets))
    used_t, used_d = set(), set()
    for dist, ti, di in pairs:
        if dist > max_jump or ti in used_t or di in used_d:
            continue
        t = tracks[ti]
        t["v"] = alpha * vec(dets[di]) + (1 - alpha) * t["v"]
        t["lost"], t["hits"] = 0, t["hits"] + 1
        used_t.add(ti)
        used_d.add(di)
    for di, d in enumerate(dets):
        if di not in used_d:
            tracks.append({"v": vec(d), "lost": 0, "hits": 1})
    tracks[:] = [t for t in tracks if t["lost"] <= max_lost]

    out = []
    for t in tracks:
        if t["hits"] < min_hits:
            continue
        cx, cy, z, nx, ny, nz, clear = t["v"]
        n = np.array([nx, ny, nz]) / max(1e-9, np.linalg.norm([nx, ny, nz]))
        out.append({"center": (cx, cy), "depth": float(z),
                    "normal": n, "clear_px": float(clear), "lost": t["lost"]})
    out.sort(key=lambda b: b["depth"])
    return out


def read_up(value):
    """up_axis 파라미터 -> 단위벡터 또는 None. 법선과 같은 방향(nz<0 쪽)으로 맞춘다."""
    u = np.asarray(list(value), float)
    L = np.linalg.norm(u) if u.shape == (3,) else 0.0
    if L < 0.5:
        return None
    u = u / L
    return u if u[2] <= 0 else -u


def put(img, txt, org, color, scale=0.5):
    """검은 테두리 깔고 글자. 배경이 밝든 어둡든 읽히게."""
    for c, th in ((0, 0, 0), 3), (color, 1):
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, c, th, cv2.LINE_AA)


def draw(color, boxes, raw, picked, up):
    """
    result 창 그림. picked 는 지금 발행 중인 박스(빨강), 나머지 후보는 초록.
    raw 는 추적 전 원본 검출 윤곽(연분홍).
    """
    vis = color.copy()
    for b in raw:
        cv2.drawContours(vis, [b["contour"]], -1, (200, 150, 200), 1)

    for i, b in enumerate(boxes):
        c = (0, 0, 255) if b is picked else (0, 200, 0)
        u, v = int(b["center"][0]), int(b["center"][1])
        rad = max(2, int(b["clear_px"]))
        cv2.circle(vis, (u, v), rad, c, 2)              # 흡착패드 여유 반경
        cv2.circle(vis, (u, v), 3, c, -1)
        n = b["normal"]
        cv2.arrowedLine(vis, (u, v), (int(u + n[0] * 120), int(v + n[1] * 120)),
                        (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.3)
        # 라벨은 원 안쪽에 두 줄. 원 위에 한 줄로 붙이면 박스가 촘촘할 때
        # 옆 박스 라벨과 겹치고 윗줄은 화면 밖으로 잘린다.
        tag = f"#{i}" + (f" pred{b['lost']}" if b.get("lost") else "")
        for k, txt in enumerate((f"{tag} {b['depth']:.3f}m",
                                 f"tilt{tilt_deg(n, up):.0f} clr{b['clear_px']:.0f}")):
            tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
            put(vis, txt, (u - tw // 2, v + 18 + k * 14), c, 0.4)
    return vis


def main():
    import time

    import pyrealsense2 as rs
    import rclpy
    from geometry_msgs.msg import PointStamped

    rclpy.init()
    node = rclpy.create_node("box_detector")
    for name, default in [("camera_frame", "camera_color_optical_frame"),
                          ("pad_radius_m", 0.020), ("max_tilt_deg", 25.0),
                          ("up_axis", [0.0, 0.0, 0.0]),
                          ("range_m", 0.40), ("layer_m", 0.03),
                          ("min_area_px", 3000), ("seam", 8), ("blur", 5),
                          ("show_window", True)]:
        node.declare_parameter(name, default)
    pub = node.create_publisher(PointStamped, "~/pick_point", 10)

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    align = rs.align(rs.stream.color)   # 같은 (u,v) 가 같은 지점을 가리키게
    # hole_filling 은 쓰지 말 것 — 박스 사이 틈까지 메워 두 박스가 한 덩어리가 된다.
    spatial, temporal = rs.spatial_filter(), rs.temporal_filter()

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
    node.get_logger().info(
        f"stream={intr.width}x{intr.height} depth_scale={depth_scale} "
        f"fx={intr.fx:.2f} fy={intr.fy:.2f} "
        f"cx={intr.ppx:.2f} cy={intr.ppy:.2f} model={intr.model} "
        f"coeffs={[round(c, 5) for c in intr.coeffs]}")
    if max(abs(c) for c in intr.coeffs) > 0.01:
        # 3D 환산은 librealsense 가 왜곡을 반영하지만 평면 피팅은 핀홀 가정이다.
        node.get_logger().warn("왜곡계수가 큽니다. 가장자리 박스에서 오차가 생길 수 있습니다.")

    tracks, n_frame, fps, t0, seam_warned = [], 0, 0.0, time.time(), False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)      # ros2 param set 수신
            P = node.get_parameter                    # 매 프레임 다시 읽는다
            up = read_up(P("up_axis").value)

            seam = P("seam").value
            if seam <= 0 and not seam_warned:         # 0 으로 바뀐 순간 한 번만
                node.get_logger().warn(
                    "seam=0 이면 이웃 박스가 한 덩어리가 되어 하나만 남습니다. 8 근처로 두세요.")
            seam_warned = seam <= 0

            frames = align.process(pipe.wait_for_frames())
            df, cf = frames.get_depth_frame(), frames.get_color_frame()
            if not df or not cf:
                continue
            df = temporal.process(spatial.process(df))
            color = np.asanyarray(cf.get_data())
            depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

            boxes, binary, dropped = find_boxes(
                color, depth_m, K,
                range_m=P("range_m").value, layer_m=P("layer_m").value,
                min_area=P("min_area_px").value,
                seam=seam, blur_k=P("blur").value, up=up)
            stable = track_boxes(tracks, boxes)

            # 가장 높은 것부터, 패드가 실제로 붙을 수 있는 첫 박스.
            # clear_px 를 미터로 환산해 패드 반경과 비교한다.
            b = next((b for b in stable
                      if tilt_deg(b["normal"], up) <= P("max_tilt_deg").value
                      and b["clear_px"] * b["depth"] / K["fx"] >= P("pad_radius_m").value),
                     None)
            n_frame += 1
            now = time.time()
            fps, t0 = 0.9 * fps + 0.1 / max(1e-6, now - t0), now

            if P("show_window").value:
                vis = draw(color, stable, boxes, b, up)
                put(vis, f"{fps:4.1f}fps raw={len(boxes)} out={len(stable)} drop={dropped}",
                    (10, 24), (255, 255, 255), 0.6)
                cv2.imshow("result", vis)
                cv2.imshow("binary", binary)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if b is None:
                continue

            # 왜곡 보정은 librealsense 역투영에 맡긴다. 색 스트림의 왜곡 모델을
            # 그대로 반영하므로 직접 구현할 이유가 없다.
            x, y, z = rs.rs2_deproject_pixel_to_point(
                intr, [float(b["center"][0]), float(b["center"][1])], b["depth"])
            msg = PointStamped()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = P("camera_frame").value
            msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
            pub.publish(msg)

            if n_frame % FPS == 0:                    # 1초에 한 번만 로그
                node.get_logger().info(
                    f"pick ({x:+.3f},{y:+.3f},{z:.3f})m tilt={tilt_deg(b['normal'], up):.1f} "
                    f"clear={b['clear_px']:.0f}px boxes={len(stable)} drop={dropped}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
