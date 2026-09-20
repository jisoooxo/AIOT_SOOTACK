#!/usr/bin/env python3
"""
box_detect_ransac_ver_3.py — top-down 박스 인식 + 흡착점 발행 (ROS2 노드).

ver_2 의 검출 로직 그대로.

창   box_detector : 검출 결과. q 또는 ESC 로 종료.

발행
    ~/pick_point  geometry_msgs/PointStamped  다음에 집을 흡착점 하나.
                  frame_id = camera_frame (카메라 optical frame)
                  왜곡계수가 반영된 3D 좌표. 핸드아이 변환은 안 한다.
                  공압 그리퍼라 자세는 안 보낸다. 수직으로 내려오면 된다.

파라미터 (전부 ros2 param set 으로 런타임 변경 가능)
    show_window  결과 창 표시 여부. 기본 True.
    camera_frame 발행 헤더의 frame_id
    pad_radius_m 흡착패드 반경. 경계 여유가 이보다 작으면 안 내보낸다.
    max_tilt_deg 윗면이 이보다 기울면 수직으로 내려와도 진공이 안 붙는다.
    range_m      이보다 먼 건 배경. 팔레트 바닥 바로 앞에 맞춘다.
    layer_m      RANSAC 후보 깊이 범위. 박스 높이보다 작게.
                 카메라가 가까울수록 줄여야 아래층이 안 딸려온다.
    min_area_px  이보다 작은 덩어리 무시. 해상도와 화각에 딸린 값이라
                 스트림 설정을 바꾸면 다시 잡아야 한다.
    seam         박스 사이 어두운 틈 감도. 0 이면 깊이만으로 자름.
    blur         가우시안 블러. 0 이면 노이즈가 전부 틈으로 잡힌다.

검출 파라미터를 눈으로 맞춰야 하면 ver_2 를 트랙바로 잡은 뒤 그 값을 넘긴다.
"""

import cv2
import numpy as np

W, H, FPS = 640, 480, 30
PLANE_TOL = 0.008          # RANSAC 인라이어 허용 오차 (m)
K3, K5 = np.ones((3, 3), np.uint8), np.ones((5, 5), np.uint8)
_rays = {}


def deproject(uu, vv, zz, K):
    """픽셀+깊이 -> 카메라 optical frame 3D."""
    return np.stack([(uu - K["cx"]) * zz / K["fx"],
                     (vv - K["cy"]) * zz / K["fy"], zz], -1)


def plane_distance(depth_m, nrm, d, K):

    key = (depth_m.shape, K["fx"], K["cx"])
    if key not in _rays:
        v, u = np.mgrid[0:depth_m.shape[0], 0:depth_m.shape[1]].astype(np.float32)
        _rays[key] = ((u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"])
    rx, ry = _rays[key]
    return depth_m * (nrm[0] * rx + nrm[1] * ry + nrm[2]) - d


def fit_plane(pts, iters=40):
    """RANSAC 평면. pts (N,3) -> (normal, d) 또는 None. n·p=d, |n|=1, nz<0."""
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


def tilt_deg(nrm):
    """윗면 법선이 광축에서 몇 도 기울었나. 0 이면 정면으로 마주봄."""
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, -nrm[2])))))


def regrow(mask, seeds, depth_m, valid, min_area):
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


def find_boxes(color, depth_m, K, range_m, layer_m, min_area, seam=8, blur_k=5):
    """반환 (boxes, binary). boxes 는 가까운(=높은) 순."""
    empty = np.zeros(depth_m.shape, np.uint8)

    # 0) depth 구멍 메우기. 안 하면 0 픽셀이 전부 경계로 잡혀 마스크가 벌집이 된다.
    holes = depth_m <= 0
    if holes.any():
        depth_m = np.where(holes, cv2.dilate(depth_m, K3), depth_m)   # max=보수적
    depth_m = cv2.medianBlur(depth_m, 3)

    # 1) 배경 컷 -> 2) 후보 영역에서 RANSAC 평면.
    #    단순 깊이 밴드만 쓰면 카메라가 4도만 기울어도 윗면의 40%가 잘려나간다.
    valid = (depth_m > 0.2) & (depth_m < range_m)
    if valid.sum() < 500:
        return [], empty
    top = float(np.percentile(depth_m[valid], 2)) - 0.01
    rough = valid & (depth_m <= top + layer_m)
    if rough.sum() < 500:
        return [], empty
    vv, uu = np.nonzero(rough)
    fit = fit_plane(deproject(uu, vv, depth_m[rough], K))
    if fit is None:
        return [], empty

    sd = plane_distance(depth_m, *fit, K)
    mask = (valid & (sd > -layer_m) & (sd < PLANE_TOL * 2)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K5)

    # 3) 딱 붙은 박스는 깊이로 안 갈라진다. 사이의 어두운 틈으로 자른다.
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

    boxes = []
    for li in range(1, labels.max() + 1):
        seg = (labels == li).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        (rw, rh) = cv2.minAreaRect(c)[1]
        if area < min_area or rw * rh <= 0 or area / (rw * rh) < 0.70:
            continue                                    # 사각형답지 않으면 버림
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > 6.0:
            continue

        sel = (seg > 0) & valid
        if sel.sum() < 50:
            continue
        sv, su = np.nonzero(sel)
        f = fit_plane(deproject(su, sv, depth_m[sel], K), iters=20)
        if f is None:
            continue

        # 파지점은 사각형 중심이 아니라 '경계에서 가장 먼 점'.
        # 흡착패드가 모서리나 틈에 걸치면 진공이 안 잡힌다.
        _, clear, _, (gu, gv) = cv2.minMaxLoc(cv2.distanceTransform(seg, cv2.DIST_L2, 5))
        ray = np.array([(gu - K["cx"]) / K["fx"], (gv - K["cy"]) / K["fy"], 1.0])
        den = float(f[0] @ ray)
        if abs(den) < 1e-6:
            continue

        boxes.append({"center": (float(gu), float(gv)), "depth": float(f[1] / den),
                      "normal": f[0], "clear_px": float(clear), "contour": c})

    boxes.sort(key=lambda b: b["depth"])
    return boxes, binary


def track_boxes(tracks, dets, alpha=0.20, max_jump=80.0, max_lost=8, min_hits=3):
    """
    검출을 프레임 간에 이어붙여 떨림을 잡는다. tracks 는 제자리에서 갱신된다.
    칼만은 안 쓴다. 박스가 제자리에 있어 속도 예측이 무의미하고,
    실측으로 EMA 만 쓴 것과 결과가 같았다.
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


def draw(color, boxes, picked):
    """검출 결과 그림. 창은 띄우지 않고 토픽으로만 내보낸다."""
    vis = color.copy()
    for b in boxes:
        hit = picked is not None and b is picked
        c = (0, 0, 255) if hit else (0, 200, 0)
        u, v = int(b["center"][0]), int(b["center"][1])
        cv2.circle(vis, (u, v), max(2, int(b["clear_px"])), c, 2)   # 패드 여유 반경
        cv2.circle(vis, (u, v), 3, c, -1)
        txt = f"{b['depth']:.3f}m tilt{tilt_deg(b['normal']):.0f} clr{b['clear_px']:.0f}"
        for col, th in ((0, 0, 0), 3), (c, 1):
            cv2.putText(vis, txt, (u - 60, v - int(b["clear_px"]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
    return vis


def main():
    import pyrealsense2 as rs
    import rclpy
    from geometry_msgs.msg import PointStamped

    rclpy.init()
    node = rclpy.create_node("box_detector")
    for name, default in [("camera_frame", "camera_color_optical_frame"),
                          ("pad_radius_m", 0.020), ("max_tilt_deg", 25.0),
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
        node.get_logger().warning("왜곡계수가 큽니다. 가장자리 박스에서 오차가 생길 수 있습니다.")

    tracks, n_frame = [], 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)      # ros2 param set 수신
            P = node.get_parameter                    # 매 프레임 다시 읽는다

            frames = align.process(pipe.wait_for_frames())
            df, cf = frames.get_depth_frame(), frames.get_color_frame()
            if not df or not cf:
                continue
            df = temporal.process(spatial.process(df))
            color = np.asanyarray(cf.get_data())
            depth_m = np.asanyarray(df.get_data()).astype(np.float32) * depth_scale

            boxes, _ = find_boxes(color, depth_m, K,
                                  range_m=P("range_m").value, layer_m=P("layer_m").value,
                                  min_area=P("min_area_px").value,
                                  seam=P("seam").value, blur_k=P("blur").value)
            stable = track_boxes(tracks, boxes)

            # 가장 높은 것부터, 패드가 실제로 붙을 수 있는 첫 박스.
            # clear_px 를 미터로 환산해 패드 반경과 비교한다.
            b = next((b for b in stable
                      if tilt_deg(b["normal"]) <= P("max_tilt_deg").value
                      and b["clear_px"] * b["depth"] / K["fx"] >= P("pad_radius_m").value),
                     None)
            n_frame += 1
            stamp = node.get_clock().now().to_msg()
            frame = P("camera_frame").value
            if P("show_window").value:
                cv2.imshow("box_detector", draw(color, stable, b))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if b is None:
                continue

            # 왜곡 보정은 librealsense 역투영에 맡긴다. 색 스트림의 왜곡 모델을
            # 그대로 반영하므로 직접 구현할 이유가 없다.
            x, y, z = rs.rs2_deproject_pixel_to_point(
                intr, [float(b["center"][0]), float(b["center"][1])], b["depth"])
            msg = PointStamped()
            msg.header.stamp, msg.header.frame_id = stamp, frame
            msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
            pub.publish(msg)

            if n_frame % FPS == 0:                    # 1초에 한 번만 로그
                node.get_logger().info(
                    f"pick ({x:+.3f},{y:+.3f},{z:.3f})m tilt={tilt_deg(b['normal']):.1f} "
                    f"clear={b['clear_px']:.0f}px boxes={len(stable)}")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():          # Ctrl-C 면 이미 내려가 있다. 두 번 부르면 예외.
            rclpy.shutdown()


if __name__ == "__main__":
    main()
