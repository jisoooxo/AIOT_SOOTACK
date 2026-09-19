"""
현재 패킹 상태를 3D와 탑뷰로 랜더링하고 영상으로 저장
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aiot_heuristic_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


RENDER_ROOT = "render_output"
FRAME_RATE = 1

SAVE_PNG = False
SAVE_MP4 = False
SAVE_WEBM = True
SHOW_GUI = True

MM_TO_M = 0.001


class PackingRenderer:
    def __init__(
        self,
        container,
        output_root=RENDER_ROOT,
        save_png=SAVE_PNG,
        save_mp4=SAVE_MP4,
        save_webm=SAVE_WEBM,
        frame_rate=FRAME_RATE,
        show_gui=SHOW_GUI,
    ):
        self.container = container
        self.save_png = save_png
        self.save_mp4 = save_mp4
        self.save_webm = save_webm
        self.frame_rate = frame_rate
        self.show_gui = show_gui
        self.viewer_process = None

        # 노드를 켤 때마다 새로운 시간 폴더 생성
        started_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self.output_directory = Path(output_root) / started_at
        self.frame_directory = self.output_directory / "frames"
        self.current_path = self.output_directory / "packing_current.png"

        self.frame_directory.mkdir(parents=True, exist_ok=True)

        self.step = 0
        self.start_viewer()

    def start_viewer(self):
        # 별도 process라서 GUI가 ROS worker와 DFS를 막지 않음
        display_available = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")

        if not self.show_gui or not display_available:
            return

        command = [sys.executable, "-m", "aiot_heuristic_pkg.packing_viewer", str(self.current_path)]
        self.viewer_process = subprocess.Popen(command)

    def stop_viewer(self):
        if self.viewer_process is None or self.viewer_process.poll() is not None:
            return

        self.viewer_process.terminate()

        try:
            self.viewer_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.viewer_process.kill()
            self.viewer_process.wait()

    def box_style(self, box, highlight_box, completed_boxes):
        # 진한 파랑=완료, 주황=현재 목표, 연한 파랑=아직 놓지 않은 미래 계획
        is_target = highlight_box is not None and box == highlight_box
        is_completed = box in completed_boxes

        if is_target:
            return "#F58518", 0.9

        if is_completed:
            return "#4C78A8", 0.85

        return "#9ECAE1", 0.35

    def draw_container_3d(self, axis):
        width = self.container.width_mm * MM_TO_M
        depth = self.container.depth_mm * MM_TO_M
        height = self.container.height_mm * MM_TO_M

        corners = (
            (0, 0, 0),
            (width, 0, 0),
            (width, depth, 0),
            (0, depth, 0),
            (0, 0, height),
            (width, 0, height),
            (width, depth, height),
            (0, depth, height),
        )

        edges = (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        )

        for start_index, end_index in edges:
            start = corners[start_index]
            end = corners[end_index]

            axis.plot(
                (start[0], end[0]),
                (start[1], end[1]),
                (start[2], end[2]),
                color="black",
                linewidth=1,
            )

        axis.set_xlim(0, width)
        axis.set_ylim(0, depth)
        axis.set_zlim(0, height)

        axis.set_xlabel("Container X (m)")
        axis.set_ylabel("Container Y (m)")
        axis.set_zlabel("Container Z (m)")

        axis.set_box_aspect((width, depth, height))

    def draw_3d(
        self,
        axis,
        state,
        highlight_box,
        completed_boxes,
    ):
        self.draw_container_3d(axis)

        for box in state.placed_boxes:
            color, alpha = self.box_style(box, highlight_box, completed_boxes)

            x = box.x_mm * MM_TO_M
            y = box.y_mm * MM_TO_M
            z = box.z_mm * MM_TO_M

            width = box.width_mm * MM_TO_M
            depth = box.depth_mm * MM_TO_M
            height = box.height_mm * MM_TO_M

            axis.bar3d(
                x,
                y,
                z,
                width,
                depth,
                height,
                color=color,
                edgecolor="black",
                alpha=alpha,
                shade=True,
            )

            center_x = x + width / 2
            center_y = y + depth / 2
            top_z = z + height

            if highlight_box is not None and box == highlight_box:
                axis.scatter(
                    center_x,
                    center_y,
                    top_z,
                    color="red",
                    s=45,
                    depthshade=False,
                )

                label = (
                    f"target {box.box_index}\n"
                    f"({center_x:.3f}, {center_y:.3f}, {top_z:.3f}) m"
                )

                axis.text(
                    center_x,
                    center_y,
                    top_z + 0.002,
                    label,
                    color="red",
                    weight="bold",
                    ha="center",
                    va="bottom",
                )
            else:
                axis.text(
                    center_x,
                    center_y,
                    top_z,
                    str(box.box_index),
                    ha="center",
                    va="bottom",
                )

        axis.set_title("3D view")
        axis.view_init(elev=25, azim=-55)

    def draw_top(
        self,
        axis,
        state,
        highlight_box,
        completed_boxes,
    ):
        # 화면 세로=X, 화면 가로=Y
        # Y+가 왼쪽이므로 로봇팔은 화면 왼쪽에 있다고 보면 됨
        container_x = self.container.width_mm * MM_TO_M
        container_y = self.container.depth_mm * MM_TO_M

        axis.set_xlim(container_y, 0)
        axis.set_ylim(0, container_x)
        axis.set_aspect("equal")

        axis.xaxis.tick_top()
        axis.xaxis.set_label_position("top")

        axis.set_xlabel("Container Y (m) | Y+ robot side")
        axis.set_ylabel("Container X (m)")
        axis.set_title("Top view")

        container_rectangle = Rectangle(
            (0, 0),
            container_y,
            container_x,
            fill=False,
            edgecolor="black",
            linewidth=2,
        )
        axis.add_patch(container_rectangle)

        for box in state.placed_boxes:
            color, alpha = self.box_style(box, highlight_box, completed_boxes)

            x = box.x_mm * MM_TO_M
            y = box.y_mm * MM_TO_M

            width = box.width_mm * MM_TO_M
            depth = box.depth_mm * MM_TO_M
            top_z = (box.z_mm + box.height_mm) * MM_TO_M

            rectangle = Rectangle(
                (y, x),
                depth,
                width,
                facecolor=color,
                edgecolor="black",
                alpha=alpha,
            )
            axis.add_patch(rectangle)

            center_y = y + depth / 2
            center_x = x + width / 2

            if highlight_box is not None and box == highlight_box:
                axis.scatter(
                    center_y,
                    center_x,
                    color="red",
                    s=35,
                    zorder=5,
                )

                label = (
                    f"target {box.box_index}\n"
                    f"top Z={top_z:.3f}m"
                )

                axis.text(
                    center_y,
                    center_x,
                    label,
                    color="red",
                    weight="bold",
                    ha="center",
                    va="center",
                )
            else:
                axis.text(
                    center_y,
                    center_x,
                    str(box.box_index),
                    ha="center",
                    va="center",
                )

        axis.grid(True, alpha=0.3)

    def render(
        self,
        state,
        highlight_box=None,
        completed_boxes=(),
        title=None,
    ):
        # 호출할 때마다 영상용 frame 하나 추가
        self.step += 1

        figure = plt.figure(figsize=(13, 6))

        axis_3d = figure.add_subplot(121, projection="3d")
        axis_top = figure.add_subplot(122)

        self.draw_3d(axis_3d, state, highlight_box, completed_boxes)
        self.draw_top(axis_top, state, highlight_box, completed_boxes)

        if title is None:
            title = f"Packing step {self.step}"

        figure.suptitle(title)
        figure.tight_layout()

        frame_path = self.frame_directory / f"step_{self.step:03d}.png"

        figure.savefig(frame_path, dpi=130)
        plt.close(figure)

        # 실시간 확인용 최신 그림
        current_temp = self.output_directory / ".packing_current.tmp"
        shutil.copyfile(frame_path, current_temp)
        os.replace(current_temp, self.current_path)

        return str(self.current_path)

    def run_ffmpeg(self, command):
        ffmpeg = shutil.which("ffmpeg")

        if ffmpeg is None:
            return False

        full_command = [ffmpeg]

        for value in command:
            full_command.append(value)

        subprocess.run(full_command, check=True)

        return True

    def finish(self):
        # 노드 종료할 때 지금까지의 frame으로 영상 생성
        self.stop_viewer()

        first_frame = self.frame_directory / "step_001.png"

        if not first_frame.exists():
            return {}

        outputs = {}

        if self.save_mp4:
            mp4_path = self.output_directory / "packing_plan.mp4"

            command = [
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(self.frame_rate),
                "-start_number",
                "1",
                "-i",
                str(self.frame_directory / "step_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ]

            if self.run_ffmpeg(command):
                outputs["mp4"] = str(mp4_path)

        if self.save_webm:
            webm_path = self.output_directory / "packing_plan.webm"

            command = [
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(self.frame_rate),
                "-start_number",
                "1",
                "-i",
                str(self.frame_directory / "step_%03d.png"),
                "-c:v",
                "libvpx-vp9",
                "-crf",
                "32",
                "-b:v",
                "0",
                str(webm_path),
            ]

            if self.run_ffmpeg(command):
                outputs["webm"] = str(webm_path)

        if not self.save_png:
            for frame_path in self.frame_directory.glob("step_*.png"):
                frame_path.unlink()

            self.current_path.unlink(missing_ok=True)

        return outputs
