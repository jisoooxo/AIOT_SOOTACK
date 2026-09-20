#!/usr/bin/env python3
"""
main2.py

Mode 1: Pick-up zone(Shelf) -> Basket
- 기존 A/C/F/H 종류/수량 기반 Pick & Place 흐름 유지
- 실행 전에 터미널에서 A/C/F/H 수량을 입력한다.
- /main/setting_start = True를 받으면 시작한다.

Mode 2: Setting interaction
- 기존 Basket -> Rail + Keep -> Rail을 하나로 합친 흐름
- /main/keep_set : std_msgs/String(JSON)
    {"keep": true, "keep_count": N}
    {"keep": false}
- /main/setting_start : std_msgs/Bool(True)
- Keep가 있으면 Keep -> Rail을 먼저 수행한다.
- Keep 작업 완료 후 /main2/keep_set_done = True를 단발 발행한다.
- 남은 수량만큼 Basket -> Rail을 수행해 Rail 총 3개를 채운다.
- 전체 완료 후 /main2/setting_done = True를 단발 발행한다.

주의:
- /ui/select_box는 사용하지 않는다.
- Mode 2에서는 setting_start와 keep_set 중 어느 것이 먼저 와도 둘 다 준비되면 시작한다.
"""

import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String

from alot_config import (
    ARM_NAMED_POSE_TICKS,
    BASKET_DROP_XYZ_M,
    MAIN2_ARM_COMMAND_TOPIC,
    MAIN2_ARM_RESULT_TOPIC,
    MAIN2_BOX_ORDER,
    MAIN2_CALIBRATION_COMPLETE,
    MAIN2_KEEP_SET_DONE_TOPIC,
    MAIN2_KEEP_SET_TOPIC,
    MAIN2_MAX_COUNT_PER_TYPE,
    MAIN2_SETTING_DONE_TOPIC,
    MAIN2_SETTING_START_TOPIC,
    MAIN2_SETTING_TOTAL_COUNT,
    MAIN2_STATUS_TOPIC,
    OCR_SHOW_WAIT_SEC,
    RAIL_FIRST_PLACE_XYZ_M,
    RAIL_MAX_BOX_COUNT,
    RAIL_PLACE_OFFSET_AXIS,
    RAIL_PLACE_OFFSET_M,
    SCOUT_MOVE_WAIT_SEC,
    SHELF_Z_M,
    STACK_SAMPLE_COUNT,
    STACK_SAMPLE_MAX_SPREAD_M,
    STACK_SAMPLE_TIMEOUT_SEC,
    VISION_PICK_BASE_TOPIC,
)

MODE_PICKUP_TO_BASKET = 1
MODE_SETTING = 2

MODE_NAMES = {
    MODE_PICKUP_TO_BASKET: "Pick-up zone -> Basket",
    MODE_SETTING: "Keep/Basket -> Rail Setting",
}

BOX_VIEW = {
    "A": "DEPTH_VIEW_AB",
    "C": "DEPTH_VIEW_CD",
    "F": "DEPTH_VIEW_EF",
    "H": "DEPTH_VIEW_GH",
}


class Main2Node(Node):
    def __init__(self, mode, selected_counts=None):
        super().__init__("main2_node")

        self.mode = int(mode)
        self.selected_counts = dict(selected_counts or {})

        self.arm_command_pub = self.create_publisher(
            String, MAIN2_ARM_COMMAND_TOPIC, 10
        )
        self.setting_done_pub = self.create_publisher(
            Bool, MAIN2_SETTING_DONE_TOPIC, 10
        )
        self.keep_set_done_pub = self.create_publisher(
            Bool, MAIN2_KEEP_SET_DONE_TOPIC, 10
        )
        self.status_pub = self.create_publisher(
            String, MAIN2_STATUS_TOPIC, 10
        )

        self.create_subscription(
            String, MAIN2_ARM_RESULT_TOPIC, self.arm_result_callback, 10
        )
        self.create_subscription(
            PointStamped,
            VISION_PICK_BASE_TOPIC,
            self.vision_base_callback,
            10,
        )
        self.create_subscription(
            Bool, MAIN2_SETTING_START_TOPIC, self.start_callback, 10
        )

        if self.mode == MODE_SETTING:
            self.create_subscription(
                String, MAIN2_KEEP_SET_TOPIC, self.keep_set_callback, 10
            )

        self.running = False
        self.session_done = False
        self.start_requested = False

        self.command_serial = 0
        self.waiting_command_id = None
        self.after_arm_success = None

        self.delay_timer = None
        self.sample_timeout_timer = None

        self.sampling_kind = None
        self.sampling_box = None
        self.vision_samples = []
        self.sample_done_callback = None

        # Mode 1 state
        self.active_stack = None
        self.stack_done_callback = None

        # Mode 2 state
        self.keep_received = False
        self.keep_enabled = False
        self.keep_count = 0
        self.phase = None
        self.phase_target_count = 0
        self.phase_index = 0
        self.rail_index = 0
        self.active_view_pose = None

        self.get_logger().info(
            f"Main2 ready | MODE {self.mode}: {MODE_NAMES[self.mode]}"
        )

        if self.mode == MODE_PICKUP_TO_BASKET:
            self.get_logger().info(
                f"Mode 1 target counts | {self.selected_counts}"
            )
        else:
            self.get_logger().info(
                "Mode 2 waits for /main/keep_set and /main/setting_start"
            )

        self._publish_status(
            "IDLE",
            mode=self.mode,
            mode_name=MODE_NAMES[self.mode],
        )

    # ============================================================
    # Common start trigger
    # ============================================================
    def start_callback(self, msg):
        if not msg.data:
            return

        if self.running:
            self.get_logger().warning("Start ignored: already running")
            return

        self.start_requested = True
        self.get_logger().info("Setting start received | data=True")

        if self.mode == MODE_PICKUP_TO_BASKET:
            self._try_start_mode1()
        else:
            if not self.keep_received:
                self.get_logger().info(
                    "Mode 2 start is waiting for /main/keep_set"
                )
            self._try_start_mode2()

    # ============================================================
    # Mode 1: Pick-up zone -> Basket
    # ============================================================
    def _try_start_mode1(self):
        if self.running or not self.start_requested:
            return

        errors = self._mode1_configuration_errors()
        if errors:
            for error in errors:
                self.get_logger().error(error)
            self._publish_status("CONFIGURATION_REQUIRED", errors=errors)
            self.start_requested = False
            return

        self.running = True
        self.start_requested = False

        self._publish_status(
            "MODE1_START",
            counts=self.selected_counts,
        )
        self.get_logger().info(
            f"Main2 MODE 1 START | {MODE_NAMES[MODE_PICKUP_TO_BASKET]}"
        )
        self._send_move("OCR_VIEW", self._wait_first_ocr)

    def _mode1_configuration_errors(self):
        errors = []

        if not MAIN2_CALIBRATION_COMPLETE:
            errors.append("MAIN2_CALIBRATION_COMPLETE is False")

        for name in MAIN2_BOX_ORDER:
            count = int(self.selected_counts.get(name, 0))
            if count <= 0:
                continue

            if not np.isfinite(float(SHELF_Z_M[name])):
                errors.append(f"SHELF_Z_M[{name}] is not calibrated")

            drop = np.asarray(BASKET_DROP_XYZ_M[name], dtype=float)
            if drop.shape != (3,) or not np.all(np.isfinite(drop)):
                errors.append(f"BASKET_DROP_XYZ_M[{name}] is not calibrated")

        required_poses = {
            "OCR_VIEW",
            "DEPTH_VIEW_AB",
            "DEPTH_VIEW_CD",
            "DEPTH_VIEW_EF",
            "DEPTH_VIEW_GH",
            "ARM_SAFE",
        }
        missing = required_poses - set(ARM_NAMED_POSE_TICKS)
        if missing:
            errors.append(f"missing named poses: {sorted(missing)}")

        return errors

    def _wait_first_ocr(self):
        self._publish_status("OCR_ABCD")
        self._start_delay(OCR_SHOW_WAIT_SEC, self._move_ab_view)

    def _move_ab_view(self):
        self._send_move(
            "DEPTH_VIEW_AB",
            lambda: self._process_stack("A", self._move_cd_view),
        )

    def _move_cd_view(self):
        self._send_move(
            "DEPTH_VIEW_CD",
            lambda: self._process_stack("C", self._prepare_scout_move),
        )

    def _prepare_scout_move(self):
        self._send_move("ARM_SAFE", self._wait_scout_move)

    def _wait_scout_move(self):
        self._publish_status("SCOUT_MOVING", wait_sec=SCOUT_MOVE_WAIT_SEC)
        self._start_delay(SCOUT_MOVE_WAIT_SEC, self._move_second_ocr)

    def _move_second_ocr(self):
        self._send_move("OCR_VIEW", self._wait_second_ocr)

    def _wait_second_ocr(self):
        self._publish_status("OCR_EFGH")
        self._start_delay(OCR_SHOW_WAIT_SEC, self._move_ef_view)

    def _move_ef_view(self):
        self._send_move(
            "DEPTH_VIEW_EF",
            lambda: self._process_stack("F", self._move_gh_view),
        )

    def _move_gh_view(self):
        self._send_move(
            "DEPTH_VIEW_GH",
            lambda: self._process_stack("H", self._finish_mode1),
        )

    def _process_stack(self, box_name, done_callback):
        count = int(self.selected_counts.get(box_name, 0))

        self.get_logger().info(
            f"{box_name}: stack process start | target_count={count} | "
            f"view={BOX_VIEW[box_name]}"
        )

        if count <= 0:
            self.get_logger().info(f"{box_name}: SKIP")
            done_callback()
            return

        self.stack_done_callback = done_callback
        self._start_vision_sampling(
            kind="STACK",
            box_name=box_name,
            done_callback=self._finish_stack_sampling,
        )

    def _finish_stack_sampling(self, median):
        box_name = self.sampling_box
        count = int(self.selected_counts[box_name])
        shelf_z = float(SHELF_Z_M[box_name])
        box_height = (float(median[2]) - shelf_z) / count

        if box_height <= 0.0:
            self._abort(
                f"{box_name} invalid box height: "
                f"{box_height * 1000.0:.1f} mm"
            )
            return

        self.active_stack = {
            "box": box_name,
            "target_count": count,
            "current_index": 0,
            "x": float(median[0]),
            "y": float(median[1]),
            "top_z": float(median[2]),
            "box_height": box_height,
        }

        self.get_logger().info(
            f"{box_name} stack locked | xyz_mm="
            f"{np.round(median * 1000.0, 1).tolist()} | "
            f"height={box_height * 1000.0:.1f} mm"
        )
        self._execute_next_stack_item()

    def _execute_next_stack_item(self):
        stack = self.active_stack
        index = stack["current_index"]

        if index >= stack["target_count"]:
            box_name = stack["box"]
            callback = self.stack_done_callback

            self.active_stack = None
            self.stack_done_callback = None

            self.get_logger().info(f"{box_name}: STACK DONE")
            callback()
            return

        pick = [
            stack["x"],
            stack["y"],
            stack["top_z"] - index * stack["box_height"],
        ]

        box_name = stack["box"]
        place = np.asarray(
            BASKET_DROP_XYZ_M[box_name],
            dtype=float,
        ).tolist()

        self._publish_status(
            "PICK_PLACE",
            box=box_name,
            number=index + 1,
            pick_xyz_m=pick,
            place_xyz_m=place,
        )

        self._send_arm_command(
            {
                "command": "pick_place",
                "pick_xyz_m": pick,
                "place_xyz_m": place,
                "return_pose": BOX_VIEW[box_name],
            },
            self._stack_item_done,
        )

    def _stack_item_done(self):
        self.active_stack["current_index"] += 1
        self._execute_next_stack_item()

    def _finish_mode1(self):
        self.running = False

        msg = Bool()
        msg.data = True
        self.setting_done_pub.publish(msg)

        self.get_logger().info(
            f"Publish {MAIN2_SETTING_DONE_TOPIC} = True"
        )
        self.get_logger().info("Mode 1 COMPLETE")

        self._publish_status("MODE1_DONE")
        self.session_done = True

    # ============================================================
    # Mode 2: Keep/Basket -> Rail Setting
    # ============================================================
    def keep_set_callback(self, msg):
        if self.running:
            self.get_logger().warning(
                "keep_set ignored: Setting is already running"
            )
            return

        try:
            payload = json.loads(msg.data)
            keep_enabled, keep_count = self._parse_keep_set(payload)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"Invalid /main/keep_set: {exc}")
            self._publish_status("KEEP_SET_INVALID", reason=str(exc))
            return

        self.keep_enabled = keep_enabled
        self.keep_count = keep_count
        self.keep_received = True

        self.get_logger().info(
            f"Keep set received | keep={self.keep_enabled} | "
            f"keep_count={self.keep_count}"
        )
        self._publish_status(
            "KEEP_SET_READY",
            keep=self.keep_enabled,
            keep_count=self.keep_count,
        )

        self._try_start_mode2()

    @staticmethod
    def _parse_keep_set(payload):
        if not isinstance(payload, dict):
            raise ValueError("keep_set must be a JSON object")

        if "keep" not in payload:
            raise ValueError("'keep' field is required")

        keep_value = payload["keep"]
        if not isinstance(keep_value, bool):
            raise ValueError("'keep' must be true or false")

        if not keep_value:
            return False, 0

        if "keep_count" not in payload:
            raise ValueError("'keep_count' is required when keep=true")

        keep_count = int(payload["keep_count"])

        if not 1 <= keep_count <= MAIN2_SETTING_TOTAL_COUNT:
            raise ValueError(
                f"keep_count must be 1~{MAIN2_SETTING_TOTAL_COUNT}"
            )

        return True, keep_count

    def _try_start_mode2(self):
        if self.running or not self.start_requested or not self.keep_received:
            return

        errors = self._mode2_configuration_errors()
        if errors:
            for error in errors:
                self.get_logger().error(error)
            self._publish_status("CONFIGURATION_REQUIRED", errors=errors)
            self.start_requested = False
            return

        self.running = True
        self.start_requested = False
        self.rail_index = 0

        self.get_logger().info(
            "Main2 MODE 2 START | "
            f"keep={self.keep_enabled} | keep_count={self.keep_count} | "
            f"total={MAIN2_SETTING_TOTAL_COUNT}"
        )
        self._publish_status(
            "MODE2_START",
            keep=self.keep_enabled,
            keep_count=self.keep_count,
            total_count=MAIN2_SETTING_TOTAL_COUNT,
        )

        if self.keep_enabled:
            self._start_keep_phase()
        else:
            self._publish_keep_set_done()
            self._start_basket_phase()

    def _mode2_configuration_errors(self):
        errors = []

        for pose_name in ("KEEP_VIEW", "BASKET_VIEW"):
            if pose_name not in ARM_NAMED_POSE_TICKS:
                errors.append(f"missing named pose: {pose_name}")

        first = np.asarray(RAIL_FIRST_PLACE_XYZ_M, dtype=float)
        if first.shape != (3,) or not np.all(np.isfinite(first)):
            errors.append("RAIL_FIRST_PLACE_XYZ_M is not calibrated")

        if not np.isfinite(float(RAIL_PLACE_OFFSET_M)):
            errors.append("RAIL_PLACE_OFFSET_M is not calibrated")

        axis = str(RAIL_PLACE_OFFSET_AXIS).upper()
        if axis not in ("X", "Y"):
            errors.append("RAIL_PLACE_OFFSET_AXIS must be 'X' or 'Y'")

        if MAIN2_SETTING_TOTAL_COUNT > RAIL_MAX_BOX_COUNT:
            errors.append(
                "MAIN2_SETTING_TOTAL_COUNT exceeds RAIL_MAX_BOX_COUNT"
            )

        return errors

    def _start_keep_phase(self):
        self.phase = "KEEP"
        self.phase_target_count = self.keep_count
        self.phase_index = 0
        self.active_view_pose = "KEEP_VIEW"

        self.get_logger().info(
            f"Keep phase START | count={self.phase_target_count}"
        )
        self._publish_status(
            "KEEP_PHASE_START",
            count=self.phase_target_count,
        )
        self._send_move(self.active_view_pose, self._sample_current_source)

    def _finish_keep_phase(self):
        self.get_logger().info(
            f"Keep phase COMPLETE | placed={self.phase_target_count}"
        )
        self._publish_keep_set_done()

        if self.rail_index >= MAIN2_SETTING_TOTAL_COUNT:
            self._finish_mode2()
            return

        self._start_basket_phase()

    def _start_basket_phase(self):
        remaining = MAIN2_SETTING_TOTAL_COUNT - self.rail_index

        if remaining <= 0:
            self._finish_mode2()
            return

        self.phase = "BASKET"
        self.phase_target_count = remaining
        self.phase_index = 0
        self.active_view_pose = "BASKET_VIEW"

        self.get_logger().info(
            f"Basket phase START | repeat_count={remaining}"
        )
        self._publish_status(
            "BASKET_PHASE_START",
            count=remaining,
            rail_start_index=self.rail_index,
        )
        self._send_move(self.active_view_pose, self._sample_current_source)

    def _sample_current_source(self):
        if self.phase_index >= self.phase_target_count:
            if self.phase == "KEEP":
                self._finish_keep_phase()
            else:
                self._finish_mode2()
            return

        self._start_vision_sampling(
            kind=f"{self.phase}_SOURCE",
            box_name=None,
            done_callback=self._execute_rail_pick_place,
        )

    def _execute_rail_pick_place(self, median):
        place = self._rail_place_xyz(self.rail_index)

        self._publish_status(
            "RAIL_PICK_PLACE",
            phase=self.phase,
            phase_number=self.phase_index + 1,
            rail_number=self.rail_index + 1,
            pick_xyz_m=median.tolist(),
            place_xyz_m=place.tolist(),
        )

        self._send_arm_command(
            {
                "command": "pick_place",
                "pick_xyz_m": median.tolist(),
                "place_xyz_m": place.tolist(),
                "return_pose": self.active_view_pose,
            },
            self._rail_item_done,
        )

    def _rail_item_done(self):
        self.phase_index += 1
        self.rail_index += 1

        self.get_logger().info(
            f"Rail item COMPLETE | phase={self.phase} | "
            f"phase_index={self.phase_index}/{self.phase_target_count} | "
            f"rail_index={self.rail_index}/{MAIN2_SETTING_TOTAL_COUNT}"
        )
        self._sample_current_source()

    @staticmethod
    def _rail_place_xyz(index):
        xyz = np.asarray(RAIL_FIRST_PLACE_XYZ_M, dtype=float).copy()
        offset = float(index) * float(RAIL_PLACE_OFFSET_M)
        axis = str(RAIL_PLACE_OFFSET_AXIS).upper()

        if axis == "X":
            xyz[0] += offset
        elif axis == "Y":
            xyz[1] += offset
        else:
            raise RuntimeError("RAIL_PLACE_OFFSET_AXIS must be 'X' or 'Y'")

        return xyz

    def _publish_keep_set_done(self):
        msg = Bool()
        msg.data = True
        self.keep_set_done_pub.publish(msg)

        self.get_logger().info(
            f"Publish {MAIN2_KEEP_SET_DONE_TOPIC} = True"
        )
        self._publish_status(
            "KEEP_SET_DONE",
            keep=self.keep_enabled,
            keep_count=self.keep_count,
        )

    def _finish_mode2(self):
        self.running = False

        msg = Bool()
        msg.data = True
        self.setting_done_pub.publish(msg)

        self.get_logger().info(
            f"Publish {MAIN2_SETTING_DONE_TOPIC} = True"
        )
        self.get_logger().info(
            f"Mode 2 COMPLETE | rail_total={self.rail_index}"
        )

        self._publish_status(
            "MODE2_DONE",
            rail_total=self.rail_index,
        )
        self.session_done = True

    # ============================================================
    # Shared Vision sampling
    # ============================================================
    def _start_vision_sampling(self, kind, box_name, done_callback):
        self.sampling_kind = kind
        self.sampling_box = box_name
        self.vision_samples = []
        self.sample_done_callback = done_callback

        self._publish_status(
            "SAMPLING_VISION",
            kind=kind,
            box=box_name,
            required=STACK_SAMPLE_COUNT,
        )
        self._start_sample_timeout()

    def vision_base_callback(self, msg):
        if not self.running or self.sampling_kind is None:
            return

        xyz = np.array(
            [msg.point.x, msg.point.y, msg.point.z],
            dtype=float,
        )
        if not np.all(np.isfinite(xyz)):
            return

        self.vision_samples.append(xyz)

        if len(self.vision_samples) >= STACK_SAMPLE_COUNT:
            self._finish_vision_sampling()

    def _finish_vision_sampling(self):
        kind = self.sampling_kind
        box_name = self.sampling_box
        callback = self.sample_done_callback

        self.sampling_kind = None
        self.sample_done_callback = None
        self._cancel_sample_timeout()

        samples = np.asarray(self.vision_samples, dtype=float)
        median = np.median(samples, axis=0)
        spread = np.ptp(samples, axis=0)

        if np.any(spread > STACK_SAMPLE_MAX_SPREAD_M):
            label = box_name if box_name is not None else kind
            self.sampling_box = None
            self._abort(
                f"{label} vision unstable: spread_mm="
                f"{np.round(spread * 1000.0, 1).tolist()}"
            )
            return

        self.get_logger().info(
            f"Vision locked | kind={kind} | "
            f"xyz_mm={np.round(median * 1000.0, 1).tolist()}"
        )

        callback(median)

        if kind != "STACK":
            self.sampling_box = None

    # ============================================================
    # Arm request / result
    # ============================================================
    def _send_move(self, pose_name, success_callback):
        ticks = np.asarray(
            ARM_NAMED_POSE_TICKS[pose_name],
            dtype=int,
        )
        self.get_logger().info(
            f"Move request | pose={pose_name} | ticks={ticks.tolist()}"
        )

        self._send_arm_command(
            {
                "command": "move_named_pose",
                "pose": pose_name,
            },
            success_callback,
        )

    def _send_arm_command(self, payload, success_callback):
        if self.waiting_command_id is not None:
            self._abort("internal error: overlapping arm command")
            return

        self.command_serial += 1
        payload = dict(payload)
        payload["id"] = self.command_serial

        self.waiting_command_id = self.command_serial
        self.after_arm_success = success_callback

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.arm_command_pub.publish(msg)

        self.get_logger().info(f"Arm command -> {payload}")

    def arm_result_callback(self, msg):
        try:
            result = json.loads(msg.data)
            command_id = int(result["id"])
            success = bool(result["success"])
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            self.get_logger().warning("Malformed arm result ignored")
            return

        if command_id != self.waiting_command_id:
            return

        callback = self.after_arm_success
        self.waiting_command_id = None
        self.after_arm_success = None

        if not success:
            self._abort(
                f"arm command failed: {result.get('message', '')}"
            )
            return

        callback()

    # ============================================================
    # Timer / status / error
    # ============================================================
    def _start_delay(self, seconds, callback):
        self._cancel_delay()

        def on_timer():
            self._cancel_delay()
            callback()

        self.delay_timer = self.create_timer(float(seconds), on_timer)

    def _cancel_delay(self):
        if self.delay_timer is not None:
            self.destroy_timer(self.delay_timer)
            self.delay_timer = None

    def _start_sample_timeout(self):
        self._cancel_sample_timeout()

        def on_timeout():
            received = len(self.vision_samples)
            label = self.sampling_box or self.sampling_kind or "VISION"

            self.sampling_kind = None
            self.sampling_box = None
            self.sample_done_callback = None
            self._cancel_sample_timeout()

            self._abort(
                f"{label} vision timeout: "
                f"{received}/{STACK_SAMPLE_COUNT}"
            )

        self.sample_timeout_timer = self.create_timer(
            STACK_SAMPLE_TIMEOUT_SEC,
            on_timeout,
        )

    def _cancel_sample_timeout(self):
        if self.sample_timeout_timer is not None:
            self.destroy_timer(self.sample_timeout_timer)
            self.sample_timeout_timer = None

    def _abort(self, reason):
        self.running = False
        self.sampling_kind = None
        self.sampling_box = None
        self.sample_done_callback = None
        self.waiting_command_id = None
        self.after_arm_success = None

        self._cancel_delay()
        self._cancel_sample_timeout()

        self.get_logger().error(f"Main2 ABORT: {reason}")
        self._publish_status(
            "ERROR",
            mode=self.mode,
            reason=reason,
        )

        self.session_done = True

    def _publish_status(self, state, **extra):
        payload = {
            "state": state,
            "time": time.time(),
            **extra,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def destroy_node(self):
        self._cancel_delay()
        self._cancel_sample_timeout()
        super().destroy_node()


def _prompt_mode():
    while True:
        print()
        print("=" * 64)
        print("SOOMAC MAIN2 MODE SELECT")
        print("=" * 64)
        print("1 : Pick-up zone -> Basket")
        print("2 : Keep/Basket -> Rail Setting")
        print("q : Exit")

        value = input("MODE > ").strip().lower()

        if value == "q":
            return None, None

        if value == "1":
            while True:
                print()
                print("Mode 1에서 옮길 박스 수량을 입력하세요. [0~3]")

                counts = {}
                valid = True

                for name in MAIN2_BOX_ORDER:
                    text = input(f"{name} 개수 > ").strip()

                    try:
                        count = int(text)
                    except ValueError:
                        print(f"{name}: 숫자로 입력하세요.")
                        valid = False
                        break

                    if not 0 <= count <= MAIN2_MAX_COUNT_PER_TYPE:
                        print(
                            f"{name}: 0~{MAIN2_MAX_COUNT_PER_TYPE} "
                            "범위로 입력하세요."
                        )
                        valid = False
                        break

                    counts[name] = count

                if not valid:
                    continue

                if sum(counts.values()) == 0:
                    print("최소 한 종류는 1개 이상 선택해야 합니다.")
                    continue

                print(f"Mode 1 선택 수량: {counts}")
                return MODE_PICKUP_TO_BASKET, counts

        if value == "2":
            return MODE_SETTING, None

        print("1, 2 또는 q를 입력하세요.")


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        while rclpy.ok():
            mode, selected_counts = _prompt_mode()

            if mode is None:
                break

            node = Main2Node(
                mode=mode,
                selected_counts=selected_counts,
            )

            try:
                while rclpy.ok() and not node.session_done:
                    rclpy.spin_once(node, timeout_sec=0.10)
            finally:
                node.destroy_node()
                node = None

    except KeyboardInterrupt:
        if node is not None and rclpy.ok():
            node.get_logger().warning(
                "Ctrl+C detected -> Main2 shutdown"
            )
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
