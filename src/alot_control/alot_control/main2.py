#!/usr/bin/env python3
"""
SOOMAC Main2 manager.
Mode 1: Shelf -> Basket
Mode 2: Keep/Basket -> Rail Setting
"""
import json
import time
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String
from alot_control.alot_config import *
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
    def __init__(self):
        super().__init__("main2_node")
        self.mode = None
        self.selected_counts = {}
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
        # Mode 선택 전부터 Main 토픽을 받을 수 있도록 항상 구독한다.
        # callback에서 현재 mode를 기준으로 Mode 1 / Mode 2를 분기한다.
        self.create_subscription(
            String, MAIN2_KEEP_SET_TOPIC, self.keep_set_callback, 10
        )
        self.running = False
        self.session_done = False
        self.start_requested = False
        self.ending_session = False
        self.end_success = None
        self.end_reason = None
        self.end_status = None
        self.command_serial = 0
        self.waiting_command_id = None
        self.after_arm_success = None
        self.delay_timer = None
        self.sample_timeout_timer = None
        self.sampling_kind = None
        self.sampling_box = None
        self.vision_samples = []
        self.sample_done_callback = None
        self.active_stack = None
        self.stack_done_callback = None
        # Mode 2 cycle state.
        # keep_set은 현재 cycle이 아니라 다음 cycle에서 사용한다.
        self.mode2_cycle_number = 0
        self.pending_keep_received = False
        self.pending_keep_enabled = False
        self.pending_keep_count = 0
        self.pending_keep_done_sent = False
        self.keep_enabled = False
        self.keep_count = 0
        self.keep_set_done_sent = False
        self.mode2_home_returning = False
        self.phase = None
        self.phase_target_count = 0
        self.phase_index = 0
        self.rail_index = 0
        self.active_view_pose = None
        self.get_logger().info(
            "Main2 ready | waiting for CLI mode selection | "
            f"setting_start={MAIN2_SETTING_START_TOPIC} | "
            f"keep_set={MAIN2_KEEP_SET_TOPIC}"
        )
        self._publish_status("WAIT_MODE_SELECT")

    def configure_mode(self, mode, selected_counts=None):
        if self.running or self.mode2_home_returning or self.ending_session:
            raise RuntimeError("cannot change mode while task is active")

        self.mode = int(mode)
        self.selected_counts = dict(selected_counts or {})
        self.session_done = False
        self.start_requested = False

        if self.mode == MODE_SETTING:
            self.mode2_cycle_number = 0
            self.pending_keep_received = False
            self.pending_keep_enabled = False
            self.pending_keep_count = 0
            self.pending_keep_done_sent = False
            self.keep_enabled = False
            self.keep_count = 0
            self.keep_set_done_sent = False
            self.phase = None
            self.phase_target_count = 0
            self.phase_index = 0
            self.rail_index = 0
            self.active_view_pose = None

        self.get_logger().info(
            f"Mode selected | MODE {self.mode}: {MODE_NAMES[self.mode]}"
        )
        if self.mode == MODE_PICKUP_TO_BASKET:
            self.get_logger().info(
                f"Mode 1 target counts | {self.selected_counts}"
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
        if self.mode is None:
            self.get_logger().warning(
                "setting_start ignored: mode is not selected yet"
            )
            return
        # Mode 1 / Mode 2를 callback 단계에서 명확하게 분기한다.
        # Mode 1은 기존 동작 유지
        if self.mode == MODE_PICKUP_TO_BASKET:
            if self.running:
                self.get_logger().warning("Start ignored: already running")
                return
            self.start_requested = True
            self._try_start_mode1()
            return
        # Mode 2
        # Cycle 1:
        #   최초 setting_start 하나만으로 즉시 시작한다.
        #
        # Cycle 2+:
        #   keep_set과 setting_start는 거의 동시에 올 수 있으므로
        #   callback 처리 순서는 보장하지 않는다.
        #   두 신호를 각각 저장해 두고 둘 다 준비되면 시작한다.
        #
        # 단, 현재 cycle 실행 중이거나 HOME 복귀 중에 들어오는
        # setting_start는 다음 cycle용으로 저장하지 않고 무시한다.
        if self.running or self.mode2_home_returning:
            self.get_logger().warning(
                "Mode 2 setting_start ignored: cycle is running or returning HOME"
            )
            return
        self.start_requested = True
        self.get_logger().info(
            "Mode 2 setting_start received | "
            f"cycle={self.mode2_cycle_number + 1} | "
            f"pending_keep={self.pending_keep_received}"
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
        for name in MAIN2_BOX_ORDER:
            if name not in BASKET_PLACE_BASE_XYZ_M:
                errors.append(
                    f"BASKET_PLACE_BASE_XYZ_M[{name}] is missing"
                )
                continue
            xyz = np.asarray(
                BASKET_PLACE_BASE_XYZ_M[name],
                dtype=float,
            )
            if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
                errors.append(
                    f"BASKET_PLACE_BASE_XYZ_M[{name}] is not calibrated"
                )
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
            BASKET_PLACE_BASE_XYZ_M[box_name],
            dtype=float,
        ).copy()
        place[2] += index * stack["box_height"]
        place = place.tolist()
        self._publish_status(
            "PICK_PLACE",
            box=box_name,
            number=index + 1,
            pick_xyz_m=pick,
            place_xyz_m=place,
        )
        is_last_pick = self._is_last_mode1_pick(box_name, index)
        is_last_cd_pick = (
            box_name == "C"
            and index + 1 >= stack["target_count"]
            and not is_last_pick
        )

        if is_last_pick:
            return_pose = "HOME"
            success_callback = self._finish_mode1_from_home
        elif is_last_cd_pick:
            return_pose = "ARM_SAFE"
            success_callback = self._finish_cd_stack_from_arm_safe
        else:
            return_pose = BOX_VIEW[box_name]
            success_callback = self._stack_item_done

        self._send_arm_command(
            {
                "command": "pick_place",
                "pick_xyz_m": pick,
                "place_xyz_m": place,
                "return_pose": return_pose,
            },
            success_callback,
        )
    def _finish_cd_stack_from_arm_safe(self):
        self.active_stack["current_index"] += 1

        box_name = self.active_stack["box"]
        self.active_stack = None
        self.stack_done_callback = None

        self.get_logger().info(
            f"{box_name}: STACK DONE | direct -> ARM_SAFE"
        )
        self._wait_scout_move()

    def _is_last_mode1_pick(self, box_name, index):
        stack = self.active_stack
        if index + 1 < stack["target_count"]:
            return False
        current_order = MAIN2_BOX_ORDER.index(box_name)
        for later_box in MAIN2_BOX_ORDER[current_order + 1:]:
            if int(self.selected_counts.get(later_box, 0)) > 0:
                return False
        return True
    def _finish_mode1_from_home(self):
        self.running = False
        self.active_stack = None
        self.stack_done_callback = None
        self.sampling_kind = None
        self.sampling_box = None
        self.sample_done_callback = None
        msg = Bool()
        msg.data = True
        self.setting_done_pub.publish(msg)
        self.get_logger().info(
            "Mode 1 COMPLETE | last Pick & Place -> HOME direct"
        )
        self._publish_status(
            "MODE1_DONE",
            home_reached=True,
        )
        self.session_done = True
    def _stack_item_done(self):
        self.active_stack["current_index"] += 1
        self._execute_next_stack_item()
    def _finish_mode1(self):
        self._return_home_before_session_end(
            success=True,
            status="MODE1_DONE",
            reason=None,
        )
    # ============================================================
    # Mode 2: Keep/Basket -> Rail Setting
    # ============================================================
    def keep_set_callback(self, msg):
        if self.mode != MODE_SETTING:
            self.get_logger().warning(
                "keep_set ignored: current mode is not Mode 2"
            )
            return
        try:
            payload = json.loads(msg.data)
            keep_enabled, keep_count = self._parse_keep_set(payload)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self.get_logger().error(f"Invalid /main/keep_set: {exc}")
            self._publish_status("KEEP_SET_INVALID", reason=str(exc))
            return
        # keep_set은 다음 cycle용으로 저장한다.
        # 현재 cycle 실행 중에 수신되어도 다음 cycle에서 사용한다.
        self.pending_keep_received = True
        self.pending_keep_enabled = keep_enabled
        self.pending_keep_count = keep_count
        self.pending_keep_done_sent = False
        self.get_logger().info(
            "Keep set stored for NEXT cycle | "
            f"keep={self.pending_keep_enabled} | "
            f"keep_count={self.pending_keep_count} | "
            f"setting_start_waiting={self.start_requested}"
        )
        if not self.pending_keep_enabled:
            self._publish_pending_keep_set_done()
        # setting_start가 먼저 도착해 대기 중이었다면
        # keep_set 저장 직후 두 조건이 모두 충족되므로 바로 시작한다.
        if self.mode2_cycle_number > 0 and self.start_requested:
            self._try_start_mode2()
            return
        self._publish_status(
            "WAIT_SETTING_START",
            cycle=self.mode2_cycle_number + 1,
            keep=self.pending_keep_enabled,
            keep_count=self.pending_keep_count,
        )
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
        if self.running or self.mode2_home_returning:
            return
        if not self.start_requested:
            return
        first_cycle = self.mode2_cycle_number == 0
        if first_cycle:
            # 첫 번째 cycle은 최초 setting_start 하나만으로 바로 시작한다.
            # 같은 시점에 들어온 keep_set은 pending에 남겨 다음 cycle에서 사용한다.
            active_keep_enabled = False
            active_keep_count = 0
            active_keep_done_sent = False
        else:
            # 두 번째 cycle부터는 keep_set + setting_start 두 신호가 모두 필요하다.
            # 둘은 어느 callback이 먼저 실행되어도 상관없다.
            if not self.pending_keep_received:
                self.get_logger().info(
                    f"Mode 2 cycle {self.mode2_cycle_number + 1} "
                    "waiting for keep_set"
                )
                self._publish_status(
                    "WAIT_KEEP_SET",
                    cycle=self.mode2_cycle_number + 1,
                )
                return
            active_keep_enabled = self.pending_keep_enabled
            active_keep_count = self.pending_keep_count
            active_keep_done_sent = self.pending_keep_done_sent
            # 이번 cycle에서 사용할 keep_set을 active 값으로 옮겼으므로
            # pending 슬롯은 비운다.
            self.pending_keep_received = False
            self.pending_keep_enabled = False
            self.pending_keep_count = 0
            self.pending_keep_done_sent = False
        errors = self._mode2_configuration_errors()
        if errors:
            for error in errors:
                self.get_logger().error(error)
            self._publish_status("CONFIGURATION_REQUIRED", errors=errors)
            self.start_requested = False
            return
        self.mode2_cycle_number += 1
        self.running = True
        self.start_requested = False
        self.rail_index = 0
        self.keep_enabled = active_keep_enabled
        self.keep_count = active_keep_count
        self.keep_set_done_sent = active_keep_done_sent
        self.get_logger().info(
            f"Main2 MODE 2 CYCLE {self.mode2_cycle_number} START | "
            f"keep={self.keep_enabled} | keep_count={self.keep_count} | "
            f"total={MAIN2_SETTING_TOTAL_COUNT}"
        )
        if self.keep_enabled:
            self._start_keep_phase()
        else:
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
        is_last_pick = self.rail_index + 1 >= MAIN2_SETTING_TOTAL_COUNT
        self._send_arm_command(
            {
                "command": "pick_place",
                "pick_xyz_m": median.tolist(),
                "place_xyz_m": place.tolist(),
                "return_pose": "HOME" if is_last_pick else self.active_view_pose,
            },
            self._rail_last_item_done_from_home if is_last_pick else self._rail_item_done,
        )
    def _rail_last_item_done_from_home(self):
        self.phase_index += 1
        self.rail_index += 1
        self.get_logger().info(
            f"Rail final item COMPLETE | phase={self.phase} | "
            f"rail_index={self.rail_index}/{MAIN2_SETTING_TOTAL_COUNT} | "
            "HOME direct"
        )
        self._finish_mode2_after_home()
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
    def _publish_pending_keep_set_done(self):
        if self.pending_keep_done_sent:
            return
        msg = Bool()
        msg.data = True
        self.keep_set_done_pub.publish(msg)
        self.pending_keep_done_sent = True
        self.get_logger().info(
            f"Publish {MAIN2_KEEP_SET_DONE_TOPIC} = True | "
            "next-cycle keep=False"
        )
    def _publish_keep_set_done(self):
        if self.keep_set_done_sent:
            return
        msg = Bool()
        msg.data = True
        self.keep_set_done_pub.publish(msg)
        self.keep_set_done_sent = True
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
        self.mode2_home_returning = True
        self.get_logger().info(
            f"Mode 2 cycle {self.mode2_cycle_number} end -> HOME 이동"
        )
        self._send_move("HOME", self._finish_mode2_after_home)
    def _finish_mode2_after_home(self):
        # 마지막 Pick & Place가 HOME으로 직접 끝나는 경로에서도
        # 다음 cycle의 setting_start를 정상 수신할 수 있도록
        # Mode 2 실행 상태를 반드시 해제한다.
        self.running = False
        self.mode2_home_returning = False
        msg = Bool()
        msg.data = True
        self.setting_done_pub.publish(msg)
        self.get_logger().info(
            f"Mode 2 CYCLE {self.mode2_cycle_number} COMPLETE | "
            "wait next setting_start"
        )
        self.phase = None
        self.phase_target_count = 0
        self.phase_index = 0
        self.rail_index = 0
        self.active_view_pose = None
        self.keep_enabled = False
        self.keep_count = 0
        self.keep_set_done_sent = False
        self.start_requested = False
        if self.pending_keep_received:
            self._publish_status(
                "WAIT_SETTING_START",
                cycle=self.mode2_cycle_number + 1,
                keep=self.pending_keep_enabled,
                keep_count=self.pending_keep_count,
            )
        else:
            self._publish_status(
                "WAIT_KEEP_SET",
                cycle=self.mode2_cycle_number + 1,
            )
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
        self.get_logger().info(
            f"Vision waiting until detected | kind={kind} | box={box_name}"
        )
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
            self.get_logger().warning(
                f"{label} vision unstable -> retry until detected | spread_mm="
                f"{np.round(spread * 1000.0, 1).tolist()}"
            )
            self._start_vision_sampling(kind, box_name, callback)
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
        self.get_logger().info(
            f"Arm command | id={payload['id']} | command={payload['command']}"
        )
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
            if self.ending_session:
                self.get_logger().error(
                    "HOME return failed before MODE SELECT | "
                    f"{result.get('message', '')}"
                )
                self._finalize_session_end(home_reached=False)
                return
            if self.mode2_home_returning:
                self.mode2_home_returning = False
                self.get_logger().error(
                    "Mode 2 HOME return failed | "
                    f"{result.get('message', '')}"
                )
                self.session_done = True
                return
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
        # 인식 timeout으로 Task를 종료하지 않는다.
        # 인식될 때까지 현재 위치에서 계속 기다린다.
        self._cancel_sample_timeout()
    def _cancel_sample_timeout(self):
        if self.sample_timeout_timer is not None:
            self.destroy_timer(self.sample_timeout_timer)
            self.sample_timeout_timer = None
    def _return_home_before_session_end(self, success, status, reason):
        if self.ending_session:
            return
        self.running = False
        self.sampling_kind = None
        self.sampling_box = None
        self.sample_done_callback = None
        self._cancel_delay()
        self._cancel_sample_timeout()
        self.ending_session = True
        self.end_success = bool(success)
        self.end_reason = reason
        self.end_status = status
        self.waiting_command_id = None
        self.after_arm_success = None
        self.get_logger().info(
            "Mode end -> HOME 이동 후 MODE SELECT로 복귀"
        )
        self._send_move("HOME", self._home_return_done)
    def _home_return_done(self):
        self._finalize_session_end(home_reached=True)
    def _finalize_session_end(self, home_reached):
        success = bool(self.end_success)
        reason = self.end_reason
        status = self.end_status
        if success:
            msg = Bool()
            msg.data = True
            self.setting_done_pub.publish(msg)
            if self.mode == MODE_PICKUP_TO_BASKET:
                self.get_logger().info("Mode 1 COMPLETE")
                self._publish_status(
                    status,
                    home_reached=bool(home_reached),
                )
            else:
                self.get_logger().info(
                    f"Mode 2 COMPLETE | rail_total={self.rail_index}"
                )
                self._publish_status(
                    status,
                    rail_total=self.rail_index,
                    home_reached=bool(home_reached),
                )
        else:
            self.get_logger().error(f"Main2 ABORT: {reason}")
            self._publish_status(
                "ERROR",
                mode=self.mode,
                reason=reason,
                home_reached=bool(home_reached),
            )
        self.ending_session = False
        self.end_success = None
        self.end_reason = None
        self.end_status = None
        self.session_done = True
    def _abort(self, reason):
        self._return_home_before_session_end(
            success=False,
            status="ERROR",
            reason=reason,
        )
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
    node = Main2Node()
    try:
        while rclpy.ok():
            # Subscriber를 먼저 생성한 상태에서 기존 CLI 방식으로 mode를 선택한다.
            # mode 선택 중 도착한 Main topic도 subscriber queue에 받을 수 있다.
            mode, selected_counts = _prompt_mode()
            if mode is None:
                break
            node.configure_mode(
                mode=mode,
                selected_counts=selected_counts,
            )
            while rclpy.ok() and not node.session_done:
                rclpy.spin_once(node, timeout_sec=0.10)
            # Mode 1 완료/오류 종료 후에도 Node는 파괴하지 않고
            # 같은 subscriber를 유지한 채 다시 mode 선택으로 돌아간다.
            node.mode = None
            node.selected_counts = {}
            node.session_done = False
            node._publish_status("WAIT_MODE_SELECT")
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().warning(
                "Ctrl+C detected -> Main2 shutdown"
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
if __name__ == "__main__":
    main()
