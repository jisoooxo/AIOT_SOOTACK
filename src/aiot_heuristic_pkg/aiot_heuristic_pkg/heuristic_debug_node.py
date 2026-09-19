"""
18개 샘플 박스를 3개씩 패킹하고 전체 랜더링 저장
"""

import json
from collections import deque
from pathlib import Path
from time import perf_counter

from .packing_data_class import BoxSpec, ContainerSpec
from .packing_render import PackingRenderer
from .packing_session import PackingSession

# 메인 노드와 동일하게 renderer가 Matplotlib/NumPy 조합을 먼저 고른다.
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node


MAX_BATCHES = 100
M_TO_MM = 1000.0


class HeuristicDebugNode(Node):
    def __init__(self):
        super().__init__("heuristic_debug_node")

        self.container = ContainerSpec.from_settings()
        self.session = PackingSession(self.container)
        self.renderer = PackingRenderer(self.container, save_png=True, save_mp4=True, save_webm=True)

        share_directory = Path(get_package_share_directory("aiot_heuristic_pkg"))
        self.input_path = share_directory / "config" / "boxes_sample_m.json"

    def load_boxes(self):
        values = json.loads(self.input_path.read_text(encoding="utf-8"))
        boxes = []

        for value in values:
            box = BoxSpec(int(value["idx"]), float(value["x"]) * M_TO_MM, float(value["y"]) * M_TO_MM, float(value["z"]) * M_TO_MM)
            boxes.append(box)

        return tuple(boxes)

    def run_debug(self):
        boxes = self.load_boxes()
        pending = deque(boxes)
        placed_indices = []
        batch_records = []
        boxes_per_container = []
        reset_count = 0
        failed = False
        started_at = perf_counter()

        self.renderer.render(self.session.state, title="Start: empty container")

        for batch_number in range(1, MAX_BATCHES + 1):
            if not pending:
                break

            window = []

            while pending and len(window) < 3:
                window.append(pending.popleft())

            batch_id = f"debug_{batch_number}"
            wall_start = perf_counter()
            self.session.start_batch(batch_id, tuple(window))
            planner_seconds = perf_counter() - wall_start

            plan = self.session.plan
            search = self.session.last_search
            execution = tuple(self.session.execution_placements)
            keep_boxes = tuple(self.session.keep_boxes)
            preview_state = self.session.state

            if execution:
                preview_state = execution[-1].state_after

            input_indices = []
            execution_indices = []
            keep_indices = []

            for box in window:
                input_indices.append(box.box_index)

            for candidate in execution:
                execution_indices.append(candidate.box_index)

            for box in keep_boxes:
                keep_indices.append(box.box_index)

            input_indices = tuple(input_indices)
            execution_indices = tuple(execution_indices)
            keep_indices = tuple(keep_indices)

            record = {
                "batch": batch_number,
                "input": input_indices,
                "execution_order": execution_indices,
                "keep": keep_indices,
                "score": plan.score,
                "visited_nodes": plan.visited_nodes,
                "planner_seconds": planner_seconds,
                "search_seconds": search.elapsed_seconds,
                "exact": search.exact,
                "termination_reason": search.termination_reason,
            }
            batch_records.append(record)

            self.get_logger().info(
                f"batch={batch_number} input={input_indices} place={execution_indices} keep={keep_indices} "
                f"time={planner_seconds:.3f}s exact={search.exact}"
            )

            first_highlight = execution[0].placed_box if execution else None
            title = f"Batch {batch_number} | input={input_indices} | place={execution_indices} | keep={keep_indices}"
            self.renderer.render(preview_state, first_highlight, self.session.state.placed_boxes, title)

            pick_position = 0
            pack_reset = False

            while True:
                task = self.session.next_task()

                if task is None:
                    break

                if task["status"] == "pack":
                    had_boxes = len(self.session.state.placed_boxes) > 0
                    boxes_per_container.append(len(self.session.state.placed_boxes))
                    self.session.reset_pack()
                    reset_count += 1
                    pack_reset = True

                    self.renderer.render(self.session.state, title=f"Container reset {reset_count}")

                    if not had_boxes:
                        failed = True

                    break

                if task["status"] == "keep":
                    self.session.confirm_done(task["task_id"])
                    continue

                candidate = execution[pick_position]
                self.session.confirm_done(task["task_id"])
                placed_indices.append(candidate.box_index)
                pick_position += 1

                next_highlight = None

                if pick_position < len(execution):
                    next_highlight = execution[pick_position].placed_box

                title = f"Placed {len(placed_indices)}/{len(boxes)}: box {candidate.box_index} | batch {batch_number}"
                self.renderer.render(preview_state, next_highlight, self.session.state.placed_boxes, title)

            for box in keep_boxes:
                pending.append(box)

            if failed:
                break

            if pack_reset:
                continue

        boxes_per_container.append(len(self.session.state.placed_boxes))
        videos = self.renderer.finish()
        elapsed_seconds = perf_counter() - started_at

        if failed:
            status = "IMPOSSIBLE_ON_EMPTY_CONTAINER"
        elif pending:
            status = "CYCLE_LIMIT"
        else:
            status = "ALL_PLACED"

        remaining_indices = []

        for box in pending:
            remaining_indices.append(box.box_index)

        summary = {
            "status": status,
            "input_json": str(self.input_path),
            "input_box_count": len(boxes),
            "placed_box_count": len(placed_indices),
            "placed_order": placed_indices,
            "remaining": tuple(remaining_indices),
            "batch_count": len(batch_records),
            "container_reset_count": reset_count,
            "boxes_per_container": boxes_per_container,
            "elapsed_seconds": elapsed_seconds,
            "videos": videos,
            "batches": batch_records,
        }

        summary_path = self.renderer.output_directory / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        self.get_logger().info(f"debug status={status} placed={len(placed_indices)}/{len(boxes)}")
        self.get_logger().info(f"debug output={self.renderer.output_directory}")

        return 0 if status == "ALL_PLACED" else 2


def main(args=None):
    rclpy.init(args=args)
    node = HeuristicDebugNode()

    try:
        return node.run_debug()
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
