# 지수, 준미랑 통신할 메인 노드 ㄹㅊㄱ
# 토픽, 스레드 분리, 세션, 렌더링 연결
"""
패킹 토픽 입출력, worker, session, 랜더링 연결
"""

import json
import queue
import threading

from .packing_data_class import BoxSpec, ContainerSpec
from .packing_render import PackingRenderer
from .packing_session import PackingSession

# packing_render가 일관된 Matplotlib/NumPy 조합을 먼저 선택한 뒤 ROS를
# import해야 ~/.local의 NumPy 2와 Ubuntu Matplotlib가 섞이지 않는다.
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


WORK_QUEUE_SIZE = 16
M_TO_MM = 1000.0
MM_TO_M = 0.001
PLACE_Z_OFFSET_MM = 0.0  # 로봇에 보낼 박스 윗면 Z 보정값


class HeuristicMainNode(Node):
    def __init__(self):
        super().__init__("heuristic_main_node")

        self.session = PackingSession(ContainerSpec.from_settings())
        self.renderer = PackingRenderer(self.session.container)

        # 발행
        self.plan_pick_pub = self.create_publisher(String, "/heuristic/plan_pick", 10)
        self.plan_place_pub = self.create_publisher(String, "/heuristic/plan_place", 10)
        self.pack_reset_done_pub = self.create_publisher(Bool, "/heuristic/pack_reset_done", 10)


        # 구독
        # self.create_subscription(String, "/main/box_sizes", self.box_sizes_callback, 10)
        self.create_subscription(String, "/vision/box_sizes", self.box_sizes_callback, 10)
        self.create_subscription(Bool, "/main/next_box", self.next_box_callback, 10)
        self.create_subscription(Bool, "/control/pick_done", self.pick_done_callback, 10)
        self.create_subscription(Bool, "/main/pack_reset", self.pack_reset_callback, 10)

        self.work_queue = queue.Queue(maxsize=WORK_QUEUE_SIZE)
        self.batch_count = 0
        self.current_task = None
        self.place_sent = False
        self.next_requested = False
        self.shutdown_started = False
        self.lookahead_count = 3
        self.task_sent_count = 0
        self.plan_place_sent_count = 0
        self.batch_active = False

        # state를 수정하는 worker는 하나만 사용
        self.worker = threading.Thread(target=self.worker_loop, daemon=False)
        self.worker.start()

        self.get_logger().info(f"heuristic ready | render={self.renderer.output_directory}")

    def put_event(self, name, value):
        # callback에서는 계산하지 않고 queue에만 넣음
        try:
            self.work_queue.put_nowait((name, value))
        except queue.Full:
            self.get_logger().warning(f"work queue full: {name}")

    def box_sizes_callback(self, msg):
        try:
            values = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().error(f"/vision/box_sizes JSON 오류: {error}")
            return

        self.put_event("box_sizes", values)

    def next_box_callback(self, msg):
        if msg.data:
            self.put_event("next_box", True)

    def pick_done_callback(self, msg):
        if msg.data:
            self.put_event("pick_done", True)

    def pack_reset_callback(self, msg):
        if msg.data:
            self.put_event("pack_reset", True)

    def parse_boxes(self, values):
        # 입력은 [{"idx":1,"x":0.1,"y":0.05,"z":0.03}, ...], 단위는 m
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError("lookahead 박스는 JSON 배열 3개 필요")

        boxes = []

        for value in values:
            box = BoxSpec(int(value["idx"]), float(value["x"]) * M_TO_MM, float(value["y"]) * M_TO_MM, float(value["z"]) * M_TO_MM)
            boxes.append(box)

        self.batch_count += 1
        return f"batch_{self.batch_count}", tuple(boxes) # 배치 카운트 1개, 박스 정보 3개 다

    def handle_box_sizes(self, values):
        if self.batch_active and self.current_task is not None:
            if self.current_task["sequence"] != self.lookahead_count:
                self.get_logger().warning(
                    "이전 batch가 끝나기 전에 새 box_sizes가 들어옴"
                )
                return

            if (
                self.current_task["status"] == "pick"
                and not self.place_sent
            ):
                self.get_logger().warning(
                    "마지막 plan_place 발행 전 새 box_sizes 거부"
                )
                return

            if self.current_task["status"] == "pack":
                self.get_logger().warning(
                    "pack_reset 전에 새 box_sizes 거부"
                )
                return

            if not self.session.confirm_done(
                self.current_task["task_id"]
            ):
                self.get_logger().warning(
                    "이전 batch 마지막 작업 완료 반영 실패"
                )
                return

            self.current_task = None
            self.place_sent = False
            self.batch_active = False

        batch_id, boxes = self.parse_boxes(values)

        input_boxes = [
            {
                "idx": box.box_index,
                "size_mm": [box.size_x_mm, box.size_y_mm, box.size_z_mm],
            }
            for box in boxes
        ] # box index랑 size다 받음.
        self.get_logger().info(
            f"[INPUT] batch={batch_id} boxes={json.dumps(input_boxes)}"
        )

        if not self.session.start_batch(batch_id, boxes):
            self.get_logger().warning(f"중복 batch 무시: {batch_id}")
            return

        self.lookahead_count = len(boxes)
        self.task_sent_count = 0
        self.plan_place_sent_count = 0
        self.batch_active = True

        plan = self.session.last_plan
        search = self.session.last_search

        log = f"[PACKING] batch={batch_id} placed={len(plan.placements)}/{len(boxes)} score={plan.score} visited={plan.visited_nodes}"
        self.get_logger().info(f"{log} elapsed={search.elapsed_seconds:.3f}s exact={search.exact} stop={search.termination_reason}")

        execution_order = [
            candidate.box_index
            for candidate in self.session.execution_placements
        ]
        keep_indices = [
            box.box_index
            for box in self.session.keep_boxes
        ]
        self.get_logger().info(
            f"[PLAN] batch={batch_id} execution_order={execution_order} "
            f"keep={keep_indices}"
        )

        for sequence, candidate in enumerate(
            self.session.execution_placements,
            start=1,
        ):
            placed = candidate.placed_box
            target_x, target_y, target_z = placed.top_center_mm
            self.get_logger().info(
                f"[PLAN_DETAIL] sequence={sequence} idx={candidate.box_index} "
                f"top_axis={candidate.orientation.top_axis} "
                f"align_axis={candidate.orientation.container_x_axis} "
                f"place_top_center_mm=({target_x:.1f}, {target_y:.1f}, {target_z:.1f})"
            )

        # 상자 크기가 들어와 계획이 완성되는 즉시 전체 계획을 먼저 보여준다.
        self.render_plan_preview(batch_id, boxes)

        self.next_requested = False
        self.publish_next_task()

    def handle_next_box(self):
        # 아직 box_sizes가 없으면 요청만 기억
        if self.session.batch_id is None and self.current_task is None:
            self.next_requested = True
            return

        # 다음 요청이 왔다는 것은 이전 작업이 끝났다는 뜻
        if self.current_task is not None:
            status = self.current_task["status"]

            if status == "pick" and not self.place_sent:
                self.get_logger().warning("plan_place 발행 전 next_box 거부")
                return

            if status == "pack":
                self.get_logger().warning("pack_reset 대기 중")
                return

            if not self.session.confirm_done(self.current_task["task_id"]):
                self.get_logger().warning(f"작업 완료 반영 실패: {self.current_task['task_id']}")
                return

            self.current_task = None
            self.place_sent = False

        self.publish_next_task()

    def publish_next_task(self):
        task = self.session.next_task()

        if task is None:
            self.get_logger().info("현재 batch 작업 완료")
            return

        payload = {"idx": task["index"], "face": task["face"], "axis": task["alignment"], "status": task["status"]}
        message = String()
        message.data = json.dumps(payload)
        self.plan_pick_pub.publish(message)
        if task["status"] in ("pick", "keep"):
            self.task_sent_count += 1

        self.current_task = task
        self.place_sent = False

        self.get_logger().info(
            f"/heuristic/plan_pick "
            f"task=({self.task_sent_count}/{self.lookahead_count}): "
            f"{message.data}"
        )
        self.render_task(task)

    def handle_pick_done(self):
        if self.current_task is None:
            self.get_logger().warning("현재 pick 작업 없음")
            return

        if self.current_task["status"] != "pick":
            self.get_logger().warning(f"pick_done 무시: status={self.current_task['status']}")
            return

        if self.place_sent:
            self.get_logger().warning("중복 pick_done 무시")
            return

        target_mm = self.current_task["target_mm"]
        payload = {"x": target_mm[0] * MM_TO_M, "y": target_mm[1] * MM_TO_M, "z": (target_mm[2] + PLACE_Z_OFFSET_MM) * MM_TO_M}

        message = String()
        message.data = json.dumps(payload)
        self.plan_place_pub.publish(message)
        self.plan_place_sent_count += 1
        self.place_sent = True

        self.get_logger().info(
            f"/heuristic/plan_place "
            f"place_count={self.plan_place_sent_count}, "
            f"task={self.current_task['sequence']}/{self.lookahead_count}: "
            f"{message.data}"
        )

    def handle_pack_reset(self):
        self.session.reset_pack()
        self.current_task = None
        self.place_sent = False
        self.next_requested = False
        self.task_sent_count = 0
        self.plan_place_sent_count = 0
        self.batch_active = False

        message = Bool()
        message.data = True
        self.pack_reset_done_pub.publish(message)

        self.get_logger().info("/heuristic/pack_reset_done: true")

    def render_task(self, task):
        if self.session.plan is None:
            return

        planned_state = self.session.state
        highlight_box = None

        if self.session.execution_placements:
            planned_state = self.session.execution_placements[-1].state_after

        if task["status"] == "pick":
            for candidate in self.session.execution_placements:
                if candidate.box_index == task["index"]:
                    highlight_box = candidate.placed_box
                    break

        title = f"batch={task['batch_id']} sequence={task['sequence']} status={task['status']} box={task['index']}"
        output_path = self.renderer.render(planned_state, highlight_box, self.session.state.placed_boxes, title)

        self.get_logger().info(f"render: {output_path}")

    def render_plan_preview(self, batch_id, boxes):
        planned_state = self.session.state
        highlight_box = None

        if self.session.execution_placements:
            planned_state = self.session.execution_placements[-1].state_after
            highlight_box = self.session.execution_placements[0].placed_box

        input_indices = [box.box_index for box in boxes]
        execution_order = [
            candidate.box_index
            for candidate in self.session.execution_placements
        ]
        keep_indices = [box.box_index for box in self.session.keep_boxes]
        title = (
            f"batch={batch_id} input={input_indices} "
            f"order={execution_order} keep={keep_indices}"
        )
        output_path = self.renderer.render(
            planned_state,
            highlight_box,
            self.session.state.placed_boxes,
            title,
        )
        self.get_logger().info(f"render plan: {output_path}")

    def worker_loop(self):
        while True:
            event = self.work_queue.get()

            try:
                if event is None:
                    return

                name, value = event

                if name == "box_sizes":
                    self.handle_box_sizes(value)
                elif name == "next_box":
                    self.handle_next_box()
                elif name == "pick_done":
                    self.handle_pick_done()
                elif name == "pack_reset":
                    self.handle_pack_reset()

            except Exception as error:
                self.get_logger().error(f"worker 실패: {type(error).__name__}: {error}")
            finally:
                self.work_queue.task_done()

    def close(self):
        if self.shutdown_started:
            return

        self.shutdown_started = True

        # 기존 작업 뒤에 sentinel을 넣고 worker 종료
        self.work_queue.put(None)
        self.work_queue.join()
        self.worker.join()

        try:
            outputs = self.renderer.finish()

            if outputs:
                self.get_logger().info(f"render video: {outputs}")
        except Exception as error:
            self.get_logger().error(f"영상 저장 실패: {type(error).__name__}: {error}")


def main(args=None):
    rclpy.init(args=args)
    node = HeuristicMainNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
