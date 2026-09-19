# 3개 입력 → 실시간 계획 → 큰 박스·먼 X·작은 Y 순서로 작업 반환 → 완료된 배치만 state에 반영

"""
lookahead 한 묶음의 계획, 실행 순서, 실제 완료 상태 관리
"""

from math import isclose

from .packing_data_class import BoxSpec, ContainerSpec, PlacementCandidate
from .packing_place import PlacementEngine
from .packing_search import PackingSearch


TOP_FACE = {
    "x": "yz",
    "y": "xz",
    "z": "xy",
}


class PackingSession:
    def __init__(self, container=None, placement_engine=None, packing_search=None):
        if container is None:
            container = ContainerSpec.from_settings()

        if placement_engine is None:
            placement_engine = PlacementEngine(container)

        if packing_search is None:
            packing_search = PackingSearch(placement_engine)

        self.container = container
        self.engine = placement_engine
        self.search = packing_search

        # 실제 배치 완료된 박스만 들어가는 state
        self.state = self.engine.empty_state()

        self.batch_id = None
        self.plan = None
        self.execution_placements = ()
        self.keep_boxes = ()
        self.all_keep = False
        self.position = 0

        # 같은 batch가 두 번 실행되는 것 방지
        self.seen_batches = set()

        # batch 완료 뒤에도 랜더링과 로그에서 확인
        self.last_plan = None
        self.last_search = None

    def find_box(self, boxes, box_index):
        # 박스 번호로 원본 BoxSpec 찾기
        for box in boxes:
            if box.box_index == box_index:
                return box

        return None

    def order_for_execution(self, initial_state, placements, boxes):
        # 받침이 먼저 놓일 수 있는 순서 중 큰 박스 -> 먼 X -> 작은 Y 선택
        if not placements:
            return ()

        best_order = None
        best_priority = None

        def visit(
            state,
            remaining,
            ordered,
            priority_values,
        ):
            nonlocal best_order
            nonlocal best_priority

            if not remaining:
                priority_key = tuple(priority_values)

                if best_priority is None or priority_key > best_priority:
                    best_order = tuple(ordered)
                    best_priority = priority_key

                return

            for position in range(len(remaining)):
                candidate = remaining[position]
                box = self.find_box(boxes, candidate.box_index)

                if box is None:
                    raise RuntimeError(f"box not found: {candidate.box_index}")

                # 이 순서에서 실제로 계획된 위치에 놓을 수 있는지 다시 검사
                result = self.engine.try_place_box(
                    state,
                    box,
                    candidate.orientation,
                    candidate.x_cells,
                    candidate.y_cells,
                )

                if result is None:
                    continue

                state_after, z_cells, placed = result

                # 받침이 아직 없으면 계획된 Z와 달라지므로 실행 불가
                if z_cells != candidate.z_cells:
                    continue

                if not isclose(placed.z_mm, candidate.placed_box.z_mm):
                    continue

                rebuilt = PlacementCandidate(
                    box_index=candidate.box_index,
                    orientation=candidate.orientation,
                    x_cells=candidate.x_cells,
                    y_cells=candidate.y_cells,
                    z_cells=z_cells,
                    state_after=state_after,
                    placed_box=placed,
                )

                center_x = placed.x_mm + placed.width_mm / 2
                center_y = placed.y_mm + placed.depth_mm / 2

                priority = (
                    box.volume_mm3,
                    center_x,
                    -center_y,
                )

                next_remaining = (
                    remaining[:position]
                    + remaining[position + 1:]
                )

                visit(
                    state_after,
                    next_remaining,
                    ordered + (rebuilt,),
                    priority_values + (priority,),
                )

        visit(
            initial_state,
            placements,
            (),
            (),
        )

        if best_order is None:
            raise RuntimeError("planned placements have no executable order")

        return best_order

    def start_batch(self, batch_id, boxes):
        # 같은 batch가 다시 들어오면 무시
        if batch_id in self.seen_batches:
            return False

        # 이전 batch 작업이 끝나기 전이면 새 batch 거부
        if self.batch_id is not None:
            raise RuntimeError("현재 batch 완료 후 새 batch 입력")

        plan = self.search.plan_batch(self.state, boxes)

        execution_placements = self.order_for_execution(self.state, plan.placements, boxes)

        placed_indices = set()

        for candidate in execution_placements:
            placed_indices.add(candidate.box_index)

        keep_boxes = []

        for box in boxes:
            if box.box_index not in placed_indices:
                keep_boxes.append(box)

        self.batch_id = batch_id
        self.plan = plan
        self.execution_placements = execution_placements
        self.keep_boxes = tuple(keep_boxes)

        # 입력 박스가 전부 배치 불가능하면 개별 KEEP 대신 PACK
        self.all_keep = (
            len(execution_placements) == 0
            and len(keep_boxes) == len(boxes)
        )

        self.position = 0
        self.seen_batches.add(batch_id)

        self.last_plan = plan
        self.last_search = plan.search

        return True

    def next_task(self):
        # next_box가 여러 번 와도 confirm 전에는 같은 작업 반환
        if self.batch_id is None:
            return None

        sequence = self.position + 1
        task_id = f"{self.batch_id}:{sequence}"

        task = {
            "batch_id": self.batch_id,
            "task_id": task_id,
            "sequence": sequence,
        }

        if self.all_keep:
            task.update(
                index=-1,
                face="none",
                alignment="none",
                status="pack",
            )
            return task

        if self.position < len(self.execution_placements):
            candidate = self.execution_placements[self.position]
            orientation = candidate.orientation
            placed = candidate.placed_box

            task.update(
                index=candidate.box_index,
                face=TOP_FACE[orientation.top_axis.lower()],
                alignment=orientation.container_x_axis.lower(),
                status="pick",
                target_mm=placed.top_center_mm,
            )

            return task

        keep_position = self.position - len(self.execution_placements)
        box = self.keep_boxes[keep_position]

        task.update(
            index=box.box_index,
            face="none",
            alignment="none",
            status="keep",
        )

        return task

    def confirm_done(self, task_id, success=True):
        task = self.next_task()

        if task is None:
            return False

        if task["task_id"] != task_id:
            return False

        if not success:
            return False

        # PACK은 pack_reset 신호가 와야 실제 컨테이너를 초기화
        if task["status"] == "pack":
            return False

        # pick 작업이 실제 완료된 뒤에만 canonical state 갱신
        if task["status"] == "pick":
            self.state = self.execution_placements[self.position].state_after

        self.position += 1

        total_tasks = (
            len(self.execution_placements)
            + len(self.keep_boxes)
        )

        if self.position >= total_tasks:
            self.finish_batch()

        return True

    def finish_batch(self):
        # 실제 배치 state와 마지막 랜더링 정보는 그대로 보존
        self.batch_id = None
        self.plan = None
        self.execution_placements = ()
        self.keep_boxes = ()
        self.all_keep = False
        self.position = 0

    def reset_pack(self):
        # 컨테이너가 교체됐다는 신호를 받은 뒤 전체 상태 초기화
        self.state = self.engine.empty_state()

        self.batch_id = None
        self.plan = None
        self.execution_placements = ()
        self.keep_boxes = ()
        self.all_keep = False
        self.position = 0

        self.seen_batches.clear()
        self.last_plan = None
        self.last_search = None
