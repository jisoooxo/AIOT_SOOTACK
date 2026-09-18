"""
lookahead 박스들의 배치 순서를 DFS로 전부 검사하고 가장 좋은 계획 선택
"""


"""
놓을 수 없는 후보
    → PlacementEngine에서 미리 제거

놓을 수 있는 후보
    → DFS의 다음 state가 됨

같은 state + 같은 남은 박스
    → 캐시 결과 재사용

모든 갈래 비교
    → score_plan이 큰 계획 선택

전체 점수까지 같음
    → 선택한 휴리스틱 값이 작은 계획 선택

######################

우리 전체 점수
    ↓ 동점
각 휴리스틱별 판단
    ↓ 최종 계획끼리 동점
4개 휴리스틱 교차 순위합
    ↓ 순위합도 동점
BSSF → BLSF → BAF → BL

"""
"""
lookahead 박스의 배치 순서와 위치를 DFS로 검사
"""

"""
lookahead 박스들을 실시간 DFS로 배치
"""

from time import perf_counter

from .packing_data_class import BatchPlan, SearchMetadata


PLANNER_MODE = "REALTIME_BAF"

GRACE_SECONDS = 0.5
HARD_TIMEOUT_SECONDS = 2.0 # search 상한선 ㅇㅇ 더 늘릴수도 있음.

# BAF로 좋은 후보부터 DFS → 공통 점수로 최종 선택 → 첫 3개 계획 후 0.5초 더 탐색 → 최대 2초에 종료

class SearchStopped(Exception):
    # 정상이지만 시간 제한 때문에 DFS를 빠져나오는 용도
    pass


class PackingSearch:
    def __init__(self, placement_engine, grace_seconds=GRACE_SECONDS, hard_timeout_seconds=HARD_TIMEOUT_SECONDS):
        self.engine = placement_engine
        self.grace_seconds = grace_seconds
        self.hard_timeout_seconds = hard_timeout_seconds

    def free_prism(self, state, candidate):
        # 후보 주변에서 박스 바닥보다 높지 않은 연속 공간 계산
        x = candidate.x_cells
        y = candidate.y_cells
        z = candidate.z_cells

        width = candidate.orientation.width_cells
        depth = candidate.orientation.depth_cells

        start_x = x
        end_x = x + width
        start_y = y
        end_y = y + depth

        # X- 방향
        for xx in range(x - 1, -1, -1):
            blocked = False

            for yy in range(y, y + depth):
                if state.rows[xx][yy] > z:
                    blocked = True
                    break

            if blocked:
                break

            start_x = xx

        # X+ 방향
        for xx in range(x + width, state.width_cells):
            blocked = False

            for yy in range(y, y + depth):
                if state.rows[xx][yy] > z:
                    blocked = True
                    break

            if blocked:
                break

            end_x = xx + 1

        # Y- 방향
        for yy in range(y - 1, -1, -1):
            blocked = False

            for xx in range(x, x + width):
                if state.rows[xx][yy] > z:
                    blocked = True
                    break

            if blocked:
                break

            start_y = yy

        # Y+ 방향
        for yy in range(y + depth, state.depth_cells):
            blocked = False

            for xx in range(x, x + width):
                if state.rows[xx][yy] > z:
                    blocked = True
                    break

            if blocked:
                break

            end_y = yy + 1

        free_width = end_x - start_x
        free_depth = end_y - start_y
        headroom = state.height_cells - z

        return free_width, free_depth, headroom

    def baf_key(self, state, candidate):
        # 작은 빈 공간에 잘 맞는 후보부터 DFS로 확인
        orientation = candidate.orientation
        free_width, free_depth, headroom = self.free_prism(state, candidate)

        width_slack = free_width - orientation.width_cells
        height_slack = headroom - orientation.height_cells
        short_fit = min(width_slack, height_slack)
        free_volume = free_width * free_depth * headroom

        return free_volume, short_fit

    def score_plan(self, initial_state, placements, boxes):
        # 실제 최종 계획을 고르는 공통 점수
        placed_count = len(placements)
        placed_indices = set()
        volume_mm3 = 0.0

        for candidate in placements:
            placed_indices.add(candidate.box_index)

        for box in boxes:
            if box.box_index in placed_indices:
                volume_mm3 += box.volume_mm3

        final_state = initial_state

        if placements:
            final_state = placements[-1].state_after

        continuity = (0, 0, 0)

        # 박스가 2개 이상 들어간 경우에만 다음 연속 공간까지 계산
        if placed_count >= 2:
            continuity = final_state.largest_flat_rectangle()

        max_height = 0

        for row in final_state.rows:
            for height in row:
                if height > max_height:
                    max_height = height

        roughness = final_state.roughness()

        # 왼쪽부터 비교
        # 개수 -> 부피 -> 연속 공간 -> 낮은 최대 높이 -> 낮은 거칠기
        return (
            placed_count,
            volume_mm3,
            continuity,
            -max_height,
            -roughness,
        )

    def plan_position_key(self, placements):
        # 공통 점수까지 같으면 X가 멀고 Y가 작은 묶음 선택
        if not placements:
            return 0.0, 0.0, 0.0, 0.0

        minimum_center_x = None
        center_x_sum = 0.0
        maximum_y = 0.0
        y_sum = 0.0

        for candidate in placements:
            placed = candidate.placed_box
            center_x = placed.x_mm + placed.width_mm / 2

            center_x_sum += center_x
            y_sum += placed.y_mm

            if minimum_center_x is None or center_x < minimum_center_x:
                minimum_center_x = center_x

            if placed.y_mm > maximum_y:
                maximum_y = placed.y_mm

        return (
            minimum_center_x,
            center_x_sum,
            -maximum_y,
            -y_sum,
        )

    def state_cache_key(self, state):
        # 같은 높이맵, 실물 박스, gap이면 이후 계산은 한 번만 함
        placed_boxes = []

        for box in state.placed_boxes:
            values = (
                box.x_mm,
                box.y_mm,
                box.z_mm,
                box.width_mm,
                box.depth_mm,
                box.height_mm,
            )
            placed_boxes.append(values)

        placed_boxes.sort()

        clearance_regions = []

        for region in state.clearance_regions:
            values = (
                region.x_min_mm,
                region.x_max_mm,
                region.y_min_mm,
                region.y_max_mm,
                region.z_min_mm,
                region.z_max_mm,
            )
            clearance_regions.append(values)

        clearance_regions.sort()

        return (
            state.rows,
            tuple(placed_boxes),
            tuple(clearance_regions),
        )

    def ordered_candidates(
        self,
        state,
        box,
        next_boxes,
        first_lookahead,
        stop_if_needed,
    ):
        # PlacementEngine이 충돌, 지지, gap, 그리퍼 검사까지 끝낸 후보
        candidates = list(self.engine.find_placements(state, box))

        # 후보가 많을 때 중간중간 시간 확인
        checked = 0

        for candidate in candidates:
            checked += 1

            if checked % 32 == 0:
                stop_if_needed()

        remaining_width = 0

        for remaining_box in next_boxes:
            remaining_width += self.engine.minimum_width(remaining_box)

        def candidate_key(candidate):
            floor_key = candidate.z_cells != 0
            height_key = candidate.orientation.height_cells
            baf = self.baf_key(state, candidate)

            if first_lookahead:
                reserved_width = candidate.orientation.width_cells + remaining_width

                if reserved_width <= state.width_cells:
                    target_x = state.width_cells - reserved_width
                else:
                    target_x = state.width_cells - candidate.orientation.width_cells

                return (
                    floor_key,
                    abs(candidate.x_cells - target_x),
                    candidate.y_cells,
                    height_key,
                    baf,
                )

            return (
                floor_key,
                -candidate.x_cells,
                candidate.y_cells,
                height_key,
                baf,
            )

        candidates.sort(key=candidate_key)
        return candidates

    def plan_batch(self, initial_state, boxes):
        start_time = perf_counter()
        hard_deadline = start_time + self.hard_timeout_seconds

        first_lookahead = len(initial_state.placed_boxes) == 0
        first_full_plan_seconds = None

        best_placements = ()
        best_score = None
        best_position = None

        visited_nodes = 0
        cache = set()
        termination_reason = "search_complete"

        def active_deadline():
            deadline = hard_deadline

            if first_full_plan_seconds is not None:
                grace_deadline = start_time + first_full_plan_seconds + self.grace_seconds
                deadline = min(deadline, grace_deadline)

            return deadline

        def stop_if_needed():
            nonlocal termination_reason

            now = perf_counter()

            if now < active_deadline():
                return

            if now >= hard_deadline:
                termination_reason = "hard_timeout"
            else:
                termination_reason = "grace_timeout"

            raise SearchStopped

        def update_best(placements):
            nonlocal best_placements
            nonlocal best_score
            nonlocal best_position
            nonlocal first_full_plan_seconds

            score = self.score_plan(initial_state, placements, boxes)
            position = self.plan_position_key(placements)

            better = best_score is None or score > best_score

            if score == best_score and position > best_position:
                better = True

            if better:
                best_placements = placements
                best_score = score
                best_position = position

            # 처음으로 lookahead 박스를 전부 놓은 시간
            if len(placements) == len(boxes) and first_full_plan_seconds is None:
                first_full_plan_seconds = perf_counter() - start_time

        def visit(state, remaining_boxes, placements):
            nonlocal visited_nodes

            stop_if_needed()

            key = (
                self.state_cache_key(state),
                remaining_boxes,
            )

            if key in cache:
                return

            cache.add(key)
            visited_nodes += 1

            update_best(placements)

            # 큰 박스 경로부터 들어가서 첫 full plan을 빨리 찾음
            box_positions = list(range(len(remaining_boxes)))

            def box_key(box_position):
                return remaining_boxes[box_position].volume_mm3

            box_positions.sort(key=box_key, reverse=True)

            for box_position in box_positions:
                stop_if_needed()

                box = remaining_boxes[box_position]

                next_boxes = (
                    remaining_boxes[:box_position]
                    + remaining_boxes[box_position + 1:]
                )

                candidates = self.ordered_candidates(
                    state,
                    box,
                    next_boxes,
                    first_lookahead,
                    stop_if_needed,
                )

                for candidate in candidates:
                    stop_if_needed()

                    next_placements = placements + (candidate,)

                    visit(candidate.state_after, next_boxes, next_placements)

        try:
            visit(initial_state, boxes, ())
        except SearchStopped:
            pass

        # 캐시 안의 state_after를 최종 순서에 맞게 다시 생성
        placements = self.engine.rebuild_placements(initial_state, best_placements, boxes)

        # 새 컨테이너 첫 lookahead 묶음은 X+와 Y=0 모서리에 붙임
        if first_lookahead and placements:
            placements = self.engine.anchor_to_far_corner(initial_state, placements, boxes)

        final_score = self.score_plan(initial_state, placements, boxes)

        elapsed_seconds = perf_counter() - start_time

        allowed_seconds = self.hard_timeout_seconds

        if first_full_plan_seconds is not None:
            grace_limit = first_full_plan_seconds + self.grace_seconds
            allowed_seconds = min(allowed_seconds, grace_limit)

        deadline_overshoot = max(0.0, elapsed_seconds - allowed_seconds)

        metadata = SearchMetadata(
            planner_mode=PLANNER_MODE,
            exact=termination_reason == "search_complete",
            elapsed_seconds=elapsed_seconds,
            first_full_plan_seconds=first_full_plan_seconds,
            grace_seconds=self.grace_seconds,
            hard_timeout_seconds=self.hard_timeout_seconds,
            hard_timeout_reached=termination_reason == "hard_timeout",
            deadline_overshoot_seconds=deadline_overshoot,
            termination_reason=termination_reason,
        )

        reason = (
            f"common_score; "
            f"candidate_order=floor_far_x_low_y_low_height_baf; "
            f"first_corner={str(first_lookahead).lower()}; "
            f"stop={termination_reason}"
        )

        return BatchPlan(
            placements=placements,
            score=final_score,
            visited_nodes=visited_nodes,
            heuristic="realtime_baf",
            selection_reason=reason,
            search=metadata,
        )
