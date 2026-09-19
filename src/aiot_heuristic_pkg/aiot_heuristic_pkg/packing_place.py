"""
Heightmap + 박스 하나의 배치 가능 위치 계산 용도
"""

from dataclasses import replace
from math import ceil, isclose

from .packing_data_class import BoxSpec, ClearanceRegion, ContainerSpec, Orientation, PlacementCandidate, PlacedBox

GRID_UNIT_MM = 10  # 높이맵 격자 한 칸의 실제 길이
BOX_SIZE_OFFSET_MM = 0.0  # 측정된 박스 길이에 추가할 여유
BOX_GAP_MM = 5.0  # 박스 옆면 사이에 확보할 실제 간격
WALL_GAP_MM = 0.0  # 컨테이너 벽과 박스 사이 간격
MIN_SUPPORT = 1.0  # 박스 바닥 지지 비율. 1.0이면 전체 지지


GRIPPER_SIDE_MARGIN_MM = BOX_GAP_MM + GRID_UNIT_MM  # 기본 gap보다 격자 한 칸 더 넓게 검사 (공압그리퍼 옆 폭 감안해서 -> 이건 지수랑 얘기 해야함!!!)
GRIPPER_HEIGHT_DIFFERENCE_MM = 30.0  # 옆 윗면이 목표 윗면보다 30mm 이상 높으면 거부

FLOOR_FIRST = False  # 바닥 후보를 먼저 보지만 적층 후보도 DFS에 전달
RESERVED_SUPPORT = False  # 탐색은 높이맵을 쓰고, 로봇에 줄 착지 Z는 실제 박스 윗면으로 계산


class HeightMapState: # Heightmap!!!!
    def __init__(
        self,
        container,
        unit_mm,
        rows,
        placed_boxes=(),
        clearance_regions=(),
    ): # 현재 높이맵, 완료된 박스, 계산된 gap 공간

        self.container = container
        self.unit_mm = unit_mm
        self.width_cells = int(container.width_mm // unit_mm)
        self.depth_cells = int(container.depth_mm // unit_mm) 
        self.height_cells = int(container.height_mm // unit_mm)

        # 상단 셀들은 전부 height map cells

        self.rows = rows
        self.placed_boxes = placed_boxes
        self.clearance_regions = clearance_regions


        # 예시

        """
                     Y
          0  1  2  3
       ┌─────────────
X = 0  │ 3  0  0  0
X = 1  │ 3  0  0  0
X = 2  │ 0  0  0  0
X = 3  │ 0  0  0  0
X = 4  │ 0  0  0  0
X = 5  │ 0  0  0  0
        
        """


    @classmethod
    def empty(cls, container, unit_mm=GRID_UNIT_MM): # 초기에 빈 heightmap 생성
        container_width_cells = int(container.width_mm // unit_mm) # 컨테이너 셀 정규화
        container_depth_cells = int(container.depth_mm // unit_mm)
        rows = []

        for x in range(container_width_cells): 
            row = []

            for y in range(container_depth_cells):
                row.append(0)

            rows.append(tuple(row))

        return cls(container, unit_mm, tuple(rows))

    def copy(self, rows=None, placed_boxes=None, clearance_regions=None): # 바로 갱신 안되도록 copy
        if rows is None:
            rows = self.rows

        if placed_boxes is None:
            placed_boxes = self.placed_boxes

        if clearance_regions is None:
            clearance_regions = self.clearance_regions

        return HeightMapState(self.container, self.unit_mm, rows, placed_boxes, clearance_regions)

    
    def try_place(
        self,
        x,
        y,
        width,
        depth,
        height,
        min_support=MIN_SUPPORT,
    ):
        # 셀 단위로 위치, 크기로 height map에 놓을 수 있는지 여부 검사

        if x < 0 or y < 0 or (x + width > self.width_cells) or (y + depth > self.depth_cells):
            return None

        footprint = []

        for xx in range(x, x + width):
            for yy in range(y, y+depth):
                footprint.append(self.rows[xx][yy]) 

        z = max(footprint)
        top = z + height 

        if top > self.height_cells: # container의 전체 높이보다 높을 경우 None처리.
            return None

        supported_cells = 0

        for value in footprint:
            if value == z:
                supported_cells += 1 # 지지대 계산

        support_ratio = supported_cells / len(footprint)

        if support_ratio < min_support:
            return None

        rows = list(self.rows)

        for xx in range(x, x + width):
            row = list(rows[xx])

            for yy in range(y, y + depth):
                row[yy] = top

            rows[xx] = tuple(row)

        next_state = self.copy(rows=tuple(rows))

        return next_state, z

    def roughness(self):
        # 붙어있는 높이맵 칸의 높이 차이를 전부 더함

        """
        rows[0][0]과 rows[0][1]
        |3 - 0| = 3

        rows[1][0]과 rows[1][1]
        |3 - 0| = 3

        rows[1][0]과 rows[2][0]
        |3 - 0| = 3
        
        """

        total = 0

        for x in range(self.width_cells):
            for y in range(self.depth_cells):
                current_height = self.rows[x][y] # 현재 X행의 Y칸 높이

                if x + 1 < self.width_cells:
                    total += abs(current_height - self.rows[x+1][y])

                if y + 1 < self.depth_cells:
                    total += abs(current_height - self.rows[x][y+1])

        return total


    def largest_flat_rectangle(self):
        # 같은 높이로 이어진 가장 큰 직사각형의 면적·짧은 변·위쪽 여유
        # rows에 이미 예약된 cell 크기가 들어가 있어서 여기서 offset을 또 더하진 않음
        best = (0, 0, 0)
        x_lengths = [0] * self.depth_cells

        for x in range(self.width_cells):
            for y in range(self.depth_cells):
                surface_height = self.rows[x][y]

                if surface_height >= self.height_cells:
                    x_lengths[y] = 0
                elif x > 0 and self.rows[x - 1][y] == surface_height:
                    x_lengths[y] += 1
                else:
                    x_lengths[y] = 1

            start_y = 0

            while start_y < self.depth_cells:
                surface_height = self.rows[x][start_y]

                if surface_height >= self.height_cells:
                    start_y += 1
                    continue

                end_y = start_y + 1

                while end_y < self.depth_cells and self.rows[x][end_y] == surface_height:
                    end_y += 1

                stack = [] # (이 길이가 시작된 Y위치, X방향 연속 길이)
                headroom = self.height_cells - surface_height

                for y in range(start_y, end_y + 1):
                    current_length = 0

                    if y < end_y:
                        current_length = x_lengths[y]

                    rectangle_start_y = y

                    while stack and stack[-1][1] > current_length:
                        previous_start_y, length_x = stack.pop()
                        length_y = y - previous_start_y
                        area = length_x * length_y

                        short_side = min(length_x, length_y)
                        key = (area, short_side, headroom)

                        if key > best:
                            best = key

                        rectangle_start_y = previous_start_y

                    if current_length > 0 and (not stack or stack[-1][1] < current_length):
                        stack.append((rectangle_start_y, current_length))

                start_y = end_y

        return best # (면적, 짧은 변, 위쪽 여유)




class PlacementEngine: # 박스 자세 생성부터 후보 좌표, 충돌, gap, 그리퍼 검사까지 처리
    def __init__(
        self,
        container,
        unit_mm=GRID_UNIT_MM,
        box_size_offset_mm=BOX_SIZE_OFFSET_MM,
        box_gap_mm=BOX_GAP_MM,
        wall_gap_mm=WALL_GAP_MM,
        min_support=MIN_SUPPORT,
        gripper_margin_mm=GRIPPER_SIDE_MARGIN_MM,
        gripper_height_mm=GRIPPER_HEIGHT_DIFFERENCE_MM,
        floor_first=FLOOR_FIRST,
        reserved_support=RESERVED_SUPPORT,
    ):
        self.container = container
        self.unit_mm = unit_mm
        self.box_size_offset_mm = box_size_offset_mm
        self.box_gap_mm = box_gap_mm
        self.wall_gap_mm = wall_gap_mm
        self.min_support = min_support
        self.gripper_margin_mm = gripper_margin_mm
        self.gripper_height_mm = gripper_height_mm
        self.floor_first = floor_first
        self.reserved_support = reserved_support

    def empty_state(self):
        # 현재 컨테이너 크기에 맞는 빈 heightmap 생성
        return HeightMapState.empty(self.container, self.unit_mm)

    def collides(self, state, box, include_clearance=True):
        # 컨테이너 밖, 기존 박스, 기존 gap 영역과 겹치는지 확인
        x_max = box.x_mm + box.width_mm
        y_max = box.y_mm + box.depth_mm
        z_max = box.z_mm + box.height_mm

        if box.x_mm < self.wall_gap_mm or x_max > self.container.width_mm - self.wall_gap_mm:
            return True

        if box.y_mm < self.wall_gap_mm or y_max > self.container.depth_mm - self.wall_gap_mm:
            return True

        if box.z_mm < 0 or z_max > self.container.height_mm:
            return True

        regions = []

        if include_clearance:
            for region in state.clearance_regions:
                regions.append(region)

        for placed in state.placed_boxes:
            region = ClearanceRegion(
                placed.x_mm,
                placed.x_mm + placed.width_mm,
                placed.y_mm,
                placed.y_mm + placed.depth_mm,
                placed.z_mm,
                placed.z_mm + placed.height_mm,
            )
            regions.append(region)

        for region in regions:
            overlap_x = box.x_mm < region.x_max_mm and x_max > region.x_min_mm
            overlap_y = box.y_mm < region.y_max_mm and y_max > region.y_min_mm
            overlap_z = box.z_mm < region.z_max_mm and z_max > region.z_min_mm

            if overlap_x and overlap_y and overlap_z: # X/Y/Z가 전부 겹쳐야 실제 충돌 ㅇㅇ
                return True

        return False

    def subtract_region(self, region, covered):
        # 새 gap에서 이미 예약된 gap을 빼고 남은 조각만 반환
        x0 = max(region.x_min_mm, covered.x_min_mm)
        x1 = min(region.x_max_mm, covered.x_max_mm)
        y0 = max(region.y_min_mm, covered.y_min_mm)
        y1 = min(region.y_max_mm, covered.y_max_mm)
        z0 = max(region.z_min_mm, covered.z_min_mm)
        z1 = min(region.z_max_mm, covered.z_max_mm)

        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            return (region,)

        bounds = (
            (region.x_min_mm, x0, region.y_min_mm, region.y_max_mm, region.z_min_mm, region.z_max_mm),
            (x1, region.x_max_mm, region.y_min_mm, region.y_max_mm, region.z_min_mm, region.z_max_mm),
            (x0, x1, region.y_min_mm, y0, region.z_min_mm, region.z_max_mm),
            (x0, x1, y1, region.y_max_mm, region.z_min_mm, region.z_max_mm),
            (x0, x1, y0, y1, region.z_min_mm, z0),
            (x0, x1, y0, y1, z1, region.z_max_mm),
        )
        pieces = []

        for xa, xb, ya, yb, za, zb in bounds:
            if xa < xb and ya < yb and za < zb:
                pieces.append(ClearanceRegion(xa, xb, ya, yb, za, zb))

        return tuple(pieces)

    def make_clearance(self, state, box):
        # 박스의 X-/X+/Y-/Y+ 옆면에 gap 영역 생성
        if self.box_gap_mm <= 0:
            return ()

        x = box.x_mm
        y = box.y_mm
        z = box.z_mm
        x_max = x + box.width_mm
        y_max = y + box.depth_mm
        z_max = z + box.height_mm
        gap = self.box_gap_mm

        bounds = (
            (max(0.0, x - gap), x, y, y_max),
            (x_max, min(self.container.width_mm, x_max + gap), y, y_max),
            (x, x_max, max(0.0, y - gap), y),
            (x, x_max, y_max, min(self.container.depth_mm, y_max + gap)),
        )
        regions = []

        for x_min, x_end, y_min, y_end in bounds:
            if x_min < x_end and y_min < y_end:
                regions.append(ClearanceRegion(x_min, x_end, y_min, y_end, z, z_max))

        for covered in state.clearance_regions:
            remaining = []

            for region in regions:
                for piece in self.subtract_region(region, covered):
                    remaining.append(piece)

            regions = remaining

        return tuple(regions)

    def landing_height(
        self,
        state,
        x_mm,
        y_mm,
        width_mm,
        depth_mm,
    ):
        # 실제 박스 윗면으로 착지 Z 계산. RESERVED_SUPPORT=False일 때 사용
        x_max = x_mm + width_mm
        y_max = y_mm + depth_mm
        landing_z = 0.0
        supports = []

        for placed in state.placed_boxes:
            overlap_x = min(x_max, placed.x_mm + placed.width_mm) - max(x_mm, placed.x_mm)
            overlap_y = min(y_max, placed.y_mm + placed.depth_mm) - max(y_mm, placed.y_mm)

            if overlap_x <= 0 or overlap_y <= 0:
                continue

            top_z = placed.z_mm + placed.height_mm
            supports.append((top_z, overlap_x * overlap_y))

            if top_z > landing_z:
                landing_z = top_z

        if not supports:
            return 0.0

        supported_area = 0.0

        for top_z, area in supports:
            if isclose(top_z, landing_z):
                supported_area += area

        support_ratio = supported_area / (width_mm * depth_mm)

        if support_ratio < self.min_support and not isclose(support_ratio, self.min_support):
            return None

        return landing_z

    def gripper_can_reach(
        self,
        state,
        x,
        y,
        width,
        depth,
        target_top,
    ):
        # 박스 주변에서 목표 윗면보다 30mm 이상 높은 칸이 있으면 out
        if self.gripper_margin_mm <= 0 or self.gripper_height_mm <= 0:
            return True

        side_cells = ceil(self.gripper_margin_mm / self.unit_mm)
        start_x = max(0, x - side_cells)
        end_x = min(state.width_cells, x + width + side_cells)
        start_y = max(0, y - side_cells)
        end_y = min(state.depth_cells, y + depth + side_cells)
        target_top_mm = target_top * self.unit_mm

        for xx in range(start_x, end_x):
            for yy in range(start_y, end_y):
                inside_box = x <= xx < x + width and y <= yy < y + depth

                if inside_box:
                    continue

                neighbor_top_mm = state.rows[xx][yy] * self.unit_mm

                if neighbor_top_mm - target_top_mm >= self.gripper_height_mm:
                    return False

        return True

    def try_place_box(
        self,
        state,
        box,
        orientation,
        x,
        y,
    ):
        # heightmap 예약 -> 그리퍼 -> 실물 충돌 -> gap 순서로 박스 하나 최종 검사
        width_mm, depth_mm, height_mm = self.rotated_size_mm(box, orientation)
        width_cells = orientation.width_cells
        depth_cells = orientation.depth_cells
        height_cells = orientation.height_cells

        reserved = state.try_place(x, y, width_cells, depth_cells, height_cells, self.min_support)

        if reserved is None:
            return None

        state_after, z_cells = reserved
        target_top = z_cells + height_cells

        if not self.gripper_can_reach(state, x, y, width_cells, depth_cells, target_top):
            return None

        x_mm = x * self.unit_mm
        y_mm = y * self.unit_mm

        if self.reserved_support:
            z_mm = z_cells * self.unit_mm
        else:
            z_mm = self.landing_height(state, x_mm, y_mm, width_mm, depth_mm)

        if z_mm is None:
            return None

        placed = PlacedBox(box.box_index, x_mm, y_mm, z_mm, width_mm, depth_mm, height_mm)

        if self.collides(state, placed):
            return None

        regions = self.make_clearance(state, placed)

        for region in regions:
            probe = PlacedBox(
                box.box_index,
                region.x_min_mm,
                region.y_min_mm,
                region.z_min_mm,
                region.x_max_mm - region.x_min_mm,
                region.y_max_mm - region.y_min_mm,
                region.z_max_mm - region.z_min_mm,
            )

            if self.collides(state, probe, include_clearance=False):
                return None

        placed_boxes = state.placed_boxes + (placed,)
        clearance_regions = state.clearance_regions + regions
        state_after = state_after.copy(placed_boxes=placed_boxes, clearance_regions=clearance_regions)

        return state_after, z_cells, placed

    def ceil_cells(self, value_mm):
        # 실제 박스보다 작게 잡히지 않도록 mm를 cell로 올림 변환
        return ceil((value_mm + self.box_size_offset_mm) / self.unit_mm)

    def build_orientations(self, box):
        # 박스 원본 XYZ를 컨테이너 XYZ에 맞춘 6가지 자세 생성
        sizes_cells = {
            "X": self.ceil_cells(box.size_x_mm),
            "Y": self.ceil_cells(box.size_y_mm),
            "Z": self.ceil_cells(box.size_z_mm),
        }

        axis_orders = (
            ("Y", "Z", "X"),
            ("Z", "Y", "X"),
            ("X", "Z", "Y"),
            ("Z", "X", "Y"),
            ("X", "Y", "Z"),
            ("Y", "X", "Z"),
        )
        orientations = []

        for container_x_axis, container_y_axis, top_axis in axis_orders:
            orientation = Orientation(
                top_axis,
                container_x_axis,
                container_y_axis,
                sizes_cells[container_x_axis],
                sizes_cells[container_y_axis],
                sizes_cells[top_axis],
            )
            orientations.append(orientation)

        return tuple(orientations)

    def rotated_size_mm(self, box, orientation):
        # 선택한 자세의 실제 크기를 컨테이너 X/Y/Z 순서로 반환
        sizes_mm = {
            "X": box.size_x_mm,
            "Y": box.size_y_mm,
            "Z": box.size_z_mm,
        }

        width_mm = sizes_mm[orientation.container_x_axis]
        depth_mm = sizes_mm[orientation.container_y_axis]
        height_mm = sizes_mm[orientation.top_axis]
        return width_mm, depth_mm, height_mm

    def minimum_width(self, box):
        # 6가지 자세 중 컨테이너 X방향을 가장 적게 쓰는 폭
        minimum = None

        for orientation in self.build_orientations(box):
            if minimum is None or orientation.width_cells < minimum:
                minimum = orientation.width_cells

        return minimum

    def find_placements(self, state, box):
        # 6가지 자세와 모든 X/Y 좌표를 검사해서 가능한 후보 전부 생성
        candidates = []
        floor_candidates = []

        for orientation in self.build_orientations(box):
            last_x = state.width_cells - orientation.width_cells
            last_y = state.depth_cells - orientation.depth_cells

            if last_x < 0 or last_y < 0:
                continue

            for x in range(last_x + 1):
                for y in range(last_y + 1):
                    result = self.try_place_box(state, box, orientation, x, y)

                    if result is None:
                        continue

                    state_after, z, placed = result
                    candidate = PlacementCandidate(box.box_index, orientation, x, y, z, state_after, placed)
                    candidates.append(candidate)

                    if z == 0:
                        floor_candidates.append(candidate)

        if self.floor_first and floor_candidates:
            selected = floor_candidates
        else:
            selected = candidates

        # 바닥부터 -> X가 먼 곳부터 -> 같은 X면 Y가 작은 곳부터 반환
        selected.sort(key=lambda candidate: (candidate.z_cells != 0, -candidate.x_cells, candidate.y_cells))
        return tuple(selected)

    def rebuild_placements(self, initial_state, placements, boxes):
        # 저장된 좌표를 첫 state부터 다시 놓아서 state_after를 순서대로 재생성
        state = initial_state
        rebuilt = []

        for candidate in placements:
            box = None

            for item in boxes:
                if item.box_index == candidate.box_index:
                    box = item
                    break

            if box is None:
                raise RuntimeError(f"box not found: {candidate.box_index}")

            result = self.try_place_box(state, box, candidate.orientation, candidate.x_cells, candidate.y_cells)

            if result is None:
                raise RuntimeError(f"placement replay failed: {candidate.box_index}")

            state_after, z, placed = result
            rebuilt_candidate = PlacementCandidate(box.box_index, candidate.orientation, candidate.x_cells, candidate.y_cells, z, state_after, placed)
            rebuilt.append(rebuilt_candidate)
            state = state_after

        return tuple(rebuilt)

    def anchor_to_far_corner(self, initial_state, placements, boxes):
        # 첫 lookahead 묶음 전체를 X+ 끝과 Y=0 모서리 쪽으로 이동
        if not placements:
            return placements

        max_x = 0
        min_y = initial_state.depth_cells

        for candidate in placements:
            end_x = candidate.x_cells + candidate.orientation.width_cells

            if end_x > max_x:
                max_x = end_x

            if candidate.y_cells < min_y:
                min_y = candidate.y_cells

        move_x = initial_state.width_cells - max_x
        move_y = -min_y
        shifted = []

        for candidate in placements:
            shifted_candidate = replace(candidate, x_cells=candidate.x_cells + move_x, y_cells=candidate.y_cells + move_y)
            shifted.append(shifted_candidate)

        return self.rebuild_placements(initial_state, tuple(shifted), boxes)
    
