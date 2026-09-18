# 패킹시 공통으로 사용되는 데이터 모음집. 그냥 Config 느낌으로 생각하면 편함.

from __future__ import annotations # return 힌트용,아직 정의되지 않은 클래스를 type hint에 사용할 수 있게 함

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .packing_place import HeightMapState



DEFAULT_CONTAINER_PRESET = "2호"

CONTAINER_PRESETS_MM = {"0호": (170, 130, 90), "1호": (220, 190, 90), "2호": (270, 180, 150), "2-1호": (350, 250, 100)}


@dataclass(frozen=True)
class ContainerSpec: # Container spec. 기존 프리셋 가능. 덮어쓰기도 가능함.
    width_mm: int
    depth_mm: int
    height_mm: int

    @classmethod
    def from_settings(cls, preset: str = DEFAULT_CONTAINER_PRESET, override_mm: tuple[int, int, int] | None = None):
        if override_mm is not None:
            return cls(*override_mm)

        return cls(*CONTAINER_PRESETS_MM[preset])


@dataclass(frozen=True)
class BoxSpec: # 준미한테 받는 박스 스펙 ㅇㅇ
    box_index: int
    size_x_mm: float
    size_y_mm: float
    size_z_mm: float

    @property
    def volume_mm3(self): # volume 값. (부피)
        return self.size_x_mm * self.size_y_mm * self.size_z_mm 



@dataclass(frozen=True)
class PlacedBox: # 놓인 박스에 대한 state용
    box_index: int
    x_mm: float
    y_mm: float
    z_mm: float
    width_mm: float
    depth_mm: float
    height_mm: float

    @property
    def top_center_mm(self): 
        center_x = self.x_mm + self.width_mm / 2
        center_y = self.y_mm + self.depth_mm / 2
        top_z = self.z_mm + self.height_mm
        return center_x, center_y, top_z



@dataclass(frozen=True) # 박스 사이의 offset 여유 공간 기록
class ClearanceRegion:
    x_min_mm: float
    x_max_mm: float
    y_min_mm: float
    y_max_mm: float
    z_min_mm: float
    z_max_mm: float


@dataclass(frozen=True)
class Orientation: # 회전 후 박스 축과 컨테이너 점유 칸
    top_axis: str # 위쪽 법선으로 향하는 박스 원본 축
    container_x_axis: str
    container_y_axis: str
    width_cells: int
    depth_cells: int
    height_cells: int



@dataclass(frozen=True)
class PlacementCandidate: # 배치하는 후보 좌표 넣어두는 ㅇㅇ
    box_index: int
    orientation: Orientation
    x_cells: int
    y_cells: int
    z_cells: int
    state_after: "HeightMapState"
    placed_box: PlacedBox


@dataclass(frozen=True)
class SearchMetadata: # 이건 랜더링용 + 디버깅용
    planner_mode: str
    exact: bool
    elapsed_seconds: float
    first_full_plan_seconds: float | None
    grace_seconds: float
    hard_timeout_seconds: float
    hard_timeout_reached: bool
    deadline_overshoot_seconds: float
    termination_reason: str


@dataclass(frozen=True)
class BatchPlan: # DFS가 최종 선택한 lookahead 배치 결과
    placements: tuple[PlacementCandidate, ...]
    score: tuple[int, float, tuple[int, int, int], int, int]
    visited_nodes: int
    heuristic: str = "realtime"
    heuristic_key: tuple[tuple[int, ...], ...] = ()
    rank_sums: tuple[tuple[str, int], ...] = ()
    selection_reason: str = ""
    search: SearchMetadata | None = None