# AIOT Heuristic Package

상자 3개를 lookahead 입력으로 받아 컨테이너 안의 배치 위치와 실행 순서를 계산하는 ROS 2 Humble 패키지다. 내부 단위는 mm이며, ROS 토픽으로 주고받는 상자 크기와 배치 좌표의 단위는 m다.

## 빌드 및 실행

워크스페이스 루트에서 빌드한다.

```bash
cd ~/AIOT_SOOTACK
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

실제 토픽 연동용 메인 노드:

```bash
ros2 run aiot_heuristic_pkg heuristic_main_node
```

샘플 18개를 자동 실행하는 디버그 노드:

```bash
ros2 run aiot_heuristic_pkg heuristic_debug_node
```

18개 샘플을 메인 노드의 실제 토픽으로 검증하려면 워크스페이스 루트의 스크립트를 실행한다.

```bash
./test_heuristic_full_cycle.sh
```

## 노드별 역할

### `heuristic_main_node`

실제 Main/Control 노드와 통신하는 운영용 노드다.

- 현재 임시 연결된 `/vision/box_sizes`로 상자 3개의 크기를 받는다.
- DFS 기반 패킹 계획과 실제 실행 순서를 계산한다.
- `/main/next_box`를 받을 때 `/heuristic/plan_pick`을 발행한다.
- pick 작업에서 `/control/pick_done`을 받으면 `/heuristic/plan_place`를 발행한다.
- 이전 작업은 다음 `/main/next_box`가 들어올 때 실제 완료 상태로 반영된다.
- 세 상자를 전부 배치할 수 없으면 `status=pack`을 보내고 컨테이너 reset을 기다린다.
- 각 작업을 보낼 때 최신 패킹 이미지를 렌더링한다.
- ROS 콜백은 계산하지 않고 최대 16개의 내부 queue에 이벤트를 넣는다. 하나의 worker 스레드가 이벤트를 순서대로 처리한다.

입력 토픽:

| 토픽 | 타입 | 설명 |
| --- | --- | --- |
| `/vision/box_sizes` | `std_msgs/msg/String` | 상자 3개 JSON 배열, 크기 단위 m. 현재 비전 직결용 임시 설정 |
| `/main/next_box` | `std_msgs/msg/Bool` | 첫 작업 요청 또는 이전 작업 완료 확정 |
| `/control/pick_done` | `std_msgs/msg/Bool` | 픽 완료 및 place 좌표 요청 |
| `/main/pack_reset` | `std_msgs/msg/Bool` | 컨테이너 교체 완료 및 내부 상태 초기화 요청 |

출력 토픽:

| 토픽 | 타입 | 설명 |
| --- | --- | --- |
| `/heuristic/plan_pick` | `std_msgs/msg/String` | 상자 번호, 면, 정렬 축, 작업 상태 |
| `/heuristic/plan_place` | `std_msgs/msg/String` | 배치할 상자 윗면 중심 좌표, 단위 m |
| `/heuristic/pack_reset_done` | `std_msgs/msg/Bool` | 내부 패킹 상태 초기화 완료 |

상자 입력 예시:

```json
[
  {"idx": 1, "x": 0.158, "y": 0.030, "z": 0.030},
  {"idx": 2, "x": 0.142, "y": 0.036, "z": 0.032},
  {"idx": 3, "x": 0.152, "y": 0.036, "z": 0.036}
]
```

`plan_pick` 예시:

```json
{"idx": 3, "face": "XZ", "axis": "Z", "status": "pick"}
```

`status`의 의미:

| 상태 | 의미 |
| --- | --- |
| `pick` | 해당 상자를 집어 계산된 위치에 배치 |
| `keep` | 현재 컨테이너에 넣지 않고 keep 영역으로 이동 |
| `pack` | 세 상자를 모두 넣을 수 없으므로 컨테이너 교체 요청 |

`plan_place` 예시:

```json
{"x": 0.198, "y": 0.076, "z": 0.036}
```

### `heuristic_debug_node`

ROS 토픽 연결 없이 패킹 알고리즘과 렌더링만 단독으로 확인하는 노드다.

- 설치된 패키지의 `config/boxes_sample_m.json`에서 상자 18개를 읽는다.
- 상자를 3개씩 패킹하고 pick/keep 완료를 내부에서 자동 처리한다.
- 컨테이너가 꽉 차면 자동 reset하고 keep된 상자를 다시 대기열에 넣는다.
- 최대 `MAX_BATCHES=100` batch까지 실행한다.
- 실행 결과와 렌더 파일을 `render_output/<실행시간>/`에 저장한다.
- `summary.json`, 프레임 PNG, MP4, WebM을 저장한다.
- 토픽을 publish하거나 subscribe하지 않는다.

디버그 노드는 `PackingRenderer` 생성 시 `save_png`, `save_mp4`, `save_webm`을 모두 `True`로 전달한다. 따라서 아래 렌더 기본 토글과 관계없이 디버그 실행에서는 모든 결과 형식을 저장한다.

## 내부 모듈

| 파일 | 역할 |
| --- | --- |
| `packing_data_class.py` | 상자, 컨테이너, 자세, 배치 결과 및 검색 메타데이터 정의 |
| `packing_place.py` | Heightmap 관리, 6가지 자세 생성, 충돌·간격·지지·그리퍼 검사, 배치 후보 생성 |
| `packing_search.py` | 상자 3개의 순서와 위치를 DFS로 탐색하고 최종 계획 선택 |
| `packing_session.py` | batch, 실행 순서, pick/keep/pack 상태와 실제 완료된 canonical state 관리 |
| `packing_render.py` | 현재 계획을 3D view와 top view PNG로 렌더링하고 영상 생성 |
| `packing_viewer.py` | `packing_current.png`를 Tkinter 창에서 주기적으로 갱신 |

## 내부 설정값

아래 값들은 ROS 2 parameter가 아니라 각 Python 파일 상단의 소스 상수다. 값을 변경한 뒤 일반 빌드를 사용했다면 다시 빌드해야 한다. `colcon build --symlink-install`을 사용하면 Python 소스 변경은 대체로 바로 반영되지만, 새 터미널에서는 항상 `source install/setup.bash`가 필요하다.

### 컨테이너 설정 — `packing_data_class.py`

```python
DEFAULT_CONTAINER_PRESET = "2호"
```

| 프리셋 | X × Y × Z (mm) |
| --- | --- |
| `0호` | `170 × 130 × 90` |
| `1호` | `220 × 190 × 90` |
| `2호` | `270 × 180 × 150` |
| `2-1호` | `350 × 250 × 100` |

기본 컨테이너를 바꾸려면 `DEFAULT_CONTAINER_PRESET`을 프리셋 이름 중 하나로 변경한다.

### 배치 설정 — `packing_place.py`

| 상수 | 기본값 | 설명 |
| --- | ---: | --- |
| `GRID_UNIT_MM` | `10` | Heightmap 한 셀의 실제 길이. 작을수록 위치가 정밀하지만 탐색량 증가 |
| `BOX_SIZE_OFFSET_MM` | `0.0` | 측정 상자 크기에 추가하는 안전 여유 |
| `BOX_GAP_MM` | `5.0` | 배치된 상자 옆면 사이에 예약하는 간격 |
| `WALL_GAP_MM` | `0.0` | 컨테이너 벽과 상자 사이 최소 간격 |
| `MIN_SUPPORT` | `1.0` | 상자 바닥의 최소 지지 비율. `1.0`은 전체 면 지지 필요 |
| `GRIPPER_SIDE_MARGIN_MM` | `BOX_GAP_MM + GRID_UNIT_MM` | 그리퍼 접근 검사에 사용하는 상자 주변 범위 |
| `GRIPPER_HEIGHT_DIFFERENCE_MM` | `30.0` | 주변이 목표 윗면보다 이 값 이상 높으면 그리퍼 접근 불가 |

Boolean 토글:

| 토글 | 기본값 | `True` | `False` |
| --- | --- | --- | --- |
| `FLOOR_FIRST` | `False` | 바닥 후보가 하나라도 있으면 적층 후보를 버리고 바닥 후보만 탐색 | 바닥 후보를 우선 정렬하지만 적층 후보도 DFS에 전달 |
| `RESERVED_SUPPORT` | `True` | Heightmap이 예약한 셀 높이를 실제 착지 Z로 사용 | 이미 배치된 상자의 실제 윗면으로 착지 Z와 지지율을 다시 계산 |

`BOX_GAP_MM <= 0`이면 gap 영역을 만들지 않는다. `GRIPPER_SIDE_MARGIN_MM <= 0` 또는 `GRIPPER_HEIGHT_DIFFERENCE_MM <= 0`이면 그리퍼 주변 높이 검사를 통과 처리한다.

### 검색 설정 — `packing_search.py`

| 상수 | 기본값 | 설명 |
| --- | ---: | --- |
| `PLANNER_MODE` | `"REALTIME_BAF"` | 결과 메타데이터에 기록되는 planner 이름 |
| `GRACE_SECONDS` | `0.5` | 상자 3개의 첫 full plan을 찾은 후 추가 탐색 시간 |
| `HARD_TIMEOUT_SECONDS` | `2.0` | 한 batch DFS의 절대 최대 탐색 시간 |

계획 비교 우선순위는 다음과 같다.

1. 배치 가능한 상자 개수
2. 배치한 상자의 총부피
3. 남은 연속 평면 공간
4. 낮은 최대 높이
5. 낮은 Heightmap 거칠기
6. 동점이면 먼 X, 작은 Y 쪽 계획

### 렌더 및 뷰어 설정 — `packing_render.py`, `packing_viewer.py`

| 상수 | 기본값 | 설명 |
| --- | ---: | --- |
| `RENDER_ROOT` | `"render_output"` | 실행 위치 기준 출력 루트 |
| `FRAME_RATE` | `1` | MP4/WebM 생성 시 초당 프레임 수 |
| `SAVE_PNG` | `False` | 종료 후 개별 프레임과 `packing_current.png` 보존 여부 |
| `SAVE_MP4` | `False` | 종료 시 MP4 생성 여부 |
| `SAVE_WEBM` | `True` | 종료 시 WebM 생성 여부 |
| `SHOW_GUI` | `True` | Tkinter 실시간 뷰어 실행 여부 |
| `REFRESH_MS` | `200` | 뷰어 이미지 확인 주기(ms) |

Boolean 토글 동작:

- `SHOW_GUI=True`: `DISPLAY` 또는 `WAYLAND_DISPLAY`가 있을 때 별도 Tkinter 뷰어 프로세스를 실행한다.
- `SHOW_GUI=False`: 파일 렌더링은 수행하지만 GUI 창은 띄우지 않는다.
- `SAVE_PNG=True`: `frames/step_*.png`와 `packing_current.png`를 종료 후에도 남긴다.
- `SAVE_PNG=False`: 영상 생성 후 프레임 PNG와 현재 이미지를 삭제한다.
- `SAVE_MP4=True`: 노드 종료 시 ffmpeg로 `packing_plan.mp4`를 만든다.
- `SAVE_WEBM=True`: 노드 종료 시 ffmpeg로 `packing_plan.webm`을 만든다.

메인 노드는 위 기본값을 그대로 사용한다. 디버그 노드는 PNG, MP4, WebM 저장을 모두 강제로 활성화한다.

### 노드 설정

| 파일 | 상수 | 기본값 | 설명 |
| --- | --- | ---: | --- |
| `heuristic_main_node.py` | `WORK_QUEUE_SIZE` | `16` | ROS 콜백 이벤트 내부 queue 최대 크기 |
| `heuristic_debug_node.py` | `MAX_BATCHES` | `100` | 디버그 실행의 무한 재시도 방지 batch 상한 |

## 렌더 출력

기본 출력 구조:

```text
render_output/<YYYYMMDD_HHMMSS_microseconds>/
├── frames/
│   ├── step_001.png
│   └── ...
├── packing_current.png
├── packing_plan.mp4
├── packing_plan.webm
└── summary.json
```

실제 생성되는 파일은 실행 노드와 저장 토글에 따라 달라진다. 영상은 노드가 정상 종료될 때 생성되므로 터미널에서 `Ctrl+C`로 종료할 때까지 기다려야 한다.

## 주의사항

- `/vision/box_sizes`는 현재 정확히 3개짜리 JSON 배열만 허용한다.
- 새 batch는 이전 batch의 세 작업이 모두 완료된 이후에 보내야 한다.
- pick 작업은 `/control/pick_done`에서 place 좌표를 발행하지만, 내부 배치 상태 반영은 다음 `/main/next_box`에서 수행된다.
- GUI는 데스크톱 세션의 `DISPLAY` 또는 `WAYLAND_DISPLAY`가 필요하다.
- 사용자 pip Matplotlib와 Ubuntu `python3-matplotlib`이 섞이면 `Unknown projection '3d'`가 발생할 수 있다. 이 경우 `PYTHONNOUSERSITE=1` 환경에서 실행하면 시스템 패키지만 사용하도록 격리할 수 있다.
