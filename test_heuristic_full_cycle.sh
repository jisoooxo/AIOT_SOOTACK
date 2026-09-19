#!/usr/bin/env bash
set -Eeo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${WORKSPACE_DIR}/heuristic_full_cycle.log"

# ~/.local의 pip Matplotlib과 Ubuntu의 mpl_toolkits가 섞이는 것을 방지한다.
export PYTHONNOUSERSITE=1

source /opt/ros/humble/setup.bash

if [[ ! -f "${WORKSPACE_DIR}/install/setup.bash" ]]; then
    echo "[ERROR] install/setup.bash가 없습니다. 먼저 워크스페이스를 빌드하세요." >&2
    exit 1
fi

source "${WORKSPACE_DIR}/install/setup.bash"
set -u
cd "${WORKSPACE_DIR}"

ros2 run aiot_heuristic_pkg heuristic_main_node >"${LOG_FILE}" 2>&1 &
HEURISTIC_PID=$!

cleanup() {
    if kill -0 "${HEURISTIC_PID}" 2>/dev/null; then
        kill -INT "${HEURISTIC_PID}" 2>/dev/null || true

        for _ in {1..30}; do
            if ! kill -0 "${HEURISTIC_PID}" 2>/dev/null; then
                wait "${HEURISTIC_PID}" 2>/dev/null || true
                return
            fi
            sleep 0.1
        done

        kill -TERM "${HEURISTIC_PID}" 2>/dev/null || true
        wait "${HEURISTIC_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

python3 - <<'PY'
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String


BOXES = [
    {"idx": 1, "x": 0.158, "y": 0.030, "z": 0.030},
    {"idx": 2, "x": 0.142, "y": 0.036, "z": 0.032},
    {"idx": 3, "x": 0.152, "y": 0.036, "z": 0.036},
    {"idx": 4, "x": 0.100, "y": 0.050, "z": 0.049},
    {"idx": 5, "x": 0.120, "y": 0.066, "z": 0.023},
    {"idx": 6, "x": 0.127, "y": 0.020, "z": 0.015},
    {"idx": 7, "x": 0.090, "y": 0.055, "z": 0.090},
    {"idx": 8, "x": 0.155, "y": 0.038, "z": 0.025},
    {"idx": 9, "x": 0.073, "y": 0.020, "z": 0.048},
    {"idx": 10, "x": 0.103, "y": 0.020, "z": 0.038},
    {"idx": 11, "x": 0.066, "y": 0.020, "z": 0.105},
    {"idx": 12, "x": 0.090, "y": 0.050, "z": 0.090},
    {"idx": 13, "x": 0.065, "y": 0.055, "z": 0.130},
    {"idx": 14, "x": 0.051, "y": 0.050, "z": 0.200},
    {"idx": 15, "x": 0.041, "y": 0.040, "z": 0.100},
    {"idx": 16, "x": 0.050, "y": 0.035, "z": 0.150},
    {"idx": 17, "x": 0.035, "y": 0.022, "z": 0.105},
    {"idx": 18, "x": 0.045, "y": 0.045, "z": 0.091},
]


class FullCycleTester(Node):
    def __init__(self):
        super().__init__("heuristic_full_cycle_tester")
        self.plan_picks = []
        self.plan_places = []
        self.reset_done_count = 0

        self.box_sizes_pub = self.create_publisher(String, "/main/box_sizes", 10)
        self.next_box_pub = self.create_publisher(Bool, "/main/next_box", 10)
        self.pick_done_pub = self.create_publisher(Bool, "/control/pick_done", 10)
        self.pack_reset_pub = self.create_publisher(Bool, "/main/pack_reset", 10)

        self.create_subscription(String, "/heuristic/plan_pick", self.on_plan_pick, 10)
        self.create_subscription(String, "/heuristic/plan_place", self.on_plan_place, 10)
        self.create_subscription(Bool, "/heuristic/pack_reset_done", self.on_reset_done, 10)

    def on_plan_pick(self, message):
        self.plan_picks.append(json.loads(message.data))

    def on_plan_place(self, message):
        self.plan_places.append(json.loads(message.data))

    def on_reset_done(self, message):
        if message.data:
            self.reset_done_count += 1

    def publish_bool(self, publisher):
        message = Bool()
        message.data = True
        publisher.publish(message)

    def spin_for(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, deadline - time.monotonic()))

    def wait_until(self, condition, description, timeout=30.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError(f"timeout: {description}")

    def wait_for_heuristic(self):
        self.wait_until(
            lambda: self.box_sizes_pub.get_subscription_count() > 0
            and self.next_box_pub.get_subscription_count() > 0
            and self.pick_done_pub.get_subscription_count() > 0,
            "휴리스틱 노드 연결",
        )

    def run_batch(self, batch_number, boxes):
        while True:
            print(f"\n[BATCH {batch_number}/6] input={[box['idx'] for box in boxes]}", flush=True)
            message = String()
            message.data = json.dumps(boxes, separators=(",", ":"))
            self.box_sizes_pub.publish(message)
            self.spin_for(0.5)

            completed = 0
            retry_after_reset = False

            while completed < 3:
                previous_pick_count = len(self.plan_picks)
                self.publish_bool(self.next_box_pub)
                self.wait_until(
                    lambda: len(self.plan_picks) > previous_pick_count,
                    f"batch {batch_number} plan_pick",
                )

                task = self.plan_picks[-1]
                status = str(task["status"]).lower()
                print(f"  plan_pick #{completed + 1}: {task}", flush=True)

                if status == "pack":
                    previous_reset_count = self.reset_done_count
                    self.publish_bool(self.pack_reset_pub)
                    self.wait_until(
                        lambda: self.reset_done_count > previous_reset_count,
                        "pack_reset_done",
                    )
                    print("  container reset -> 같은 batch 재시도", flush=True)
                    retry_after_reset = True
                    break

                if status == "pick":
                    previous_place_count = len(self.plan_places)
                    self.publish_bool(self.pick_done_pub)
                    self.wait_until(
                        lambda: len(self.plan_places) > previous_place_count,
                        f"box {task['idx']} plan_place",
                    )
                    print(f"  plan_place: {self.plan_places[-1]}", flush=True)
                elif status != "keep":
                    raise RuntimeError(f"알 수 없는 status: {status}")

                completed += 1

            if retry_after_reset:
                self.spin_for(0.5)
                continue

            # 마지막 작업을 canonical state에 반영하고 batch를 닫는다.
            self.publish_bool(self.next_box_pub)
            self.spin_for(0.5)
            return


def main():
    rclpy.init()
    node = FullCycleTester()

    try:
        node.wait_for_heuristic()
        print("[READY] 휴리스틱 연결 완료. GUI 뷰어와 18개 테스트를 시작합니다.", flush=True)

        for offset in range(0, len(BOXES), 3):
            node.run_batch(offset // 3 + 1, BOXES[offset:offset + 3])

        print(
            f"\n[PASS] 입력 18개 완료 | plan_pick={len(node.plan_picks)} "
            f"| plan_place={len(node.plan_places)} | reset={node.reset_done_count}",
            flush=True,
        )
    except Exception as error:
        print(f"\n[FAIL] {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
PY

LATEST_RENDER="$(find "${WORKSPACE_DIR}/render_output" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"

if [[ -z "${LATEST_RENDER}" || ! -s "${LATEST_RENDER}/packing_current.png" ]]; then
    echo "[FAIL] packing_current.png가 생성되지 않았습니다." >&2
    exit 1
fi

if grep -qE 'worker 실패|Traceback|Unknown projection' "${LOG_FILE}"; then
    echo "[FAIL] 휴리스틱 로그에서 오류를 발견했습니다: ${LOG_FILE}" >&2
    exit 1
fi

echo
echo "휴리스틱 로그: ${LOG_FILE}"
echo "렌더 결과: ${LATEST_RENDER}/packing_current.png"
