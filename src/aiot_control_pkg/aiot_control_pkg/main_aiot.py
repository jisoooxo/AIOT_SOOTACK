#!/usr/bin/env python3

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int8, Bool, String


# ============================================================
# State
# ============================================================

WAIT_UI_START = 'WAIT_UI_START'
WAIT_SETTING_DONE = 'WAIT_SETTING_DONE'

WAIT_BELT_STOP_DONE = 'WAIT_BELT_STOP_DONE'
WAIT_BOX_SIZES = 'WAIT_BOX_SIZES'
WAIT_PLAN_PICK = 'WAIT_PLAN_PICK'

WAIT_PLACE_DONE = 'WAIT_PLACE_DONE'
WAIT_KEEP_SET_DONE = 'WAIT_KEEP_SET_DONE'
WAIT_KEEP_DONE = 'WAIT_KEEP_DONE'

WAIT_RESET = 'WAIT_RESET'


class MainNode(Node):

    def __init__(self):
        super().__init__('main_node')

        self.state = WAIT_UI_START

        self.count = 0
        self.keep_count = 0

        # 첫 사이클에는 keep 영역이 비어 있으므로 바로 keep 가능
        self.keep_set_done = True

        # keep_set_done을 기다리는 동안 저장할 index
        self.pending_keep_index = None


        # SETTING: Main -> Main2
        self.setting_start_pub = self.create_publisher(Bool, '/main/setting_start', 10) ## 박스 3개 세팅해라.
        self.keep_set_pub = self.create_publisher(String, '/main/keep_set', 10) ## keep 몇 개 했는지 알려줄게

        # TRIGGER: Main -> Vision / Belt
        self.vision_start_pub = self.create_publisher(Bool, '/main/vision_start', 10) ## 비전 디택 시작해라.
        self.belt_start_pub = self.create_publisher(Bool, '/main/belt_start', 10) ## 벨트 움직여라.

        # STACKING(pick, keep): Main -> Vision
        self.box_ready_pub = self.create_publisher(Bool, '/main/box_ready', 10) ## 박스 3개 디택해라.
        self.plan_pick_pub = self.create_publisher(String, '/main/plan_pick', 10) ## pick이 왔을 때
        self.keep_ready_pub = self.create_publisher(Int8, '/main/keep_ready', 10) ## keep이 왔을 때

        # STACKING(place): Main -> Heuristic
        self.box_sizes_pub = self.create_publisher(String, '/main/box_sizes', 10) ## 박스 3개 정보 넘기기
        self.next_box_pub = self.create_publisher(Bool, '/main/next_box', 10) ## 다음으로 잡아야 하는 박스 정보 달라.

        # UI -> Main
        self.create_subscription(Bool, '/ui/start', self.ui_start_callback, 10) ## MAIN 시작 ~

        # Main2 -> Main
        self.create_subscription(Bool, '/main2/setting_done', self.setting_done_callback, 10) ## 박스 3개 세팅 끝났어 ~
        self.create_subscription(Bool, '/main2/keep_set_done', self.keep_set_done_callback, 10) ## keep 공간에 있는 박스 다 치웠어 ~

        # Belt -> Main
        self.create_subscription(Bool, '/belt/stop_done', self.belt_stop_done_callback, 10) ## 벨트 멈추기 완료

        # Vision -> Main
        self.create_subscription(String, '/vision/box_sizes', self.box_sizes_callback, 10) ## 비전이 준 박스 3개 정보

        # Heuristic -> Main
        self.create_subscription(String, '/heuristic/plan_pick', self.plan_pick_callback, 10) ## 휴리스틱이 주는 잡아야 하는 박스 정보

        # Control -> Main
        self.create_subscription(Bool, '/control/place_done', self.place_done_callback, 10) ## 박스 하나 넣었어 ~
        self.create_subscription(Bool, '/control/keep_done', self.keep_done_callback, 10) ## 박스 하나 킵했어 ~

        self.get_logger().info('Main node ready')


    def publish_true(self, publisher):

        msg = Bool()
        msg.data = True

        publisher.publish(msg)

    def ui_start_callback(self, msg):

        if self.state != WAIT_UI_START:
            return

        if not msg.data:
            return

        # 첫 사이클에서는 Main2 setting 바로 시작
        self.state = WAIT_SETTING_DONE

        self.publish_true(self.setting_start_pub)

    def setting_done_callback(self, msg):

        if self.state != WAIT_SETTING_DONE:
            return

        if not msg.data:
            return

        self.start_trigger()

    def start_trigger(self):

        self.count = 0
        self.keep_count = 0

        self.publish_true(self.vision_start_pub)
        self.publish_true(self.belt_start_pub)

        self.state = WAIT_BELT_STOP_DONE


    def belt_stop_done_callback(self, msg):

        if self.state != WAIT_BELT_STOP_DONE:
            return

        if not msg.data:
            return

        self.publish_true(self.box_ready_pub)

        self.state = WAIT_BOX_SIZES

    def box_sizes_callback(self, msg):
        if self.state != WAIT_BOX_SIZES:
            return

        try:
            json.loads(msg.data)

        except json.JSONDecodeError as exc:
            self.get_logger().error(
                f'/vision/box_sizes JSON 파싱 실패: {exc}'
            )
            return

        self.state = WAIT_PLAN_PICK
        self.box_sizes_pub.publish(msg)
        
    def plan_pick_callback(self, msg): ## 휴리스틱이 알려주는 pick 해야하는 인덱스

        if self.state != WAIT_PLAN_PICK:
            return

        data = json.loads(msg.data)

        index = int(data['idx'])
        face = data['face']
        axis = data['axis']
        command = str(data['status']).strip().lower()

        ## command에 따라 분기 !!
        if command == 'pick':

            plan_msg = String()

            plan_msg.data = json.dumps({
                'idx': index,
                'face': face,
                'axis': axis
            })

            self.plan_pick_pub.publish(plan_msg)

            self.state = WAIT_PLACE_DONE

        elif command == 'keep':

            self.pending_keep_index = index

            if self.keep_set_done:
                self.publish_keep_ready()

            else:
                self.state = WAIT_KEEP_SET_DONE

        elif command == 'reset':

            self.state = WAIT_RESET

            self.get_logger().warning(
                'reset 로직은 아직 구현되지 않았습니다.'
            )  ## 상자 바꾸는거랑 찐리셋이랑 구분 일단 안하는 걸로 짤게용

        else:
            return

    def publish_keep_ready(self):

        if self.pending_keep_index is None:
            return

        msg = Int8()
        msg.data = int(self.pending_keep_index)

        self.keep_ready_pub.publish(msg)

        self.pending_keep_index = None
        self.state = WAIT_KEEP_DONE

    def keep_set_done_callback(self, msg):

        if not msg.data:
            return

        # Main2가 keep 영역의 box 세팅을 모두 완료
        self.keep_set_done = True

        ## keep 해야하는게 있더라도 keep_set_done이 되면 진행
        if self.state == WAIT_KEEP_SET_DONE:
            self.publish_keep_ready()

    def place_done_callback(self, msg):

        if self.state != WAIT_PLACE_DONE:
            return

        if not msg.data:
            return

        self.count += 1

        self.finish_box()

    def keep_done_callback(self, msg):

        if self.state != WAIT_KEEP_DONE:
            return

        if not msg.data:
            return

        self.count += 1
        self.keep_count += 1

        self.finish_box()

    def finish_box(self):

        # 아직 3개 처리 전 -> 휴리스틱한테 인덱스 받는 곳으로 돌아감
        if self.count < 3:

            self.state = WAIT_PLAN_PICK

            self.publish_true(self.next_box_pub)

            return

        # 3개 처리 완료
        self.finish_cycle()

    def finish_cycle(self):

        # 다음 keep 동작은 Main2의 keep_set_done을 받을 때까지 금지
        self.keep_set_done = False

        # 다음 Trigger는 Main2의 setting_done을 받을 때까지 금지
        self.state = WAIT_SETTING_DONE

        # 이번 사이클 keep 정보 전달
        keep_msg = String()

        if self.keep_count > 0:
            keep_msg.data = json.dumps({
                'keep': True,
                'keep_count': self.keep_count
            })
        else:
            keep_msg.data = json.dumps({
                'keep': False
            })

        self.keep_set_pub.publish(keep_msg)

        # Main2 setting 시작
        self.publish_true(self.setting_start_pub)


def main(args=None):

    rclpy.init(args=args)

    node = MainNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()