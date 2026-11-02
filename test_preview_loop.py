"""Preview redraws and held visuals cannot manufacture fresh inference results."""

from contextlib import ExitStack, nullcontext
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from camera_capture import CameraFrame
from hand_tracking import MODEL_PATH
from main import run
from preview_tracking import PreviewTracker
from test_hand_tracking import hand_pixels, hand_pose
from test_target_selection import detection
from test_preview_support import PreviewWindowStub
from vision_worker import FrameResult


def pointing_hand():
    pixels = hand_pixels()
    return SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=x / 640, y=y / 480) for x, y in pixels]],
        hand_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in hand_pose()]],
        handedness=[[SimpleNamespace(category_name='Left')]],
    )


class PreviewLoopTests(unittest.TestCase):
    def run_loop(self, camera_sequences, *, expired_last=False):
        base = time.monotonic()
        clock = [base]
        frame = np.zeros((480, 640, 3), np.uint8)
        cup = detection(180, 300, 60, 60, 'cup')
        hand = pointing_hand()
        submissions, statuses, draws = [], [], []
        responder = Mock()
        responder.status = 'Robot: ready | Voice: ready'

        class Camera:
            actual_fps = 60
            calls = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def wait_first(self): return CameraFrame(frame, 0, base)
            def latest(self):
                index = self.calls
                self.calls += 1
                clock[0] = base + (0.4 if expired_last and index == len(camera_sequences) - 1 else index * .02)
                return CameraFrame(frame, camera_sequences[index], clock[0])

        class Worker:
            current = None
            def __init__(self, factory): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def submit(self, frame, sequence, captured_at, generation):
                submissions.append((sequence, captured_at))
                if expired_last and len(submissions) == len(camera_sequences):
                    return  # Model is still processing the last camera frame.
                self.current = FrameResult(sequence, generation, captured_at, clock[0], 10,
                                           (hand, [cup]))
            def latest(self, generation, max_age=.3):
                return self.current

        original_text, original_boxes = cv2.putText, __import__('main').draw_objects
        def text(image, value, origin, *args):
            if origin == (12, 27): statuses.append(value)
            return original_text(image, value, origin, *args)
        def boxes(image, detections, selected_index=None):
            draws.append((list(detections), selected_index))
            return original_boxes(image, detections, selected_index)

        with ExitStack() as stack:
            camera_factory = stack.enter_context(patch('main.LatestCamera', return_value=Camera()))
            stack.enter_context(patch('main.LatestFrameWorker', Worker))
            stack.enter_context(patch('main.SelectionResponder', return_value=nullcontext(responder)))
            stack.enter_context(patch('main.PreviewTracker', return_value=PreviewTracker(clock=lambda: clock[0])))
            stack.enter_context(patch('main.time.monotonic', side_effect=lambda: clock[0]))
            stack.enter_context(patch('main.PreviewWindow', PreviewWindowStub))
            stack.enter_context(patch('main.cv2.imshow'))
            stack.enter_context(patch('main.cv2.destroyAllWindows'))
            stack.enter_context(patch('main.cv2.getWindowProperty', return_value=1))
            stack.enter_context(patch('main.cv2.waitKey', side_effect=[-1] * (len(camera_sequences) - 1) + [ord('q')]))
            stack.enter_context(patch('main.cv2.putText', side_effect=text))
            stack.enter_context(patch('main.draw_objects', side_effect=boxes))
            run(0, MODEL_PATH, False, object_model=Path('models/yolo26s.onnx'),
                robot_enabled=True, speech_enabled=True, target_fps=60, pace=False)
        camera_factory.assert_called_once_with(0, target_fps=60)
        return submissions, statuses, draws, responder

    def test_repeated_redraws_do_not_resubmit_or_confirm_a_target_twice(self):
        submissions, statuses, _, responder = self.run_loop([1, 1, 1, 2, 2, 3, 3])
        self.assertEqual([item[0] for item in submissions], [1, 2, 3])
        self.assertEqual(statuses, ['POINTING'] * 7)
        targets = [call.args[0] for call in responder.update.call_args_list]
        self.assertTrue(all(target.index is None for target in targets[:3]))
        self.assertEqual([target.index for target in targets[3:]], [0, 0, 0, 0])

    def test_held_visual_survives_short_latency_but_cannot_authorize_actions(self):
        _, statuses, draws, responder = self.run_loop([1, 2, 3, 4], expired_last=True)
        requests = responder.update.call_args_list
        self.assertEqual(requests[2].args[0].index, 0)
        self.assertIsNone(requests[3].args[0].index)
        self.assertIsNone(requests[3].args[4])
        self.assertIsNone(requests[3].args[5])
        self.assertTrue(draws[-1][0], 'Brief inference latency should retain the visual box')
        self.assertIsNone(draws[-1][1], 'Held visual must not retain a selected highlight')
        self.assertNotEqual(statuses[-1], 'POINTING')

    def test_collapsed_image_index_cannot_create_pointing_or_actions(self):
        hand = pointing_hand()
        for landmark in hand.hand_landmarks[0][5:9]:
            landmark.x = landmark.y = .5
        with patch(__name__ + '.pointing_hand', return_value=hand), \
             patch('main.cv2.arrowedLine') as arrow:
            _, statuses, _, responder = self.run_loop([1, 2, 3])
        self.assertTrue(all(not status.startswith('POINTING') for status in statuses))
        self.assertTrue(all(call.args[0].index is None
                            for call in responder.update.call_args_list))
        arrow.assert_not_called()

    def test_image_pointing_does_not_require_world_landmarks(self):
        hand = pointing_hand()
        hand.hand_world_landmarks = []
        with patch(__name__ + '.pointing_hand', return_value=hand):
            _, statuses, _, responder = self.run_loop([1, 2])
        self.assertEqual(statuses, ['POINTING', 'POINTING'])
        requests = responder.update.call_args_list
        self.assertIsNone(requests[0].args[0].index)
        self.assertEqual(requests[1].args[0].index, 0)


if __name__ == '__main__':
    unittest.main()
