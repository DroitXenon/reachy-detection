"""Exercise selection, switching, ambiguity, and clearing through the preview."""

from pathlib import Path
from contextlib import nullcontext
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from hand_tracking import MODEL_PATH
from main import run
from test_hand_tracking import hand_pose
from test_target_selection import detection
from test_preview_support import PreviewWindowStub
from vision_worker import FrameResult
from camera_capture import CameraFrame


def hand_result(rotation=False, pointing=True):
    world = hand_pose() if pointing else hand_pose(())
    if rotation:
        world = world @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pixels = world[:, :2] * 30 + [200, 100]
    return SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=x / 640, y=y / 480) for x, y in pixels]],
        hand_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in world]],
        handedness=[[SimpleNamespace(category_name='Left')]],
    )


def render_scenario(**run_options):
    run_options.setdefault('pace', False)
    cup = detection(180, 300, 60, 60, 'cup')
    bottle = detection(330, 80, 80, 80, 'bottle')
    overlap = detection(190, 310, 60, 60, 'bottle')
    down, right, relaxed = hand_result(), hand_result(True), hand_result(pointing=False)
    steps = [(1, down, [cup, bottle]), (1, down, [cup, bottle]),
             (2, down, [cup, bottle]), (2, down, [cup, bottle]),
             (3, down, [cup, bottle]), (3, down, [cup, bottle]),
             (4, right, [cup, bottle]), (4, right, [cup, bottle]),
             (5, right, [cup, bottle]), (6, relaxed, [cup, bottle]),
             None, (7, down, [cup, overlap]), (8, down, [cup, overlap])]
    class Worker:
        calls = 0
        def __init__(self, factory): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, *args): pass
        def latest(self, generation, max_age=0.3):
            step = steps[self.calls]
            self.calls += 1
            if step is None: return None
            sequence, hand, objects = step
            now = time.monotonic()
            return FrameResult(sequence, generation, now, now, 10, (hand, objects))
    class Camera:
        actual_fps = 60
        sequence = 0
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def wait_first(self, *args, **kwargs):
            return CameraFrame(np.zeros((480, 640, 3), np.uint8), 0, time.monotonic())
        def latest(self):
            self.sequence += 1
            return CameraFrame(np.zeros((480, 640, 3), np.uint8), self.sequence,
                               time.monotonic())
    statuses, images = [], []
    original = cv2.putText
    def put(image, text, origin, *args):
        if origin == (12, 53): statuses.append(text)
        return original(image, text, origin, *args)
    with patch('main.LatestFrameWorker', Worker), \
         patch('main.LatestCamera', Camera), \
         patch('main.PreviewWindow', PreviewWindowStub), patch('main.cv2.destroyAllWindows'), \
         patch('main.cv2.imshow', side_effect=lambda title, image: images.append(image.copy())), \
         patch('main.cv2.putText', side_effect=put), \
         patch('main.cv2.getWindowProperty', return_value=1), \
         patch('main.cv2.waitKey', side_effect=[-1] * (len(steps) - 1) + [ord('q')]):
        run(0, MODEL_PATH, False, object_model=Path('models/yolo26s.onnx'), **run_options)
    return statuses, images


class TargetPreviewTests(unittest.TestCase):
    def test_preview_passes_confirmed_targets_and_snapshot_freshness_to_actions(self):
        responder = Mock()
        responder.status = 'Robot: ready | Voice: ready'
        with patch('main.SelectionResponder', return_value=nullcontext(responder)) as factory:
            statuses, _ = render_scenario(robot_enabled=True, speech_enabled=True)
        factory.assert_called_once_with(True, True)
        requests = responder.update.call_args_list
        self.assertEqual(len(requests), len(statuses))
        for status, request in zip(statuses, requests):
            selection, detections, size, mirror, sequence, captured_at = request.args
            self.assertEqual(selection.index is not None, status.startswith('Selected:'))
            self.assertEqual(size, (640, 480))
            self.assertFalse(mirror)
            if selection.index is not None:
                category = detections[selection.index].categories[0].category_name
                self.assertEqual(status, f'Selected: {category}')
                self.assertIsNotNone(sequence)
                self.assertGreater(captured_at, 0)
        self.assertIsNone(requests[10].args[4])  # Stale snapshots cannot authorize actions.
        self.assertIsNone(requests[10].args[5])
        responder.reset.assert_called()  # Camera-size initialization clears queued work.

    def test_switching_uses_confirmed_fresh_data_and_clears_immediately(self):
        statuses, images = render_scenario()
        self.assertEqual(statuses[:2], ['Hold target steady: cup'] * 2)
        # Immediate pointing still needs two distinct object-selection results.
        self.assertEqual(statuses[2:6], ['Selected: cup'] * 4)
        self.assertIn('Hold target steady: bottle', statuses[6])
        self.assertIn('Hold target steady: bottle', statuses[7])
        self.assertEqual(statuses[8], 'Selected: bottle')
        self.assertNotIn('Selected:', statuses[9])  # Relaxed hand clears.
        self.assertNotIn('Selected:', statuses[10])  # Stale/missing snapshot clears.
        self.assertIn('Unclear', statuses[11])
        self.assertIn('Unclear', statuses[12])
        self.assertTrue(any(np.all(images[4] == (0, 255, 80), axis=-1).ravel()))


if __name__ == '__main__':
    unittest.main()
