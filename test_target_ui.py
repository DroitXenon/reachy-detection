"""Exercise selection, switching, ambiguity, and clearing through the preview."""

from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from hand_tracking import MODEL_PATH
from main import run
from test_hand_tracking import hand_pose
from test_target_selection import detection
from vision_worker import FrameResult


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


def render_scenario():
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
        def latest(self, generation):
            step = steps[self.calls]
            self.calls += 1
            if step is None: return None
            sequence, hand, objects = step
            now = time.monotonic()
            return FrameResult(sequence, generation, now, now, 10, (hand, objects))
    class Camera:
        def isOpened(self): return True
        def set(self, *args): pass
        def read(self): return True, np.zeros((480, 640, 3), np.uint8)
        def release(self): pass
    statuses, images = [], []
    original = cv2.putText
    def put(image, text, origin, *args):
        if origin == (12, 53): statuses.append(text)
        return original(image, text, origin, *args)
    with tempfile.TemporaryDirectory() as temporary, \
         patch('main.LatestFrameWorker', Worker), \
         patch('main.cv2.VideoCapture', return_value=Camera()), \
         patch('main.cv2.namedWindow'), patch('main.cv2.destroyAllWindows'), \
         patch('main.cv2.imshow', side_effect=lambda title, image: images.append(image.copy())), \
         patch('main.cv2.putText', side_effect=put), \
         patch('main.cv2.getWindowProperty', return_value=1), \
         patch('main.cv2.waitKey', side_effect=[-1] * (len(steps) - 1) + [ord('q')]):
        run(0, MODEL_PATH, False, object_model=Path('models/yolo26s.onnx'),
            calibration_path=Path(temporary) / 'profile.json')
    return statuses, images


class TargetPreviewTests(unittest.TestCase):
    def test_switching_uses_confirmed_fresh_data_and_clears_immediately(self):
        statuses, images = render_scenario()
        self.assertNotIn('Selected:', statuses[0])  # Amber preview cannot select.
        self.assertIn('Hold target steady', statuses[2])
        self.assertIn('Hold target steady', statuses[3])  # Repeated inference result.
        self.assertEqual(statuses[4:6], ['Selected: cup'] * 2)
        self.assertIn('Hold target steady: bottle', statuses[6])
        self.assertIn('Hold target steady: bottle', statuses[7])
        self.assertEqual(statuses[8], 'Selected: bottle')
        self.assertNotIn('Selected:', statuses[9])  # Relaxed hand clears.
        self.assertNotIn('Selected:', statuses[10])  # Stale/missing snapshot clears.
        self.assertIn('Unclear', statuses[12])
        self.assertTrue(any(np.all(images[4] == (0, 255, 80), axis=-1).ravel()))


if __name__ == '__main__':
    unittest.main()
