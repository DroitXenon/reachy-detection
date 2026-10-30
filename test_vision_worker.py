"""Deterministic scheduling checks without a camera or model."""

from contextlib import contextmanager
from threading import Event
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from vision_worker import LatestFrameWorker


def factory(function):
    @contextmanager
    def context():
        yield function
    return context


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError("Timed out waiting for worker")


class WorkerTests(unittest.TestCase):
    def test_pending_frames_are_replaced_and_input_is_copied(self):
        started, release = Event(), Event()
        processed = []
        def process(frame, sequence):
            if sequence == 1:
                started.set()
                if not release.wait(2):
                    raise RuntimeError("Test did not release worker")
            processed.append((sequence, int(frame[0])))
            return sequence
        with LatestFrameWorker(factory(process)) as worker:
            frame = np.array([7])
            worker.submit(frame, 1, time.monotonic(), 0)
            self.assertTrue(started.wait(2))
            frame[0] = 9
            worker.submit(frame, 2, time.monotonic(), 0)
            worker.submit(frame, 3, time.monotonic(), 0)
            release.set()
            wait_until(lambda: worker.latest(0, 5) is not None and worker.latest(0, 5).sequence == 3)
            self.assertEqual(processed, [(1, 7), (3, 9)])
        self.assertFalse(worker._thread.is_alive())

    def test_old_orientation_and_expired_frames_are_hidden(self):
        with LatestFrameWorker(factory(lambda frame, sequence: sequence)) as worker:
            worker.submit(np.zeros(1), 1, time.monotonic() - 1, 0)
            wait_until(lambda: worker.latest(0, 5) is not None)
            self.assertIsNone(worker.latest(0, 0.3))
            self.assertIsNone(worker.latest(1, 5))

    def test_errors_reach_the_preview(self):
        def fail(frame, sequence):
            raise ValueError("model unavailable")
        with LatestFrameWorker(factory(fail)) as worker:
            worker.submit(np.zeros(1), 1, time.monotonic(), 0)
            def failed():
                try:
                    worker.latest(0)
                except RuntimeError as error:
                    self.assertIn("model unavailable", str(error))
                    return True
                return False
            wait_until(failed)

    def test_gesture_confirmation_counts_inference_results_not_display_frames(self):
        import cv2
        from hand_tracking import run, MODEL_PATH
        from test_hand_tracking import hand_pixels, hand_pose
        from vision_worker import FrameResult

        pixels = hand_pixels()
        hand = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=x / 640, y=y / 480) for x, y in pixels]],
            hand_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in hand_pose()]],
            handedness=[[SimpleNamespace(category_name="Left")]],
        )
        class Worker:
            calls = 0
            def __init__(self, factory): pass
            def __enter__(self): return self
            def __exit__(self, *exc): pass
            def submit(self, *args): pass
            def latest(self, generation):
                sequence = [1, 1, 2, None][self.calls]
                self.calls += 1
                if sequence is None: return None
                now = time.monotonic()
                return FrameResult(sequence, generation, now, now, 10, (hand, []))
        class Camera:
            def isOpened(self): return True
            def set(self, *args): pass
            def read(self): return True, np.zeros((480, 640, 3), np.uint8)
            def release(self): pass
        statuses = []
        original = cv2.putText
        def draw(image, text, origin, *args):
            if origin == (12, 27): statuses.append(text)
            return original(image, text, origin, *args)
        with patch('hand_tracking.LatestFrameWorker', Worker), \
             patch('hand_tracking.cv2.VideoCapture', return_value=Camera()), \
             patch('hand_tracking.cv2.namedWindow'), patch('hand_tracking.cv2.imshow'), \
             patch('hand_tracking.cv2.destroyAllWindows'), \
             patch('hand_tracking.cv2.putText', side_effect=draw), \
             patch('hand_tracking.cv2.getWindowProperty', return_value=1), \
             patch('hand_tracking.cv2.waitKey', side_effect=[-1, -1, -1, ord('q')]):
            run(0, MODEL_PATH, False)
        self.assertEqual(statuses, ['Hold pointing gesture...', 'Hold pointing gesture...',
                                    'POINTING', 'Waiting for fresh detection...'])


if __name__ == "__main__":
    unittest.main()
