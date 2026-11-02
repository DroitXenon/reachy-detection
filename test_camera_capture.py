"""Camera scheduling checks without a physical device or GUI."""

from collections import deque
from threading import Condition, Event, Thread
import time
import unittest

import cv2
import numpy as np

from camera_capture import CameraError, LatestCamera


class Camera:
    def __init__(self, *, opened=True, reported_fps=30):
        self.opened = opened
        self.reported_fps = reported_fps
        self.settings = []
        self.released = 0
        self.reading = Event()
        self._condition = Condition()
        self._frames = deque()

    def isOpened(self):
        return self.opened

    def set(self, property_id, value):
        self.settings.append((property_id, value))

    def get(self, property_id):
        return self.reported_fps

    def push(self, frame):
        with self._condition:
            self._frames.append(frame)
            self._condition.notify()

    def read(self):
        self.reading.set()
        with self._condition:
            self._condition.wait_for(lambda: self._frames or self.released)
            if self.released:
                return False, None
            frame = self._frames.popleft()
            return frame is not None, frame

    def release(self):
        with self._condition:
            self.released += 1
            self._condition.notify_all()


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.002)
    raise AssertionError('Timed out waiting for capture thread')


class LatestCameraTests(unittest.TestCase):
    def test_requests_sixty_fps_but_reports_the_actual_device_setting(self):
        camera = Camera(reported_fps=30)
        camera.push(np.zeros((10, 20, 3), np.uint8))
        with LatestCamera(3, camera_factory=lambda index: camera) as capture:
            first = capture.wait_first()
            self.assertIn((cv2.CAP_PROP_FRAME_WIDTH, 960), camera.settings)
            self.assertIn((cv2.CAP_PROP_FRAME_HEIGHT, 540), camera.settings)
            self.assertIn((cv2.CAP_PROP_FPS, 60), camera.settings)
            self.assertIn((cv2.CAP_PROP_BUFFERSIZE, 1), camera.settings)
            self.assertEqual(capture.actual_fps, 30)
            self.assertEqual(first.sequence, 1)
            self.assertGreater(first.captured_at, 0)
        self.assertEqual(camera.released, 1)

    def test_latest_replaces_frames_and_returns_none_for_unchanged_identity(self):
        camera = Camera()
        source = np.full((10, 20, 3), 7, np.uint8)
        camera.push(source)
        with LatestCamera(0, camera_factory=lambda index: camera) as capture:
            first = capture.wait_first()
            self.assertIsNone(capture.latest(after_sequence=first.sequence))
            source[:] = 9
            camera.push(source)
            camera.push(np.full_like(source, 11))
            wait_until(lambda: capture.latest().sequence == 3)
            self.assertEqual(int(capture.latest().frame[0, 0, 0]), 11)
            self.assertEqual(int(first.frame[0, 0, 0]), 7)
            self.assertFalse(first.frame.flags.writeable)

    def test_camera_read_does_not_block_latest_or_drawing_thread(self):
        camera = Camera()
        with LatestCamera(0, camera_factory=lambda index: camera) as capture:
            self.assertTrue(camera.reading.wait(1))
            done = Event()
            result = []
            def fetch():
                result.append(capture.latest())
                done.set()
            Thread(target=fetch, daemon=True).start()
            self.assertTrue(done.wait(.3), 'Preview waited for camera.read()')
            self.assertEqual(result, [None])
        self.assertFalse(capture._thread.is_alive())

    def test_large_frames_are_resized_once_before_publication(self):
        camera = Camera()
        camera.push(np.zeros((1080, 1920, 3), np.uint8))
        with LatestCamera(0, camera_factory=lambda index: camera) as capture:
            self.assertEqual(capture.wait_first().frame.shape, (540, 960, 3))

    def test_open_failure_reaches_first_frame_wait_and_latest(self):
        camera = Camera(opened=False)
        with LatestCamera(0, camera_factory=lambda index: camera) as capture:
            with self.assertRaisesRegex(CameraError, 'Cannot open webcam'):
                capture.wait_first()
            with self.assertRaisesRegex(CameraError, 'Cannot open webcam'):
                capture.latest()
        self.assertEqual(camera.released, 1)

    def test_read_failure_is_not_reported_as_a_frozen_old_frame(self):
        camera = Camera()
        camera.push(np.zeros((10, 20, 3), np.uint8))
        with LatestCamera(0, camera_factory=lambda index: camera) as capture:
            capture.wait_first()
            camera.push(None)
            def failed():
                try:
                    capture.latest()
                except CameraError:
                    return True
                return False
            wait_until(failed)
            with self.assertRaisesRegex(CameraError, 'stopped returning frames'):
                capture.latest()
        self.assertEqual(camera.released, 1)

    def test_wait_timeout_and_close_interrupt_an_empty_camera(self):
        camera = Camera()
        capture = LatestCamera(0, camera_factory=lambda index: camera)
        with capture:
            with self.assertRaisesRegex(CameraError, 'within'):
                capture.wait_first(timeout=.01)
        capture.close()
        self.assertFalse(capture._thread.is_alive())
        self.assertEqual(camera.released, 1)

    def test_unknown_device_fps_is_zero(self):
        camera = Camera(reported_fps=float('nan'))
        camera.push(np.zeros((10, 20, 3), np.uint8))
        with LatestCamera(0, camera_factory=lambda index: camera) as capture:
            capture.wait_first()
            self.assertEqual(capture.actual_fps, 0)

    def test_backend_fps_query_failure_does_not_stop_capture(self):
        camera = Camera()
        def unsupported(property_id):
            raise OSError('Frame rate query is unsupported')
        camera.get = unsupported
        camera.push(np.zeros((10, 20, 3), np.uint8))
        with LatestCamera(0, camera_factory=lambda index: camera) as capture:
            self.assertIsNotNone(capture.wait_first())
            self.assertEqual(capture.actual_fps, 0)


if __name__ == '__main__':
    unittest.main()
