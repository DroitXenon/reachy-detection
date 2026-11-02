"""Keep only the newest camera frame without blocking preview drawing."""

from dataclasses import dataclass
from math import isfinite
from threading import Condition, Event, Lock, Thread
import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    """Camera startup or capture failed."""


@dataclass(frozen=True)
class CameraFrame:
    """One owned, read-only BGR image and its capture identity."""

    frame: np.ndarray
    sequence: int
    captured_at: float


class LatestCamera:
    """Read on a background thread and replace the latest frame, never queue it.

    ``actual_fps`` is the device-reported frame rate, which may differ from both
    the requested rate and the measured rate. Callers copy ``CameraFrame.frame``
    before mirroring or drawing. A slow read never blocks ``latest``.
    """

    def __init__(self, camera_index, target_fps=60, max_width=960, *, camera_factory=None):
        if not isfinite(target_fps) or not 1 <= target_fps <= 240:
            raise ValueError('Camera target FPS must be between 1 and 240')
        if isinstance(max_width, bool) or not isinstance(max_width, int) or max_width < 1:
            raise ValueError('Camera maximum width must be a positive integer')
        self.camera_index = camera_index
        self.target_fps = float(target_fps)
        self.max_width = max_width
        self._factory = camera_factory or cv2.VideoCapture
        self._condition = Condition()
        self._release_lock = Lock()
        self._stop = Event()
        self._thread = None
        self._camera = None
        self._released = False
        self._result = None
        self._error = None
        self._actual_fps = 0.0

    def __enter__(self):
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError('A camera capture instance can only be started once')
        self._thread = Thread(target=self._run, name='camera-capture', daemon=True)
        self._thread.start()
        return self

    @property
    def actual_fps(self):
        with self._condition:
            return self._actual_fps

    def _raise_error(self):
        if self._error is not None:
            raise CameraError(str(self._error)) from self._error

    def latest(self, after_sequence=None):
        """Return immediately with the latest frame, or None if it is unchanged."""
        with self._condition:
            self._raise_error()
            result = self._result
        if result is None or (after_sequence is not None and result.sequence <= after_sequence):
            return None
        return result

    def wait_first(self, timeout=5):
        """Wait at startup for the first frame; ongoing preview uses ``latest``."""
        with self._condition:
            available = self._condition.wait_for(
                lambda: self._result is not None or self._error is not None or self._stop.is_set(),
                timeout=timeout,
            )
            self._raise_error()
            if self._result is not None:
                return self._result
            if self._stop.is_set():
                raise CameraError('Camera closed before the first frame arrived')
            if not available:
                raise CameraError(f'Camera did not return a frame within {timeout:g} seconds')

    def _release_camera(self):
        # release() also interrupts backends whose next read is waiting for a frame.
        with self._release_lock:
            camera = self._camera
            if camera is None or self._released:
                return
            self._released = True
        camera.release()

    def _run(self):
        try:
            camera = self._factory(self.camera_index)
            with self._release_lock:
                self._camera = camera
            if self._stop.is_set():
                return
            if not camera.isOpened():
                raise CameraError(
                    'Cannot open webcam. Allow camera access for your terminal in macOS '
                    'System Settings > Privacy & Security > Camera, close other camera '
                    'apps, or select another --camera index.'
                )
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, max(1, round(self.max_width * 9 / 16)))
            camera.set(cv2.CAP_PROP_FPS, self.target_fps)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                reported = float(camera.get(cv2.CAP_PROP_FPS))
            except (AttributeError, TypeError, ValueError, OverflowError, OSError, cv2.error):
                reported = 0.0
            with self._condition:
                self._actual_fps = reported if isfinite(reported) and reported > 0 else 0.0
            sequence = 0
            while not self._stop.is_set():
                success, frame = camera.read()
                captured_at = time.monotonic()
                if self._stop.is_set():
                    break
                if not success or frame is None:
                    raise CameraError('Webcam stopped returning frames')
                if frame.shape[1] > self.max_width:
                    height = max(1, round(frame.shape[0] * self.max_width / frame.shape[1]))
                    frame = cv2.resize(frame, (self.max_width, height), interpolation=cv2.INTER_AREA)
                else:
                    # Own the data even if a capture backend reuses its buffer.
                    frame = frame.copy()
                frame.setflags(write=False)
                sequence += 1
                with self._condition:
                    self._result = CameraFrame(frame, sequence, captured_at)
                    self._condition.notify_all()
        except Exception as error:
            if not self._stop.is_set():
                with self._condition:
                    self._error = error
                    self._condition.notify_all()
        finally:
            self._release_camera()

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._release_camera()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def __exit__(self, *exc):
        self.close()
