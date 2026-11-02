"""Respond to confirmed targets without blocking camera capture or drawing."""

from dataclasses import dataclass
import logging
from math import hypot, isfinite
from threading import Condition, Thread
import time

from robot_control import RobotController
from speech import Speaker
from target_selection import box_iou, target_name


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetRequest:
    token: int
    episode: int
    name: str
    box: tuple
    image_size: tuple
    mirror: bool
    captured_at: float


def _bounds(box):
    x, y, width, height = box
    return x, y, x + width, y + height


def _centre(request):
    x, y, width, height = request.box
    return ((x + width / 2) / request.image_size[0],
            (y + height / 2) / request.image_size[1])


class SelectionResponder:
    """One action in progress and one replaceable pending target.

    Updates only copy small target records. The worker owns SDK calls and speech.
    Losing selection cancels queued actions and suppresses speech after an
    already-started movement. An in-progress head movement finishes normally.
    """

    def __init__(self, robot_enabled=True, speech_enabled=True, *,
                 robot_factory=None, speaker_factory=None, clock=time.monotonic):
        self.robot_enabled = robot_enabled
        self.speech_enabled = speech_enabled
        self._robot_factory = robot_factory or RobotController
        self._speaker_factory = speaker_factory or Speaker
        self._clock = clock
        self._condition = Condition()
        self._stopped = False
        self._desired = self._pending = self._last_command = None
        self._identity = None
        self._lost_at = None
        self._token = self._episode = 0
        self._announced_episode = None
        self._last_dispatch = float('-inf')
        self._retry_needed = False
        self._robot_status = 'waiting for selection' if robot_enabled else 'off'
        self._voice_status = 'starting' if speech_enabled else 'off'
        self._robot = self._speaker = self._thread = None

    def __enter__(self):
        if self.robot_enabled or self.speech_enabled:
            self._thread = Thread(target=self._run, name='reachy-actions', daemon=True)
            self._thread.start()
        return self

    @property
    def status(self):
        with self._condition:
            return f'Robot: {self._robot_status} | Voice: {self._voice_status}'

    def _clear(self, now, reset_identity=False):
        if self._desired is not None:
            self._token += 1
        self._desired = self._pending = self._last_command = None
        if self._lost_at is None:
            self._lost_at = now
        if reset_identity:
            self._identity = None

    def reset(self):
        """Invalidate pending actions after mirroring, calibration, or size changes."""
        with self._condition:
            self._clear(self._clock(), reset_identity=True)

    def update(self, selection, detections, image_size, mirror, sequence, captured_at):
        if not (self.robot_enabled or self.speech_enabled):
            return
        now = self._clock()
        try:
            index = selection.index
            if (not isinstance(index, int) or isinstance(index, bool)
                    or not 0 <= index < len(detections)
                    or sequence is None or captured_at is None):
                raise ValueError('No confirmed target')
            detection = detections[index]
            name = target_name(detection)
            box = detection.bounding_box
            coordinates = tuple(float(getattr(box, field)) for field in
                                ('origin_x', 'origin_y', 'width', 'height'))
            size = tuple(float(value) for value in image_size)
            if (not name or len(size) != 2 or min(*size, *coordinates[2:]) <= 0
                    or not all(isfinite(value) for value in (*size, *coordinates, captured_at))
                    or now - captured_at > 0.5):
                raise ValueError('Invalid or expired target')
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            with self._condition:
                self._clear(now)
            return

        with self._condition:
            if self._stopped:
                return
            same = (self._identity is not None and self._identity[0] == name
                    and box_iou(_bounds(coordinates), _bounds(self._identity[1])) >= 0.3)
            new_episode = not same or (self._lost_at is not None and now - self._lost_at >= 0.75)
            if new_episode:
                self._episode += 1
            if self._desired is None or new_episode:
                self._token += 1
            request = TargetRequest(self._token, self._episode, name, coordinates,
                                    size, bool(mirror), float(captured_at))
            self._desired = request
            self._identity, self._lost_at = (name, coordinates), None
            moved = self._last_command is None
            if not moved:
                previous, current = _centre(self._last_command), _centre(request)
                moved = (self._last_command.mirror != request.mirror
                         or self._last_command.image_size != request.image_size
                         or hypot(current[0] - previous[0], current[1] - previous[1]) >= 0.05)
            retry = self._retry_needed and now - self._last_dispatch >= 3.0
            if (new_episode or self._last_command is None or retry
                    or (moved and now - self._last_dispatch >= 0.6)):
                self._pending = request
                self._last_command, self._last_dispatch = request, now
                self._condition.notify()

    def _current(self, request):
        with self._condition:
            return (not self._stopped and self._desired is not None
                    and self._desired.token == request.token
                    and self._clock() - self._desired.captured_at <= 0.5)

    def _set_status(self, component, text):
        with self._condition:
            setattr(self, f'_{component}_status', text)

    def _run(self):
        try:
            if self.speech_enabled:
                try:
                    self._speaker = self._speaker_factory()
                    self._set_status('voice', 'ready')
                except Exception as error:
                    self._set_status('voice', f'unavailable ({error})')
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopped or self._pending is not None)
                    if self._stopped:
                        return
                    request, self._pending = self._pending, None
                if not self._current(request):
                    continue
                failed = False
                if self.robot_enabled:
                    try:
                        if self._robot is None:
                            self._robot = self._robot_factory()
                        self._set_status('robot', f'looking at {request.name}')
                        self._robot.connect()
                        if not self._current(request):
                            continue
                        yaw, pitch = self._robot.look_at(request.box, request.image_size, request.mirror)
                        self._set_status('robot', f'{request.name} (yaw {yaw:.0f}, pitch {pitch:.0f})')
                    except Exception as error:
                        failed = True
                        self._set_status('robot', f'unavailable ({error})')
                if (self._current(request) and self._speaker is not None
                        and self._announced_episode != request.episode):
                    try:
                        self._set_status('voice', f'saying {request.name}')
                        self._speaker.speak(request.name)
                        self._announced_episode = request.episode
                        self._set_status('voice', f'said {request.name}')
                    except Exception as error:
                        failed = True
                        self._set_status('voice', f'error ({error})')
                with self._condition:
                    self._retry_needed = failed
        finally:
            for resource in (self._speaker, self._robot):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception:
                        LOGGER.exception('Could not close action resource')

    def close(self):
        with self._condition:
            self._stopped = True
            self._desired = self._pending = None
            self._condition.notify()
        if self._speaker is not None:
            self._speaker.close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def __exit__(self, *exc):
        self.close()
