"""Bounded head movement toward image targets in the local MuJoCo simulator.

This is an image-relative demonstration, not camera-to-robot 3D calibration.
The camera should face the scene from approximately Reachy's neutral direction.
"""

import json
from math import isfinite
from urllib.request import urlopen


SIMULATOR_COMMAND = 'mjpython simulator.py'
MAX_YAW = 30.0
MAX_PITCH = 15.0


def gaze_angles(box, image_size, mirror=True):
    """Map a bounding-box centre to (yaw, pitch) degrees, undoing preview mirroring.

    ``box`` may have origin_x/origin_y/width/height attributes or be an xywh
    sequence. Image coordinates increase right and down. In Reachy's frame,
    positive yaw turns left and positive pitch turns down. Targets outside the
    image are clamped to +/-30 degrees yaw and +/-15 degrees pitch.
    """
    try:
        width, height = (float(value) for value in image_size)
        if hasattr(box, 'origin_x'):
            x, y, box_width, box_height = (
                float(getattr(box, field))
                for field in ('origin_x', 'origin_y', 'width', 'height')
            )
        else:
            x, y, box_width, box_height = (float(value) for value in box)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('Target needs a finite xywh box and image width/height') from error
    values = (width, height, x, y, box_width, box_height)
    if not all(isfinite(value) for value in values) or min(width, height, box_width, box_height) <= 0:
        raise ValueError('Target needs positive image/box sizes and finite coordinates')
    centre_x = x + box_width / 2
    centre_y = y + box_height / 2
    horizontal = max(-1.0, min(1.0, 2 * centre_x / width - 1))
    vertical = max(-1.0, min(1.0, 2 * centre_y / height - 1))
    if mirror:
        horizontal = -horizontal
    return -horizontal * MAX_YAW, vertical * MAX_PITCH


def _require_simulation(status):
    def field(name):
        return status.get(name) if isinstance(status, dict) else getattr(status, name, None)

    state = field('state')
    state = getattr(state, 'value', state)
    if (field('simulation_enabled') is not True
            or field('mockup_sim_enabled') is True or state != 'running'):
        raise RuntimeError(f'Full MuJoCo simulator is not running. Start: {SIMULATOR_COMMAND}')


def _sdk():
    # Camera/vision-only runs do not import or initialize the robot SDK.
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
    return ReachyMini, create_head_pose


def _close_robot(robot):
    try:
        robot.__exit__(None, None, None)
    except Exception:
        # Release the connection even if the SDK's media cleanup fails.
        try:
            robot.client.disconnect()
        except Exception:
            pass


class RobotController:
    """Synchronous SDK owner intended for a single action worker thread.

    Connection or movement errors are reported to the caller. No daemon is
    automatically started, stopped, or switched from another mode.
    """

    def __init__(self, port=8000, robot_name='reachy_mini'):
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError('Simulator port must be an integer from 1 to 65535')
        if not isinstance(robot_name, str) or not robot_name.strip():
            raise ValueError('Simulator robot name must be a nonempty string')
        self.port = port
        self.robot_name = robot_name
        self._robot = None
        self._create_pose = None

    def _check_simulator(self):
        try:
            status_url = f'http://127.0.0.1:{self.port}/api/daemon/status'
            with urlopen(status_url, timeout=0.75) as response:
                status = json.load(response)
        except Exception as error:
            raise RuntimeError(f'Cannot reach local simulator. Start: {SIMULATOR_COMMAND}') from error
        _require_simulation(status)

    def connect(self):
        if self._robot is not None:
            return
        self._check_simulator()
        robot = None
        try:
            robot_class, create_pose = _sdk()
            robot = robot_class(
                connection_mode='localhost_only', media_backend='no_media',
                automatic_body_yaw=False, timeout=2.0, port=self.port,
                robot_name=self.robot_name,
            )
            # Check again after connecting so the HTTP preflight is not the only gate.
            _require_simulation(robot.client.get_status(timeout=2.0))
        except Exception as error:
            if robot is not None:
                _close_robot(robot)
            raise RuntimeError(f'Cannot connect to full MuJoCo simulator: {error}') from error
        self._robot, self._create_pose = robot, create_pose

    def look_at(self, box, image_size, mirror=True):
        yaw, pitch = gaze_angles(box, image_size, mirror)
        self.connect()
        try:
            self._check_simulator()
            # SDK status events publish once per second and get_status clears
            # the event. Rewaiting per move races the next publication. HTTP
            # above checks fresh state; inspect the already-received SDK value.
            _require_simulation(self._robot.client.get_status(wait=False))
            pose = self._create_pose(yaw=yaw, pitch=pitch, degrees=True)
            self._robot.goto_target(head=pose, duration=0.6, body_yaw=0.0)
        except Exception as error:
            self.close()
            raise RuntimeError(f'Reachy movement failed: {error}') from error
        return yaw, pitch

    def close(self):
        robot, self._robot = self._robot, None
        self._create_pose = None
        if robot is not None:
            _close_robot(robot)
