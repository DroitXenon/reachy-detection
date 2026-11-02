"""Image-to-gaze geometry and simulator-only SDK ownership."""

import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from robot_control import RobotController, gaze_angles


def status(**changes):
    return dict(simulation_enabled=True, mockup_sim_enabled=False, state='running', **changes)


def response(value):
    return io.BytesIO(json.dumps(value).encode())


class GazeAnglesTests(unittest.TestCase):
    def test_centre_and_edges_use_bounded_yaw_pitch(self):
        self.assertEqual(gaze_angles((300, 220, 40, 40), (640, 480), False), (0.0, 0.0))
        self.assertEqual(gaze_angles((-10, -10, 20, 20), (640, 480), False), (30.0, -15.0))
        self.assertEqual(gaze_angles((630, 470, 20, 20), (640, 480), False), (-30.0, 15.0))
        self.assertEqual(gaze_angles((1000, -1000, 20, 20), (640, 480), False), (-30.0, -15.0))

    def test_mirror_changes_horizontal_direction_only(self):
        normal = gaze_angles((100, 300, 40, 40), (640, 480), False)
        mirror = gaze_angles((100, 300, 40, 40), (640, 480), True)
        self.assertEqual(mirror, (-normal[0], normal[1]))
        # A mirrored version of one physical box yields identical physical gaze.
        self.assertEqual(normal, gaze_angles((500, 300, 40, 40), (640, 480), True))

    def test_detection_box_and_xywh_sequence_are_equivalent(self):
        box = SimpleNamespace(origin_x=100, origin_y=120, width=40, height=50)
        self.assertEqual(gaze_angles(box, (640, 480)), gaze_angles((100, 120, 40, 50), (640, 480)))

    def test_invalid_boxes_and_image_sizes_are_rejected(self):
        cases = [((1, 2, 0, 4), (640, 480)), ((1, 2, 3, -1), (640, 480)),
                 ((1, 2, 3, 4), (0, 480)), ((1, 2, 3, 4), (640, float('inf'))),
                 ((1, float('nan'), 3, 4), (640, 480)), ((1, 2, 3), (640, 480)),
                 ((1, 2, 3, 4), (640,)), (None, (640, 480))]
        for box, size in cases:
            with self.subTest(box=box, size=size), self.assertRaises(ValueError):
                gaze_angles(box, size)


class RobotControllerTests(unittest.TestCase):
    def setUp(self):
        self.robot = MagicMock()
        self.robot.client.get_status.return_value = SimpleNamespace(**status())
        self.robot_class, self.create_pose = Mock(return_value=self.robot), Mock(return_value='pose')
        self.sdk = self.enterContext(patch('robot_control._sdk', return_value=(self.robot_class, self.create_pose)))
        self.http = self.enterContext(patch('robot_control.urlopen', side_effect=lambda *args, **kwargs: response(status())))
        self.controller = RobotController()
        self.addCleanup(self.controller.close)

    def test_connect_is_local_no_media_and_idempotent(self):
        self.controller.connect()
        self.controller.connect()
        self.robot_class.assert_called_once_with(connection_mode='localhost_only',
            media_backend='no_media', automatic_body_yaw=False, timeout=2.0, port=8000,
            robot_name='reachy_mini')
        self.http.assert_called_once()

    def test_custom_local_port_is_used_by_http_and_sdk(self):
        controller = RobotController(port=8765, robot_name='reachy_native_test')
        self.addCleanup(controller.close)
        controller.connect()
        self.http.assert_called_once_with('http://127.0.0.1:8765/api/daemon/status', timeout=0.75)
        self.assertEqual(self.robot_class.call_args.kwargs['port'], 8765)
        self.assertEqual(self.robot_class.call_args.kwargs['robot_name'], 'reachy_native_test')

    def test_http_gate_rejects_physical_mockup_and_stopped_daemons_before_sdk(self):
        for changes in ({'simulation_enabled': False}, {'simulation_enabled': None},
                        {'mockup_sim_enabled': True}, {'state': 'stopped'}):
            value = status(); value.update(changes)
            self.http.side_effect = lambda *args, value=value, **kwargs: response(value)
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, 'Full MuJoCo'):
                self.controller.connect()
        self.sdk.assert_not_called()

    def test_missing_daemon_is_actionable_and_does_not_load_sdk(self):
        self.http.side_effect = OSError('Connection refused')
        with self.assertRaisesRegex(RuntimeError, r'mjpython simulator\.py'):
            self.controller.connect()
        self.sdk.assert_not_called()

    def test_sdk_gate_rechecks_and_releases_rejected_connection(self):
        self.robot.client.get_status.return_value = SimpleNamespace(
            simulation_enabled=False, mockup_sim_enabled=False, state='running')
        with self.assertRaisesRegex(RuntimeError, 'full MuJoCo'):
            self.controller.connect()
        self.robot.__exit__.assert_called_once_with(None, None, None)
        self.assertIsNone(self.controller._robot)

    def test_look_at_sends_pose_then_close_releases_once(self):
        self.assertEqual(self.controller.look_at((300, 220, 40, 40), (640, 480)), (0.0, 0.0))
        self.create_pose.assert_called_once_with(yaw=0.0, pitch=0.0, degrees=True)
        self.robot.goto_target.assert_called_once_with(head='pose', duration=0.6, body_yaw=0.0)
        self.controller.close()
        self.controller.close()
        self.robot.__exit__.assert_called_once_with(None, None, None)

    def test_each_move_checks_http_without_rewaiting_for_sdk_status_event(self):
        self.controller.connect()
        self.robot.client.get_status.assert_called_once_with(timeout=2.0)
        self.robot.client.get_status.reset_mock()

        def cached_status(*, wait=True, timeout=5):
            if wait:
                raise TimeoutError('Next periodic status event has not arrived')
            return SimpleNamespace(**status())

        self.robot.client.get_status.side_effect = cached_status
        self.controller.look_at((300, 220, 40, 40), (640, 480))
        self.robot.client.get_status.assert_called_once_with(wait=False)
        self.assertEqual(self.http.call_count, 2)  # connect and a fresh movement preflight
        self.robot.goto_target.assert_called_once()

    def test_fresh_http_mode_change_blocks_motion_even_if_sdk_cache_is_simulation(self):
        self.controller.connect()
        value = status(); value['simulation_enabled'] = False
        self.http.side_effect = lambda *args, **kwargs: response(value)
        with self.assertRaisesRegex(RuntimeError, 'Full MuJoCo'):
            self.controller.look_at((300, 220, 40, 40), (640, 480))
        self.robot.goto_target.assert_not_called()
        self.robot.__exit__.assert_called_once_with(None, None, None)

    def test_failed_movement_clears_connection_for_retry(self):
        self.robot.goto_target.side_effect = TimeoutError('movement timed out')
        with self.assertRaisesRegex(RuntimeError, 'Reachy movement failed'):
            self.controller.look_at((300, 220, 40, 40), (640, 480))
        self.robot.__exit__.assert_called_once_with(None, None, None)
        self.assertIsNone(self.controller._robot)

    def test_daemon_mode_change_blocks_motion_and_closes_connection(self):
        self.controller.connect()
        self.robot.client.get_status.return_value = SimpleNamespace(
            simulation_enabled=False, mockup_sim_enabled=True, state='running')
        with self.assertRaisesRegex(RuntimeError, 'Full MuJoCo'):
            self.controller.look_at((300, 220, 40, 40), (640, 480))
        self.robot.goto_target.assert_not_called()
        self.robot.__exit__.assert_called_once_with(None, None, None)


if __name__ == '__main__':
    unittest.main()
