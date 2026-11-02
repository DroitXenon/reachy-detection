"""The project launcher preserves full simulation and avoids unused rendering."""

from contextlib import redirect_stderr
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import simulator


class Backend:
    run = Mock(name='inherited_viewer_and_physics')
    rendering_loop = Mock(name='sdk_virtual_camera')


class SimulatorLauncherTests(unittest.TestCase):
    def setUp(self):
        self.daemon = SimpleNamespace(MujocoBackend=Backend)
        self.run = Mock()
        self.version = self.enterContext(patch('simulator.version', return_value='1.8.0'))
        self.loader = self.enterContext(patch('simulator._load_sdk', return_value=(self.daemon, self.run)))
        self.original_argv = sys.argv

    def test_default_disables_only_virtual_camera_and_forces_full_sim_no_media(self):
        observed = {}
        def run():
            observed['backend'] = self.daemon.MujocoBackend
            observed['argv'] = sys.argv.copy()
            backend = self.daemon.MujocoBackend.__new__(self.daemon.MujocoBackend)
            backend.logger = Mock()
            backend._get_renderer = Mock(side_effect=AssertionError('Unused camera renderer created'))
            backend.rendering_loop('eye_camera', 5005)
            backend._get_renderer.assert_not_called()
            backend.logger.info.assert_called_once()
        self.run.side_effect = run
        simulator.main([])
        self.assertTrue(issubclass(observed['backend'], Backend))
        self.assertIs(observed['backend'].run, Backend.run)
        self.assertIsNot(observed['backend'].rendering_loop, Backend.rendering_loop)
        self.assertEqual(observed['argv'], [self.original_argv[0], '--sim', '--no-media'])
        self.assertIs(self.daemon.MujocoBackend, Backend)
        self.assertIs(sys.argv, self.original_argv)

    def test_sdk_options_are_passed_through(self):
        arguments = ['--headless', '--fastapi-port', '8765', '--robot-name', 'isolated',
                     '--dataset-update-interval', '0']
        observed = []
        self.run.side_effect = lambda: observed.extend(sys.argv[1:])
        simulator.main(arguments)
        self.assertEqual(observed, ['--sim', '--no-media', *arguments])

    def test_virtual_camera_keeps_stock_backend_and_consumes_wrapper_flag(self):
        def run():
            self.assertIs(self.daemon.MujocoBackend, Backend)
            self.assertEqual(sys.argv[1:], ['--sim', '--no-media', '--headless'])
        self.run.side_effect = run
        simulator.main(['--virtual-camera', '--headless'])

    def test_backend_and_arguments_restore_even_when_daemon_exits(self):
        for error in (SystemExit(2), RuntimeError('daemon failed'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                self.run.side_effect = error
                with self.assertRaises(type(error)):
                    simulator.main([])
                self.assertIs(self.daemon.MujocoBackend, Backend)
                self.assertIs(sys.argv, self.original_argv)

    def test_version_mismatch_does_not_load_or_start_sdk(self):
        self.version.return_value = '1.9.0'
        with self.assertRaisesRegex(RuntimeError, 'supports reachy-mini 1.8.0'):
            simulator.main([])
        self.loader.assert_not_called()
        self.run.assert_not_called()

    def test_missing_render_hook_is_rejected_before_start(self):
        original = self.daemon.MujocoBackend = type('ChangedBackend', (), {})
        with self.assertRaisesRegex(RuntimeError, 'rendering hook is missing'):
            simulator.main([])
        self.run.assert_not_called()
        self.assertIs(self.daemon.MujocoBackend, original)
        self.assertIs(sys.argv, self.original_argv)

    def test_hardware_and_mockup_arguments_are_rejected(self):
        for arguments in (['--mockup-sim'], ['--serialport', '/dev/example'],
                          ['--serialport=/dev/example'], ['-p/dev/example'],
                          ['--wireless-version'], ['--no-sim'],
                          ['--hardware-config-filepath', 'hardware.yaml']):
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    simulator.main(arguments)
                self.assertEqual(error.exception.code, 2)
        self.loader.assert_not_called()
        self.run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
