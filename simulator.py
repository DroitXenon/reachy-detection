"""Run full MuJoCo without the unused virtual eye-camera stream.

Start with ``mjpython simulator.py`` on macOS. The visible viewer, physics,
head control, and SDK server are unchanged. This project uses an external camera,
so the separate virtual-camera rendering/UDP loop is disabled by default.
Use ``--virtual-camera`` to restore the SDK's virtual-camera rendering.
Other arguments are passed through to the Reachy Mini daemon.
"""

import argparse
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import sys


SUPPORTED_SDK_VERSION = '1.8.0'


def _load_sdk():
    """Load the SDK only when starting the simulator, not when importing this file."""
    daemon = import_module('reachy_mini.daemon.daemon')
    app = import_module('reachy_mini.daemon.app.main')
    return daemon, app.main


def _check_sdk_version():
    try:
        installed = version('reachy-mini')
    except PackageNotFoundError as error:
        raise RuntimeError('Reachy Mini is not installed. Activate the project Conda environment.') from error
    if installed != SUPPORTED_SDK_VERSION:
        raise RuntimeError(
            f'This simulator launcher supports reachy-mini {SUPPORTED_SDK_VERSION}; '
            f'found {installed}. Install the pinned requirements before using it.'
        )


def _without_virtual_camera(base):
    if not callable(getattr(base, 'rendering_loop', None)):
        raise RuntimeError('Unsupported MuJoCo backend: virtual-camera rendering hook is missing')

    class ExternalCameraMujocoBackend(base):
        """Keep the inherited viewer and physics; skip the unused eye stream."""

        def rendering_loop(self, camera_name, port):
            self.logger.info(
                'Virtual eye-camera rendering disabled; using the external camera. '
                'Add --virtual-camera to restore the SDK stream.'
            )

    return ExternalCameraMujocoBackend


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--virtual-camera', action='store_true')
    options, forwarded = parser.parse_known_args(arguments)
    hardware_flags = {'--mockup-sim', '--no-sim', '--wireless-version',
                      '--serialport', '--hardware-config-filepath'}
    for argument in forwarded:
        flag = argument.split('=', 1)[0]
        short_serialport = argument.startswith('-p') and not argument.startswith('--')
        if flag in hardware_flags or short_serialport:
            parser.error(f'{flag} is incompatible with this full MuJoCo simulator launcher')
    _check_sdk_version()
    daemon, run_daemon = _load_sdk()
    original_backend = daemon.MujocoBackend
    original_argv = sys.argv
    try:
        if not options.virtual_camera:
            daemon.MujocoBackend = _without_virtual_camera(original_backend)
        # Explicitly select full simulation and leave the external camera/audio
        # with this project's preview and speech processes.
        sys.argv = [original_argv[0], '--sim', '--no-media', *forwarded]
        if '--help' in forwarded or '-h' in forwarded:
            print(__doc__)
            print('\nDaemon options:')
        return run_daemon()
    finally:
        daemon.MujocoBackend = original_backend
        sys.argv = original_argv


if __name__ == '__main__':
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(f'Error: {error}') from error
