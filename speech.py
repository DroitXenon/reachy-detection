"""Offline spoken labels using the operating system's speech command.

``Speaker.speak`` blocks until one short label finishes. Call it from the
interaction worker, never from the camera/preview thread.
"""

import shutil
import subprocess
from threading import Lock


class Speaker:
    """Speak through macOS ``say`` or an installed Linux eSpeak command."""

    def __init__(self):
        self._command = None
        for name, arguments in (
            ('say', ('-f', '-')),
            ('espeak-ng', ('--stdin',)),
            ('espeak', ('--stdin',)),
        ):
            executable = shutil.which(name)
            if executable is not None:
                self.backend = name
                self._command = (executable, *arguments)
                break
        if self._command is None:
            raise RuntimeError('Speech unavailable: no say, espeak-ng, or espeak command found')
        self._state = Lock()
        self._speaking = Lock()
        self._process = None
        self._closed = False

    def speak(self, text):
        """Speak one label, sending it as data rather than shell arguments."""
        if (not isinstance(text, str) or not text.strip() or len(text) > 200
                or any(ord(character) < 32 for character in text)):
            raise ValueError('Speech text must be a nonempty, single-line label of at most 200 characters')
        with self._speaking:
            with self._state:
                if self._closed:
                    raise RuntimeError('Speaker is closed')
                try:
                    process = subprocess.Popen(
                        self._command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE, text=True,
                    )
                except OSError as error:
                    raise RuntimeError(f'Speech could not start: {error}') from error
                self._process = process
            try:
                _, error_output = process.communicate(input=text.strip(), timeout=10)
            except subprocess.TimeoutExpired as error:
                self._stop(process)
                raise RuntimeError('Speech timed out after 10 seconds') from error
            except (OSError, ValueError) as error:
                self._stop(process)
                with self._state:
                    cancelled = self._closed
                if not cancelled:
                    raise RuntimeError(f'Speech failed: {error}') from error
                return
            finally:
                with self._state:
                    self._process = None
                    cancelled = self._closed
            if process.returncode != 0 and not cancelled:
                detail = (error_output or '').strip()[:200]
                raise RuntimeError(f'Speech failed ({self.backend}): {detail or process.returncode}')

    @staticmethod
    def _stop(process):
        """Bound shutdown even if a backend stops responding."""
        try:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass

    def close(self):
        """Stop active speech; may be called from the preview thread."""
        with self._state:
            self._closed = True
            process = self._process
        if process is not None:
            self._stop(process)
