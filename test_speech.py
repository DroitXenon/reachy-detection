"""Speech stays local, treats names as data, and closes bounded processes."""

from threading import Event, Thread
import subprocess
import unittest
from unittest.mock import Mock, patch

from speech import Speaker


class SpeechTests(unittest.TestCase):
    def speaker(self):
        with patch('speech.shutil.which', side_effect=lambda name: '/usr/bin/say' if name == 'say' else None):
            return Speaker()

    def process(self):
        process = Mock()
        process.communicate.return_value = ('', '')
        process.returncode = 0
        process.poll.return_value = 0
        return process

    def test_model_label_is_stdin_data_and_never_shell_or_options(self):
        speaker, process = self.speaker(), self.process()
        with patch('speech.subprocess.Popen', return_value=process) as popen:
            speaker.speak('--output-file=/tmp/wrong; $(echo bad)')
        command = popen.call_args.args[0]
        self.assertEqual(command, ('/usr/bin/say', '-f', '-'))
        self.assertNotIn('shell', popen.call_args.kwargs)
        process.communicate.assert_called_once_with(input='--output-file=/tmp/wrong; $(echo bad)', timeout=10)
        self.assertIsNone(speaker._process)

    def test_linux_fallback_and_unavailable_backend(self):
        with patch('speech.shutil.which', side_effect=lambda name: '/usr/bin/espeak-ng' if name == 'espeak-ng' else None):
            speaker = Speaker()
        self.assertEqual(speaker._command, ('/usr/bin/espeak-ng', '--stdin'))
        with patch('speech.shutil.which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Speech unavailable'):
                Speaker()

    def test_invalid_or_empty_labels_do_not_start_a_process(self):
        speaker = self.speaker()
        with patch('speech.subprocess.Popen') as popen:
            for label in (None, '', '   ', 'cup\nfile', '\x00', 'x' * 201):
                with self.subTest(label=label), self.assertRaises(ValueError):
                    speaker.speak(label)
        popen.assert_not_called()

    def test_start_and_exit_failures_are_reported(self):
        speaker, process = self.speaker(), self.process()
        with patch('speech.subprocess.Popen', side_effect=OSError('not executable')):
            with self.assertRaisesRegex(RuntimeError, 'could not start'):
                speaker.speak('cup')
        process.returncode = 1
        process.communicate.return_value = ('', 'No audio device')
        with patch('speech.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'No audio device'):
                speaker.speak('cup')
        self.assertIsNone(speaker._process)

    def test_timeout_terminates_and_escalates_to_kill(self):
        speaker, process = self.speaker(), self.process()
        process.communicate.side_effect = subprocess.TimeoutExpired('say', 10)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('say', 0.5), 0]
        with patch('speech.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                speaker.speak('cup')
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertIsNone(speaker._process)

    def test_close_interrupts_active_speech_and_rejects_future_labels(self):
        speaker, process = self.speaker(), self.process()
        started, released = Event(), Event()
        process.poll.return_value = None
        process.returncode = -15

        def communicate(**arguments):
            started.set()
            self.assertTrue(released.wait(2))
            return '', ''

        process.communicate.side_effect = communicate
        process.terminate.side_effect = released.set
        errors = []

        def speak():
            try:
                speaker.speak('bottle')
            except Exception as error:
                errors.append(error)

        with patch('speech.subprocess.Popen', return_value=process):
            thread = Thread(target=speak)
            thread.start()
            self.assertTrue(started.wait(2))
            speaker.close()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            speaker.close()
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                speaker.speak('cup')
        self.assertEqual(errors, [])
        process.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
