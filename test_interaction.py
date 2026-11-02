"""Observable action scheduling without a daemon, camera, or audio device."""

from threading import Event, Thread
import time
import unittest
from unittest.mock import Mock

from interaction import SelectionResponder
from target_selection import Selection
from test_target_selection import detection


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError('Timed out waiting for background actions')


class Robot:
    def __init__(self, gate=None, fail_connect=False):
        self.gate = gate
        self.fail_connect = fail_connect
        self.moves = []
        self.attempts = 0
        self.closed = 0
        self.started = Event()
        self.finished = Event()
        self.second_attempt = Event()
        self.second_move = Event()

    def connect(self):
        self.attempts += 1
        if self.attempts > 1:
            self.second_attempt.set()
        if self.fail_connect:
            raise RuntimeError('simulator absent')

    def look_at(self, box, image_size, mirror):
        self.moves.append((tuple(box), tuple(image_size), mirror))
        if len(self.moves) > 1:
            self.second_move.set()
        self.started.set()
        if self.gate is not None and len(self.moves) == 1 and not self.gate.wait(2):
            raise RuntimeError('test did not release movement')
        self.finished.set()
        return 10, -5

    def close(self):
        self.closed += 1


class Voice:
    def __init__(self, fail_first=False):
        self.texts = []
        self.attempts = 0
        self.closed = 0
        self.fail_first = fail_first
        self.repeated = Event()

    def speak(self, text):
        self.attempts += 1
        if self.fail_first and self.attempts == 1:
            raise RuntimeError('voice failed')
        self.texts.append(text)
        if len(self.texts) > 1:
            self.repeated.set()

    def close(self):
        self.closed += 1


class SelectionResponderTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.cup = detection(180, 150, 60, 60, 'cup')
        self.bottle = detection(360, 150, 60, 60, 'bottle')
        self.book = detection(480, 150, 60, 60, 'book')

    def responder(self, robot=None, voice=None, **options):
        robot = robot if robot is not None else Robot()
        voice = voice if voice is not None else Voice()
        return SelectionResponder(robot_factory=lambda: robot, speaker_factory=lambda: voice,
                                  clock=lambda: self.now, **options), robot, voice

    def select(self, responder, item=None, sequence=1):
        item = self.cup if item is None else item
        responder.update(Selection(0, 'Selected'), [item], (640, 480), True,
                         sequence, self.now)

    def clear(self, responder):
        responder.update(Selection(None, 'No target'), [], (640, 480), True, None, None)

    def test_disabled_actions_never_create_resources(self):
        robot, voice = Mock(), Mock()
        with SelectionResponder(False, False, robot_factory=robot, speaker_factory=voice) as responder:
            self.select(responder)
            responder.reset()
        robot.assert_not_called()
        voice.assert_not_called()

    def test_repeated_frames_jitter_and_brief_loss_do_not_repeat_name(self):
        responder, robot, voice = self.responder()
        with responder:
            self.select(responder)
            wait_until(lambda: voice.texts == ['cup'])
            for _ in range(10):
                self.select(responder)
            self.select(responder, detection(183, 151, 60, 60), sequence=2)
            self.clear(responder)
            self.now += 0.2
            self.select(responder, sequence=3)
            self.assertTrue(robot.second_move.wait(1))
            self.assertFalse(voice.repeated.wait(0.1))
            self.assertEqual(voice.texts, ['cup'])
        self.assertGreaterEqual(robot.closed, 1)
        self.assertGreaterEqual(voice.closed, 1)

    def test_long_loss_or_separate_same_label_object_starts_new_announcement(self):
        responder, _, voice = self.responder()
        with responder:
            self.select(responder)
            wait_until(lambda: voice.texts == ['cup'])
            self.select(responder, detection(440, 150, 60, 60), sequence=2)
            wait_until(lambda: voice.texts == ['cup', 'cup'])
            self.clear(responder)
            self.now += 0.8
            self.select(responder, detection(440, 150, 60, 60), sequence=3)
            wait_until(lambda: voice.texts == ['cup', 'cup', 'cup'])

    def test_updates_do_not_wait_for_movement_and_latest_target_replaces_pending(self):
        gate = Event()
        responder, robot, voice = self.responder(robot=Robot(gate))
        with responder:
            try:
                self.select(responder)
                self.assertTrue(robot.started.wait(1))
                updated = Event()
                def update():
                    self.select(responder, self.bottle, sequence=2)
                    self.select(responder, self.book, sequence=3)
                    updated.set()
                updater = Thread(target=update, daemon=True)
                updater.start()
                self.assertTrue(updated.wait(0.5), 'GUI update waited for an SDK movement')
            finally:
                gate.set()
            wait_until(lambda: voice.texts == ['book'])
            self.assertEqual([move[0][0] for move in robot.moves], [180, 480])
            self.assertEqual(robot.moves[-1][1:], ((640, 480), True))

    def test_selection_loss_cancels_pending_target_and_suppresses_old_speech(self):
        gate = Event()
        responder, robot, voice = self.responder(robot=Robot(gate))
        with responder:
            try:
                self.select(responder)
                self.assertTrue(robot.started.wait(1))
                self.select(responder, self.bottle, sequence=2)
                self.clear(responder)
            finally:
                gate.set()
            self.assertTrue(robot.finished.wait(1))
            self.assertFalse(robot.second_move.wait(0.1))
            self.assertEqual(voice.texts, [])

    def test_reset_suppresses_speech_after_inflight_movement(self):
        gate = Event()
        responder, robot, voice = self.responder(robot=Robot(gate))
        with responder:
            try:
                self.select(responder)
                self.assertTrue(robot.started.wait(1))
                responder.reset()
            finally:
                gate.set()
            self.assertTrue(robot.finished.wait(1))
        self.assertEqual(voice.texts, [])

    def test_invalid_and_expired_snapshots_never_trigger_actions(self):
        for selection, items, size, sequence, captured_at in (
                (Selection(None, 'Unclear'), [self.cup], (640, 480), 1, self.now),
                (Selection(0, 'Selected'), [self.cup], (640, 480), None, self.now),
                (Selection(0, 'Selected'), [self.cup], (640, 480), 1, self.now - 0.6),
                (Selection(0, 'Selected'), [self.cup], (640, 480), 1, float('nan')),
                (Selection(0, 'Selected'), [self.cup], (0, 480), 1, self.now),
                (Selection(-1, 'Selected'), [self.cup], (640, 480), 1, self.now),
                (Selection(2, 'Selected'), [self.cup], (640, 480), 1, self.now),
                (Selection(0, 'Selected'), [detection(100, 100, -1, 60)], (640, 480), 1, self.now),
        ):
            with self.subTest(index=selection.index, size=size, sequence=sequence,
                              captured_at=captured_at):
                responder, robot, voice = self.responder()
                with responder:
                    responder.update(selection, items, size, True, sequence, captured_at)
                    self.assertFalse(robot.started.wait(0.1))
                self.assertEqual(robot.moves, [])
                self.assertEqual(voice.texts, [])

    def test_robot_failure_allows_speech_and_retries_after_backoff(self):
        responder, robot, voice = self.responder(robot=Robot(fail_connect=True))
        with responder:
            self.select(responder)
            wait_until(lambda: voice.texts == ['cup'])
            self.assertIn('unavailable', responder.status)
            robot.fail_connect = False
            self.now += 0.2
            self.select(responder, sequence=2)
            self.assertFalse(robot.second_attempt.wait(0.1))
            self.now += 3
            self.select(responder, sequence=3)
            self.assertTrue(robot.started.wait(1))
            self.assertFalse(voice.repeated.wait(0.1))
        self.assertEqual(voice.texts, ['cup'])

    def test_speech_error_can_retry_without_crashing_preview(self):
        responder, _, voice = self.responder(voice=Voice(fail_first=True), robot_enabled=False)
        with responder:
            self.select(responder)
            wait_until(lambda: 'voice failed' in responder.status)
            self.now += 3.1
            self.select(responder, sequence=2)
            wait_until(lambda: voice.texts == ['cup'])
        self.assertEqual(voice.attempts, 2)


if __name__ == '__main__':
    unittest.main()
