"""Gesture confirmation without camera access or native model inference."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from gesture_recog import PointingRecognizer
from test_hand_tracking import hand_pixels, hand_pose


class PointingRecognizerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.recognizer = PointingRecognizer(Path(temporary.name) / 'pointing.json')

    def update(self, sequence, label='Left', pixels=None, world=None):
        return self.recognizer.update(
            [hand_pose() if world is None else world],
            [hand_pixels() if pixels is None else pixels], [label], sequence, 640, 480,
        )

    def test_only_distinct_results_confirm_and_duplicates_retain_confirmation(self):
        self.assertFalse(self.update(1).stable)
        self.assertFalse(self.update(1).stable)
        self.assertTrue(self.update(2).stable)
        self.assertTrue(self.update(2).stable)

    def test_changed_handedness_or_distant_wrist_needs_confirmation_again(self):
        self.update(1)
        self.assertTrue(self.update(2).stable)
        self.assertFalse(self.update(3, label='Right').stable)
        self.assertTrue(self.update(4, label='Right').stable)
        moved = hand_pixels() + [200, 0]
        self.assertFalse(self.update(5, label='Right', pixels=moved).stable)
        self.assertTrue(self.update(6, label='Right', pixels=moved).stable)

    def test_stale_result_clears_confirmation_and_requires_new_results(self):
        self.update(1)
        self.assertTrue(self.update(2).stable)
        stale = self.recognizer.update([], [], [], None, 640, 480)
        self.assertFalse(stale.stable)
        self.assertIsNone(stale.index)
        # Redisplaying the old inference cannot count as a new confirmation.
        self.assertFalse(self.update(2).stable)
        self.assertFalse(self.update(3).stable)
        self.assertTrue(self.update(4).stable)

    def test_reset_does_not_recount_the_current_inference(self):
        self.update(1)
        self.assertTrue(self.update(2).stable)
        self.recognizer.reset_confirmation()
        self.assertFalse(self.update(2).stable)
        self.assertFalse(self.update(3).stable)
        self.assertTrue(self.update(4).stable)

    def test_rejected_or_ambiguous_pose_clears_confirmation(self):
        self.update(1)
        self.assertTrue(self.update(2).stable)
        self.assertFalse(self.update(3, world=hand_pose(())).stable)
        self.assertFalse(self.update(4).stable)
        self.assertTrue(self.update(5).stable)
        ambiguous = self.recognizer.update(
            [hand_pose(), hand_pose()], [hand_pixels(), hand_pixels() + [200, 0]],
            ['Left', 'Right'], 6, 640, 480,
        )
        self.assertFalse(ambiguous.stable)
        self.assertIsNone(ambiguous.index)
        self.assertIn('two hands', ambiguous.status)

    def test_calibration_counts_only_new_results_and_never_confirms_during_capture(self):
        self.recognizer.pose.start()
        self.recognizer.reset_confirmation()
        for sequence in (1, 1, 2, 2):
            result = self.update(sequence)
            self.assertFalse(result.stable)
            self.assertIn('CALIBRATING', result.status)
        self.assertEqual(len(self.recognizer.pose.samples), 2)
        np.testing.assert_allclose(self.recognizer.pose.capture_wrist,
                                   hand_pixels()[0] / [640, 480])

    def test_tracking_loss_cancels_calibration(self):
        self.recognizer.pose.start()
        self.update(1)
        self.recognizer.update([], [], [], None, 640, 480)
        self.assertFalse(self.recognizer.pose.capturing)
        self.assertIn('tracking lost', self.recognizer.pose.notice)

    def test_missing_hand_labels_or_invalid_pixels_clear_without_crashing(self):
        for pixels, labels in (([hand_pixels()], []), ([np.zeros((3, 2))], ['Left'])):
            result = self.recognizer.update([hand_pose()], pixels, labels, 1, 640, 480)
            self.assertFalse(result.stable)
            self.assertIsNone(result.index)


if __name__ == '__main__':
    unittest.main()
