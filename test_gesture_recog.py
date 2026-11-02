"""Immediate pointing from 2D finger extension, with source-identity resets."""

import unittest

import numpy as np

from gesture_recog import PointingRecognizer
from test_hand_tracking import hand_pixels, hand_pose, index_pixels


class PointingRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = PointingRecognizer()

    def update(self, sequence, pixels=None):
        return self.recognizer.update(
            [index_pixels() if pixels is None else pixels], sequence,
        )

    def test_first_fresh_extended_finger_is_pointing_immediately(self):
        result = self.update(1)
        self.assertTrue(result.stable)
        self.assertEqual(result.index, 0)
        self.assertEqual(result.status, 'POINTING')
        np.testing.assert_allclose(result.direction, [0, 1])

    def test_duplicate_source_can_redraw_without_an_additional_confirmation(self):
        first = self.update(1)
        duplicate = self.update(1)
        self.assertTrue(first.stable)
        self.assertTrue(duplicate.stable)
        self.assertEqual(first.index, duplicate.index)
        np.testing.assert_allclose(first.direction, duplicate.direction)

    def test_reset_suppresses_cached_source_until_one_new_result(self):
        self.assertTrue(self.update(1).stable)
        self.recognizer.reset_confirmation()
        self.assertFalse(self.update(1).stable)
        self.assertFalse(self.update(1).stable)
        self.assertTrue(self.update(2).stable)

    def test_stale_source_clears_and_cannot_be_reused(self):
        self.assertTrue(self.update(1).stable)
        stale = self.recognizer.update([], None)
        self.assertFalse(stale.stable)
        self.assertIsNone(stale.index)
        self.assertFalse(self.update(1).stable)
        self.assertTrue(self.update(2).stable)

    def test_bending_stops_pointing_and_extension_recovers_immediately(self):
        self.assertTrue(self.update(1).stable)
        for sequence, kind in enumerate(('pip_bent', 'dip_bent', 'folded'), 2):
            with self.subTest(kind=kind):
                result = self.update(sequence, index_pixels(kind))
                self.assertFalse(result.stable)
                self.assertIsNone(result.index)
        self.assertTrue(self.update(5).stable)

    def test_open_palm_v_sign_and_missing_other_fingers_are_accepted(self):
        for sequence, extended in enumerate(((5,), (5, 9), (5, 9, 13, 17)), 1):
            points = hand_pose(extended)[:, :2] * 30 + [200, 100]
            self.assertTrue(self.update(sequence, points).stable)
        only_index = np.full((21, 2), np.nan)
        only_index[5:9] = hand_pixels()[5:9]
        self.assertTrue(self.update(4, only_index).stable)

    def test_screen_position_changes_do_not_add_a_hand_switch_delay(self):
        self.assertTrue(self.update(1).stable)
        self.assertTrue(self.update(2, index_pixels() + [200, 50]).stable)

    def test_one_qualified_hand_is_selected_but_two_are_ambiguous(self):
        result = self.recognizer.update([index_pixels('folded'), index_pixels()], 1)
        self.assertTrue(result.stable)
        self.assertEqual(result.index, 1)
        ambiguous = self.recognizer.update([index_pixels(), index_pixels() + [200, 0]], 2)
        self.assertFalse(ambiguous.stable)
        self.assertIsNone(ambiguous.index)
        self.assertIn('two hands', ambiguous.status.lower())
        self.assertTrue(self.update(3).stable)

    def test_no_hand_invalid_index_or_short_projection_cannot_point(self):
        result = self.recognizer.update([], 1)
        self.assertFalse(result.stable)
        self.assertIsNone(result.index)
        invalid = index_pixels(); invalid[8] = np.nan
        short = index_pixels(); short[5:9] = [200, 200]
        for sequence, pixels in enumerate((invalid, short, np.zeros((3, 2))), 2):
            with self.subTest(sequence=sequence):
                result = self.update(sequence, pixels)
                self.assertFalse(result.stable)
                self.assertIsNone(result.index)
                self.assertIsNone(result.direction)


if __name__ == '__main__':
    unittest.main()
