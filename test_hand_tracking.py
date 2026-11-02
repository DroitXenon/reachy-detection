"""Distance-only 2D finger extension and direct arrow geometry."""

import unittest

import numpy as np

from gesture_recog import EXTENDED_RATIO, finger_extension, pointing_direction


def hand_pose(extended=(5,)):
    points = np.zeros((21, 3), dtype=float)
    for number, base in enumerate((5, 9, 13, 17)):
        x = number * 0.4
        if base in extended:
            points[base:base + 4] = [(x, 1, 0), (x, 2, 0), (x, 3, 0), (x, 4, 0)]
        else:
            points[base:base + 4] = [(x, 1, 0), (x, 2, 0), (x, 2, 1), (x, 1, 1)]
    return points


def hand_pixels():
    return hand_pose()[:, :2] * 30 + [200, 100]



def index_pixels(kind='straight'):
    """Independent screen-coordinate examples of extended and folded fingers."""
    points = hand_pixels()
    examples = {
        'straight': [(0, 0), (0, 30), (0, 60), (0, 90)],
        'slight': [(0, 0), (0, 30), (10, 58), (20, 86)],
        'pip_bent': [(0, 0), (0, 30), (30, 30), (60, 30)],
        'dip_bent': [(0, 0), (0, 30), (0, 60), (30, 60)],
        'folded': [(0, 0), (0, 30), (30, 30), (30, 0)],
    }
    points[5:9] = np.asarray(examples[kind]) + [200, 100]
    return points


class FingerExtensionTests(unittest.TestCase):
    def test_straight_finger_has_full_extension(self):
        self.assertAlmostEqual(finger_extension(index_pixels()), 1.0)
        self.assertGreaterEqual(finger_extension(index_pixels()), EXTENDED_RATIO)

    def test_mostly_straight_finger_is_tolerated(self):
        ratio = finger_extension(index_pixels('slight'))
        self.assertGreaterEqual(ratio, EXTENDED_RATIO)
        self.assertLess(ratio, 1.0)

    def test_fold_at_either_joint_reduces_extension(self):
        for kind in ('pip_bent', 'dip_bent', 'folded'):
            with self.subTest(kind=kind):
                self.assertLess(finger_extension(index_pixels(kind)), EXTENDED_RATIO)
        expected = np.sqrt(60 ** 2 + 30 ** 2) / 90
        self.assertAlmostEqual(finger_extension(index_pixels('pip_bent')), expected)
        self.assertAlmostEqual(finger_extension(index_pixels('dip_bent')), expected)

    def test_other_fingers_and_wrist_do_not_affect_extension(self):
        reference = index_pixels()
        only_index = np.full((21, 2), np.nan)
        only_index[5:9] = reference[5:9]
        self.assertAlmostEqual(finger_extension(only_index), 1.0)
        np.testing.assert_allclose(pointing_direction(only_index), [0, 1])
        for extended in ((5,), (5, 9), (5, 9, 13, 17), (5, 13)):
            points = hand_pose(extended)[:, :2] * 30 + [200, 100]
            self.assertAlmostEqual(finger_extension(points), 1.0)

    def test_translation_rotation_reflection_and_visible_scale_are_invariant(self):
        pixels = index_pixels('slight')
        rotation = np.array([[0, -1], [1, 0]])
        reference = finger_extension(pixels)
        for transformed in (pixels + [230, -80], pixels @ rotation,
                            pixels * [-1, 1], pixels * .5, pixels * 2):
            with self.subTest(example=transformed[8].tolist()):
                self.assertAlmostEqual(finger_extension(transformed), reference)

    def test_malformed_nonfinite_collapsed_and_short_projection_are_invalid(self):
        nonfinite = index_pixels(); nonfinite[7] = np.nan
        collapsed = index_pixels(); collapsed[5:9] = [200, 200]
        short = index_pixels(); short[5:9] = [(0, 0), (0, 1), (0, 2), (0, 3)]
        for pixels in (None, np.zeros((4, 2)), np.zeros((21, 3)),
                       nonfinite, collapsed, short):
            with self.subTest(shape=np.shape(pixels)):
                self.assertIsNone(finger_extension(pixels))
                self.assertIsNone(pointing_direction(pixels))

    def test_arrow_uses_base_to_tip_even_for_a_bent_finger(self):
        pixels = index_pixels('pip_bent')
        expected = pixels[8] - pixels[5]
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(pointing_direction(pixels), expected)

    def test_arrow_rotates_and_reflects_with_the_index(self):
        pixels = index_pixels()
        np.testing.assert_allclose(pointing_direction(pixels), [0, 1])
        np.testing.assert_allclose(pointing_direction(pixels[:, ::-1]), [1, 0])
        pixels[:, 1] *= -1
        np.testing.assert_allclose(pointing_direction(pixels), [0, -1])


if __name__ == '__main__':
    unittest.main()
