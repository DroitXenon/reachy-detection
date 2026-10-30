"""Geometry checks; webcam accuracy still requires a live gesture test."""

import unittest

import numpy as np

from hand_tracking import is_pointing, pointing_direction


def hand_pose(extended):
    points = np.zeros((21, 3), dtype=float)
    for finger, base in enumerate((5, 9, 13, 17)):
        x = float(finger)
        if base in extended:
            points[base:base + 4] = [(x, 1, 0), (x, 2, 0), (x, 3, 0), (x, 4, 0)]
        else:
            points[base:base + 4] = [(x, 1, 0), (x, 2, 0), (x, 2, 1), (x, 1, 1)]
    return points


class PointingGeometryTests(unittest.TestCase):
    def test_pointing_survives_rotation_scale_translation_and_reflection(self):
        points = hand_pose({5})
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for pose in (points, points @ rotation, points * 0.01 + 3,
                     points * [-1, 1, 1]):
            self.assertTrue(is_pointing(pose))

    def test_fist_open_palm_and_two_extended_fingers_are_rejected(self):
        for extended in (set(), {5, 9, 13, 17}, {5, 9}):
            self.assertFalse(is_pointing(hand_pose(extended)))

    def test_degenerate_coordinates_are_rejected(self):
        self.assertFalse(is_pointing(np.zeros((21, 3))))

    def test_arrow_follows_index_and_reflects_with_preview(self):
        pixels = hand_pose({5})[:, :2] * 50 + 200
        np.testing.assert_allclose(pointing_direction(pixels), [0, 1])
        pixels = pixels[:, ::-1]
        np.testing.assert_allclose(pointing_direction(pixels), [1, 0])
        pixels[:, 0] = 960 - pixels[:, 0]
        np.testing.assert_allclose(pointing_direction(pixels), [-1, 0])

    def test_foreshortened_projection_is_rejected(self):
        pixels = hand_pose({5})[:, :2] * 50
        pixels[8] = pixels[5] + [1, 1]
        self.assertIsNone(pointing_direction(pixels))


if __name__ == "__main__":
    unittest.main()
