"""Personal-pose, direction, persistence, and calibration safeguards."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gesture_recognition import PointingPose, choose_pointing_hand, pointing_direction, pose_features


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


class PersonalPoseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'pointing.json'
        self.pose = PointingPose(self.path, sample_count=3)

    def teach(self, world):
        self.pose.start()
        for sequence in range(3):
            self.pose.observe([world], ['Left'], [[0.5, 0.5]], sequence)

    def test_default_pointing_is_accepted(self):
        self.assertTrue(self.pose.classify(hand_pose())[0])

    def test_fist_palm_and_v_sign_are_rejected(self):
        for extended in ((), (5, 9, 13, 17), (5, 9)):
            self.assertFalse(self.pose.classify(hand_pose(extended))[0])

    def test_features_and_decisions_are_rotation_scale_and_reflection_invariant(self):
        world = hand_pose()
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for transformed in (world @ rotation, world * 0.01 + 5, world * [-1, 1, 1]):
            np.testing.assert_allclose(pose_features(world), pose_features(transformed), atol=1e-10)
            self.assertTrue(self.pose.classify(transformed)[0])

    def test_calibration_learns_a_pose_missed_by_default_and_persists(self):
        world = hand_pose()
        world[7:9] = [[0, 2, 1], [0, 2, 2]]  # A less extended personal pose.
        self.assertFalse(self.pose.classify(world)[0])
        self.teach(world)
        self.assertFalse(self.pose.capturing)
        self.assertTrue(self.path.is_file())
        self.assertTrue(self.pose.classify(world)[0])
        loaded = PointingPose(self.path)
        self.assertTrue(loaded.classify(world)[0])
        self.assertFalse(loaded.classify(hand_pose(()))[0])
        self.assertFalse(loaded.classify(hand_pose((5, 9, 13, 17)))[0])

    def test_preview_duplicates_do_not_complete_calibration(self):
        self.pose.start()
        for _ in range(10):
            self.pose.observe([hand_pose()], ['Left'], [[0.5, 0.5]], 1)
        self.assertEqual(len(self.pose.samples), 1)
        self.assertFalse(self.path.exists())

    def test_ambiguous_or_changed_hand_cancels_calibration(self):
        for labels, wrists in ((['Right'], [[0.5, 0.5]]), (['Left'], [[0.9, 0.9]]), ([], [])):
            self.pose.start()
            self.pose.observe([hand_pose()], ['Left'], [[0.5, 0.5]], 1)
            worlds = [hand_pose()] if labels else []
            self.pose.observe(worlds, labels, wrists, 2)
            self.assertFalse(self.pose.capturing)
            self.assertFalse(self.path.exists())

    def test_invalid_and_open_palm_samples_cannot_calibrate(self):
        self.pose.start()
        for seq in range(4):
            self.pose.observe([hand_pose((5, 9, 13, 17))], ['Left'], [[0.5, 0.5]], seq)
        self.assertEqual(self.pose.samples, [])
        self.assertFalse(self.path.exists())
        self.assertIsNone(pose_features(np.zeros((21, 3))))
        world = hand_pose(); world[8] = np.nan
        self.assertIsNone(pose_features(world))

    def test_invalid_profile_falls_back_and_reset_removes_saved_profile(self):
        self.path.write_text(json.dumps({'version': 1, 'centre': [0], 'tolerance': [0]}))
        loaded = PointingPose(self.path)
        self.assertIsNone(loaded.centre)
        self.assertIn('invalid', loaded.notice)
        self.teach(hand_pose())
        self.pose.reset()
        self.assertFalse(self.path.exists())
        self.assertIsNone(self.pose.centre)

    def test_resting_second_hand_and_two_pointing_hands(self):
        pointing, resting = hand_pose(), hand_pose(())
        for worlds, expected in (([pointing, resting], 0), ([resting, pointing], 1)):
            pixels = [hand[:, :2] * 30 + [200, 100] for hand in worlds]
            self.assertEqual(choose_pointing_hand(worlds, pixels, self.pose)[0], expected)
        self.assertIsNone(choose_pointing_hand([pointing, pointing], [hand_pixels()] * 2, self.pose)[0])

    def test_direction_rotates_and_reflects_without_classification(self):
        pixels = hand_pixels()
        np.testing.assert_allclose(pointing_direction(pixels), [0, 1], atol=1e-6)
        np.testing.assert_allclose(pointing_direction(pixels[:, ::-1]), [1, 0], atol=1e-6)
        pixels[:, 1] *= -1
        np.testing.assert_allclose(pointing_direction(pixels), [0, -1], atol=1e-6)

    def test_short_or_invalid_direction_is_hidden(self):
        pixels = hand_pixels(); pixels[8] = pixels[5]
        self.assertIsNone(pointing_direction(pixels))
        pixels[8] = np.nan
        self.assertIsNone(pointing_direction(pixels))


if __name__ == '__main__':
    unittest.main()
