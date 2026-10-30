"""Deterministic pointing-ray and target-selection geometry checks."""

from types import SimpleNamespace
import unittest

import numpy as np

from target_selection import Selection, TargetTracker, ray_box_intersection, select_target


def box(x, y, width, height):
    return SimpleNamespace(origin_x=x, origin_y=y, width=width, height=height)


def detection(x, y, width, height, label='cup'):
    return SimpleNamespace(
        bounding_box=box(x, y, width, height),
        categories=[SimpleNamespace(category_name=label, score=0.9)],
    )


class RayIntersectionTests(unittest.TestCase):
    def test_cardinal_rays_have_distances_in_pixels(self):
        for direction, bounds in (
            ((1, 0), box(150, 90, 20, 20)),
            ((-1, 0), box(30, 90, 20, 20)),
            ((0, 1), box(90, 150, 20, 20)),
            ((0, -1), box(90, 30, 20, 20)),
        ):
            with self.subTest(direction=direction):
                np.testing.assert_allclose(
                    ray_box_intersection((100, 100), direction, bounds), (50, 70)
                )

    def test_direction_is_normalized_for_diagonal_and_nonunit_rays(self):
        bounds = box(150, 150, 20, 20)
        expected = (50 * np.sqrt(2), 70 * np.sqrt(2))
        np.testing.assert_allclose(
            ray_box_intersection((100, 100), (5, 5), bounds), expected
        )
        np.testing.assert_allclose(
            ray_box_intersection((100, 100), (10, 0), box(150, 90, 20, 20)),
            (50, 70),
        )

    def test_parallel_outside_and_objects_behind_are_misses(self):
        self.assertIsNone(ray_box_intersection((100, 100), (1, 0), box(150, 120, 20, 20)))
        self.assertIsNone(ray_box_intersection((100, 100), (1, 0), box(30, 90, 20, 20)))
        self.assertIsNone(ray_box_intersection((100, 100), (0, 1), box(120, 150, 20, 20)))

    def test_origin_inside_and_parallel_boundary_are_supported(self):
        np.testing.assert_allclose(
            ray_box_intersection((100, 100), (1, 0), box(90, 90, 20, 20)), (0, 10)
        )
        np.testing.assert_allclose(
            ray_box_intersection((100, 100), (0, 1), box(100, 150, 20, 20)), (50, 70)
        )


class TargetSelectionTests(unittest.TestCase):
    def test_nearest_forward_object_wins_independent_of_array_order(self):
        far = detection(400, 90, 40, 30, 'bottle')
        near = detection(150, 90, 40, 30)
        for items, expected in (([far, near], 1), ([near, far], 0)):
            with self.subTest(expected=expected):
                selection = select_target((100, 100), (1, 0), items, (640, 480))
                self.assertEqual(selection.index, expected)
                self.assertIn(expected, selection.candidates)

    def test_overlapping_candidates_are_ambiguous_even_with_same_label(self):
        items = [detection(150, 90, 80, 30), detection(200, 90, 80, 30)]
        selection = select_target((100, 100), (1, 0), items, (640, 480))
        self.assertIsNone(selection.index)
        self.assertIn('Unclear', selection.status)
        self.assertEqual(set(selection.candidates), {0, 1})

    def test_close_along_ray_candidates_use_the_ambiguity_margin(self):
        # The 10px gap lies within the 16px margin for a 640x480 image,
        # even before the small box padding is considered.
        items = [detection(150, 90, 30, 30), detection(190, 90, 30, 30, 'bottle')]
        selection = select_target((100, 100), (1, 0), items, (640, 480))
        self.assertIsNone(selection.index)
        self.assertIn('Unclear', selection.status)

    def test_no_intersection_and_no_objects_leave_nothing_selected(self):
        for items in ([], [detection(200, 200, 30, 30)], [detection(30, 90, 20, 20)]):
            with self.subTest(items=items):
                self.assertIsNone(select_target((100, 100), (1, 0), items, (640, 480)).index)

    def test_invalid_ray_is_rejected(self):
        item = detection(150, 90, 40, 30)
        for origin, direction in (
            ((100, 100), (0, 0)),
            ((np.nan, 100), (1, 0)),
            ((100, 100), (np.inf, 0)),
        ):
            with self.subTest(origin=origin, direction=direction):
                self.assertIsNone(select_target(origin, direction, [item], (640, 480)).index)

    def test_person_and_invalid_detection_data_do_not_block_valid_object(self):
        invalid = [
            detection(110, 90, 20, 30, 'person'),
            SimpleNamespace(bounding_box=box(110, 90, 20, 30), categories=[]),
            SimpleNamespace(categories=[SimpleNamespace(category_name='cup', score=0.9)]),
            detection(110, 90, -20, 30),
            detection(np.nan, 90, 20, 30),
        ]
        valid = detection(200, 90, 30, 30, 'bottle')
        selection = select_target((100, 100), (1, 0), invalid + [valid], (640, 480))
        self.assertEqual(selection.index, len(invalid))

    def test_horizontal_mirror_preserves_selected_identity(self):
        items = [detection(400, 90, 40, 30, 'bottle'), detection(200, 90, 40, 30)]
        mirrored = [detection(640 - d.bounding_box.origin_x - d.bounding_box.width,
                              d.bounding_box.origin_y, d.bounding_box.width,
                              d.bounding_box.height, d.categories[0].category_name)
                    for d in items]
        original = select_target((100, 100), (1, 0), items, (640, 480))
        reflection = select_target((540, 100), (-1, 0), mirrored, (640, 480))
        self.assertEqual(original.index, 1)
        self.assertEqual(reflection.index, original.index)


class TargetConfirmationTests(unittest.TestCase):
    def test_duplicate_display_frames_do_not_confirm(self):
        tracker = TargetTracker()
        items = [detection(200, 90, 40, 30)]
        proposal = Selection(0, 'candidate', (0,))
        for _ in range(10):
            self.assertIsNone(tracker.update(proposal, items, 1).index)
        self.assertEqual(tracker.update(proposal, items, 2).index, 0)

    def test_matching_jittered_box_can_change_detection_order(self):
        tracker = TargetTracker()
        cup, bottle = detection(200, 90, 40, 30), detection(400, 90, 40, 30, 'bottle')
        tracker.update(Selection(0, 'candidate'), [cup, bottle], 1)
        moved = detection(202, 91, 40, 30)
        self.assertEqual(tracker.update(Selection(1, 'candidate'), [bottle, moved], 2).index, 1)

    def test_switching_to_a_different_same_label_object_requires_confirmation(self):
        tracker = TargetTracker()
        first, second = detection(200, 90, 40, 30), detection(400, 90, 40, 30)
        tracker.update(Selection(0, 'candidate'), [first], 1)
        tracker.update(Selection(0, 'candidate'), [first], 2)
        self.assertIsNone(tracker.update(Selection(0, 'candidate'), [second], 3).index)
        self.assertEqual(tracker.update(Selection(0, 'candidate'), [second], 4).index, 0)

    def test_ambiguity_or_reset_clears_confirmation(self):
        tracker = TargetTracker()
        items = [detection(200, 90, 40, 30)]
        tracker.update(Selection(0, 'candidate'), items, 1)
        tracker.update(Selection(0, 'candidate'), items, 2)
        self.assertIsNone(tracker.update(Selection(None, 'Unclear'), items, 3).index)
        self.assertIsNone(tracker.update(Selection(0, 'candidate'), items, 4).index)
        tracker.reset()
        self.assertIsNone(tracker.update(Selection(0, 'candidate'), items, 5).index)


if __name__ == '__main__':
    unittest.main()
