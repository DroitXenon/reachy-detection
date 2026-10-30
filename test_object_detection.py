"""Check confidence filtering without needing a camera or downloaded model."""

from types import SimpleNamespace
import unittest
import numpy as np

from object_detection import decode_yolo, visible_objects


def detection(label, score):
    return SimpleNamespace(categories=[SimpleNamespace(category_name=label, score=score)])


class DetectionFilteringTests(unittest.TestCase):
    def test_lower_threshold_recovers_weak_object_without_person(self):
        bottle = detection("bottle", 0.28)
        results = [detection("person", 0.99), bottle]
        self.assertEqual(visible_objects(results, 0.4), [])
        self.assertEqual(visible_objects(results, 0.25), [bottle])

    def test_threshold_inclusive_and_results_sorted(self):
        cup, book = detection("cup", 0.25), detection("book", 0.8)
        self.assertEqual(visible_objects([cup, book], 0.25), [book, cup])

    def test_unknown_and_empty_categories_are_ignored(self):
        self.assertEqual(visible_objects([
            detection("???", 0.9), SimpleNamespace(categories=[])
        ], 0), [])

    def test_display_limit_applies_after_person_exclusion(self):
        items = [detection("person", 1)] + [detection("cup", 0.5) for _ in range(12)]
        self.assertEqual(len(visible_objects(items, 0.25)), 10)

    def test_yolo_letterbox_coordinates_person_filter_and_duplicate_suppression(self):
        # 1280x720 -> 640x360 with 140px top padding. Two overlapping bottle
        # predictions should become one original-image box, and person disappears.
        rows = np.array([[100, 215, 100, 100, 0.01, 0.9],
                         [101, 216, 100, 100, 0.01, 0.8],
                         [400, 300, 100, 100, 0.99, 0.01]], dtype=np.float32)
        result = decode_yolo(rows.T[None], {0: "person", 1: "bottle"},
                             1280, 720, 0.5, 0.5, 0, 140, 0.1)
        self.assertEqual(len(result), 1)
        box = result[0].bounding_box
        self.assertEqual((box.origin_x, box.origin_y, box.width, box.height), (100, 50, 200, 200))
        self.assertEqual(result[0].categories[0].category_name, "bottle")

    def test_yolo_portrait_coordinates_and_clipping(self):
        rows = np.array([[215, 100, 100, 100, 0.9]], dtype=np.float32)
        result = decode_yolo(rows.T[None], {0: "cup"}, 720, 1280,
                             0.5, 0.5, 140, 0, 0.1)
        box = result[0].bounding_box
        self.assertEqual((box.origin_x, box.origin_y, box.width, box.height), (50, 100, 200, 200))


if __name__ == "__main__":
    unittest.main()
