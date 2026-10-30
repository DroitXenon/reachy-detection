"""Check confidence filtering without needing a camera or downloaded model."""

from types import SimpleNamespace
import unittest

from object_detection import visible_objects


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


if __name__ == "__main__":
    unittest.main()
