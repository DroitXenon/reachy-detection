"""Decode YOLO26 end-to-end results without a camera or model download."""

import unittest

import numpy as np

from object_detection import decode_yolo_end_to_end


NAMES = {0: "person", 1: "cup", 2: "bottle"}


def box_tuple(detection):
    box = detection.bounding_box
    return box.origin_x, box.origin_y, box.width, box.height


class Yolo26EndToEndTests(unittest.TestCase):
    def test_landscape_letterbox_is_reversed(self):
        # 1280x720 -> 640x360, padded by 140 pixels vertically.
        output = np.array([[[50, 165, 150, 265, 0.9, 2]]], dtype=np.float32)
        detections = decode_yolo_end_to_end(
            output, NAMES, 1280, 720, 0.5, 0.5, 0, 140, 0.5
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(box_tuple(detections[0]), (100, 50, 200, 200))
        self.assertEqual(detections[0].categories[0].category_name, "bottle")

    def test_portrait_letterbox_is_reversed(self):
        # 720x1280 -> 360x640, padded by 140 pixels horizontally.
        output = np.array([[[165, 50, 265, 150, 0.9, 1]]], dtype=np.float32)
        detections = decode_yolo_end_to_end(
            output, NAMES, 720, 1280, 0.5, 0.5, 140, 0, 0.5
        )
        self.assertEqual(box_tuple(detections[0]), (50, 100, 200, 200))

    def test_coordinates_are_clipped_and_inverted_boxes_rejected(self):
        output = np.array([[
            [-10, -10, 700, 700, 0.9, 1],
            [100, 100, 90, 90, 0.9, 2],
        ]], dtype=np.float32)
        detections = decode_yolo_end_to_end(
            output, NAMES, 640, 640, 1, 1, 0, 0, 0.5
        )
        self.assertEqual(len(detections), 1)
        left, top, width, height = box_tuple(detections[0])
        self.assertEqual((left, top), (0, 0))
        self.assertGreater(width, 600)
        self.assertGreater(height, 600)
        self.assertLessEqual(left + width, 640)
        self.assertLessEqual(top + height, 640)

    def test_person_and_low_scores_are_excluded(self):
        output = np.array([[
            [10, 10, 100, 100, 0.99, 0],
            [10, 10, 100, 100, 0.49, 1],
            [10, 10, 100, 100, 0.50, 2],
        ]], dtype=np.float32)
        detections = decode_yolo_end_to_end(
            output, NAMES, 640, 640, 1, 1, 0, 0, 0.5
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].categories[0].category_name, "bottle")

    def test_nonfinite_rows_and_invalid_class_ids_are_rejected(self):
        output = np.array([[
            [10, 10, 100, 100, np.nan, 1],
            [np.inf, 10, 100, 100, 0.9, 1],
            [10, 10, 100, 100, 0.9, np.nan],
            [10, 10, 100, 100, 0.9, -1],
            [10, 10, 100, 100, 0.9, 99],
            [10, 10, 100, 100, 0.9, 1.5],
            [10, 10, 100, 100, 0.9, 2],
        ]], dtype=np.float32)
        detections = decode_yolo_end_to_end(
            output, NAMES, 640, 640, 1, 1, 0, 0, 0.5
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].categories[0].category_name, "bottle")

    def test_end_to_end_predictions_do_not_receive_extra_nms(self):
        output = np.array([[
            [10, 10, 100, 100, 0.9, 1],
            [11, 11, 101, 101, 0.8, 1],
        ]], dtype=np.float32)
        detections = decode_yolo_end_to_end(
            output, NAMES, 640, 640, 1, 1, 0, 0, 0.5
        )
        self.assertEqual(len(detections), 2)

    def test_empty_output_is_valid(self):
        output = np.empty((1, 0, 6), dtype=np.float32)
        self.assertEqual(decode_yolo_end_to_end(
            output, NAMES, 640, 640, 1, 1, 0, 0, 0.5
        ), [])

    def test_incompatible_output_shapes_fail_explicitly(self):
        for shape in ((300, 6), (1, 6, 300), (1, 300, 7), (2, 300, 6)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                decode_yolo_end_to_end(
                    np.zeros(shape, dtype=np.float32), NAMES,
                    640, 640, 1, 1, 0, 0, 0.5
                )


if __name__ == "__main__":
    unittest.main()
