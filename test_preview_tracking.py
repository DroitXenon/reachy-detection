"""Visual tracking is bounded, source-aligned, and never edits model results."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from preview_tracking import PreviewTracker
from vision_worker import FrameResult


def detection(x=60, y=50, width=70, height=70, label='cup', score=.9):
    return SimpleNamespace(
        bounding_box=SimpleNamespace(origin_x=x, origin_y=y, width=width, height=height),
        categories=[SimpleNamespace(category_name=label, score=score)],
    )


def snapshot(sequence, captured_at, objects=(), pixels=None, generation=0):
    hands = SimpleNamespace(hand_landmarks=[], handedness=[])
    if pixels is not None:
        hands.hand_landmarks = [[SimpleNamespace(x=x / 240, y=y / 180) for x, y in pixels]]
        hands.handedness = [[SimpleNamespace(category_name='Left')]]
    return FrameResult(sequence, generation, captured_at, captured_at + .05, 50,
                       (hands, list(objects)))


def texture():
    random = np.random.default_rng(3)
    frame = random.integers(0, 255, (180, 240, 3), dtype=np.uint8)
    return cv2.GaussianBlur(frame, (3, 3), .4)


def shifted(frame, x, y=0):
    return cv2.warpAffine(frame, np.float32([[1, 0, x], [0, 1, y]]),
                          (frame.shape[1], frame.shape[0]))


class PreviewTrackingTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.tracker = PreviewTracker(clock=lambda: self.now)
        self.frame = texture()

    def update(self, sequence=1, frame=None, model=None, generation=0):
        return self.tracker.update(self.frame if frame is None else frame,
                                   sequence, generation, self.now, model)

    def test_optical_flow_moves_hands_and_boxes_between_inferences(self):
        pixels = np.array([[65 + (i % 5) * 10, 55 + (i // 5) * 10] for i in range(21)])
        first = self.update(model=snapshot(1, self.now, [detection()], pixels))
        self.now += 1 / 60
        second = self.update(2, shifted(self.frame, 5, 3))
        np.testing.assert_allclose(second.pixels[0] - first.pixels[0], np.tile([5, 3], (21, 1)), atol=.25)
        box = second.detections[0].bounding_box
        self.assertEqual((box.origin_x, box.origin_y), (65, 53))
        self.assertEqual(second.detections[0].preview_source_box, (60, 50, 70, 70))
        self.assertEqual(second.hand_source_sequences, (1,))
        self.assertTrue(second.held)

    def test_delayed_prediction_is_projected_from_its_source_frame(self):
        self.update(1)
        self.now += .1
        result = self.update(2, shifted(self.frame, 8), snapshot(1, 10.0, [detection()]))
        self.assertEqual(result.detections[0].bounding_box.origin_x, 68)
        self.assertEqual(result.detections[0].preview_source_sequence, 1)

    def test_duplicate_camera_frame_skips_flow_but_accepts_late_model(self):
        self.update()
        with patch.object(self.tracker, '_flow', wraps=self.tracker._flow) as flow:
            result = self.update(model=snapshot(1, self.now, [detection()]))
            self.update(model=snapshot(1, self.now, [detection()]))
        flow.assert_not_called()
        self.assertEqual(len(result.detections), 1)

    def test_small_model_jitter_is_smoothed_and_raw_input_is_unchanged(self):
        original = detection()
        self.update(model=snapshot(1, self.now, [original]))
        self.now += .02
        moved = detection(64)
        result = self.update(2, model=snapshot(2, self.now, [moved]))
        self.assertGreater(result.detections[0].bounding_box.origin_x, 60)
        self.assertLess(result.detections[0].bounding_box.origin_x, 64)
        result.detections[0].categories[0].score = .1
        self.assertEqual(original.bounding_box.origin_x, 60)
        self.assertEqual(moved.bounding_box.origin_x, 64)
        self.assertEqual(moved.categories[0].score, .9)

    def test_brief_missing_prediction_is_retained_then_clears(self):
        self.update(model=snapshot(1, self.now, [detection()]))
        self.now += .1
        brief = self.update(2, model=snapshot(2, self.now))
        self.assertEqual(len(brief.detections), 1)
        self.assertTrue(brief.detections[0].preview_held)
        self.now += .23
        self.assertEqual(self.update(3).detections, [])

    def test_no_visual_survives_half_second_without_positive_model_data(self):
        self.update(model=snapshot(1, self.now, [detection()]))
        self.now += .51
        # Duplicate camera frames must still expire on wall-clock time.
        result = self.update(1)
        self.assertEqual(result.detections, [])
        self.assertIsNone(result.age)

    def test_generation_reset_and_missing_source_history_do_not_reuse_coordinates(self):
        self.update(model=snapshot(1, self.now, [detection()]))
        self.now += .1
        result = self.update(2, generation=1, model=snapshot(1, 10, [detection()]))
        self.assertEqual(result.detections, [])
        result = self.update(3, generation=1, model=snapshot(99, self.now, [detection()], generation=1))
        self.assertEqual(result.detections, [])

    def test_order_and_class_filter_match_current_model_and_held_metadata(self):
        items = [detection(label='person'), detection(label='bottle', score=.8),
                 detection(150, label='cup', score=.9), detection(score=.2)]
        result = self.update(model=snapshot(1, self.now, items))
        self.assertEqual([item.categories[0].category_name for item in result.detections], ['cup', 'bottle'])
        self.assertTrue(all(isinstance(item.bounding_box.origin_x, int) for item in result.detections))
        self.tracker.reset()
        self.assertEqual(self.update(2).detections, [])

    def test_class_change_replaces_old_visual_instead_of_drawing_duplicate_boxes(self):
        self.update(model=snapshot(1, self.now, [detection(label='cup')]))
        self.now += .05
        result = self.update(2, model=snapshot(2, self.now, [detection(label='bottle')]))
        self.assertEqual([item.categories[0].category_name for item in result.detections], ['bottle'])

    def test_novel_detections_do_not_accumulate_unbounded_retained_tracks(self):
        for sequence in range(1, 8):
            self.now += .02
            objects = [detection(10 + index * 12, 10 + sequence * 15, 5, 5,
                                 label=f'object-{sequence}-{index}') for index in range(10)]
            result = self.update(sequence, model=snapshot(sequence, self.now, objects))
            self.assertEqual(len(result.detections), 10)
            self.assertTrue(all(item.categories[0].category_name.startswith(f'object-{sequence}-')
                                for item in result.detections))
            self.assertLessEqual(len(self.tracker._tracks), 12)


if __name__ == '__main__':
    unittest.main()
