"""Hand-model lifecycle and the boundary between tracking and the app."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from hand_tracking import HandTracker, MODEL_PATH, extract_hands
from main import VisionProcessor


class HandTrackerTests(unittest.TestCase):
    def test_landmarks_keep_hand_order_and_image_coordinates(self):
        left = SimpleNamespace(x=0.25, y=0.75, z=-0.1)
        right = SimpleNamespace(x=0.8, y=0.2, z=0.3)
        result = SimpleNamespace(
            hand_landmarks=[[left] * 21, [right] * 21],
            hand_world_landmarks=[[right] * 21, [left] * 21],
            handedness=[[SimpleNamespace(category_name='Left')],
                        [SimpleNamespace(category_name='Right')]],
        )
        hands = extract_hands(result, (640, 480))
        self.assertEqual(hands.labels, ('Left', 'Right'))
        np.testing.assert_allclose(hands.pixels[0][8], [160, 360])
        np.testing.assert_allclose(hands.pixels[1][8], [512, 96])
        np.testing.assert_allclose(hands.worlds[0][8], [0.8, 0.2, 0.3])
        self.assertEqual(extract_hands(None, (640, 480)).pixels, ())

    def test_hand_model_closes_even_when_processing_fails(self):
        landmarker = Mock()
        landmarker.detect_for_video.side_effect = ValueError('bad frame')
        with patch('hand_tracking.vision.HandLandmarker.create_from_options',
                   return_value=landmarker):
            with self.assertRaisesRegex(ValueError, 'bad frame'):
                with HandTracker(MODEL_PATH) as tracker:
                    tracker.detect_for_video('image', 10)
        landmarker.close.assert_called_once_with()

    def test_pipeline_shares_image_and_increasing_timestamps(self):
        hand, detector = Mock(), Mock()
        hand.detect_for_video.return_value = 'hand result'
        detector.detect_for_video.return_value = SimpleNamespace(detections=['object'])
        with patch('main.HandTracker', return_value=nullcontext(hand)), \
             patch('main.create_detector', return_value=nullcontext(detector)), \
             patch('main.time.monotonic_ns', return_value=100_000_000):
            with VisionProcessor(MODEL_PATH, 'model.onnx') as processor:
                first = processor(np.full((2, 2, 3), [1, 2, 3], dtype=np.uint8), 1)
                processor(np.zeros((2, 2, 3), dtype=np.uint8), 2)
        self.assertEqual(first, ('hand result', ['object']))
        calls = hand.detect_for_video.call_args_list
        self.assertEqual([call.args[1] for call in calls], [100, 101])
        for hand_call, object_call in zip(calls, detector.detect_for_video.call_args_list):
            self.assertIs(hand_call.args[0], object_call.args[0])
            self.assertEqual(hand_call.args[1], object_call.args[1])
        np.testing.assert_array_equal(calls[0].args[0].numpy_view()[0, 0], [3, 2, 1])

    def test_object_model_failure_closes_already_open_hand_model(self):
        hand = Mock()
        context = Mock()
        context.__enter__ = Mock(return_value=hand)
        context.__exit__ = Mock(return_value=False)
        with patch('main.HandTracker', return_value=context), \
             patch('main.create_detector', side_effect=RuntimeError('missing object model')):
            with self.assertRaisesRegex(RuntimeError, 'missing object model'):
                with VisionProcessor(MODEL_PATH, 'missing.onnx'):
                    pass
        context.__exit__.assert_called_once()


if __name__ == '__main__':
    unittest.main()
