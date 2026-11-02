"""CLI modes load only the requested models and keep their previews separate."""

from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
import io
from itertools import count
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from main import VisionProcessor, main, run
from test_target_selection import detection
from test_preview_support import PreviewWindowStub
from vision_worker import FrameResult
from camera_capture import CameraFrame


class ModeCliTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        self.hand_model = stack.enter_context(patch('main.ensure_model'))
        self.object_model = stack.enter_context(patch('main.ensure_object_model'))
        self.run = stack.enter_context(patch('main.run'))

    def test_omitting_mode_flags_enables_both_models(self):
        for arguments, camera in (([], 0), (['--camera', '1'], 1)):
            with self.subTest(arguments=arguments):
                self.hand_model.reset_mock()
                self.object_model.reset_mock()
                self.run.reset_mock()
                main(arguments)
                self.hand_model.assert_called_once()
                self.object_model.assert_called_once()
                self.run.assert_called_once()
                self.assertEqual(self.run.call_args.args[0], camera)
                self.assertIsNotNone(self.run.call_args.args[3])
                self.assertTrue(self.run.call_args.kwargs['hand_enabled'])
                self.assertTrue(self.run.call_args.kwargs['robot_enabled'])
                self.assertTrue(self.run.call_args.kwargs['speech_enabled'])
                self.assertEqual(self.run.call_args.kwargs['target_fps'], 60)

    def test_object_flag_selects_only_object_model(self):
        path = Path('custom-object.onnx')
        main(['--object', '--camera', '1', '--object-model', str(path)])
        self.hand_model.assert_not_called()
        self.object_model.assert_called_once_with(path)
        self.assertEqual(self.run.call_args.args[0], 1)
        self.assertEqual(self.run.call_args.args[3], path)
        self.assertFalse(self.run.call_args.kwargs['hand_enabled'])
        self.assertFalse(self.run.call_args.kwargs['robot_enabled'])
        self.assertFalse(self.run.call_args.kwargs['speech_enabled'])

    def test_preview_fps_can_be_overridden_and_invalid_rates_do_not_open_models(self):
        main(['--fps', '30'])
        self.assertEqual(self.run.call_args.kwargs['target_fps'], 30)
        for value in ('0', '121', 'nan'):
            with self.subTest(value=value):
                self.hand_model.reset_mock()
                self.object_model.reset_mock()
                self.run.reset_mock()
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(['--fps', value])
                self.assertEqual(error.exception.code, 2)
                self.hand_model.assert_not_called()
                self.object_model.assert_not_called()
                self.run.assert_not_called()

    def test_hand_flag_selects_only_hand_model(self):
        path = Path('custom-hand.task')
        main(['--hand', '--model', str(path)])
        self.hand_model.assert_called_once_with(path)
        self.object_model.assert_not_called()
        self.assertEqual(self.run.call_args.args[1], path)
        self.assertIsNone(self.run.call_args.args[3])
        self.assertTrue(self.run.call_args.kwargs['hand_enabled'])
        self.assertFalse(self.run.call_args.kwargs['robot_enabled'])
        self.assertFalse(self.run.call_args.kwargs['speech_enabled'])

    def test_robot_and_voice_can_be_disabled_independently(self):
        for flags, robot, voice in ((['--mute'], True, False),
                                    (['--no-robot'], False, True),
                                    (['--mute', '--no-robot'], False, False)):
            with self.subTest(flags=flags):
                main(flags)
                self.assertEqual(self.run.call_args.kwargs['robot_enabled'], robot)
                self.assertEqual(self.run.call_args.kwargs['speech_enabled'], voice)

    def test_both_flags_enable_both_models(self):
        main(['--hand', '--object'])
        self.hand_model.assert_called_once()
        self.object_model.assert_called_once()
        self.assertIsNotNone(self.run.call_args.args[3])
        self.assertTrue(self.run.call_args.kwargs['hand_enabled'])

    def test_download_only_prepares_selected_models_without_camera(self):
        for flags, hand_count, object_count in (([], 1, 1),
                                                (['--hand'], 1, 0),
                                                (['--object'], 0, 1),
                                                (['--hand', '--object'], 1, 1)):
            with self.subTest(flags=flags):
                self.hand_model.reset_mock()
                self.object_model.reset_mock()
                main(flags + ['--download-model-only'])
                self.assertEqual(self.hand_model.call_count, hand_count)
                self.assertEqual(self.object_model.call_count, object_count)
                self.run.assert_not_called()

    def test_label_listing_remains_available_without_a_mode(self):
        output = io.StringIO()
        with patch('main.supported_labels', return_value=['bottle', 'cup']), \
             redirect_stdout(output):
            main(['--list-object-labels'])
        self.assertIn('bottle, cup', output.getvalue())
        self.hand_model.assert_not_called()
        self.object_model.assert_called_once()
        self.run.assert_not_called()


class ModePipelineTests(unittest.TestCase):
    def test_object_only_never_constructs_hand_tracker(self):
        detector = Mock()
        detector.detect_for_video.return_value = SimpleNamespace(detections=['cup'])
        with patch('main.HandTracker') as hand_tracker, \
             patch('main.create_detector', return_value=nullcontext(detector)):
            with VisionProcessor(None, Path('object.onnx')) as processor:
                result = processor(np.zeros((2, 2, 3), dtype=np.uint8), 1)
        hand_tracker.assert_not_called()
        detector.detect_for_video.assert_called_once()
        self.assertEqual(result, (None, ['cup']))

    def test_hand_only_never_constructs_object_detector(self):
        tracker = Mock()
        tracker.detect_for_video.return_value = 'hand landmarks'
        with patch('main.HandTracker', return_value=nullcontext(tracker)), \
             patch('main.create_detector') as object_detector:
            with VisionProcessor(Path('hand.task'), None) as processor:
                result = processor(np.zeros((2, 2, 3), dtype=np.uint8), 1)
        object_detector.assert_not_called()
        tracker.detect_for_video.assert_called_once()
        self.assertEqual(result, ('hand landmarks', []))

    def test_object_preview_ignores_gesture_controls_and_never_selects_a_target(self):
        cup = detection(200, 200, 60, 60, 'cup')
        processors = []

        class Worker:
            def __init__(self, factory):
                processors.append(factory())
                self.sequence = 0
            def __enter__(self): return self
            def __exit__(self, *exc): pass
            def submit(self, *args): pass
            def latest(self, generation, max_age=0.3):
                self.sequence += 1
                now = time.monotonic()
                return FrameResult(self.sequence, generation, now, now, 10, (None, [cup]))

        camera = Mock()
        camera.actual_fps = 60
        camera.__enter__ = Mock(return_value=camera)
        camera.__exit__ = Mock(side_effect=lambda *args: camera.close())
        camera.wait_first.return_value = CameraFrame(
            np.zeros((480, 640, 3), np.uint8), 0, time.monotonic())
        sequences = count(1)
        camera.latest.side_effect = lambda: CameraFrame(
            np.zeros((480, 640, 3), np.uint8), next(sequences), time.monotonic())
        with patch('main.LatestFrameWorker', Worker), \
             patch('main.LatestCamera', return_value=camera), \
             patch('main.PreviewWindow', PreviewWindowStub), patch('main.cv2.destroyAllWindows'), \
             patch('main.cv2.imshow'), patch('main.cv2.getWindowProperty', return_value=1), \
             patch('main.cv2.waitKey', side_effect=[ord('c'), ord('r'), ord('m'), ord('d'), ord('q')]), \
             patch('main.PointingRecognizer') as recognizer, \
             patch('main.draw_hand') as draw_hand, \
             patch('main.select_target') as select_target, \
             patch('main.draw_objects') as draw_objects:
            run(0, Path('unused-hand.task'), False, object_model=Path('object.onnx'),
                hand_enabled=False, pace=False)
        recognizer.assert_not_called()
        draw_hand.assert_not_called()
        select_target.assert_not_called()
        self.assertEqual(len(processors), 1)
        self.assertIsNone(processors[0].model_path)
        self.assertEqual(draw_objects.call_count, 5)
        for call in draw_objects.call_args_list:
            displayed = call.args[1]
            self.assertEqual(len(displayed), 1)
            self.assertIsNot(displayed[0], cup)
            self.assertEqual(vars(displayed[0].bounding_box),
                             {'origin_x': 200, 'origin_y': 200, 'width': 60, 'height': 60})
            self.assertEqual(displayed[0].categories[0].category_name, 'cup')
            self.assertEqual(displayed[0].categories[0].score, 0.9)
            self.assertIsNone(call.kwargs['selected_index'])
        self.assertEqual(vars(cup.bounding_box),
                         {'origin_x': 200, 'origin_y': 200, 'width': 60, 'height': 60})
        self.assertEqual(vars(cup.categories[0]), {'category_name': 'cup', 'score': 0.9})
        self.assertEqual(set(vars(cup)), {'bounding_box', 'categories'})
        camera.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
