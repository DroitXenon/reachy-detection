"""Combine hand tracking, gesture recognition, and object selection in a webcam app."""

import argparse
from collections import deque
from contextlib import ExitStack
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np

from hand_tracking import MODEL_PATH, HandTracker, draw_hand, ensure_model, extract_hands
from gesture_recog import PROFILE_PATH, PointingRecognizer, pointing_direction
from object_detection import (
    OBJECT_MODEL_PATH, OBJECT_MODEL_URL, create_detector, draw_objects,
    ranked_objects, supported_labels, visible_objects,
)
from target_selection import Selection, TargetTracker, select_target
from vision_worker import LatestFrameWorker


WINDOW = "Reachy - Vision"


def put_panel_text(panel, text, origin, font, scale, color, thickness):
    """Keep diagnostics readable within narrower webcam previews."""
    text_width = cv2.getTextSize(text, font, scale, thickness)[0][0]
    available = max(1, panel.shape[1] - origin[0] - 12)
    fitted_scale = scale * min(1.0, available / max(1, text_width))
    cv2.putText(panel, text, origin, font, fitted_scale, color, thickness)


class VisionProcessor:
    """All models process the same clean frame, entirely on the worker thread."""

    def __init__(self, model_path, object_model):
        self.model_path = model_path
        self.object_model = object_model
        self.timestamp = -1

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.hand = (self.stack.enter_context(HandTracker(self.model_path))
                         if self.model_path is not None else None)
            self.detector = (self.stack.enter_context(create_detector(self.object_model, 0.1))
                             if self.object_model is not None else None)
            return self
        except Exception:
            self.stack.close()
            raise

    def __call__(self, frame, sequence):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp = max(self.timestamp + 1, time.monotonic_ns() // 1_000_000)
        hand = (self.hand.detect_for_video(image, self.timestamp)
                if self.hand is not None else None)
        objects = (self.detector.detect_for_video(image, self.timestamp).detections
                   if self.detector is not None else [])
        return hand, objects

    def __exit__(self, *exc):
        self.stack.close()


def ensure_object_model(path):
    if path.suffix == ".onnx":
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise RuntimeError("YOLO model missing. Run: python prepare_yolo.py")
    elif path.suffix == ".tflite":
        ensure_model(path, OBJECT_MODEL_URL)
    else:
        raise ValueError("Object model must be an exported .onnx or a .tflite file")


def run(camera_index, model_path, mirror, object_model=None, score_threshold=0.35,
        calibration_path=PROFILE_PATH, *, hand_enabled=True):
    """Run the chosen models; target selection needs both hand and object modes."""
    if not hand_enabled and object_model is None:
        raise ValueError("Choose --hand, --object, or both")
    cv2.setNumThreads(4)  # Leave CPU capacity for capture, GUI, and hand inference.
    camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                "Cannot open webcam. Allow camera access for your terminal in macOS "
                "System Settings > Privacy & Security > Camera, close other camera "
                "apps, or try --camera 1."
            )
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        show_debug = False
        sequence = 0
        generation = 0
        previous_shape = None
        frame_times = deque(maxlen=60)
        recognizer = PointingRecognizer(calibration_path) if hand_enabled else None
        pose = recognizer.pose if recognizer is not None else None
        target_tracker = TargetTracker()
        with LatestFrameWorker(
            lambda: VisionProcessor(model_path if hand_enabled else None, object_model)
        ) as worker:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Webcam stopped returning frames.")
                captured_at = time.monotonic()
                # Some iPhone cameras ignore capture-size requests. Bound processing size.
                if frame.shape[1] > 960:
                    new_height = max(1, round(frame.shape[0] * 960 / frame.shape[1]))
                    frame = cv2.resize(frame, (960, new_height), interpolation=cv2.INTER_AREA)
                if mirror:
                    frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                if previous_shape != frame.shape:
                    if pose is not None:
                        pose.cancel('Calibration cancelled: camera size changed; press C again')
                        recognizer.reset_confirmation()
                    generation += 1
                    target_tracker.reset()
                    previous_shape = frame.shape
                sequence += 1
                worker.submit(frame, sequence, captured_at, generation)
                snapshot = worker.latest(generation)
                result, raw_detections = snapshot.value if snapshot is not None else (None, [])
                detections = visible_objects(raw_detections, score_threshold)
                hands = extract_hands(result if hand_enabled else None, (width, height))
                pixel_hands = hands.pixels
                for pixels in pixel_hands:
                    draw_hand(frame, pixels)
                if recognizer is not None:
                    gesture = recognizer.update(
                        hands.worlds, pixel_hands, hands.labels,
                        snapshot.sequence if snapshot is not None else None, width, height,
                    )
                    index, direction = gesture.index, gesture.direction
                    status, debug, stable = gesture.status, gesture.debug, gesture.stable
                else:
                    index = direction = None
                    status, debug, stable = "Object detection only", "", False
                if stable and object_model is not None:
                    pixels = pixel_hands[index]
                    proposal = select_target(pixels[8], direction, detections, (width, height))
                    target = target_tracker.update(proposal, detections, snapshot.sequence)
                else:
                    target_tracker.reset()
                    reason = ('Target selection requires --hand --object'
                              if object_model is None or not hand_enabled
                              else 'No target: confirm pointing first')
                    target = Selection(None, reason)
                draw_objects(frame, detections, selected_index=target.index)
                if stable:
                    pixels = pixel_hands[index]
                    status = "POINTING"
                    origin = tuple(np.rint(pixels[8]).astype(int))
                    endpoint = tuple(np.rint(pixels[8] + direction * 2 * np.hypot(width, height)).astype(int))
                    visible, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
                    if visible:
                        cv2.arrowedLine(frame, start, end, (0, 255, 80), 3, tipLength=0.06)
                elif len(pixel_hands) == 1:
                    # Show that tracking works even when the pose is unconfirmed.
                    # This amber preview is never a confirmed pointing result.
                    pixels = pixel_hands[0]
                    preview_direction = pointing_direction(pixels)
                    if preview_direction is not None:
                        origin = tuple(np.rint(pixels[8]).astype(int))
                        endpoint = tuple(np.rint(pixels[8] + preview_direction * 2 * np.hypot(width, height)).astype(int))
                        visible, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
                        if visible:
                            cv2.arrowedLine(frame, start, end, (0, 190, 255), 2, tipLength=0.06)
                if pose is not None and pose.capturing:
                    status = pose.notice
                # Put controls above the image so they never hide object labels.
                panel = np.full((224 if show_debug else 198, width, 3), 25, dtype=np.uint8)
                color = (0, 255, 80) if stable else (220, 220, 220)
                put_panel_text(panel, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                target_color = ((0, 255, 80) if target.index is not None else
                                (0, 190, 255) if 'Unclear' in target.status else (220, 220, 220))
                put_panel_text(panel, target.status, (12, 53), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, target_color, 2)
                controls = "Q: quit | M: mirror"
                if hand_enabled:
                    controls = "C: teach pointing | R: reset | " + controls
                if object_model is not None:
                    controls += " | D: details"
                if object_model is not None:
                    object_status = f"{Path(object_model).stem} | Objects: {len(detections)} | Minimum score: {score_threshold:.0%} | - / +: adjust"
                else:
                    object_status = "Objects OFF: add --object to detect items"
                put_panel_text(panel, object_status, (12, 79),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 200, 120), 1)
                put_panel_text(panel, controls, (12, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                frame_times.append(time.monotonic())
                fps = ((len(frame_times) - 1) / max(1e-6, frame_times[-1] - frame_times[0]))
                metrics = f"Preview: {fps:.0f} FPS"
                if snapshot is not None:
                    age = (time.monotonic() - snapshot.captured_at) * 1000
                    metrics += f" | Inference: {snapshot.inference_ms:.0f} ms | Overlay age: {age:.0f} ms"
                put_panel_text(panel, metrics, (12, 131), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (190, 190, 190), 1)
                if hand_enabled:
                    hint = 'Amber ray: preview | Green ray: pointing'
                    if object_model is not None:
                        hint += ' | Green box: selected'
                else:
                    hint = 'Blue boxes: detected objects | - / +: adjust minimum score'
                if pose is not None and pose.notice:
                    hint += ' | ' + pose.notice
                put_panel_text(panel, hint, (12, 157),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                put_panel_text(panel, debug, (12, 183), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (190, 190, 190), 1)
                if show_debug:
                    low_scores = []
                    for detection in ranked_objects(raw_detections):
                        category = max(detection.categories, key=lambda item: item.score)
                        if category.score < score_threshold:
                            low_scores.append(f"{category.category_name} {category.score:.0%}")
                    detail = "Below threshold (unconfirmed): " + (", ".join(low_scores[:3]) or "none")
                    put_panel_text(panel, detail, (12, 209), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (190, 190, 190), 1)
                cv2.imshow(WINDOW, np.vstack((panel, frame)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("m"):
                    mirror = not mirror
                    generation += 1
                    if pose is not None:
                        pose.cancel('Calibration cancelled: mirror changed; press C again')
                        recognizer.reset_confirmation()
                    target_tracker.reset()
                if object_model is not None and key == ord("d"):
                    show_debug = not show_debug
                if pose is not None and key == ord('c'):
                    pose.start()
                    target_tracker.reset()
                    recognizer.reset_confirmation()
                if pose is not None and key == ord('r'):
                    pose.reset()
                    target_tracker.reset()
                    recognizer.reset_confirmation()
                if object_model is not None and key in (ord("-"), ord("_")):
                    score_threshold = max(0.1, round(score_threshold - 0.05, 2))
                    target_tracker.reset()
                if object_model is not None and key in (ord("+"), ord("=")):
                    score_threshold = min(1.0, round(score_threshold + 0.05, 2))
                    target_tracker.reset()
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="Default: enable both hand gestures and object detection when neither mode flag is supplied.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--hand", action="store_true", help="Enable hand tracking and gesture recognition")
    parser.add_argument("--object", action="store_true", help="Enable object detection")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Hand landmark model (used with --hand)")
    parser.add_argument('--calibration', type=Path, default=PROFILE_PATH,
                        help='Personal pointing profile (saved when C calibration finishes)')
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--download-model-only", action="store_true",
                        help="Prepare models for the selected modes without opening the camera")
    parser.add_argument("--object-model", type=Path, default=OBJECT_MODEL_PATH)
    parser.add_argument("--list-object-labels", action="store_true",
                        help="Print recognised object categories without opening the camera")
    parser.add_argument("--score-threshold", type=float, default=0.35,
                        help="Minimum object detection score, from 0.1 to 1 (default: 0.35)")
    args = parser.parse_args(argv)
    if not args.hand and not args.object:
        args.hand = args.object = True
    if not 0.1 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0.1 and 1")
    try:
        if args.list_object_labels:
            ensure_object_model(args.object_model)
            print("Supported object labels (person excluded):")
            print(", ".join(supported_labels(args.object_model)))
            return
        if args.hand:
            ensure_model(args.model)
        if args.object:
            ensure_object_model(args.object_model)
        if not args.download_model_only:
            run(args.camera, args.model, not args.no_mirror,
                args.object_model if args.object else None, args.score_threshold,
                args.calibration, hand_enabled=args.hand)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
