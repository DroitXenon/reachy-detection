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
from gesture_recog import PointingRecognizer, pointing_direction
from object_detection import (
    OBJECT_MODEL_PATH, OBJECT_MODEL_URL, create_detector, draw_objects,
    ranked_objects, supported_labels, visible_objects,
)
from target_selection import Selection, TargetTracker, box_iou, select_target, target_name
from vision_worker import LatestFrameWorker
from interaction import SelectionResponder
from camera_capture import LatestCamera
from preview_tracking import PreviewTracker
from preview_window import PreviewWindow


WINDOW = "Reachy - Vision"


def put_panel_text(panel, text, origin, font, scale, color, thickness):
    """Keep diagnostics readable within narrower webcam previews."""
    text_width = cv2.getTextSize(text, font, scale, thickness)[0][0]
    available = max(1, panel.shape[1] - origin[0] - 12)
    fitted_scale = scale * min(1.0, available / max(1, text_width))
    cv2.putText(panel, text, origin, font, fitted_scale, color, thickness)


def preview_selection_index(target, detections, display_detections, snapshot):
    """Match a fresh selected object to its independently tracked display box."""
    if target.index is None or snapshot is None:
        return None
    raw = detections[target.index]
    box = raw.bounding_box
    bounds = (box.origin_x, box.origin_y,
              box.origin_x + box.width, box.origin_y + box.height)
    best, overlap = None, 0.3
    for index, item in enumerate(display_detections):
        source = getattr(item, 'preview_source_box', None)
        if (source is None or item.preview_source_sequence != snapshot.sequence
                or getattr(item, 'preview_held', False) or target_name(item) != target_name(raw)):
            continue
        x, y, width, height = source
        score = box_iou(bounds, (x, y, x + width, y + height))
        if score >= overlap:
            best, overlap = index, score
    return best


def measured_rate(timestamps, now):
    if len(timestamps) < 2 or now - timestamps[-1] > 0.5:
        return 0.0
    return (len(timestamps) - 1) / max(1e-6, timestamps[-1] - timestamps[0])


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
        *, hand_enabled=True,
        robot_enabled=False, speech_enabled=False, target_fps=60, pace=True):
    """Run the chosen models; target selection needs both hand and object modes."""
    if not hand_enabled and object_model is None:
        raise ValueError("Choose --hand, --object, or both")
    if not 1 <= target_fps <= 120:
        raise ValueError("Target FPS must be between 1 and 120")
    cv2.setNumThreads(4)  # Leave CPU capacity for capture, GUI, and hand inference.
    resources = ExitStack()
    try:
        camera = resources.enter_context(LatestCamera(camera_index, target_fps=target_fps))
        captured = camera.wait_first()
        window = resources.enter_context(PreviewWindow(WINDOW))
        show_debug = False
        generation = 0
        previous_shape = None
        frame_times = deque(maxlen=120)
        capture_times = deque(maxlen=120)
        inference_times = deque(maxlen=60)
        last_capture_sequence = last_inference_sequence = None
        submitted = None
        next_preview_at = time.monotonic()
        preview_tracker = PreviewTracker(score_threshold)
        recognizer = PointingRecognizer() if hand_enabled else None
        target_tracker = TargetTracker()
        with (
            LatestFrameWorker(
                lambda: VisionProcessor(model_path if hand_enabled else None, object_model)
            ) as worker,
            SelectionResponder(robot_enabled, speech_enabled) as responder,
        ):
            while True:
                newest = camera.latest()
                if newest is not None:
                    captured = newest
                sequence, captured_at = captured.sequence, captured.captured_at
                if sequence != last_capture_sequence:
                    capture_times.append(captured_at)
                    last_capture_sequence = sequence
                frame = captured.frame.copy()
                if mirror:
                    frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                if previous_shape != frame.shape:
                    if recognizer is not None:
                        recognizer.reset_confirmation()
                    generation += 1
                    target_tracker.reset()
                    responder.reset()
                    preview_tracker.reset()
                    previous_shape = frame.shape
                if submitted != (generation, sequence):
                    worker.submit(frame, sequence, captured_at, generation)
                    submitted = generation, sequence
                visual_snapshot = worker.latest(generation, max_age=0.5)
                now = time.monotonic()
                snapshot = (visual_snapshot if visual_snapshot is not None
                            and now - visual_snapshot.captured_at <= 0.30 else None)
                if visual_snapshot is not None and visual_snapshot.sequence != last_inference_sequence:
                    inference_times.append(visual_snapshot.completed_at)
                    last_inference_sequence = visual_snapshot.sequence
                preview = preview_tracker.update(frame, sequence, generation, captured_at, visual_snapshot)
                result, raw_detections = snapshot.value if snapshot is not None else (None, [])
                detections = visible_objects(raw_detections, score_threshold)
                hands = extract_hands(result if hand_enabled else None, (width, height), include_world=False)
                pixel_hands = hands.pixels
                for pixels in preview.pixels:
                    draw_hand(frame, pixels)
                if recognizer is not None:
                    gesture = recognizer.update(
                        pixel_hands, snapshot.sequence if snapshot is not None else None,
                    )
                    index, direction = gesture.index, gesture.direction
                    status, debug, stable = gesture.status, gesture.debug, gesture.stable
                else:
                    index = direction = None
                    status, debug, stable = "Object detection only", "", False
                if stable and direction is not None and object_model is not None:
                    pixels = pixel_hands[index]
                    proposal = select_target(pixels[8], direction, detections, (width, height))
                    target = target_tracker.update(proposal, detections, snapshot.sequence)
                else:
                    target_tracker.reset()
                    reason = ('Target selection requires --hand --object'
                              if object_model is None or not hand_enabled
                              else 'No target: point across the image' if stable
                              else 'No target: confirm pointing first')
                    target = Selection(None, reason)
                responder.update(
                    target, detections, (width, height), mirror,
                    snapshot.sequence if snapshot is not None else None,
                    snapshot.captured_at if snapshot is not None else None,
                )
                selected_display = preview_selection_index(target, detections, preview.detections, snapshot)
                draw_objects(frame, preview.detections, selected_index=selected_display)
                if stable and direction is not None:
                    pixels = pixel_hands[index]
                    current_visuals = [item for item, source in
                                       zip(preview.pixels, preview.hand_source_sequences)
                                       if source == snapshot.sequence]
                    if current_visuals:
                        pixels = min(current_visuals, key=lambda item: np.linalg.norm(item[0] - pixels[0]))
                    display_direction = pointing_direction(pixels)
                    if display_direction is None:
                        display_direction = direction
                    status = "POINTING"
                    origin = tuple(np.rint(pixels[8]).astype(int))
                    endpoint = tuple(np.rint(pixels[8] + display_direction * 2 * np.hypot(width, height)).astype(int))
                    visible, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
                    if visible:
                        cv2.arrowedLine(frame, start, end, (0, 255, 80), 3, tipLength=0.06)
                elif not stable and len(preview.pixels) == 1:
                    # Show that tracking works even when the pose is unconfirmed.
                    # This amber preview is never a confirmed pointing result.
                    pixels = preview.pixels[0]
                    preview_direction = pointing_direction(pixels)
                    if preview_direction is not None:
                        origin = tuple(np.rint(pixels[8]).astype(int))
                        endpoint = tuple(np.rint(pixels[8] + preview_direction * 2 * np.hypot(width, height)).astype(int))
                        visible, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
                        if visible:
                            cv2.arrowedLine(frame, start, end, (0, 190, 255), 2, tipLength=0.06)
                # Put controls above the image so they never hide object labels.
                panel = np.full((250 if show_debug else 224, width, 3), 25, dtype=np.uint8)
                color = (0, 255, 80) if stable else (220, 220, 220)
                put_panel_text(panel, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                target_color = ((0, 255, 80) if target.index is not None else
                                (0, 190, 255) if 'Unclear' in target.status else (220, 220, 220))
                put_panel_text(panel, target.status, (12, 53), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, target_color, 2)
                controls = "Q: quit | M: mirror"
                if hand_enabled:
                    controls = "R: reset pointing | " + controls
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
                now = time.monotonic()
                frame_times.append(now)
                fps = measured_rate(frame_times, now)
                camera_fps = measured_rate(capture_times, now)
                neural_fps = measured_rate(inference_times, now)
                metrics = f"Preview: {fps:.0f}/{target_fps:g} FPS | Camera: {camera_fps:.0f} FPS | Models: {neural_fps:.1f} Hz"
                if visual_snapshot is not None:
                    age = (now - visual_snapshot.captured_at) * 1000
                    metrics += f" | Inference: {visual_snapshot.inference_ms:.0f} ms | Source: {age:.0f} ms"
                put_panel_text(panel, metrics, (12, 131), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (190, 190, 190), 1)
                if hand_enabled:
                    hint = 'Amber ray: preview | Green ray: pointing'
                    if object_model is not None:
                        hint += ' | Green box: selected'
                else:
                    hint = 'Blue boxes: detected objects | - / +: adjust minimum score'
                if preview.held:
                    hint += ' | Visual tracking between predictions'
                put_panel_text(panel, hint, (12, 157),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                put_panel_text(panel, responder.status, (12, 183), cv2.FONT_HERSHEY_SIMPLEX,
                               0.45, (190, 190, 190), 1)
                put_panel_text(panel, debug, (12, 209), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (190, 190, 190), 1)
                if show_debug:
                    low_scores = []
                    for detection in ranked_objects(raw_detections):
                        category = max(detection.categories, key=lambda item: item.score)
                        if category.score < score_threshold:
                            low_scores.append(f"{category.category_name} {category.score:.0%}")
                    detail = "Below threshold (unconfirmed): " + (", ".join(low_scores[:3]) or "none")
                    put_panel_text(panel, detail, (12, 235), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (190, 190, 190), 1)
                window.show(np.vstack((panel, frame)))
                key = window.poll_key()
                if key in (ord("q"), 27) or not window.is_open():
                    break
                if key == ord("m"):
                    mirror = not mirror
                    preview_tracker.reset()
                    generation += 1
                    if recognizer is not None:
                        recognizer.reset_confirmation()
                    target_tracker.reset()
                    responder.reset()
                if object_model is not None and key == ord("d"):
                    show_debug = not show_debug
                if recognizer is not None and key == ord('r'):
                    preview_tracker.reset()
                    target_tracker.reset()
                    responder.reset()
                    recognizer.reset_confirmation()
                if object_model is not None and key in (ord("-"), ord("_")):
                    score_threshold = max(0.1, round(score_threshold - 0.05, 2))
                    preview_tracker.score_threshold = score_threshold
                    preview_tracker.reset()
                    target_tracker.reset()
                    responder.reset()
                if object_model is not None and key in (ord("+"), ord("=")):
                    score_threshold = min(1.0, round(score_threshold + 0.05, 2))
                    preview_tracker.score_threshold = score_threshold
                    preview_tracker.reset()
                    target_tracker.reset()
                    responder.reset()
                if pace:
                    next_preview_at += 1 / target_fps
                    remaining = next_preview_at - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                    elif remaining < -2 / target_fps:
                        # Recover from a long stall without letting small
                        # scheduling delays lower the sustained redraw rate.
                        next_preview_at = time.monotonic()
    finally:
        resources.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        epilog="Default: enable both hand gestures and object detection when neither mode flag is supplied.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--fps", type=float, default=60, help="Target preview/camera FPS (default: 60)")
    parser.add_argument("--hand", action="store_true", help="Enable hand tracking and gesture recognition")
    parser.add_argument("--object", action="store_true", help="Enable object detection")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Hand landmark model (used with --hand)")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-robot", action="store_true", help="Disable simulated head movement")
    parser.add_argument("--mute", action="store_true", help="Disable spoken object names")
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
    if not 1 <= args.fps <= 120:
        parser.error("--fps must be between 1 and 120")
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
                hand_enabled=args.hand,
                robot_enabled=args.hand and args.object and not args.no_robot,
                speech_enabled=args.hand and args.object and not args.mute,
                target_fps=args.fps)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
