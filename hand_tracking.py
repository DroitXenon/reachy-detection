"""Webcam landmarks, personally calibrated pointing, and object detection."""

import argparse
from collections import deque
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from object_detection import (
    OBJECT_MODEL_PATH, OBJECT_MODEL_URL, create_detector, draw_objects,
    ranked_objects, supported_labels, visible_objects,
)
from vision_worker import LatestFrameWorker
from gesture_recognition import (
    PROFILE_PATH, PointingPose, choose_pointing_hand, pointing_direction,
)


MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WINDOW = "Reachy - Hand tracking"
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def ensure_model(path, url=MODEL_URL):
    """Download the official model once; never leave a partial model file."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model to {path}...", flush=True)
    temporary = None
    try:
        with urlopen(url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
                temporary = Path(output.name)
                shutil.copyfileobj(response, output)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def draw_hand(frame, pixels):
    points = np.rint(pixels).astype(int)
    for start, end in CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), (230, 180, 40), 2)
    for point in points:
        cv2.circle(frame, tuple(point), 4, (0, 230, 255), -1)


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
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.hand = self.stack.enter_context(vision.HandLandmarker.create_from_options(options))
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
        hand = self.hand.detect_for_video(image, self.timestamp)
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
        calibration_path=PROFILE_PATH):
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
        consecutive = 0
        previous_hand = None
        previous_wrist = None
        show_debug = False
        sequence = 0
        generation = 0
        previous_shape = None
        last_processed = -1
        frame_times = deque(maxlen=60)
        pose = PointingPose(calibration_path)
        with LatestFrameWorker(lambda: VisionProcessor(model_path, object_model)) as worker:
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
                    pose.cancel('Calibration cancelled: camera size changed; press C again')
                    generation += 1
                    previous_shape = frame.shape
                sequence += 1
                worker.submit(frame, sequence, captured_at, generation)
                snapshot = worker.latest(generation)
                new_result = snapshot is not None and snapshot.sequence != last_processed
                result, raw_detections = snapshot.value if snapshot is not None else (None, [])
                detections = visible_objects(raw_detections, score_threshold)
                draw_objects(frame, detections)
                hands = result.hand_landmarks if result is not None else []
                pixel_hands = []
                for landmarks in hands:
                    pixels = np.array([(p.x * width, p.y * height) for p in landmarks])
                    pixel_hands.append(pixels)
                    draw_hand(frame, pixels)
                worlds = [np.array([(p.x, p.y, p.z) for p in landmarks])
                          for landmarks in result.hand_world_landmarks] if result is not None else []
                if pose.capturing and snapshot is None:
                    pose.cancel('Calibration cancelled: tracking lost; press C again')
                if new_result:
                    labels = [categories[0].category_name for categories in result.handedness]
                    wrists = [pixels[0] / [width, height] for pixels in pixel_hands]
                    pose.observe(worlds, labels, wrists, snapshot.sequence)
                index, direction, status, debug = choose_pointing_hand(worlds, pixel_hands, pose)
                if index is not None and new_result:
                    pixels = pixel_hands[index]
                    hand = result.handedness[index][0].category_name
                    # Do not accumulate confirmation across different hands.
                    same_hand = (hand == previous_hand and previous_wrist is not None
                                 and np.linalg.norm(pixels[0] - previous_wrist) < 0.25 * width)
                    consecutive = min(consecutive + 1, 2) if same_hand else 1
                    previous_hand, previous_wrist = hand, pixels[0].copy()
                elif index is None:
                    consecutive = 0
                    previous_hand = previous_wrist = None
                if snapshot is not None:
                    last_processed = snapshot.sequence
                else:
                    status = "Waiting for fresh detection..."
                stable = consecutive >= 2 and not pose.capturing
                if stable:
                    pixels = pixel_hands[index]
                    status = "POINTING"
                    origin = tuple(np.rint(pixels[8]).astype(int))
                    endpoint = tuple(np.rint(pixels[8] + direction * max(width, height)).astype(int))
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
                        endpoint = tuple(np.rint(pixels[8] + preview_direction * max(width, height)).astype(int))
                        visible, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
                        if visible:
                            cv2.arrowedLine(frame, start, end, (0, 190, 255), 2, tipLength=0.06)
                if pose.capturing:
                    status = pose.notice
                # Put controls above the image so they never hide object labels.
                panel = np.full((198 if show_debug else 172, width, 3), 25, dtype=np.uint8)
                color = (0, 255, 80) if stable else (220, 220, 220)
                put_panel_text(panel, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                controls = "C: teach pointing | R: reset | Q: quit | M: mirror | D: details"
                if object_model is not None:
                    object_status = f"{Path(object_model).stem} | Objects: {len(detections)} | Minimum score: {score_threshold:.0%} | - / +: adjust"
                else:
                    object_status = "Objects OFF: restart with --objects to detect items"
                put_panel_text(panel, object_status, (12, 53),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 200, 120), 1)
                put_panel_text(panel, controls, (12, 79),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                frame_times.append(time.monotonic())
                fps = ((len(frame_times) - 1) / max(1e-6, frame_times[-1] - frame_times[0]))
                metrics = f"Preview: {fps:.0f} FPS"
                if snapshot is not None:
                    age = (time.monotonic() - snapshot.captured_at) * 1000
                    metrics += f" | Inference: {snapshot.inference_ms:.0f} ms | Overlay age: {age:.0f} ms"
                put_panel_text(panel, metrics, (12, 105), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (190, 190, 190), 1)
                hint = 'Amber: unconfirmed direction | Green: confirmed pointing'
                if pose.notice:
                    hint += ' | ' + pose.notice
                put_panel_text(panel, hint, (12, 131),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                put_panel_text(panel, debug, (12, 157), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (190, 190, 190), 1)
                if show_debug:
                    low_scores = []
                    for detection in ranked_objects(raw_detections):
                        category = max(detection.categories, key=lambda item: item.score)
                        if category.score < score_threshold:
                            low_scores.append(f"{category.category_name} {category.score:.0%}")
                    detail = "Below threshold (unconfirmed): " + (", ".join(low_scores[:3]) or "none")
                    put_panel_text(panel, detail, (12, 183), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (190, 190, 190), 1)
                cv2.imshow(WINDOW, np.vstack((panel, frame)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("m"):
                    mirror = not mirror
                    generation += 1
                    pose.cancel('Calibration cancelled: mirror changed; press C again')
                    consecutive = 0
                    previous_hand = previous_wrist = None
                if key == ord("d"):
                    show_debug = not show_debug
                if key == ord('c'):
                    pose.start()
                    consecutive = 0
                    previous_hand = previous_wrist = None
                if key == ord('r'):
                    pose.reset()
                    consecutive = 0
                    previous_hand = previous_wrist = None
                if object_model is not None and key in (ord("-"), ord("_")):
                    score_threshold = max(0.1, round(score_threshold - 0.05, 2))
                if object_model is not None and key in (ord("+"), ord("=")):
                    score_threshold = min(1.0, round(score_threshold + 0.05, 2))
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument('--calibration', type=Path, default=PROFILE_PATH,
                        help='Personal pointing profile (saved when C calibration finishes)')
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--download-model-only", action="store_true")
    parser.add_argument("--objects", action="store_true", help="Also detect and label common objects")
    parser.add_argument("--object-model", type=Path, default=OBJECT_MODEL_PATH)
    parser.add_argument("--list-object-labels", action="store_true",
                        help="Print recognised object categories without opening the camera")
    parser.add_argument("--score-threshold", type=float, default=0.35,
                        help="Minimum object detection score, from 0.1 to 1 (default: 0.35)")
    args = parser.parse_args()
    if not 0.1 <= args.score_threshold <= 1:
        parser.error("--score-threshold must be between 0.1 and 1")
    try:
        if args.list_object_labels:
            ensure_object_model(args.object_model)
            print("Supported object labels (person excluded):")
            print(", ".join(supported_labels(args.object_model)))
            return
        ensure_model(args.model)
        if args.objects:
            ensure_object_model(args.object_model)
        if not args.download_model_only:
            run(args.camera, args.model, not args.no_mirror,
                args.object_model if args.objects else None, args.score_threshold,
                args.calibration)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
