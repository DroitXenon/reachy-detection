"""Webcam landmarks and a conservative, rule-based pointing gesture demo."""

import argparse
from collections import deque
from contextlib import ExitStack
import math
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


def joint_angle(a, b, c):
    """Angle at joint b, in degrees. Degenerate coordinates are invalid."""
    u, v = a - b, c - b
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator < 1e-10:
        return float("nan")
    cosine = np.clip(np.dot(u, v) / denominator, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def finger_metrics(points, base):
    """Measure straightness by joint angles and tip distance / bone length."""
    finger = points[base:base + 4]
    bones = np.linalg.norm(np.diff(finger, axis=0), axis=1)
    if not np.isfinite(finger).all() or np.any(bones < 1e-10):
        return float("nan"), float("nan"), float("nan")
    reach = np.linalg.norm(finger[-1] - finger[0]) / bones.sum()
    return joint_angle(*finger[:3]), joint_angle(*finger[1:]), reach


def pointing_feedback(world_points):
    """Accept a relaxed index point, but reject a palm or a clear V sign."""
    metrics = [finger_metrics(world_points, base) for base in (5, 9, 13, 17)]
    if not np.isfinite(metrics).all():
        return False, "Hand unclear: keep fingers visible", "Invalid landmarks"
    pip, dip, reach = metrics[0]
    debug = f"Index joints: {pip:.0f}/{dip:.0f} deg | reach: {reach:.2f}"
    if pip < 140 or dip < 135 or reach < 0.86:
        return False, "Straighten your index finger a little", debug
    # Other fingers need not be tightly folded. Only clearly extended fingers
    # count against pointing; the thumb is deliberately ignored.
    extended = [a >= 155 and b >= 155 and r >= 0.92 for a, b, r in metrics[1:]]
    middle_pip, middle_dip, middle_reach = metrics[1]
    middle_similarly_straight = (
        middle_reach > reach - 0.03
        and min(middle_pip, middle_dip) > min(pip, dip) - 15
    )
    if extended[0] or middle_similarly_straight:
        return False, "Relax your middle finger (avoid a V sign)", debug
    if sum(extended) > 1:
        return False, "Relax your ring and little fingers", debug
    return True, "Pointing pose detected", debug


def is_pointing(world_points):
    return pointing_feedback(world_points)[0]


def choose_pointing_hand(world_hands, pixel_hands):
    """Allow a resting second hand, but never choose between two pointing hands."""
    candidates = []
    feedback = []
    for index, (world, pixels) in enumerate(zip(world_hands, pixel_hands)):
        pointing, reason, debug = pointing_feedback(world)
        feedback.append((reason, debug))
        if pointing:
            candidates.append((index, pointing_direction(pixels), debug))
    if len(candidates) > 1:
        return None, None, "Unclear: two hands pointing", ""
    if candidates:
        index, direction, debug = candidates[0]
        if direction is None:
            return None, None, "Unclear: point across the image", debug
        return index, direction, "Hold pointing gesture...", debug
    if feedback:
        reason = feedback[0][0] if len(feedback) == 1 else "No pointing pose: extend one index finger"
        return None, None, reason, feedback[0][1]
    return None, None, "No hand detected", ""


def pointing_direction(pixels):
    """Return a unit screen-space direction, or reject a short projection."""
    direction = pixels[8] - pixels[5]
    length = np.linalg.norm(direction)
    palm_length = np.linalg.norm(pixels[9] - pixels[0])
    if length < max(15, 0.3 * palm_length):
        return None
    return direction / length


def draw_hand(frame, pixels):
    points = np.rint(pixels).astype(int)
    for start, end in CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), (230, 180, 40), 2)
    for point in points:
        cv2.circle(frame, tuple(point), 4, (0, 230, 255), -1)


class VisionProcessor:
    """Both models process the same clean frame, entirely on the worker thread."""

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


def run(camera_index, model_path, mirror, object_model=None, score_threshold=0.35):
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
                world_hands = [np.array([(p.x, p.y, p.z) for p in landmarks])
                               for landmarks in result.hand_world_landmarks] if result is not None else []
                index, direction, status, debug = choose_pointing_hand(world_hands, pixel_hands)
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
                stable = consecutive >= 2
                if stable:
                    pixels = pixel_hands[index]
                    status = "POINTING"
                    origin = tuple(np.rint(pixels[8]).astype(int))
                    endpoint = tuple(np.rint(pixels[8] + direction * max(width, height)).astype(int))
                    visible, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
                    if visible:
                        cv2.arrowedLine(frame, start, end, (0, 255, 80), 3, tipLength=0.06)
                # Put controls above the image so they never hide object labels.
                panel = np.full((172 if show_debug else 120, width, 3), 25, dtype=np.uint8)
                color = (0, 255, 80) if stable else (220, 220, 220)
                cv2.putText(panel, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                controls = "Q / Esc: quit | M: mirror | D: details"
                if object_model is not None:
                    object_status = f"Objects: {len(detections)} | Minimum score: {score_threshold:.0%} | - / +: adjust"
                else:
                    object_status = "Objects OFF: restart with --objects to detect items"
                cv2.putText(panel, object_status, (12, 53),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 200, 120), 1)
                cv2.putText(panel, controls, (12, 79),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                frame_times.append(time.monotonic())
                fps = ((len(frame_times) - 1) / max(1e-6, frame_times[-1] - frame_times[0]))
                metrics = f"Preview: {fps:.0f} FPS"
                if snapshot is not None:
                    age = (time.monotonic() - snapshot.captured_at) * 1000
                    metrics += f" | Inference: {snapshot.inference_ms:.0f} ms | Overlay age: {age:.0f} ms"
                cv2.putText(panel, metrics, (12, 105), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (190, 190, 190), 1)
                if show_debug:
                    cv2.putText(panel, debug, (12, 131), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (190, 190, 190), 1)
                    low_scores = []
                    for detection in ranked_objects(raw_detections):
                        category = max(detection.categories, key=lambda item: item.score)
                        if category.score < score_threshold:
                            low_scores.append(f"{category.category_name} {category.score:.0%}")
                    detail = "Below threshold (unconfirmed): " + (", ".join(low_scores[:3]) or "none")
                    cv2.putText(panel, detail, (12, 157), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (190, 190, 190), 1)
                cv2.imshow(WINDOW, np.vstack((panel, frame)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("m"):
                    mirror = not mirror
                    generation += 1
                    consecutive = 0
                    previous_hand = previous_wrist = None
                if key == ord("d"):
                    show_debug = not show_debug
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
                args.object_model if args.objects else None, args.score_threshold)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
