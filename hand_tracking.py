"""Webcam landmarks and a conservative, rule-based pointing gesture demo."""

import argparse
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


def ensure_model(path):
    """Download the official model once; never leave a partial model file."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Hand Landmarker model to {path}...", flush=True)
    temporary = None
    try:
        with urlopen(MODEL_URL, timeout=60) as response:
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


def is_pointing(world_points):
    """Use 3D joint angles so the rule does not depend on image orientation."""
    index_straight = (
        joint_angle(*world_points[[5, 6, 7]]) >= 155
        and joint_angle(*world_points[[6, 7, 8]]) >= 155
    )
    other_fingers_bent = all(
        joint_angle(*world_points[[base, base + 1, base + 2]]) <= 140
        for base in (9, 13, 17)
    )
    return index_straight and other_fingers_bent


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


def run(camera_index, model_path, mirror):
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
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
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        consecutive = 0
        timestamp = -1
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Webcam stopped returning frames.")
                if mirror:
                    frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = max(timestamp + 1, time.monotonic_ns() // 1_000_000)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp
                )
                hands = result.hand_landmarks
                candidate = False
                direction = None
                pixels = None
                status = "No hand detected"
                for landmarks in hands:
                    pixels = np.array([(p.x * width, p.y * height) for p in landmarks])
                    draw_hand(frame, pixels)
                if len(hands) > 1:
                    status = "Unclear: show one hand only"
                elif len(hands) == 1:
                    world = np.array([
                        (p.x, p.y, p.z) for p in result.hand_world_landmarks[0]
                    ])
                    status = "Extend index finger; curl the other fingers"
                    if is_pointing(world):
                        direction = pointing_direction(pixels)
                        candidate = direction is not None
                        status = (
                            "Hold pointing gesture..." if candidate
                            else "Unclear: point across the image"
                        )
                consecutive = min(consecutive + 1, 3) if candidate else 0
                stable = consecutive >= 3
                if stable:
                    status = "POINTING"
                    origin = tuple(np.rint(pixels[8]).astype(int))
                    endpoint = tuple(np.rint(pixels[8] + direction * max(width, height)).astype(int))
                    visible, start, end = cv2.clipLine((0, 0, width, height), origin, endpoint)
                    if visible:
                        cv2.arrowedLine(frame, start, end, (0, 255, 80), 3, tipLength=0.06)
                cv2.rectangle(frame, (0, 0), (width, 68), (25, 25, 25), -1)
                color = (0, 255, 80) if stable else (220, 220, 220)
                cv2.putText(frame, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, "Q / Esc: quit | M: mirror", (12, 53),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)
                cv2.imshow(WINDOW, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("m"):
                    mirror = not mirror
                    consecutive = 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--download-model-only", action="store_true")
    args = parser.parse_args()
    try:
        ensure_model(args.model)
        if not args.download_model_only:
            run(args.camera, args.model, not args.no_mirror)
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
