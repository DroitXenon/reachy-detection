"""MediaPipe hand landmarks and drawing, independent of gesture recognition.

Run ``python main.py --hand`` for the hand and gesture application. The old
``python hand_tracking.py`` command also forwards to that entry point.
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def ensure_model(path, url=MODEL_URL):
    """Download the model once without leaving a partial model file."""
    path = Path(path)
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


@dataclass(frozen=True)
class TrackedHands:
    """Aligned image landmarks, world landmarks, and left/right hand labels."""

    pixels: tuple
    worlds: tuple
    labels: tuple


def extract_hands(result, image_size):
    """Convert a MediaPipe result into arrays without deciding any gesture.

    ``image_size`` is (width, height) of the exact frame used for inference.
    A missing snapshot produces empty hands so stale gestures can be cleared.
    """
    if result is None:
        return TrackedHands((), (), ())
    width, height = image_size
    return TrackedHands(
        pixels=tuple(np.array([(p.x * width, p.y * height) for p in hand])
                     for hand in result.hand_landmarks),
        worlds=tuple(np.array([(p.x, p.y, p.z) for p in hand])
                     for hand in result.hand_world_landmarks),
        labels=tuple(categories[0].category_name for categories in result.handedness),
    )


class HandTracker:
    """Own the hand model; callers supply an RGB MediaPipe image and timestamp.

    Use on a single inference thread in a ``with`` block. The combined pipeline
    supplies increasing timestamps shared with object detection on the same frame.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = Path(model_path)

    def __enter__(self):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        return self

    def detect_for_video(self, image, timestamp_ms):
        return self.landmarker.detect_for_video(image, timestamp_ms)

    def __exit__(self, *exc):
        self.landmarker.close()


def draw_hand(frame, pixels):
    points = np.rint(pixels).astype(int)
    for start, end in CONNECTIONS:
        cv2.line(frame, tuple(points[start]), tuple(points[end]), (230, 180, 40), 2)
    for point in points:
        cv2.circle(frame, tuple(point), 4, (0, 230, 255), -1)


if __name__ == "__main__":
    from main import main
    main()
