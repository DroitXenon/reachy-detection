"""EfficientDet-Lite0 detection and labelled boxes for the camera preview."""

from pathlib import Path

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OBJECT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "efficientdet_lite0.tflite"
OBJECT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
)


def create_detector(model_path, score_threshold=0.4):
    """Create a video detector; the caller must close it (or use a with block)."""
    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        score_threshold=score_threshold,
        max_results=10,
    )
    return vision.ObjectDetector.create_from_options(options)


def draw_objects(frame, detections):
    """Draw boxes in the same pixel coordinate system used by hand tracking."""
    height, width = frame.shape[:2]
    color = (255, 170, 40)
    for detection in detections:
        if not detection.categories:
            continue
        box = detection.bounding_box
        left = max(0, box.origin_x)
        top = max(0, box.origin_y)
        right = min(width - 1, box.origin_x + box.width)
        bottom = min(height - 1, box.origin_y + box.height)
        if right <= left or bottom <= top:
            continue
        category = max(detection.categories, key=lambda item: item.score)
        name = category.category_name or category.display_name or "object"
        label = f"{name} {category.score:.0%}"
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_x = max(0, min(left, width - text_width - 8))
        text_y = max(text_height + 6, top - 6)
        cv2.rectangle(frame, (text_x, text_y - text_height - 5),
                      (min(width - 1, text_x + text_width + 8), text_y + baseline + 3),
                      color, -1)
        cv2.putText(frame, label, (text_x + 4, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 1, cv2.LINE_AA)
