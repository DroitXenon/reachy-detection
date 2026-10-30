"""YOLO11 ONNX detection, with the earlier MediaPipe detector as a fallback."""

import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OBJECT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolo11s.onnx"
OBJECT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
)


def supported_labels(model_path):
    """Read the actual label map bundled with this detector model."""
    if Path(model_path).suffix == ".onnx":
        labels = json.loads(Path(model_path).with_suffix(".json").read_text())["names"].values()
        return sorted(set(labels) - {"person"})
    with ZipFile(model_path) as model:
        labels = model.read("labels.txt").decode("utf-8").splitlines()
    return sorted({label.strip() for label in labels} - {"", "???", "person"})


def create_detector(model_path, score_threshold=0.35):
    """Create a video detector; the caller must close it (or use a with block)."""
    if Path(model_path).suffix == ".onnx":
        return YoloDetector(model_path, max(0.1, score_threshold))
    options = vision.ObjectDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        score_threshold=score_threshold,
        max_results=-1,
        category_denylist=["person"],
    )
    return vision.ObjectDetector.create_from_options(options)


def decode_yolo(output, names, width, height, scale_x, scale_y, pad_x, pad_y, threshold):
    """Map YOLO's letterboxed xywh output back to the source image, then NMS."""
    rows = output[0].T  # Static YOLO11 export: [1, 4 + 80, 8400].
    class_ids = rows[:, 4:].argmax(axis=1)
    scores = rows[np.arange(len(rows)), class_ids + 4]
    boxes, confidences, classes = [], [], []
    # Filter in NumPy before Python iteration; walking all 8400 anchors holds the
    # GIL and can stall the GUI even though inference runs in another thread.
    for index in np.flatnonzero(scores >= threshold):
        row, class_id, score = rows[index], class_ids[index], scores[index]
        label = names[int(class_id)]
        if score < threshold or label == "person" or not np.isfinite(row[:4]).all():
            continue
        cx, cy, bw, bh = row[:4]
        left = np.clip((cx - bw / 2 - pad_x) / scale_x, 0, width - 1)
        top = np.clip((cy - bh / 2 - pad_y) / scale_y, 0, height - 1)
        right = np.clip((cx + bw / 2 - pad_x) / scale_x, 0, width - 1)
        bottom = np.clip((cy + bh / 2 - pad_y) / scale_y, 0, height - 1)
        if right - left < 2 or bottom - top < 2:
            continue
        boxes.append([int(left), int(top), int(right - left), int(bottom - top)])
        confidences.append(float(score))
        classes.append(label)
    # Class-agnostic suppression avoids several competing labels on one item.
    keep = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.5) if boxes else []
    detections = []
    for index in np.asarray(keep).reshape(-1)[:30]:
        x, y, w, h = boxes[index]
        detections.append(SimpleNamespace(
            bounding_box=SimpleNamespace(origin_x=x, origin_y=y, width=w, height=h),
            categories=[SimpleNamespace(category_name=classes[index], display_name=None,
                                        score=confidences[index])],
        ))
    return detections


class YoloDetector:
    """OpenCV runs the exported model; no PyTorch needed in the app environment."""

    def __init__(self, path, threshold):
        metadata = json.loads(Path(path).with_suffix(".json").read_text())
        self.names = {int(key): value for key, value in metadata["names"].items()}
        self.size = metadata["input_size"]
        self.threshold = threshold
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.net = None

    def detect_for_video(self, image, timestamp_ms):
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        scale = min(self.size / width, self.size / height)
        resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
        pad_x, pad_y = (self.size - resized_w) // 2, (self.size - resized_h) // 2
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = cv2.resize(rgb, (resized_w, resized_h))
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=False))
        output = self.net.forward()
        detections = decode_yolo(output, self.names, width, height, resized_w / width,
                                 resized_h / height, pad_x, pad_y, self.threshold)
        return SimpleNamespace(detections=detections)


def ranked_objects(detections):
    """Return non-person candidates in descending confidence order."""
    candidates = []
    for detection in detections:
        if not detection.categories:
            continue
        category = max(detection.categories, key=lambda item: item.score)
        if category.category_name not in (None, "", "???", "person"):
            candidates.append(detection)
    return sorted(candidates, key=lambda item: max(c.score for c in item.categories), reverse=True)


def visible_objects(detections, threshold):
    return [d for d in ranked_objects(detections)
            if max(c.score for c in d.categories) >= threshold][:10]


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
