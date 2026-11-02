"""Display-only optical flow and short retention between neural predictions.

These estimates are for drawing. They must never be supplied to gesture
confirmation, target selection, calibration, or robot actions.
"""

from collections import OrderedDict
from dataclasses import dataclass
from math import exp
from types import SimpleNamespace
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class PreviewResult:
    pixels: list
    detections: list
    age: float | None
    held: bool
    source_sequence: int | None
    hand_source_sequences: tuple = ()


@dataclass
class _Track:
    kind: str
    geometry: np.ndarray
    points: np.ndarray
    label: str
    categories: list
    source_sequence: int
    source_time: float
    source_box: tuple | None = None
    missed_at: float | None = None


def _iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    area = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return area / max(1e-9, aw * ah + bw * bh - area)


class PreviewTracker:
    """Track a bounded number of visual landmarks and boxes at camera rate.

    Frames are retained for 0.75 seconds so delayed model coordinates can be
    projected from their actual source frame. Missing neural detections get a
    0.22-second drawing grace period; no visual survives more than 0.5 seconds
    after its last positive source frame. Duplicate camera frames do no LK work.
    ``score_threshold`` can be changed alongside the app's confidence control.
    """

    def __init__(self, score_threshold=0.35, *, clock=time.monotonic):
        self.score_threshold = score_threshold
        self._clock = clock
        self.reset()

    def reset(self):
        self._history = OrderedDict()
        self._tracks = []
        self._generation = None
        self._shape = None
        self._sequence = None
        self._gray = None
        self._last_snapshot = None
        self._source_sequence = None

    @staticmethod
    def _flow(old, new, points):
        """Forward/backward LK validation, with one pyramid pass per batch."""
        points = np.asarray(points, np.float32).reshape(-1, 2)
        if len(points) == 0:
            return points, np.zeros(0, bool)
        if old is new:
            return points.copy(), np.ones(len(points), bool)
        options = dict(winSize=(21, 21), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, .03))
        forward, status, error = cv2.calcOpticalFlowPyrLK(old, new, points[:, None], None, **options)
        if forward is None:
            return points.copy(), np.zeros(len(points), bool)
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(new, old, forward, None, **options)
        if backward is None:
            return points.copy(), np.zeros(len(points), bool)
        forward = forward.reshape(-1, 2)
        valid = (status.ravel().astype(bool) & backward_status.ravel().astype(bool)
                 & np.isfinite(forward).all(axis=1)
                 & (np.linalg.norm(backward.reshape(-1, 2) - points, axis=1) < 1.5)
                 & (error.ravel() < 40))
        valid &= np.linalg.norm(forward - points, axis=1) < .2 * np.hypot(*old.shape)
        return forward, valid

    @staticmethod
    def _features(gray, box):
        x, y, width, height = box
        left, top = max(0, int(x)), max(0, int(y))
        right, bottom = min(gray.shape[1], int(x + width)), min(gray.shape[0], int(y + height))
        if right <= left or bottom <= top:
            return np.empty((0, 2), np.float32)
        crop = gray[top:bottom, left:right]
        points = cv2.goodFeaturesToTrack(crop, maxCorners=12, qualityLevel=.04,
                                         minDistance=5, blockSize=3)
        if points is None:
            return np.empty((0, 2), np.float32)
        return points.reshape(-1, 2) + [left, top]

    def _project(self, tracks, old, new):
        groups = [track.geometry if track.kind == 'hand' else track.points for track in tracks]
        if not groups or not sum(len(points) for points in groups):
            return
        moved, valid = self._flow(old, new, np.concatenate(groups))
        start = 0
        for track, points in zip(tracks, groups):
            stop = start + len(points)
            passed, projected = valid[start:stop], moved[start:stop]
            start = stop
            minimum = 5 if track.kind == 'hand' else 3
            if np.count_nonzero(passed) < minimum:
                # Low-texture frames retain their recent geometry rather than
                # inventing motion; source-age expiry still bounds retention.
                continue
            offset = np.median(projected[passed] - points[passed], axis=0)
            if track.kind == 'hand':
                prediction = track.geometry + offset
                prediction[passed] = projected[passed]
                track.geometry = prediction
            else:
                track.geometry = track.geometry + [*offset, 0, 0]
                track.points = projected[passed]

    def _incoming(self, snapshot, source_gray, size):
        hands, detections = snapshot.value
        width, height = size
        tracks = []
        landmarks = getattr(hands, 'hand_landmarks', ()) if hands is not None else ()
        labels = getattr(hands, 'handedness', ()) if hands is not None else ()
        for index, hand in enumerate(landmarks[:2]):
            pixels = np.array([(point.x * width, point.y * height) for point in hand], dtype=float)
            if pixels.shape != (21, 2) or not np.isfinite(pixels).all():
                continue
            label = labels[index][0].category_name if index < len(labels) and labels[index] else str(index)
            tracks.append(_Track('hand', pixels, np.empty((0, 2)), label, [],
                                 snapshot.sequence, snapshot.captured_at))
        candidates = []
        for detection in detections:
            categories = [SimpleNamespace(category_name=getattr(category, 'category_name', None),
                                           display_name=getattr(category, 'display_name', None),
                                           score=float(category.score))
                          for category in getattr(detection, 'categories', ()) if np.isfinite(category.score)]
            if not categories:
                continue
            category = max(categories, key=lambda item: item.score)
            if category.category_name in (None, '', '???', 'person') or category.score < self.score_threshold:
                continue
            box = detection.bounding_box
            geometry = np.array([box.origin_x, box.origin_y, box.width, box.height], dtype=float)
            if not np.isfinite(geometry).all() or min(geometry[2:]) <= 0:
                continue
            candidates.append((category.score, geometry, category.category_name, categories))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, geometry, label, categories in candidates[:10]:
            tracks.append(_Track('object', geometry, self._features(source_gray, geometry), label,
                                 categories, snapshot.sequence, snapshot.captured_at, tuple(geometry)))
        return tracks

    def _merge(self, incoming, captured_at):
        previous = list(self._tracks)
        observed = list(incoming)
        used = set()
        for track in incoming:
            best, score = None, 0.0
            for index, old in enumerate(previous):
                if index in used or old.kind != track.kind or old.label != track.label:
                    continue
                if track.kind == 'object':
                    similarity = _iou(track.geometry, old.geometry)
                else:
                    distance = np.linalg.norm(track.geometry[0] - old.geometry[0])
                    similarity = max(0, 1 - distance / (0.2 * np.hypot(*self._shape)))
                if similarity > score:
                    best, score = index, similarity
            if best is not None and score >= .2:
                old = previous[best]
                used.add(best)
                dt = max(0.0, track.source_time - old.source_time)
                alpha = max(.25, 1 - exp(-dt / .045))
                track.geometry = old.geometry * (1 - alpha) + track.geometry * alpha
        for index, old in enumerate(previous):
            if index not in used:
                replaced = any(
                    track.kind == old.kind and (
                        _iou(track.geometry, old.geometry) > .5 if old.kind == 'object'
                        else np.linalg.norm(track.geometry[0] - old.geometry[0]) < 15
                    ) for track in observed
                )
                if replaced:
                    continue
                if old.missed_at is None:
                    old.missed_at = captured_at
                incoming.append(old)
        # Model order comes first; temporarily retained misses follow it.
        limits, counts = {'hand': 2, 'object': 10}, {'hand': 0, 'object': 0}
        self._tracks = []
        for track in incoming:
            if counts[track.kind] < limits[track.kind]:
                self._tracks.append(track)
                counts[track.kind] += 1

    def update(self, frame, sequence, generation, captured_at, snapshot=None):
        now = self._clock()
        shape = frame.shape[:2]
        if self._generation != generation or self._shape != shape:
            self.reset()
            self._generation, self._shape = generation, shape
        new_frame = sequence != self._sequence
        if new_frame:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._gray is not None:
                self._project(self._tracks, self._gray, gray)
            self._gray, self._sequence = gray, sequence
            self._history[sequence] = (gray, captured_at)
            while self._history and (captured_at - next(iter(self._history.values()))[1] > .75
                                     or len(self._history) > 90):
                self._history.popitem(last=False)
        accepted = False
        if (snapshot is not None and snapshot.generation == generation
                and snapshot.sequence != self._last_snapshot):
            self._last_snapshot = snapshot.sequence
            source = self._history.get(snapshot.sequence)
            if source is not None and 0 <= now - snapshot.captured_at <= .5:
                incoming = self._incoming(snapshot, source[0], (shape[1], shape[0]))
                if snapshot.sequence != sequence:
                    self._project(incoming, source[0], self._gray)
                self._merge(incoming, now)
                self._source_sequence = snapshot.sequence
                accepted = True
        self._tracks = [track for track in self._tracks
                        if 0 <= now - track.source_time <= .5
                        and (track.missed_at is None or now - track.missed_at <= .22)]
        hands, detections, hand_sources = [], [], []
        ages = []
        held = not accepted
        for track in self._tracks:
            ages.append(now - track.source_time)
            held |= track.missed_at is not None
            if track.kind == 'hand':
                hands.append(track.geometry.copy())
                hand_sources.append(track.source_sequence)
            else:
                x, y, width, height = np.rint(track.geometry).astype(int)
                detections.append(SimpleNamespace(
                    bounding_box=SimpleNamespace(origin_x=int(x), origin_y=int(y),
                                                 width=max(1, int(width)), height=max(1, int(height))),
                    categories=[SimpleNamespace(**vars(category)) for category in track.categories],
                    preview_source_sequence=track.source_sequence,
                    preview_source_box=track.source_box,
                    preview_held=track.missed_at is not None,
                ))
        return PreviewResult(hands, detections, max(ages) if ages else None,
                             bool(held and (hands or detections)), self._source_sequence,
                             tuple(hand_sources))
