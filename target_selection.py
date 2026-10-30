"""Deterministic forward-ray target selection and short temporal confirmation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Selection:
    index: int | None
    status: str
    candidates: tuple[int, ...] = ()


def ray_box_intersection(origin, direction, box, padding=0):
    """Return entry/exit distances on a normalized forward ray, or no hit."""
    origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
    if origin.shape != (2,) or direction.shape != (2,):
        return None
    bounds = np.array([box.origin_x, box.origin_y, box.width, box.height], dtype=float)
    if (not np.isfinite(origin).all() or not np.isfinite(direction).all()
            or not np.isfinite(bounds).all() or box.width <= 0 or box.height <= 0
            or not np.isfinite(padding) or padding < 0):
        return None
    length = np.linalg.norm(direction)
    if length < 1e-10:
        return None
    direction = direction / length
    lower = bounds[:2] - padding
    upper = bounds[:2] + bounds[2:] + padding
    entry, exit = 0.0, float('inf')
    for axis in (0, 1):
        if abs(direction[axis]) < 1e-10:
            if not lower[axis] <= origin[axis] <= upper[axis]:
                return None
            continue
        first, second = (np.array([lower[axis], upper[axis]]) - origin[axis]) / direction[axis]
        entry = max(entry, min(first, second))
        exit = min(exit, max(first, second))
        if exit < entry:
            return None
    return float(entry), float(exit)


def target_name(detection):
    categories = getattr(detection, 'categories', [])
    if not categories:
        return None
    valid = [category for category in categories if np.isfinite(category.score)]
    if not valid:
        return None
    category = max(valid, key=lambda item: item.score)
    name = category.category_name
    return name if name not in (None, '', '???', 'person') else None


def select_target(origin, direction, detections, image_size):
    """Choose the nearest ray hit, refusing overlapping/closely spaced hits."""
    size = np.asarray(image_size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        return Selection(None, 'No target: invalid image size')
    diagonal = float(np.linalg.norm(size))
    padding, separation = 0.005 * diagonal, 0.02 * diagonal
    hits = []
    for index, detection in enumerate(detections):
        box = getattr(detection, 'bounding_box', None)
        if box is None or target_name(detection) is None:
            continue
        if not all(hasattr(box, field) for field in ('origin_x', 'origin_y', 'width', 'height')):
            continue
        hit = ray_box_intersection(origin, direction, box, padding)
        if hit is not None:
            hits.append((hit[0], hit[1], index))
    hits.sort(key=lambda item: (item[0], item[1], item[2]))
    candidates = tuple(item[2] for item in hits)
    if not hits:
        return Selection(None, 'No object on the pointing ray')
    if len(hits) > 1 and hits[1][0] <= hits[0][1] + separation:
        return Selection(None, 'Unclear: multiple objects on the pointing ray', candidates)
    index = hits[0][2]
    return Selection(index, f'Target candidate: {target_name(detections[index])}', candidates)


def box_iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return overlap / max(1e-10, area_first + area_second - overlap)


class TargetTracker:
    """Confirm a matching label and box across distinct inference results."""

    def __init__(self, confirmations=2):
        self.confirmations = confirmations
        self.reset()

    def reset(self):
        self.name = self.box = self.sequence = None
        self.count = 0

    def update(self, proposal, detections, sequence):
        if proposal.index is None:
            self.reset()
            return proposal
        detection = detections[proposal.index]
        name, box = target_name(detection), detection.bounding_box
        coordinates = (box.origin_x, box.origin_y,
                       box.origin_x + box.width, box.origin_y + box.height)
        same = name == self.name and self.box is not None and box_iou(coordinates, self.box) >= 0.3
        if not same:
            self.count = 1
        elif sequence != self.sequence:
            self.count = min(self.count + 1, self.confirmations)
        self.name, self.box, self.sequence = name, coordinates, sequence
        if self.count < self.confirmations:
            return Selection(None, f'Hold target steady: {name}', proposal.candidates)
        return Selection(proposal.index, f'Selected: {name}', proposal.candidates)
