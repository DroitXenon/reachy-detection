"""Fast 2D pointing: compare index endpoint distance with its visible path."""

from dataclasses import dataclass
import math

import numpy as np


EXTENDED_RATIO = 0.85
MIN_FINGER_PIXELS = 5.0


def _index_points(pixels):
    try:
        points = np.asarray(pixels, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.shape != (21, 2):
        return None
    finger = points[5:9]
    return finger if np.isfinite(finger).all() else None


def finger_extension(pixels):
    """Endpoint distance / index path length: near 1 means visibly extended.

    Uses image landmarks 5, 6, 7, 8 only. No angles, world coordinates, pose
    templates, or other fingers are involved.
    """
    finger = _index_points(pixels)
    if finger is None:
        return None
    path = sum(math.hypot(*edge) for edge in np.diff(finger, axis=0))
    direct = math.hypot(*(finger[-1] - finger[0]))
    if (not math.isfinite(path) or not math.isfinite(direct)
            or path < MIN_FINGER_PIXELS or direct < MIN_FINGER_PIXELS):
        return None
    return min(1.0, direct / path)


def pointing_direction(pixels):
    """Use the direct index-base to fingertip vector for the arrow."""
    finger = _index_points(pixels)
    if finger is None:
        return None
    direction = finger[-1] - finger[0]
    length = math.hypot(*direction)
    if not math.isfinite(length) or length < MIN_FINGER_PIXELS:
        return None
    return direction / length


@dataclass(frozen=True)
class GestureResult:
    index: int | None
    direction: np.ndarray | None
    status: str
    debug: str
    stable: bool


class PointingRecognizer:
    """Recognize an extended index on the first actual detection result.

    No gesture confirmation counter is needed. Cached redraws can display the
    result, while target selection independently counts distinct model results.
    Reset/lost tracking invalidates the previous cached sequence.
    """

    def __init__(self):
        self._last_sequence = None
        self._blocked_sequence = None

    def reset_confirmation(self):
        self._blocked_sequence = self._last_sequence

    def update(self, pixel_hands, sequence):
        if sequence is None:
            self.reset_confirmation()
            return GestureResult(None, None, 'Waiting for fresh detection...', '', False)
        if sequence == self._blocked_sequence:
            return GestureResult(None, None, 'Waiting for new detection...', '', False)
        self._last_sequence = sequence
        self._blocked_sequence = None
        candidates, details = [], []
        for index, pixels in enumerate(pixel_hands):
            ratio = finger_extension(pixels)
            details.append(f'Hand {index + 1}: extension {ratio:.2f}'
                           if ratio is not None else f'Hand {index + 1}: index not visible')
            if ratio is not None and ratio >= EXTENDED_RATIO:
                candidates.append((index, pointing_direction(pixels)))
        debug = ' | '.join(details)
        if len(candidates) > 1:
            return GestureResult(None, None, 'Unclear: two hands pointing', debug, False)
        if candidates:
            index, direction = candidates[0]
            return GestureResult(index, direction, 'POINTING', debug, True)
        status = 'Extend your index finger' if len(pixel_hands) else 'No hand detected'
        return GestureResult(None, None, status, debug, False)
