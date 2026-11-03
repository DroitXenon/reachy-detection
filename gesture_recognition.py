"""Compatibility import for the 2D pointing implementation in gesture_recog."""

from gesture_recog import (
    EXTENDED_RATIO,
    MIN_FINGER_PIXELS,
    GestureResult,
    PointingRecognizer,
    finger_extension,
    pointing_direction,
)

__all__ = [
    'EXTENDED_RATIO', 'MIN_FINGER_PIXELS', 'GestureResult', 'PointingRecognizer',
    'finger_extension', 'pointing_direction',
]
