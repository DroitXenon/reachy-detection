"""Personal pointing calibration from MediaPipe landmarks; no gesture network."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROFILE_PATH = Path(__file__).resolve().parent / 'calibration' / 'pointing.json'
FINGERS = (5, 9, 13, 17)
FEATURE_FLOOR = np.array([0.12] * 4 + [0.20] * 4)


def pose_features(world):
    """Eight rotation/scale-invariant reach and protrusion measurements."""
    points = np.asarray(world, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None
    palm = np.linalg.norm(points[9] - points[0])
    if palm < 1e-8:
        return None
    reach, protrusion = [], []
    for base in FINGERS:
        finger = points[base:base + 4]
        bones = np.linalg.norm(np.diff(finger, axis=0), axis=1)
        if np.any(bones < 1e-8):
            return None
        reach.append(np.linalg.norm(finger[-1] - finger[0]) / bones.sum())
        protrusion.append((np.linalg.norm(finger[-1] - points[0])
                           - np.linalg.norm(finger[1] - points[0])) / palm)
    return np.array(reach + protrusion)


def pose_guard(features):
    """Reject clear non-pointing poses before a default or personal comparison."""
    if features is None:
        return 'Hand unclear: keep fingers visible'
    reach = features[:4]
    if reach[0] < 0.60 or features[4] < 0.05:
        return 'Extend the index finger'
    if np.all(reach > 0.90):
        return 'Open palm: relax the other fingers'
    if reach[0] > 0.90 and reach[1] > 0.90:
        return 'Two fingers extended: relax the middle finger'
    return None


class PointingPose:
    """Default reach comparison plus an optional user-labelled pose template."""

    def __init__(self, path=PROFILE_PATH, sample_count=15):
        self.path = Path(path)
        self.sample_count = sample_count
        self.centre = None
        self.tolerance = None
        self.samples = None
        self.capture_hand = None
        self.capture_wrist = None
        self.last_sequence = None
        self.notice = ''
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text())
                centre = np.asarray(data['centre'], dtype=float)
                tolerance = np.asarray(data['tolerance'], dtype=float)
                if (data.get('version') != 1 or centre.shape != (8,)
                        or tolerance.shape != (8,) or not np.isfinite(centre).all()
                        or not np.isfinite(tolerance).all()
                        or np.any(centre[:4] < 0) or np.any(centre[:4] > 1.001)
                        or np.any(tolerance < FEATURE_FLOOR)
                        or np.any(tolerance > FEATURE_FLOOR * 2)
                        or pose_guard(centre) is not None):
                    raise ValueError('invalid calibration values')
                self.centre, self.tolerance = centre, tolerance
            except (OSError, ValueError, KeyError, TypeError):
                self.notice = 'Saved calibration invalid; press C to calibrate again'

    @property
    def capturing(self):
        return self.samples is not None

    def start(self):
        self.samples = []
        self.capture_hand = self.capture_wrist = self.last_sequence = None
        self.notice = 'CALIBRATING: show one hand in your natural pointing pose'

    def cancel(self, reason):
        if self.capturing:
            self.samples = None
            self.notice = reason

    def reset(self):
        self.path.unlink(missing_ok=True)
        self.centre = self.tolerance = None
        self.samples = None
        self.notice = 'Calibration cleared; default pointing detection active'

    def observe(self, worlds, labels, wrists, sequence):
        """Capture only distinct fresh results from one consistently tracked hand."""
        if not self.capturing or sequence == self.last_sequence:
            return
        self.last_sequence = sequence
        if len(worlds) != 1 or len(labels) != 1 or len(wrists) != 1:
            self.cancel('Calibration cancelled: show one hand and press C again')
            return
        feature = pose_features(worlds[0])
        reason = pose_guard(feature)
        if reason:
            self.samples = []
            self.capture_hand = self.capture_wrist = None
            self.notice = 'CALIBRATING: ' + reason
            return
        wrist = np.asarray(wrists[0])
        if self.capture_hand is not None and (
                labels[0] != self.capture_hand
                or np.linalg.norm(wrist - self.capture_wrist) > 0.2):
            self.cancel('Calibration cancelled: hand changed; press C again')
            return
        self.capture_hand, self.capture_wrist = labels[0], wrist.copy()
        self.samples.append(feature)
        self.notice = f'CALIBRATING: hold pointing pose {len(self.samples)}/{self.sample_count}'
        if len(self.samples) < self.sample_count:
            return
        samples = np.asarray(self.samples)
        centre = np.median(samples, axis=0)
        spread = np.max(np.abs(samples - centre), axis=0)
        if np.any(spread > FEATURE_FLOOR * 1.5) or pose_guard(centre) is not None:
            self.cancel('Calibration too variable: hold steady and press C again')
            return
        tolerance = np.clip(spread * 2, FEATURE_FLOOR, FEATURE_FLOOR * 2)
        data = {'version': 1, 'centre': centre.tolist(), 'tolerance': tolerance.tolist()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        try:
            temporary.write_text(json.dumps(data, indent=2) + '\n')
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self.centre, self.tolerance = centre, tolerance
        self.samples = None
        self.notice = 'Calibration saved; point naturally to confirm green'

    def classify(self, world):
        feature = pose_features(world)
        reason = pose_guard(feature)
        if reason:
            return False, reason, ''
        if self.centre is not None:
            distance = np.sqrt(np.mean(((feature - self.centre) / self.tolerance) ** 2))
            accepted = distance <= 1
            reason = 'Personal pointing pose matched' if accepted else 'Pose differs from calibration; C to teach again'
            return accepted, reason, f'Personal pose distance: {distance:.2f} (accept <= 1)'
        reach = feature[:4]
        accepted = (reach[0] >= 0.75 and feature[4] >= 0.15
                    and reach[0] - reach[1] >= 0.08
                    and reach[0] - np.median(reach[1:]) >= 0.12)
        debug = 'Finger reach I/M/R/L: ' + '/'.join(f'{value:.2f}' for value in reach)
        reason = 'Pointing pose detected' if accepted else 'Point naturally; press C if confirmation is missed'
        return accepted, reason, debug


def pointing_direction(pixels):
    """Fit a line through the index finger, pointing toward its tip."""
    finger = pixels[[5, 6, 7, 8]]
    if not np.isfinite(finger).all() or not np.isfinite(pixels[[0, 9]]).all():
        return None
    forward = finger[-1] - finger[0]
    palm_length = np.linalg.norm(pixels[9] - pixels[0])
    if np.linalg.norm(forward) < max(10, 0.20 * palm_length):
        return None
    _, _, vectors = np.linalg.svd(finger - finger.mean(axis=0), full_matrices=False)
    direction = vectors[0]
    return -direction if np.dot(direction, forward) < 0 else direction


def choose_pointing_hand(worlds, pixel_hands, pose):
    candidates, descriptions, reasons = [], [], []
    for index, (world, pixels) in enumerate(zip(worlds, pixel_hands)):
        accepted, reason, debug = pose.classify(world)
        descriptions.append(f'Hand {index + 1}: {debug or reason}')
        reasons.append(reason)
        if accepted:
            candidates.append((index, pointing_direction(pixels)))
    debug = ' | '.join(descriptions)
    if len(candidates) > 1:
        return None, None, 'Unclear: two hands pointing', debug
    if candidates:
        index, direction = candidates[0]
        if direction is None:
            return None, None, 'Pointing pose recognised; aim across the image', debug
        return index, direction, 'Hold pointing gesture...', debug
    return None, None, reasons[0] if reasons else 'No hand detected', debug


@dataclass(frozen=True)
class GestureResult:
    """Recognition and image direction, independent of camera/model types."""

    index: int | None
    direction: np.ndarray | None
    status: str
    debug: str
    stable: bool


class PointingRecognizer:
    """Recognise and confirm pointing using only supplied hand landmarks.

    Model ownership belongs to hand tracking. A sequence identifies one
    inference result, so drawing it repeatedly never confirms a gesture twice.
    Pass ``None`` for the sequence when tracking is missing or has expired.
    """

    def __init__(self, calibration_path=PROFILE_PATH):
        self.pose = PointingPose(calibration_path)
        self._last_sequence = None
        self.reset_confirmation()

    def reset_confirmation(self):
        """Require two new results after a camera or calibration change."""
        self._consecutive = 0
        self._previous_hand = None
        self._previous_wrist = None

    def update(self, worlds, pixel_hands, labels, sequence, width, height=None):
        """Consume matched world/pixel hands and return confirmed pointing.

        ``height`` defaults to ``width`` for square inputs. Supply both actual
        image dimensions so calibration uses normalised wrist coordinates.
        """
        if sequence is None:
            self.pose.cancel('Calibration cancelled: tracking lost; press C again')
            self.reset_confirmation()
            return GestureResult(None, None, 'Waiting for fresh detection...', '', False)

        new_result = sequence != self._last_sequence
        height = width if height is None else height
        valid = (np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0
                 and len(worlds) == len(pixel_hands) == len(labels)
                 and all(np.asarray(pixels).shape == (21, 2)
                         and np.isfinite(pixels).all() for pixels in pixel_hands))
        if not valid:
            self.pose.cancel('Calibration cancelled: hand unclear; press C again')
            self.reset_confirmation()
            self._last_sequence = sequence
            return GestureResult(None, None, 'Hand unclear: keep fingers visible', '', False)

        if new_result:
            wrists = [np.asarray(pixels)[0] / [width, height] for pixels in pixel_hands]
            self.pose.observe(worlds, labels, wrists, sequence)
        index, direction, status, debug = choose_pointing_hand(worlds, pixel_hands, self.pose)
        if index is not None and new_result:
            wrist = np.asarray(pixel_hands[index])[0]
            hand = labels[index]
            same_hand = (hand == self._previous_hand and self._previous_wrist is not None
                         and np.linalg.norm(wrist - self._previous_wrist) < 0.25 * width)
            self._consecutive = min(self._consecutive + 1, 2) if same_hand else 1
            self._previous_hand, self._previous_wrist = hand, wrist.copy()
        elif index is None:
            self.reset_confirmation()
        self._last_sequence = sequence
        stable = self._consecutive >= 2 and not self.pose.capturing
        if self.pose.capturing:
            status = self.pose.notice
        elif stable:
            status = 'POINTING'
        return GestureResult(index, direction, status, debug, stable)
