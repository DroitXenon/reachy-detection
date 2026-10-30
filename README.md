# reachy-detection

Reachy Mini coursework prototype. Current milestones: head movement, webcam
hand tracking, and object detection. Target selection and spoken naming are not
yet implemented; the camera demo does not control the robot yet.

## Setup

```bash
conda create -n reachy-detection python=3.12 -y
conda activate reachy-detection
python -m pip install -r requirements.txt
```

If the environment already exists, activate it and install the requirements.

## Webcam hand tracking

```bash
python hand_tracking.py --camera 1
```

The first run downloads Google's Hand Landmarker model into `models/` (internet
required once). Frames are processed locally and are not recorded. This demo runs
independently of Reachy and does not require the simulator.

Show one hand, straighten your index finger, and curl your middle, ring, and little
fingers. Keep your hand in view and point across the image. The preview draws all
21 landmarks and shows a green arrow after three consecutive pointing frames.
An open palm, fist, or two visible hands should not produce an arrow.

- Press **Q** or **Esc** to quit, or **M** to toggle the mirrored preview.
- Use `python hand_tracking.py --camera 1` if the default camera is incorrect.
- On macOS, allow your terminal camera access in System Settings > Privacy &
  Security > Camera. Restart the terminal after changing permissions if needed.
- Close other apps using the webcam if capture fails.

## Hand tracking with object detection

```bash
python hand_tracking.py --camera 1 --objects
```

Use the same camera index that worked for your webcam/iPhone; `1` is an example,
not a fixed iPhone device ID. The first run downloads EfficientDet-Lite0 (int8,
version 1). No additional Python packages are required.

The preview adds blue boxes, category names, and confidence scores, while keeping
the hand landmarks and green pointing arrow. Place two separated objects, such as
a cup and a bottle, in good lighting. Check that both get labelled before trying
to point. Boxes identify detected objects; they do not yet indicate a selected
target. The model supports the 80 COCO categories, not arbitrary object names.

The default minimum confidence is 0.4. If objects are missed, improve lighting,
move them closer, or try `--score-threshold 0.3` (which can increase false positives).
At most 10 detections are shown. Both models process the same clean, mirrored frame
before overlays are drawn, so boxes and pointing direction share pixel coordinates.
Frames wider than 960 pixels are resized proportionally, including high-resolution
iPhone input. Processing is synchronous to avoid stale boxes from a different frame.

EfficientDet-Lite0 is the lightweight model recommended in the
[MediaPipe Object Detector guide](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector).
Model: [version 1, int8](https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite).

### How the pointing rule works

MediaPipe Hand Landmarker supplies image and 3D world landmarks. The index finger's
two middle joints must each have an angle of at least 155 degrees; the middle,
ring, and little fingers' first interphalangeal joints must each be at most 140
degrees. The thumb is ignored. Angles use world landmarks so the rule is independent
of hand rotation in the image. These are initial heuristic thresholds, not a
trained gesture classifier.

The arrow uses the screen-space vector from index knuckle (landmark 5) to fingertip
(8), and starts at the fingertip. Both tracking and drawing use the same mirrored
frame. A projection shorter than 15 pixels or 30% of wrist-to-middle-knuckle length
is rejected. This catches some foreshortened poses but cannot reliably interpret
pointing in depth. Three consecutive positive frames reduce flicker; any negative
frame clears the arrow immediately. Occlusion and unusual poses can still fail.

Manual check: point left/right/up/down with either hand, then show a fist and an
open palm, remove your hand, and try two hands. Check the arrow's direction and
that it disappears when pointing is unclear. Live gesture accuracy needs this
check on your webcam and with other users.

Uses the [MediaPipe Hand Landmarker Tasks API](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
and the [official model, version 1](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

### Checks

```bash
python -m unittest -v test_hand_tracking
```

Five geometry tests cover rotations, reflections, non-pointing poses, degenerate
landmarks, and short projected directions. These tests and blank-frame model
inference passed on the development Mac. Live webcam validation was blocked by
macOS camera permission; the preview and real-hand accuracy still need checking.
Object detection also passed a blank-frame check and detected both a cat and dog
in Google's example image. The combined preview was exercised with both real
models and a substituted camera frame; live iPhone detection remains unverified.

## Head movement

Stop the desktop app's mockup daemon before starting full MuJoCo. In terminal 1:

```bash
conda activate reachy-detection
mjpython -m reachy_mini.daemon.app.main --sim
```

Keep it running, then in terminal 2:

```bash
conda activate reachy-detection
python demo_head.py
```

`--no-media` disables the daemon's camera/audio/streaming for the movement test.
Webcam capture in `hand_tracking.py` is separate; speech is still to be added.
