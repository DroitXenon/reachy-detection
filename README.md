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

Extend your index finger and let the other fingers bend naturally; a tight fist
is not required. Keep your hand in view and point across the image. The preview
draws all 21 landmarks and shows a green arrow after two consecutive pointing
frames from the same hand. A resting second hand is allowed; two pointing hands,
an open palm, or a fist are rejected.

- Press **Q** or **Esc** to quit, or **M** to toggle the mirrored preview.
- Press **D** for index-joint angles and straightness details. The status explains
  whether to straighten the index, relax another finger, or improve visibility.
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
The `person` category is excluded inside the detector, so people do not get boxes
or count toward the displayed object total. Hand detection still runs normally.

The default minimum confidence is 0.25. Press **-** to lower the threshold by 5
percentage points or **+** to raise it while running. The threshold is shown above
the preview. Lower thresholds can recover weak detections but increase false
positives. Press **D** to see up to three below-threshold candidates, explicitly
labelled as unconfirmed. These are model guesses, not verified object identities.
If detection is disabled, the preview says **Objects OFF** and reminds you to use
`--objects`. If objects are missed, improve lighting and move them closer. To check
whether your object type is supported by the installed model:

```bash
python hand_tracking.py --list-object-labels
```

The model includes cup, bottle, book, cell phone, keyboard, and many other common
objects, but not every item (for example, pen is not a category). Threshold changes
cannot add new categories.

At most 10 detections are shown. Both models process the same clean, mirrored frame
before overlays are drawn, so boxes and pointing direction share pixel coordinates.
Frames wider than 960 pixels are resized proportionally, including high-resolution
iPhone input. Processing is synchronous to avoid stale boxes from a different frame.

EfficientDet-Lite0 is the lightweight model recommended in the
[MediaPipe Object Detector guide](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector).
Model: [version 1, int8](https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite).

### How the pointing rule works

MediaPipe Hand Landmarker supplies image and 3D world landmarks. Hand detection,
presence, and tracking confidence thresholds are 0.5. Gesture rules use world
coordinates so they are independent of image rotation and scale:

- Index PIP/DIP joint angles must be at least 140/135 degrees. Its base-to-tip
  distance must be at least 86% of the sum of its three bone lengths.
- A finger is clearly extended when both angles are at least 155 degrees and its
  base-to-tip distance is at least 92% of bone length. Reject an extended middle
  finger (V sign), or both an extended ring and little finger.
- Also reject a middle finger whose reach ratio is within 0.03 of the index's or
  greater, AND whose smaller joint angle is within 15 degrees of the index's or
  greater. This prevents a similarly curved open hand from looking like pointing.
- Other fingers can be loosely bent. The thumb is ignored. Invalid landmarks
  are rejected. These remain heuristic thresholds, not a trained gesture classifier.

The arrow uses the screen-space vector from index knuckle (landmark 5) to fingertip
(8), and starts at the fingertip. Both tracking and drawing use the same mirrored
frame. A projection shorter than 15 pixels or 30% of wrist-to-middle-knuckle length
is rejected. This catches some foreshortened poses but cannot reliably interpret
pointing in depth. Two consecutive positive frames from the same handedness and a
nearby wrist confirm the gesture; any rejected frame clears the arrow immediately.
Two pointing hands are ambiguous, but one pointing hand alongside a resting hand
is accepted. Occlusion and unusual poses can still fail.

Manual check: point left/right/up/down with either hand, then show a fist and an
open palm, remove your hand, and try a resting second hand versus two pointing
hands. Check the arrow's direction and
that it disappears when pointing is unclear. Live gesture accuracy needs this
check on your webcam and with other users.

Uses the [MediaPipe Hand Landmarker Tasks API](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
and the [official model, version 1](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

### Checks

```bash
python -m unittest -v test_hand_tracking test_object_detection
```

Twelve geometry tests cover rotations, reflections, relaxed pointing, non-pointing
poses, invalid landmarks, second-hand selection, and short projections. These tests and blank-frame model
inference passed on the development Mac. Live webcam validation was blocked by
macOS camera permission; the preview and real-hand accuracy still need checking.
Object detection also passed a blank-frame check and detected both a cat and dog
in Google's example image. The combined preview was exercised with both real
models and a substituted camera frame; live iPhone detection remains unverified.
Person exclusion was verified with the native detector on Google's pose example:
the baseline detected a person, while the filtered detector omitted it and kept
other objects. The revised gesture thresholds need a live recheck with your hand;
synthetic tests do not establish real-world accuracy.

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
