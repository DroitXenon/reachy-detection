# reachy-detection

Reachy Mini coursework prototype. Current milestones: head movement, webcam
hand tracking, and object detection. Target selection and spoken naming are not
yet implemented; the camera demo does not control the robot yet.

## Setup

```bash
conda create -n reachy-detection python=3.12 -y
conda activate reachy-detection
python -m pip install -r requirements.txt
python prepare_yolo.py
```

If the environment already exists, activate it and install the requirements.
The YOLO model is already prepared on this development Mac. On a fresh checkout,
`prepare_yolo.py` downloads official YOLO11s weights and exports a 640px ONNX model
plus its label metadata into `models/`. It installs export tools in a temporary
environment, leaving this Conda environment's OpenCV installation intact. Runtime
inference uses OpenCV; PyTorch is not a runtime dependency. Internet is required
for that first model preparation. Model files are ignored by Git.

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
version 1) only when explicitly using the legacy `.tflite` model. The default is
now YOLO11s at 640px input, prepared with `python prepare_yolo.py` during setup.
No additional runtime Python packages are required.

The preview adds blue boxes, category names, and confidence scores, while keeping
the hand landmarks and green pointing arrow. Place two separated objects, such as
a cup and a bottle, in good lighting. Check that both get labelled before trying
to point. Boxes identify detected objects; they do not yet indicate a selected
target. The model supports the 80 COCO categories, not arbitrary object names.
The `person` category is excluded inside the detector, so people do not get boxes
or count toward the displayed object total. Hand detection still runs normally.

The default minimum confidence is 0.35. Press **-** to lower the threshold by 5
percentage points or **+** to raise it while running. The threshold is shown above
the preview. Lower thresholds can recover weak detections but increase false
positives. Scores below 0.10 are discarded. Press **D** to see up to three below-threshold candidates, explicitly
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
iPhone input.

### Preview performance

Camera capture and drawing stay on the main thread. A single background worker
runs both models on the same clean frame. It keeps only the newest pending frame
instead of building a queue of old frames. The live preview reuses recent overlays;
overlays older than 300ms are hidden. Mirroring or resolution changes invalidate
old results. Gesture confirmation counts new inference results, not display frames.
During movement the boxes can trail the live image slightly; overlay age is shown
in the preview and is not a measure of recognition confidence. Any future target
selection must use the matched hand/object snapshot, not assume cached boxes are
perfectly aligned with the latest camera image.

The preview shows its FPS, inference time, and overlay age. These are separate:
a smooth 30 FPS preview does not imply 30 object detections per second. The camera
requests 960x540 at 30 FPS, though the device/Continuity Camera may choose a different
format. On the development M1 Pro, a paced synthetic 30 FPS camera kept the preview
near 30 FPS with about 95ms inference time. GUI calls were substituted in that
test; it does not measure the live iPhone connection. The earlier sequential
pipeline took about 113ms per blank frame before capture/display, limiting it to
roughly 9 FPS. A few public example images verified inference and person exclusion;
they do not establish detection accuracy on the user's objects.

For less inference work, prepare the nano model and select it explicitly:

```bash
python prepare_yolo.py --size n
python hand_tracking.py --camera 1 --objects --object-model models/yolo11n.onnx
```

Model source and published model comparisons: [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/).
The ONNX export uses opset 12, float32, fixed 640x640 input, letterboxing, and
class-agnostic nonmaximum suppression at IoU 0.5. The smaller nano variant trades
published detection accuracy for speed. Person detections are discarded before
display and suppression. Neither model recognises categories outside COCO.

The previous detector remains available for comparisons:

```bash
python hand_tracking.py --camera 1 --objects --object-model models/efficientdet_lite0.tflite
```

Legacy model source: [MediaPipe Object Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector).

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
python -m unittest -v test_hand_tracking test_object_detection test_vision_worker
```

Tests cover gesture geometry, relaxed poses, second-hand selection, confidence
filtering, person exclusion, ONNX coordinate conversion/NMS, frame dropping,
orientation changes, stale results, and worker errors. Native models are also
checked on public reference images. Live iPhone accuracy and frame rate need a
user check; synthetic frames and a few example images are not an accuracy benchmark.

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
