# reachy-detection

Reachy Mini coursework prototype. Current milestones: head movement, webcam
hand tracking, object detection, and deterministic target selection. Spoken naming
and connecting camera selections to robot movement are still pending.

## Setup

```bash
conda create -n reachy-detection python=3.12 -y
conda activate reachy-detection
python -m pip install -r requirements.txt
python prepare_yolo.py
```

If the environment already exists, activate it and install the requirements.
The YOLO model is already prepared on this development Mac. On a fresh checkout,
`prepare_yolo.py` downloads official YOLO26s weights and exports a 640px ONNX model
plus its label metadata into `models/`. It installs export tools in a temporary
environment, leaving this Conda environment's OpenCV installation intact. Runtime
inference uses OpenCV; PyTorch is not a runtime dependency. Internet is required
for that first model preparation. Model files are ignored by Git.

## Program structure

- `hand_tracking.py`: MediaPipe HandTracker, landmark coordinates, and hand drawing.
- `gesture_recog.py`: pointing recognition, personal calibration, and confirmation
  across fresh inference results. It takes landmark arrays and has no camera or GUI.
- `object_detection.py`: object models, filtering, and box drawing.
- `target_selection.py`: pointing-ray geometry and target confirmation.
- `vision_worker.py`: background inference and dropping outdated frames.
- `main.py`: combines the modules, owns camera capture, preview, controls, and CLI.

With both modes enabled, `main.py` sends a clean frame to the hand and object models, converts
hand results to landmark arrays, passes them to `PointingRecognizer`, then passes
a confirmed pointing ray and the matching object boxes to target selection.

The previous `python hand_tracking.py ...` command still forwards to `main.py`
and uses the same mode flags. `gesture_recognition.py` is a compatibility import
for the renamed `gesture_recog.py`.

## Choose a mode

Both hand gestures and object detection are enabled by default, including
pointing-based object selection. Use a mode flag to run just one function:

| Flags | Behaviour |
| --- | --- |
| No mode flags | Both modes, plus pointing-based object selection (default). |
| `--hand` | Hand landmarks, pointing gestures, and personal calibration only. |
| `--object` | Object boxes, labels, and scores only. |
| `--hand --object` | Both modes explicitly; same as the default. |

```bash
python main.py
python main.py --hand
python main.py --object
```

`python main.py` uses the default camera (index 0). Optionally add `--camera 1`
to choose another camera. Omitting both mode flags enables all current
vision functions even when other options, such as `--camera`, are supplied.
`--list-object-labels` works on its own without opening the camera.
To prepare/check only the selected model files without opening the camera, add
`--download-model-only`. With no mode flags it prepares/checks both models;
with `--hand` or `--object` it handles only that model:

```bash
python main.py --download-model-only
python main.py --hand --download-model-only
```

The hand model downloads automatically when needed. YOLO models must first be
exported with `python prepare_yolo.py`; hand-only mode does not need that export.

## Webcam hand tracking

```bash
python main.py --hand
```

The first run downloads Google's Hand Landmarker model into `models/` (internet
required once). Frames are processed locally and are not recorded. This demo runs
independently of Reachy and does not require the simulator.

Make an index-finger pointing pose with the other fingers folded naturally. Keep your hand in view and point across the image. The preview
draws all 21 landmarks and an amber, unconfirmed direction arrow for one tracked
hand. A green arrow and POINTING appear after two fresh results confirm the pose.
A resting second hand is allowed; two pointing hands are ambiguous.

- Press **Q** or **Esc** to quit, or **M** to toggle the mirrored preview.
- Press **C** while pointing naturally to calibrate your own pose. Hold it steady
  with one hand visible until the 15-sample progress completes (usually a few seconds).
- Press **R** to clear the saved calibration and return to default recognition.
- The amber arrow is only a direction preview; green indicates confirmed pointing.
- **C** and **R** are available only when `--hand` is enabled.
- Use `python main.py --camera 1 --hand` if the default camera is incorrect.
- On macOS, allow your terminal camera access in System Settings > Privacy &
  Security > Camera. Restart the terminal after changing permissions if needed.
- Close other apps using the webcam if capture fails.

## Object detection only

```bash
python main.py --object
```

This mode loads only the object detector and shows boxes, labels, and scores.
It does not load the hand model or your pointing calibration, draw hand landmarks
or arrows, or select a pointing target. Press **D** for below-threshold diagnostics
and **-** / **+** to adjust the object confidence threshold. These object controls
are available whenever `--object` is enabled, including the combined mode.

## Hand gestures with object detection and selection

```bash
python main.py
```

The first run downloads EfficientDet-Lite0 (int8,
version 1) only when explicitly using the legacy `.tflite` model. The default is
now YOLO26s at 640px input, prepared with `python prepare_yolo.py` during setup.
No additional runtime Python packages are required.

The preview adds blue boxes, category names, and confidence scores, while keeping
the hand landmarks and green pointing arrow. Place two separated objects, such as
a cup and a bottle, in good lighting. Check that both get labelled before trying
to point. Unselected objects have blue boxes; a confirmed target gets a thicker
green box and a SELECTED label. The model supports the 80 COCO categories, not
arbitrary object names.
The `person` category is excluded inside the detector, so people do not get boxes
or count toward the displayed object total. Hand detection still runs normally.

The default minimum confidence is 0.35. Press **-** to lower the threshold by 5
percentage points or **+** to raise it while running. The threshold is shown above
the preview. Lower thresholds can recover weak detections but increase false
positives. Scores below 0.10 are discarded. Press **D** to see up to three below-threshold candidates, explicitly
labelled as unconfirmed. These are model guesses, not verified object identities.
In hand-only mode, the preview says **Objects OFF**. Add `--object` to the
`--hand` command to enable object detection and target selection. If objects are missed, improve lighting and move them closer. To check
whether your object type is supported by the installed model:

```bash
python main.py --list-object-labels
```

The model includes cup, bottle, book, cell phone, keyboard, and many other common
objects, but not every item (for example, pen is not a category). Threshold changes
cannot add new categories.

At most 10 detections are shown. All models use the same clean, mirrored source frame
before overlays are drawn, so boxes and pointing direction share pixel coordinates.
Frames wider than 960 pixels are resized proportionally, including high-resolution
iPhone input.

### Select the object you point at

Run `python main.py`. Place a cup and bottle apart so their
boxes do not overlap. Point until the arrow is green, then hold it toward either
object. The target line should show **Selected: cup** or **Selected: bottle**, and
that object's box turns green. Relax your hand or point away to clear selection.
An amber arrow is only a direction preview and never selects an object.

The target method is deterministic and lives in `target_selection.py`:

1. Start the forward ray at index fingertip 8, using the fitted finger direction.
   Its hand landmarks and object boxes come from the same inference snapshot.
2. Intersect the ray with the detected boxes. Ignore boxes behind the fingertip,
   invalid detections, and people. A ray beginning inside a box has entry distance 0.
3. Allow small pointing jitter by padding each box by 0.5% of the image diagonal.
   Select the first box along the forward ray. If the next box starts before the
   first ends, or within another 2% of the image diagonal, return **Unclear**.
   This also rejects overlapping boxes with the same category; model confidence
   does not resolve pointing ambiguity.
4. Require the same object in two distinct inference results. Match its label and
   box overlap (IoU >= 0.3), so detector result reordering does not switch targets.
   Repeated display frames do not count. A different object must confirm again.
5. Clear immediately on ambiguous/no-hit rays, rejected pointing, calibration,
   expired snapshots, mirroring, camera-size changes, or confidence changes.

The two target-confirmation results are in addition to pointing confirmation, so
there is a short deliberate hold before an object turns green. The rule is 2D:
collinear objects, occlusion, large enclosing boxes, and depth pointing can remain
ambiguous. Separate objects and aim the visible ray through the desired box.

Manual check: select each of two objects in turn, relax your hand, point between
them, and try overlapping boxes. Verify that only one box is green, ambiguity
selects nothing, and an amber ray never produces a selected target. Robot movement
and spoken naming will be connected after this selection behaviour is checked.

### Preview performance

Camera capture and drawing stay on the main thread. A single background worker
runs only the enabled models; in combined mode both use one source frame.
Gesture recognition uses the hand landmarks on the main thread. The worker keeps only the newest pending frame
instead of building a queue of old frames. The live preview reuses recent overlays;
overlays older than 300ms are hidden. Mirroring or resolution changes invalidate
old results. Gesture confirmation counts new inference results, not display frames.
During movement the boxes can trail the live image slightly; overlay age is shown
in the preview and is not a measure of recognition confidence. Selection uses
matched hand/object snapshots; cached boxes can still trail the latest camera
image while objects move.

The preview shows its FPS, inference time, and overlay age. These are separate:
a smooth 30 FPS preview does not imply 30 object detections per second. The camera
requests 960x540 at 30 FPS, though the device/Continuity Camera may choose a different
format. On the development M1 Pro, a paced synthetic 30 FPS camera kept the preview
near 30 FPS with about 94ms inference time using YOLO26s and the current calibration
method. Object inference alone measured about 80ms on a public reference image.
GUI calls were substituted in that test; it does not measure the live iPhone
connection. The earlier sequential pipeline took about 113ms per blank frame
before capture/display, limiting it to roughly 9 FPS. Public example images
verified inference and person exclusion; they do not establish detection accuracy
on the user's objects.

For less inference work, prepare the nano model and select it explicitly:

```bash
python prepare_yolo.py --size n
python main.py --object-model models/yolo26n.onnx
```

Model source and published model comparisons: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).
The default export is YOLO26s with `nms=False`, opset 12, float32, and fixed
640x640 input. Its NMS-free output has shape `[1, 300, 6]`: x1, y1, x2, y2,
confidence, class ID. The decoder removes letterbox padding and rescales boxes
back to the camera image; it does not apply a second NMS pass. The accompanying
JSON records the output format so old YOLO11 exports use their separate raw-score
decoder. Person detections are excluded before display. The smaller nano variant
trades published detection accuracy for speed. Both pretrained families use COCO
categories. The preview displays the active model filename.

For a YOLO11 comparison, run `python prepare_yolo.py --family 11 --size s`, then
select `--object-model models/yolo11s.onnx`. The default remains YOLO26s.

The previous detector remains available for comparisons:

```bash
python main.py --object-model models/efficientdet_lite0.tflite
```

Legacy model source: [MediaPipe Object Detector](https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector).

### How pointing recognition works

The extra trained gesture-classification network and hand-crop matching have been
removed. MediaPipe Hand Landmarker is the only hand model. Recognition uses four
finger-reach ratios and four fingertip-protrusion measurements from its world
landmarks. These distance ratios are independent of image rotation, translation,
scale, and left/right reflection; no joint-angle thresholds are used.

Default recognition requires the index to reach at least 75% of its finger-chain
length, protrude beyond the PIP joint, and extend more than the middle finger and
the median of the other fingers. This is an initial heuristic. If it misses your
natural pose, calibrate it directly:

1. Keep **one hand** visible and make your natural pointing gesture.
2. Click the preview window and press **C**.
3. Keep pointing until **Calibration saved** appears. It collects 15 distinct
   inference results, not repeated display frames. A changed/missing hand,
   mirroring, camera-size changes, and excessive sample variation cancel capture.
4. Point normally. The saved profile compares the current distance measurements
   with your captured pose. Press **R** to remove it and calibrate again if needed.

Calibration uses median features and bounded per-feature tolerances. Clear fists,
open palms, and V signs are rejected before comparison; do not calibrate with any
of these. Your profile is stored locally as `calibration/pointing.json`, ignored by
Git. No photos or video are saved. An alternative profile can be selected with
`--calibration path/to/profile.json`. The profile contains pose measurements, not
an accuracy estimate or probability.

The arrow direction fits a line through index landmarks 5, 6, 7, and 8 in the
original camera coordinates. A projection under 10 pixels or 20% of palm length is
hidden because pointing toward the camera has no reliable 2D direction. A valid
single-hand direction is shown in **amber even when recognition is unconfirmed**;
this makes tracking visible and helps you align your pose. Green requires two fresh
recognition results from the same hand. Amber never triggers target selection. Two recognised pointing hands are ambiguous.

The classifier runs as cheap NumPy calculations; the extra gesture model no longer
runs or needs downloading. The existing background inference worker and explicit
Hand Landmarker remain. The deterministic pointing-ray/bounding-box method
selects targets as described above.

Personal calibration cannot establish an accuracy rate across poses or users.
Check your pointing pose, a fist, an open palm, and a V sign after calibrating.
Source: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

### Checks

```bash
python -m unittest discover -v
```

Tests cover pose features, calibration persistence, duplicate-frame rejection, confidence
filtering, person exclusion, both ONNX output formats, coordinate conversion, frame dropping,
orientation changes, stale results, ray geometry, target ambiguity/confirmation,
selection switching and clearing, model cleanup, shared inference timestamps,
mode-specific model loading and controls, the separated gesture interface, and
worker errors. Native models are also
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
Webcam capture in `main.py` is separate; speech is still to be added.
