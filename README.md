# reachy-detection

Reachy Mini coursework prototype. The pipeline detects pointing and objects,
selects a target deterministically, turns the simulated Reachy head toward it,
and speaks its name through the computer.

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

- `camera_capture.py`: background camera capture with one replaceable latest frame.
- `hand_tracking.py`: MediaPipe HandTracker, landmark coordinates, and hand drawing.
- `gesture_recog.py`: immediate 2D finger-extension recognition from image
  landmarks. It has no camera, GUI, calibration, or 3D pose logic.
- `object_detection.py`: object models, filtering, and box drawing.
- `target_selection.py`: pointing-ray geometry and target confirmation.
- `vision_worker.py`: background inference and dropping outdated frames.
- `preview_tracking.py`: display-only optical flow, smoothing, and short gap handling.
- `preview_window.py`: GPU texture display with nonblocking window/input events.
- `simulator.py`: launches full MuJoCo with its viewer and head control, skipping
  the unused virtual eye-camera stream for this external-camera project.
- `robot_control.py`: bounded head movement in the local full MuJoCo simulator.
- `speech.py`: offline spoken labels through macOS `say` or Linux eSpeak.
- `interaction.py`: responds to confirmed targets on a background action thread.
- `main.py`: combines the modules, owns preview pacing, controls, and CLI.

With both modes enabled, `main.py` sends a clean frame to the hand and object models, converts
hand results to pixel landmark arrays, passes them and the model-result sequence
to `PointingRecognizer.update(pixel_hands, sequence)`, then passes a valid pointing
ray and the matching object boxes to target selection.
`SelectionResponder` sends the selected box to the robot controller, then speaks
its label. Camera capture and drawing continue while those actions run.

The previous `python hand_tracking.py ...` command still forwards to `main.py`
and uses the same mode flags. `gesture_recognition.py` is a compatibility import
for the renamed `gesture_recog.py`.

## Choose a mode

Both hand gestures and object detection are enabled by default, including
pointing-based object selection, simulated head movement, and spoken naming.
Use a mode flag to run just one vision function:

| Flags | Behaviour |
| --- | --- |
| No mode flags | Hand gestures, objects, target selection, robot movement, and speech (default). |
| `--hand` | Hand landmarks and 2D finger-extension pointing only. |
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
The default preview target is 60 FPS. Optionally select another target from 1 to
120 with `--fps`, for example `python main.py --fps 30`. The camera is asked for
the same rate; the device may deliver a lower rate.
`--list-object-labels` works on its own without opening the camera.
The single modes `--hand` and `--object` do not start robot or speech actions.
In combined mode, use `--no-robot` to disable movement, `--mute` to disable
spoken names, or both to run only the vision preview:

```bash
python main.py --no-robot --mute
```

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

Extend your index finger across the image; the other fingers can be in any
position. An open palm or V sign also counts when the index finger is extended.
Bend the index finger to stop pointing. The preview draws all 21 landmarks.
POINTING and a green arrow appear on the first fresh model result that passes
the 2D extension check; there is no extra gesture-confirmation wait. The arrow
disappears when pointing is not detected. A second hand with a bent index is allowed;
two qualifying index fingers are ambiguous.

- Press **Q** or **Esc** to quit, or **M** to toggle the mirrored preview.
- Press **R** to clear pointing and ignore the cached result until a new model
  result arrives.
- The arrow appears only for a valid fresh pointing result.
- **R** is available only when `--hand` is enabled.
- Use `python main.py --camera 1 --hand` if the default camera is incorrect.
- On macOS, allow your terminal camera access in System Settings > Privacy &
  Security > Camera. Restart the terminal after changing permissions if needed.
- Close other apps using the webcam if capture fails.

## Object detection only

```bash
python main.py --object
```

This mode loads only the object detector and shows boxes, labels, and scores.
It does not load the hand model, draw hand landmarks or arrows, or select a
pointing target. Press **D** for below-threshold diagnostics
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
that object's box turns green. Bend your index finger or point away to clear selection.
With the full simulator running, the selected box also drives the head movement
and its name is spoken. See the simulator setup below.

The target method is deterministic and lives in `target_selection.py`:

1. Start the forward ray at index fingertip 8, using the direct base-5-to-tip-8
   direction.
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
5. Clear immediately on ambiguous/no-hit rays, a bent index finger, a gesture reset,
   expired snapshots, mirroring, camera-size changes, or confidence changes.

Pointing itself is immediate. Object selection still requires two distinct
fresh model results for the same object, so there is a short hold before its box
turns green. The rule is 2D:
collinear objects, occlusion, large enclosing boxes, and depth pointing can remain
ambiguous. Separate objects and aim the visible ray through the desired box.

Manual check: select each of two objects in turn, bend your index finger, point between
them, and try overlapping boxes. Verify that only one box is green, ambiguity
selects nothing, and the arrow disappears when pointing stops. With the
simulator running, also check that each confirmed object produces the expected
head direction and spoken name.

### Responding to a selected object

The action worker keeps one action in progress and one replaceable pending
target, so head movement and speech do not block the preview or build a queue.
A newly confirmed object causes a roughly 0.6-second head movement followed by
its spoken category name. The preview shows separate robot and voice status.
If the simulator is unavailable, vision keeps running and speech can still work;
robot connection is retried on fresh confirmed selection data at most once every
three seconds. The controller connects only to the local full MuJoCo simulator.

Holding a target steady does not keep repeating its name. A brief selection loss
under 0.75 seconds suppresses repeated naming; reselecting after a longer pause
can announce it again. A centre shift of at least 0.05 in normalized image
coordinates can update the head direction, at most once every 0.6 seconds,
without repeating the name. Stale, ambiguous, or bent-index pointing cancels queued
actions. An already-started head movement finishes; its following speech is
skipped if the selection is no longer current. Speech that has already begun may
finish its current word; quitting stops the speech process.

The gaze rule maps the selected box centre to at most +/-30 degrees yaw and
+/-15 degrees pitch, undoing preview mirroring first. This is an approximate
image-to-head mapping, not a 3D location estimate or calibrated camera-to-robot
transform. Place an external webcam or iPhone approximately at the robot's
neutral viewing position. Camera placement, perspective, and object depth can
affect how closely the simulated gaze appears to align with the real object.

Speech uses the Mac's built-in `say` command and plays through the computer's
selected audio output. On Linux it uses an existing `espeak-ng` or `espeak`
command. No cloud service, API key, or additional Python dependency is used.
If no local speech command is available, the voice status reports it and the
vision/robot paths keep running.

### Preview performance and smooth tracking

`python main.py` targets a 60 FPS preview and requests 960x540 camera frames at
60 FPS. `--fps 30` or `--fps 60` changes the target; accepted values are 1 to 120.
The GPU preview uploads a reusable texture and polls window events without a
timed wait. GLFW and PyOpenGL are pinned in `requirements.txt` and are already
part of this development environment. Keyboard controls are unchanged.
Actual rates depend on the camera, Continuity Camera connection, display, and
available CPU/GPU capacity. A camera delivering 30 FPS still supplies 30 distinct
images per second; a 60 FPS preview can redraw its latest image between captures.

Camera reads run in their own thread and replace one latest frame, so a slow
`camera.read()` cannot block preview drawing. A separate inference worker runs
only the enabled models and retains at most one current and one newest pending
frame. In combined mode, hand and object models process the same clean source
frame. The default remains YOLO26s at 640px; smoothing does not switch models or
change its detector confidence rules.

The preview displays separate rates:

- **Preview FPS / target:** how often the window is redrawn and its requested rate.
- **Camera FPS:** measured arrivals of distinct camera frames.
- **Models Hz:** measured completions of distinct model results.
- **Inference and overlay age:** model processing time and the source age of the
  visual landmarks/boxes, rather than recognition confidence.

Drawing at 60 FPS does not mean running the neural models 60 times per second.
Earlier native measurements around 100ms per combined result correspond to
roughly 10 model updates per second. Display tracking bridges those updates with
batched Lucas-Kanade optical flow and an exponential moving average for up to two
hands and ten boxes. A short history of clean source frames aligns delayed model
predictions with the current camera image before drawing them.

A controlled native GUI test reached 60.0 FPS with the visible optimized MuJoCo
simulator open, real Hand Landmarker/YOLO26s inference, and a synthetic 60 FPS
camera moving public hand/object reference images. The models updated at about
8.7 Hz while optical flow updated the display between predictions. This test
used the actual GPU preview window; physical webcam/iPhone delivery still needs
checking on your chosen device.

A missed detection can retain its neutral/amber visual for up to 220ms, and every
visual expires within 500ms of its last positive model source frame. This reduces
flicker during brief gaps while still clearing lost hands/objects. Optical flow
can fail during fast motion, occlusion, or camera movement; retained visuals are
estimates and can trail the image slightly.

Gesture recognition, target selection, and robot responses use only
matched raw hand/object results captured within 300ms. Display smoothing and held
visuals never count as fresh observations, confirm pointing, create a selected
target, or trigger a robot action. Held visuals stay neutral/amber rather than
showing a new green confirmation. Mirroring, resolution changes, gesture resets,
and confidence changes reset the relevant tracking state.

Manual check: move your finger and either object slowly, then quickly; check that
landmarks and boxes follow without repeated disappearance during short inference
gaps. Briefly hide a hand or object, then keep it out of view for more than half a
second. The visual should clear, and stale or held visuals must not select a new
target, preserve a green pointing confirmation, or queue a robot response. Use
the separate Camera, Preview, and Models rates to distinguish capture limits from
inference speed.

Run the simulator with `mjpython simulator.py`. In the pinned Reachy Mini SDK
1.8.0, the upstream MuJoCo backend also renders a virtual eye camera at
1280x720 and 25 FPS and sends it over UDP, including when `--no-media` is used.
This project uses an external webcam instead. That extra rendering can compete
with inference and previously made the strict 300ms overlay cutoff produce
repeated disappearing dots and boxes. The launcher disables only this unused
virtual-camera loop while retaining the visible MuJoCo viewer, physics, head
movement, and SDK connection. Short visual gaps are now handled by the display
tracker; the 300ms source-age limit for gesture decisions and robot actions stays
in effect.

In a controlled reference-frame benchmark with the visible simulator and SDK
movements active, median inference fell from about 175ms with the upstream
launcher to 103ms with this launcher. Expired-overlay frames fell from 90 to 0
over matching seven-second measurement windows. Capture was paced synthetically
and no live iPhone or preview GUI was used in this benchmark; check your actual
camera after restarting the simulator.

If pauses continue, keep the simulator viewer open and compare with:

```bash
python main.py --no-robot --mute
```

If detection still pauses, the simulator's resource use or camera/GUI scheduling
is involved rather than robot movement or speech calls. Compare the displayed
inference time and overlay age. If this command runs smoothly, the interaction
path needs further diagnosis. Return to `python main.py` for the full pipeline.

Historical reference measurements on the development M1 Pro used a paced
synthetic 30 FPS camera before the 60 FPS preview changes. They kept preview near
30 FPS with about 94ms combined inference; object inference alone measured about
80ms on one public reference image. GUI calls were substituted. These earlier
measurements do not establish a current 60 FPS rate, live iPhone performance, or
accuracy on the user's objects. Actual camera and preview rates must be checked
in the current app.

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

MediaPipe Hand Landmarker supplies the image coordinates. The pointing rule uses
only the index landmarks 5 (base), 6, 7, and 8 (tip). It compares the direct
base-to-tip distance with the total length along the three finger segments:

```text
extension = distance(5, 8) / [distance(5, 6) + distance(6, 7) + distance(7, 8)]
pointing  = extension >= 0.85
```

An extended index has a ratio close to 1; folding it reduces the ratio. This is
one image-distance check, with no 3D joint angles, trained gesture classifier,
pose template, or calibration. The other fingers are ignored, so an open palm or
V sign can count when its index is extended. The debug line shows the extension
ratio, which is a geometry measurement rather than a probability.

POINTING appears on the first valid fresh result. Duplicate preview frames can
redraw that result but do not become additional model observations. Press **R**
to clear it and wait for the next real model result. Two qualifying hands are
ambiguous. Missing, invalid, or stale results clear pointing and stop new robot
actions. Object selection independently keeps its two-result confirmation rule.

The arrow uses the direct normalized vector from base 5 to tip 8; there is no
line fitting or SVD. A projected index shorter than 5 pixels is rejected because
it cannot provide a useful screen direction. This 2D method can miss a finger
aimed toward the camera; point across the image for recognition and selection.
The arrow is hidden when pointing is not detected. Held display landmarks never
trigger target selection.

The old calibration API and `--calibration` option have been removed. An existing
local `calibration/pointing.json` file is left untouched and is not read or used.
No photos or video are saved.

The hand model, 60 FPS display tracking, deterministic ray/bounding-box object
selection, and robot response continue as described above. Check an extended
index in several orientations, then bend it and confirm pointing clears.
Source: [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python).

### Checks

```bash
python -m unittest discover -v
```

Tests cover 2D finger-extension geometry, rotation and scale invariance, direct
arrow direction, immediate pointing, cached-result reset, duplicate-frame handling,
confidence
filtering, person exclusion, both ONNX output formats, coordinate conversion, frame dropping,
orientation changes, stale results, ray geometry, target ambiguity/confirmation,
selection switching and clearing, model cleanup, shared inference timestamps,
mode-specific model loading and controls, the separated gesture interface,
background camera capture, distinct-frame counting, display smoothing and expiry,
source-frame alignment, and worker errors. Action tests cover gaze direction and simulator checks, repeated
selection suppression, cancellation, action-thread scheduling, and speech-process
cleanup. Native models are also checked on public reference images. A controlled
full MuJoCo headless check verified left/right head movement and the action
worker's cup/bottle sequence, with valid local speech saved to audio files.
Live iPhone accuracy, camera-to-head alignment, visible simulator behaviour, and
audio playback still need a user check; synthetic frames and a few example images
are not an accuracy benchmark.

## Run Reachy movement and spoken naming

Stop the desktop app's mockup daemon before starting full MuJoCo. In terminal 1:

```bash
conda activate reachy-detection
mjpython simulator.py
```

Keep it running, then in terminal 2:

```bash
conda activate reachy-detection
python main.py
```

The launcher selects `--sim --no-media` and disables the backend's unused virtual
eye-camera rendering/UDP stream. It keeps the visible full MuJoCo viewer and
normal SDK head control. The app captures the external webcam separately and
speaks through the computer. The desktop app's mockup view is not the full
MuJoCo simulator required for the head-movement demonstration.

The launcher checks for the pinned `reachy-mini==1.8.0` API before applying its
process-local rendering override; it does not edit installed SDK files. To enable
the virtual eye-camera stream explicitly:

```bash
mjpython simulator.py --virtual-camera
```

For reference, the upstream command
`mjpython -m reachy_mini.daemon.app.main --sim --no-media` also enables that
backend stream in SDK 1.8.0. Use the project launcher for the external-camera
pipeline.

Place a cup and bottle apart with both visible. Point at the cup until its box
turns green: Reachy should turn toward its image position and say "cup". Then
select the bottle and check its different head direction and name. Keep pointing
at one object to check that speech does not repeat continuously. Bend your
index finger or point between overlapping candidates and check that no new response is
queued. Confirm the head direction after toggling mirroring with **M**. These
live webcam, simulator, and speaker checks are still necessary on your setup.

For a separate movement diagnostic with the simulator already running:

```bash
python demo_head.py
```
