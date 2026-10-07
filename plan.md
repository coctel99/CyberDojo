# Desktop Fencing Technique Analyzer — Development Plan

Below is the roadmap I would use for a **MacBook-first prototype**. I would deliberately keep the first version offline and video-based rather than trying to solve real-time coaching immediately.

## 1. Define the MVP narrowly

The first version should analyze **one person, one weapon, one camera, and one technique at a time**.

Start with:

- Suburi
- Maki-uchi

Input:

```text
video.mp4
```

Output:

```text
Overall score: 84/100

Sword trajectory       91
Final sword position   88
Elbow movement         73
Shoulder stability     79
Timing                  86

Detected issues:
- Right elbow moves too far laterally during the downswing
- Kissaki finishes 8° too far to the right
- Right shoulder rises during acceleration
```

And, more importantly, produce an **annotated video** showing the skeleton, sword axis, trajectory and deviations.

The initial goal is not:

> “AI understands Katori Shinto-ryu.”

The goal is:

> “The system reliably measures motion and compares it against a known-good reference.”

---

## 2. Proposed architecture

```text
                           ┌────────────────────┐
                           │    Input Video     │
                           │  60/120 FPS ideal  │
                           └─────────┬──────────┘
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │ Frame Preprocessing    │
                        │ OpenCV                 │
                        └───────────┬────────────┘
                                    │
                   ┌────────────────┴─────────────────┐
                   │                                  │
                   ▼                                  ▼
        ┌─────────────────────┐            ┌──────────────────────┐
        │ Human Pose          │            │ Sword Pose           │
        │ RTMPose WholeBody   │            │ Custom YOLO Pose     │
        │                     │            │                      │
        │ body + hands        │            │ kashira              │
        │ keypoints           │            │ tsuba                │
        │                     │            │ kissaki               │
        └──────────┬──────────┘            └──────────┬───────────┘
                   │                                  │
                   └────────────────┬─────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Tracking / Smoothing│
                         │ Kalman / One-Euro   │
                         │ / Savitzky-Golay    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Normalized Motion   │
                         │ Representation      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Repetition / Phase Detection │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                   ┌─────────────────────────────────┐
                   │ Technique Evaluation            │
                   │                                 │
                   │ angles                          │
                   │ trajectories                    │
                   │ velocities                      │
                   │ symmetry                        │
                   │ DTW vs reference                │
                   └───────────────┬─────────────────┘
                                   │
                                   ▼
                   ┌─────────────────────────────────┐
                   │ Feedback + Visualization        │
                   │ score / plots / annotated video│
                   └─────────────────────────────────┘
```

---

## Phase 1 — Basic video analysis pipeline

Do not train anything yet.

Build:

```text
video
  ↓
frames
  ↓
human pose
  ↓
keypoint sequence
  ↓
overlay skeleton
  ↓
output video
```

For human pose, I would start with **RTMPose WholeBody through `rtmlib`**.

`rtmlib` currently supports an `mps` device, so it is particularly convenient on Apple Silicon.

You don't need to train this model yourself.

Relevant points:

```text
nose

left/right shoulder
left/right elbow
left/right wrist

left/right hip
left/right knee
left/right ankle

hand keypoints
```

The hand keypoints may later help determine grip orientation.

### Stack

```text
Python
uv

OpenCV
NumPy
SciPy

PyTorch
rtmlib / RTMPose

matplotlib
pandas
```

Your Mac can use PyTorch's `mps` backend to run supported computations on the Apple GPU.

---

## Phase 2 — Detect the sword

This is probably the first model you actually need to train.

I would formulate this as **custom pose estimation**, not merely object detection.

Sword keypoints:

```text
0 — kashira
1 — tsuba
2 — kissaki
```

Potentially later:

```text
3 — center of tsuka
```

Example:

```text
kashira             tsuba                         kissaki
   ●------------------●------------------------------●
```

Use a small **YOLO Pose** model.

Ultralytics supports custom keypoint layouts through its pose training pipeline, and training can run using `device="mps"` on Apple Silicon.

For a local prototype on an Apple Silicon MacBook, that makes the iteration loop relatively straightforward.

### Dataset

Record videos under different conditions:

```text
front
45°
side

light room
darker room

white wall
complex background

black bokken
wooden bokken
etc.
```

Extract frames.

Then annotate:

```text
bounding box

kashira
tsuba
kissaki
```

YOLO Pose's dataset format supports arbitrary keypoints associated with each object.

Use something like CVAT for annotation.

I would initially aim for **diversity rather than massive quantity**. Sword orientation and motion blur matter more than having thousands of almost identical frames.

---

## Phase 3 — Solve motion blur before anything sophisticated

This could become one of the hardest CV problems in the project.

During fast cuts:

```text
actual sword:

──────────────

camera frame:

///////////////////
```

The sword may effectively become a blurred streak.

So during dataset collection, prefer:

```text
60 FPS minimum
120 FPS if available

short exposure
good lighting
```

For the early prototype, using recorded iPhone footage and analyzing it on the Mac is probably better than restricting yourself to the MacBook webcam.

You can later support:

```text
Mac webcam
USB camera
iPhone Continuity Camera
video upload
```

but internally they should all produce the same video pipeline.

---

## Phase 4 — Create a normalized motion representation

Raw image coordinates are unsuitable.

For example:

```text
wrist = (1243, 681)
```

means almost nothing because camera distance changes.

Normalize coordinates relative to the body.

For example:

```python
body_scale = shoulder_distance
origin = pelvis_center

normalized_point = (point - origin) / body_scale
```

You can additionally normalize orientation using the shoulder/hip axis.

Then the same movement should have approximately comparable coordinates at different camera distances.

Store per frame:

```text
timestamp

body:
    shoulders
    elbows
    wrists
    hips
    knees
    ankles

sword:
    kashira
    tsuba
    kissaki

derived:
    sword_angle
    sword_length_2d
    elbow_angles
    shoulder_angles
    wrist_distance
    torso_angle
```

The result becomes something like:

```text
TechniqueSequence
    ├── body_keypoints[T, K, 2]
    ├── sword_keypoints[T, 3, 2]
    ├── confidence[T, ...]
    └── features[T, F]
```

---

## Phase 5 — Temporal smoothing

Frame-by-frame pose estimates will jitter.

Do **not** evaluate technique directly from raw keypoints.

Pipeline:

```text
raw points

   ↓

remove low-confidence detections

   ↓

interpolate short gaps

   ↓

temporal smoothing

   ↓

calculate angles / velocity
```

Candidates:

- One Euro Filter
- Kalman filter
- Savitzky-Golay filter

For offline analysis, Savitzky-Golay is particularly convenient because latency is irrelevant.

This becomes important when computing derivatives such as:

```text
angular velocity

dθ/dt
```

because tiny coordinate noise produces enormous velocity noise.

---

## Phase 6 — Segment individual repetitions

Suppose the video contains:

```text
ready
suburi
suburi
suburi
pause
suburi
```

The program should automatically produce:

```text
rep_001: frames 103–148
rep_002: frames 171–219
rep_003: frames 245–291
rep_004: frames 381–429
```

Initially this can be rule-based.

For example, detect:

```text
sword angle
+
vertical position of kissaki
+
angular velocity
```

A suburi might look approximately like:

```text
ready
  ↓
backswing
  ↓
maximum sword angle
  ↓
acceleration
  ↓
impact/cut phase
  ↓
deceleration
  ↓
final position
```

This segmentation is useful even before technique evaluation exists.

---

## Phase 7 — Build the reference technique system

This is the core of the application.

Record several technically correct repetitions.

Ideally from an instructor:

```text
reference_01
reference_02
...
reference_20
```

Convert each to the normalized representation.

Instead of choosing one recording as “perfect”, construct a **reference distribution**.

For example:

```text
mean elbow trajectory
± standard deviation

mean sword trajectory
± standard deviation

mean final position
± tolerance
```

So instead of saying:

```text
ideal elbow angle = 147.3°
```

you can say:

```text
normal range = 141–153°
```

That is much more realistic.

---

## Phase 8 — Use DTW for temporal alignment

Don't compare:

```text
student frame 20
vs
teacher frame 20
```

because movement speeds differ.

Instead:

```text
teacher

|--------|--------------|-------|

student

|-------------|-------------------|----------|
```

Use **Dynamic Time Warping** to align the sequences.

Suitable signals could include:

```text
sword_angle

right_elbow_angle
left_elbow_angle

right_wrist_y
left_wrist_y

kissaki_x
kissaki_y
```

After alignment, compare corresponding movement phases.

---

## Phase 9 — Define explicit technique metrics

This is where martial arts knowledge becomes more important than ML.

For **suburi**, you might measure:

```text
Sword path
    lateral deviation
    vertical path
    final angle

Hands
    wrist spacing
    relative motion

Elbows
    maximum flexion
    extension timing
    lateral movement

Torso
    lean
    shoulder rotation
    shoulder elevation

Lower body
    stance width
    knee angle
    center movement

Timing
    acceleration profile
    peak velocity position
    stopping characteristics
```

For **maki-uchi**, additional metrics could be:

```text
kissaki trajectory curvature

circular/elliptical trajectory shape

radius of movement

rotation timing

hand synchronization

final sword alignment
```

The critical design principle is:

**Do not let the ML model define what “correct” means.**

Encode technical principles explicitly and calibrate them using instructor examples.

Different koryu, schools and even instructors can have meaningful technical differences.

---

## Phase 10 — Build the scoring system

Avoid one opaque score at first.

Produce a score hierarchy:

```text
Overall
│
├── Sword          87
│   ├── trajectory       91
│   ├── final angle      88
│   └── stability        82
│
├── Upper body     78
│   ├── elbows           73
│   ├── shoulders        77
│   └── wrists           84
│
├── Lower body     92
│
└── Timing         85
```

Then calculate overall score from weighted components:

```text
overall =
    sword       × 0.40 +
    upper_body  × 0.25 +
    lower_body  × 0.15 +
    timing      × 0.20
```

The actual weights should eventually come from your instructor rather than from statistical convenience.

---

## Phase 11 — Build visual debugging tools

This is extremely important.

For every analyzed video, generate an overlay:

```text
body skeleton

sword axis

kissaki trail

reference trajectory

actual trajectory

joint angles
```

For example:

```text
        reference
           │
           │
           │
           │

        actual
          ╲
           ╲
            ╲
```

Also generate plots:

```text
Sword angle vs time

        teacher
       ╱──────╲
──────╯        ╲──────

        student
      ╱─────────╲
─────╯           ╲─────
```

Without these tools, debugging scoring errors will be miserable.

---

## Phase 12 — Build the first desktop UI

I wouldn't build a native macOS UI yet.

Use:

```text
Gradio
```

or possibly Streamlit.

For this particular ML application, I prefer Gradio.

UI:

```text
┌──────────────────────────────────────────────┐
│ Technique: [ Maki-uchi ▼ ]                  │
│                                              │
│ [ Upload video ]                             │
│                                              │
│ Reference: [ Instructor A ▼ ]               │
│                                              │
│             [ Analyze ]                      │
├──────────────────────────────────────────────┤
│                                              │
│        Annotated video                       │
│                                              │
├──────────────────────────────────────────────┤
│ Overall score: 84                            │
│                                              │
│ Sword trajectory        91                   │
│ Elbows                  76                   │
│ Shoulders               80                   │
│ Timing                  86                   │
│                                              │
│ Issues:                                      │
│ • right shoulder rises                       │
│ • sword finishes 8° right                    │
└──────────────────────────────────────────────┘
```

Everything can run locally.

No backend server is necessary initially.

---

## Phase 13 — Suggested repository structure

I would structure it something like:

```text
fencing-coach/
│
├── pyproject.toml
├── README.md
│
├── configs/
│   ├── suburi.yaml
│   └── maki_uchi.yaml
│
├── src/
│   └── fencing_coach/
│       │
│       ├── video/
│       │   ├── reader.py
│       │   └── preprocessing.py
│       │
│       ├── pose/
│       │   ├── human_pose.py
│       │   ├── sword_pose.py
│       │   └── smoothing.py
│       │
│       ├── motion/
│       │   ├── normalization.py
│       │   ├── features.py
│       │   ├── segmentation.py
│       │   └── alignment.py
│       │
│       ├── techniques/
│       │   ├── base.py
│       │   ├── suburi.py
│       │   └── maki_uchi.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── scoring.py
│       │   └── reference.py
│       │
│       ├── visualization/
│       │   ├── overlay.py
│       │   └── plots.py
│       │
│       └── app/
│           └── gradio_app.py
│
├── models/
│   └── sword_pose/
│
├── datasets/
│   ├── raw/
│   ├── frames/
│   └── labels/
│
├── references/
│   ├── suburi/
│   └── maki_uchi/
│
├── experiments/
│
└── tests/
```

I would specifically keep:

```text
pose estimation
```

separate from:

```text
martial arts evaluation
```

because you will probably replace models several times while the evaluation logic remains largely unchanged.

---

## Phase 14 — Add a learned temporal model only later

Once you have enough labeled repetitions, you can start learning things such as:

```text
good
shoulder too high
elbow too wide
sword too far right
poor stop
etc.
```

Then the architecture could become:

```text
pose sequence
      ↓
feature encoder
      ↓
Temporal Transformer / TCN
      ↓
error classification
```

Potential input:

```text
T × N keypoints
```

Potential output:

```text
correct              0.83

right shoulder high  0.71
right elbow wide     0.92
bad final angle      0.12
```

But I would treat this as **V2**, not MVP.

The rule/reference-based approach is much easier to debug and requires dramatically less training data.

---

## Phase 15 — Only after that consider 3D

The first version should be deliberately **2D**.

Use fixed camera positions:

```text
FRONT
45°
SIDE
```

and have separate reference profiles.

Later you can introduce:

```text
camera 1 ──┐
            ├── triangulation ──> 3D skeleton
camera 2 ──┘
```

That would significantly improve analysis of:

```text
sword plane
forward/backward motion
torso rotation
hand depth
```

I would prefer true two-camera geometry over monocular neural 3D reconstruction for evaluating subtle technique.

---

## Phase 16 — LLM/VLM feedback layer

Only add this once the deterministic evaluator works.

The LLM should receive something like:

```json
{
  "technique": "maki_uchi",
  "overall_score": 81,
  "errors": [
    {
      "metric": "right_shoulder_elevation",
      "deviation": 0.13,
      "phase": "downswing"
    },
    {
      "metric": "kissaki_lateral_deviation",
      "deviation": 0.09,
      "phase": "final"
    }
  ]
}
```

Then it can produce readable feedback:

> Your sword trajectory is generally consistent, but your right shoulder begins to rise during the acceleration phase. This coincides with the kissaki moving to the right of the reference trajectory.

That is a much better use of an LLM than asking a multimodal model to judge the original video directly.

---

## What I would build first

My actual implementation sequence would be:

1. **Video → RTMPose → annotated skeleton video.**
2. Add coordinate normalization.
3. Extract joint angles and trajectories.
4. Record several suburi videos and visualize them.
5. Implement manual repetition boundaries first.
6. Implement automatic repetition segmentation.
7. Build DTW comparison between two repetitions.
8. Create a basic suburi reference.
9. Build the first metrics and scoring system.
10. Train the sword keypoint model.
11. Merge sword + body tracking.
12. Add maki-uchi.
13. Build Gradio UI.
14. Collect instructor reference data.
15. Only then experiment with learned temporal models.

The important point is that **YOLO training doesn't actually need to be the first task**. You can make a surprising amount of progress using only body pose and manually marking the sword in a few videos.

---

## Target V1

I would call V1 complete when this works reliably:

```text
Upload 10-second video
        ↓
choose "Maki-uchi"
        ↓
system finds 5 repetitions
        ↓
tracks body + sword
        ↓
compares all 5 against reference
        ↓

Rep 1     87
Rep 2     81
Rep 3     91
Rep 4     76
Rep 5     88

Main recurring problem:
Right elbow moves outward during acceleration.

[Show trajectory]
[Show joint-angle plot]
[Export annotated video]
```

That would already be a genuinely useful training tool rather than just a pose-estimation demo.

## Notes

For later commercialization, review the licensing of any third-party model or library used in the prototype before distributing a commercial product.
