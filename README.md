# campose — Camera Calibration & 6-DoF Pose Estimation Toolkit

**Camera calibration and 6-DoF pose estimation with diagnostic reporting and
accuracy evaluation — not another `cv2.calibrateCamera()` wrapper.**

Most calibration code treats OpenCV as a black box: call one function, print a
matrix, move on. This toolkit instead exposes and *checks* every step — the
projection math is written out by hand and tested against OpenCV to
floating-point precision, calibration comes with per-image error and bootstrap
confidence, and pose estimation ships with a real accuracy and robustness
evaluation. It is a small, modular library you could `pip install` and use on
your own camera.

<p align="center">
  <img src="figures/3d_board_poses.png" width="46%" alt="Recovered board poses in 3D"/>
  <img src="figures/accuracy_vs_distance.png" width="52%" alt="Pose accuracy vs distance"/>
</p>

> **A note on the sample data.** This repository ships with a *synthetic* sample
> set rendered by the toolkit's own virtual camera (`campose.synthetic`), with
> exact ground truth recorded in `data/ground_truth.json`. That is deliberate:
> it lets every number below be checked against a known answer rather than a
> screenshot. Calibrating your own real camera is one command away — see the
> [capture guide](docs/capture_guide.md).

---

## Why this exists

Calibration and pose estimation are the "can you actually do geometry" test for
any computer-vision role touching robotics, AR, or inspection. The differentiator
here is rigour: understand the model, expose the intermediate math, report error
honestly, and quantify where things break.

The full derivations live in [`docs/theory.md`](docs/theory.md) — pinhole model,
distortion, calibration as least-squares, and the PnP solvers.

---

## Install

```bash
pip install -e .          # from a clone
# or just the runtime deps:
pip install -r requirements.txt
```

Dependencies are deliberately minimal: `numpy`, `opencv-python`, `matplotlib`,
`pyyaml`.

---

## Quick start (Python API)

```python
from campose import CameraCalibrator, CheckerboardSpec, PoseEstimator

# 1. Calibrate a camera from a folder of checkerboard images
calibrator = CameraCalibrator(CheckerboardSpec(9, 6, square_size=0.025))
calibrator.add_images("data/sample_calibration_images/")
result = calibrator.calibrate()
print(result.summary())            # K, distortion, per-image error, verdict
result_path = "calibration.json"
from campose import io; io.save(result, result_path)

# 2. Estimate an object's 6-DoF pose from a single image
import cv2
estimator = PoseEstimator.from_calibration(result_path)
pose = estimator.estimate_aruco(cv2.imread("data/sample_aruco_images/aruco_00.png"),
                                marker_size=0.08)[0]
print(pose.translation, pose.rotation_euler, pose.reprojection_error)
```

---

## Command line

```bash
# Calibrate, with bootstrap confidence on the parameters
python -m campose calibrate --images data/sample_calibration_images \
    --board-size 9x6 --square-size 0.025 --output calibration.json --bootstrap 50

# Estimate pose from one image
python -m campose estimate --calibration calibration.json \
    --image data/sample_aruco_images/aruco_00.png --target aruco --marker-size 0.08

# Live webcam demo with 3D axis overlay, pose readout, FPS, and quality light
python -m campose live --calibration calibration.json --target aruco --marker-size 0.08
```

The live overlay draws the canonical RGB=XYZ axes on each marker plus a
per-frame reprojection-error quality indicator (green → good, red → suspect):

<p align="center">
  <img src="figures/undistortion.png" width="70%" alt="Original vs undistorted"/>
</p>

---

## Calibration results (bundled synthetic set)

Recovered from 22 synthetic images against a known ground truth of
`fx=920, fy=918, cx=328, cy=244`:

| Parameter | Recovered | Ground truth |
|-----------|-----------|--------------|
| `fx, fy`  | 921.2, 919.1 | 920, 918 |
| `cx, cy`  | 329.7, 244.1 | 328, 244 |
| overall RMS reprojection error | **0.08 px** | — |

<p align="center">
  <img src="figures/reprojection_error_per_image.png" width="49%" alt="Per-image reprojection error"/>
  <img src="figures/distortion_map.png" width="49%" alt="Distortion displacement map"/>
</p>

The per-image chart flags outliers automatically (mean + 2σ); the distortion map
shows how many pixels the lens model displaces each location.

### Capture-quality assessment

Before calibrating, `CameraCalibrator.assess_capture()` inspects the detected
corners and warns about the mistakes that quietly ruin a calibration — too few
images, boards clustered in the centre, no distance variation, all views
fronto-parallel, soft/blurred frames. The `calibrate` CLI prints it automatically:

```
Capture quality assessment
--------------------------
ok image count     : 21 views (>= 15)
ok fov coverage    : corners reach 9/9 frame regions
ok edge coverage   : board reaches all four image edges
ok distance spread : apparent-size variation 0.37
ok orientation     : 86% of views are tilted
ok sharpness       : no obviously blurred views

verdict: capture looks solid
```

---

## Evaluation

Four experiments, reproducible via `evaluation/` and `scripts/make_figures.py`.

### 1. How many images do you actually need?

Error falls and focal length converges as images accumulate, with clear
diminishing returns past ~15–20 views.

<p align="center"><img src="figures/calibration_vs_count.png" width="80%" alt="Calibration vs image count"/></p>

### 2. Pose accuracy vs distance

Translation error stays sub-millimetre up close and grows with distance, as the
marker shrinks in the image. Rotation error grows too and is noisiest when a
planar marker turns to face the camera (the planar ambiguity, discussed in the
theory doc).

<p align="center"><img src="figures/accuracy_vs_distance.png" width="80%" alt="Accuracy vs distance"/></p>

### 3. PnP solver comparison

Same marker, every applicable solver, scored against ground truth:

| solver | trans err (mm) | rot err (deg) | reproj (px) | time (ms) |
|--------|---------------:|--------------:|------------:|----------:|
| ippe_square | 0.67 | 0.89 | 0.254 | **0.04** |
| ippe | 0.67 | 0.89 | 0.254 | 0.23 |
| ap3p | 0.69 | 0.30 | 0.219 | 0.24 |
| p3p | 0.69 | 0.30 | 0.219 | 0.26 |
| iterative | 0.69 | 0.30 | 0.219 | 0.38 |
| sqpnp | 0.69 | 0.30 | 0.219 | 0.44 |
| epnp | 0.69 | 0.30 | 0.219 | 0.65 |

`ippe_square` is by far the fastest for square markers; the general solvers are
marginally more accurate in rotation. `campose.solvers.compare_solvers` produces
this table for any correspondence set.

### 4. Robustness under degradation

Blur, occlusion, and low light at increasing severity. ArUco detection tolerates
heavy blur and low light but fails once ~50% of the marker is occluded — the
motivation for the ChArUco stretch goal.

<p align="center"><img src="figures/robustness.png" width="80%" alt="Robustness stress test"/></p>

---

## Temporal pose smoothing

Per-frame PnP is noisy. `campose.PoseSmoother` steadies it with a constant-velocity
Kalman filter on translation and manifold-aware SLERP smoothing on rotation, so
the live overlay stops trembling without lagging behind real motion. On a static
marker it roughly halves both translation and rotation jitter while tracking
motion to sub-millimetre lag.

<p align="center"><img src="figures/temporal_smoothing.png" width="80%" alt="Temporal pose smoothing"/></p>

```python
from campose import PoseSmoother
smoother = PoseSmoother(rotation_gain=0.3)
steady = smoother.update(pose.rvec, pose.tvec, dt=1/30)   # per frame
```

Enable it in the live demo with `python -m campose live ... --smooth`.

---

## Multi-marker board pose

A single marker vanishes the moment a hand covers it. A **marker board** — several
ArUco markers at known relative positions — pools every visible marker's corners
into one PnP solve, so the pose survives heavy occlusion. On a 3×3 grid the board
pose stays accurate (translation within ~2.7 mm, rotation under ~1.2°) even when
only one of nine markers is visible.

<p align="center"><img src="figures/board_occlusion.png" width="80%" alt="Board pose under occlusion"/></p>

```python
from campose import grid_board, PoseEstimator
board = grid_board(markers_x=3, markers_y=3, marker_length=0.04, marker_separation=0.01)
result = PoseEstimator.from_calibration("calibration.json").estimate_board(image, board)
print(result.marker_count, result.pose.translation, result.pose.reprojection_error)
```

---

## Package layout

```
campose/
├── rotations.py       # Rodrigues / matrix / Euler conversions
├── camera_model.py    # pinhole projection + distortion, by hand
├── board.py           # checkerboard spec + corner detection
├── calibrator.py      # CameraCalibrator: solve + diagnostics + bootstrap
├── results.py         # CalibrationResult / BoardPose value objects
├── io.py              # JSON / YAML / npz save + load
├── solvers.py         # unified PnP solver interface
├── pose_estimator.py  # ArUco + planar-target + marker-board 6-DoF pose
├── marker_board.py    # multi-marker board layout + grid constructor
├── visualization.py   # 3D poses, distortion maps, error charts, axis overlay
├── evaluation.py      # the four experiment runners
├── quality.py         # pre-calibration capture-quality assessment
├── smoothing.py       # constant-velocity Kalman + SLERP pose smoothing
├── synthetic.py       # virtual camera with exact ground truth
├── live.py            # real-time webcam annotation loop
└── cli.py             # calibrate / estimate / live
```

---

## Honest limitations

- **Planar targets only.** ArUco and flat feature targets are supported; full 3D
  object tracking from a CAD model is not.
- **Single camera.** No stereo calibration or depth from disparity.
- **Sample data is synthetic.** Real-camera calibration is fully supported and
  documented, but the bundled images come from the virtual camera so ground
  truth is exact.

## Extensions (natural next steps)

Stereo calibration · ChArUco boards for graceful occlusion handling.

---

## Notebooks

Three executed walkthroughs live in [`notebooks/`](notebooks/):

1. `01_calibration_walkthrough.ipynb` — calibrate, read every diagnostic, and
   check the estimate against ground truth.
2. `02_pose_estimation_demo.ipynb` — recover a marker's 6-DoF pose, draw the
   axes, and compare PnP solvers.
3. `03_evaluation_results.ipynb` — reproduce all four evaluation experiments.

## Development

```bash
pip install -e ".[dev]"
pytest                                   # 20 tests, geometry verified vs OpenCV
python scripts/generate_sample_data.py   # regenerate the synthetic sample set
python scripts/make_figures.py           # regenerate every figure in this README
```

Licensed under the MIT License.
