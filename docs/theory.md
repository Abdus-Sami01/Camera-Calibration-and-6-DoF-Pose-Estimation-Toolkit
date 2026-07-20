# The geometry behind the toolkit

This document is the "can you actually do the geometry" companion to the code.
Nothing here is decoration: every equation maps onto a function you can open and
read, and the toolkit is built so that you never have to take OpenCV's word for
a result you could derive yourself.

## 1. The pinhole camera

A camera turns a 3D point into a 2D pixel. The pinhole model captures the whole
transformation as a chain of matrix multiplications:

```
        [u]       [fx  0  cx] [r11 r12 r13 | tx] [X]
    s * [v]  =    [ 0 fy  cy] [r21 r22 r23 | ty] [Y]
        [1]       [ 0  0   1] [r31 r32 r33 | tz] [Z]
                                                 [1]

              =        K       [   R   |   t   ]  P_world
```

Reading it right to left:

1. `[R | t]` moves a world point into the camera's own frame. `R` is the 3×3
   rotation of the camera, `t` its translation. Together they are the
   **extrinsics** — six numbers, three for rotation and three for translation.
2. Dividing by the camera-frame depth `Z` performs the perspective projection:
   distant things get smaller. This is the `s` scale factor, and it is why the
   equation is written up to scale.
3. `K`, the **intrinsic matrix**, converts the projected ray into pixels. `fx`
   and `fy` are the focal length measured in pixels; `cx, cy` is the principal
   point, ideally near the image centre.

In code this is `campose.camera_model.project_points`, written out by hand. Its
test asserts agreement with `cv2.projectPoints` to floating-point precision — so
the same arithmetic that OpenCV hides is right there to inspect.

## 2. Lens distortion

Real lenses bend light non-linearly, so straight lines bow. The toolkit uses
OpenCV's five-coefficient model `(k1, k2, p1, p2, k3)`, applied in normalized
(pre-`K`) coordinates. With `r² = x² + y²`:

```
    radial     : (x, y) *= 1 + k1·r² + k2·r⁴ + k3·r⁶
    tangential : x += 2·p1·x·y + p2·(r² + 2x²)
                 y += p1·(r² + 2y²) + 2·p2·x·y
```

- **Radial** terms (`k1, k2, k3`) produce barrel or pincushion bowing — the
  dominant effect in most lenses.
- **Tangential** terms (`p1, p2`) model a lens that is not perfectly parallel to
  the sensor (decentering). Usually small.

`apply_distortion` implements exactly these lines, and `visualization.plot_distortion_map`
turns the coefficients into a picture of how far every pixel is pushed.

## 3. What calibration solves

Given many images of a checkerboard whose geometry we know, calibration finds
the single `K` and distortion vector — plus one `[R | t]` per image — that make
the model's predicted corners land on the detected corners. Formally it
minimises the **reprojection error**:

```
    minimise  Σ_i Σ_j  || detected_ij  −  project(K, dist, R_i, t_i, X_j) ||²
```

over all images `i` and board corners `j`. The board's 3D points `X_j` are known
because we defined the board (`CheckerboardSpec.object_points`), and the detected
2D points come from `cv2.findChessboardCorners` refined to sub-pixel accuracy.
OpenCV solves the non-linear least-squares with Levenberg–Marquardt; the toolkit
wraps it in `CameraCalibrator` and, crucially, reports how much to trust it.

### Why the diagnostics matter

A single RMS number hides a lot. The toolkit adds two honest checks:

- **Per-image error** flags the one blurry or non-flat board dragging the fit
  down (`CameraCalibrator.flag_outliers`).
- **Bootstrap resampling** recalibrates on random subsets and reports the
  standard deviation of every parameter (`bootstrap_confidence`). If `fx` swings
  by tens of pixels between subsets, the calibration is unstable and needs more
  or better images — a conclusion you can only reach by measuring it.

## 4. Pose estimation and the PnP problem

Once the camera is calibrated, finding an object's pose from one image is the
**Perspective-n-Point** problem: given `n` known 3D model points and their 2D
detections, recover the `[R | t]` that projects one onto the other.

The toolkit exposes every OpenCV solver behind one interface (`campose.solvers`):

| Solver        | Idea                                   | Notes                          |
|---------------|----------------------------------------|--------------------------------|
| `iterative`   | LM refinement from a homography guess  | General, accurate, the default |
| `epnp`        | Linear formulation in a control basis  | Fast for many points           |
| `p3p` / `ap3p`| Closed-form from exactly 3 (+1) points | Minimal, good RANSAC kernels   |
| `ippe`        | Planar-specific, returns two solutions | Coplanar targets only          |
| `ippe_square` | Planar, specialised to 4 square corners| Ideal for ArUco markers        |
| `sqpnp`       | Global non-minimal solver              | Robust, no initial guess       |

Because they all accept the same correspondences, `compare_solvers` runs them
side by side — the ablation most tutorials skip.

### The planar ambiguity

A flat target viewed nearly head-on has two poses that project almost
identically — related by a flip about an in-plane axis. This is not a bug; it is
geometry, and it is why `ippe` returns two candidates and why pose jitter grows
as a marker turns to face the camera. The toolkit surfaces this honestly in the
distance and robustness experiments rather than hiding it.

## 5. Coordinate conventions

The toolkit commits to OpenCV's camera frame throughout:

- **Z** points forward, into the scene.
- **X** points right, **Y** points down.

Rotations are stored as compact Rodrigues (axis-angle) vectors, the form OpenCV
returns, and converted on demand to rotation matrices or intrinsic Z-Y-X Euler
angles for reporting (`campose.rotations`). A single convention, applied
everywhere, is worth more than any clever trick.
