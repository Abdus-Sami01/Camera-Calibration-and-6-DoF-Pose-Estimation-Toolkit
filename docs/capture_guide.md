# Calibration capture guide

Calibration lives or dies on the images you feed it. Bad code on good images
gives a bad number you can debug; good code on bad images gives a
confident-looking number that is quietly wrong. This guide is the protocol for
good images, and the reason Phase 1 of the project exists at all.

> The sample images in this repository are **synthetic**, rendered by the
> toolkit's own virtual camera with exact known ground truth (see
> `data/ground_truth.json`). The protocol below is what you follow when
> calibrating a **real** camera of your own.

## The target

- Print a **checkerboard** — 9×6 inner corners, ~25 mm squares is a good
  default. The inner-corner count is what the code needs; a 10×7 grid of squares
  has 9×6 inner corners.
- Glue the print to something **rigid and flat** — foam board or stiff cardboard.
  A checkerboard that curves even slightly introduces a systematic error no
  amount of images will average out.
- **Measure the printed square** with a ruler and pass that number as
  `--square-size`. Printers rescale; do not trust the nominal size. Every
  distance the toolkit reports is in whatever unit you measure here.

## The camera

Any camera works — webcam, phone, USB camera. What matters is that you are
calibrating *that camera at that specific resolution, zoom, and focus*. If any
of those change, the calibration is no longer valid and you must recapture.

## The capture protocol

Take **20–30 images** of the board, varying:

- **Position** — centre, all four edges, and each corner of the frame. The
  distortion coefficients are estimated from how the board bends near the image
  periphery, so peripheral coverage is not optional.
- **Orientation** — tilt the board left, right, toward, and away from the
  camera. Tilt is what lets the optimiser separate focal length from distance.
- **Distance** — near, medium, and far. A single distance leaves focal length
  poorly constrained.

Aim for at least a few images in each of the four image quadrants.

## Mistakes that silently ruin a calibration

| Mistake                              | Symptom                                   |
|--------------------------------------|-------------------------------------------|
| Board only ever in the centre        | Distortion wrong at the edges             |
| All images at one distance           | Unstable focal length (`fx` varies a lot) |
| Curved / non-flat board              | Systematic reprojection error everywhere  |
| Zoom or focus changed mid-set        | Invalid — the model no longer fits        |
| Motion blur                          | Noisy corners, inflated per-image error   |
| Fewer than ~15 usable images         | High parameter variance                   |

## Checking your capture instead of guessing

Do not eyeball it — measure it. `CameraCalibrator.assess_capture()` runs the
checks in the table above automatically and prints a pass/warn verdict per
category (the `calibrate` CLI shows it before every run), so a weak set is caught
before it wastes a calibration. Beyond that:

- **Overall RMS** under ~0.5 px is good, under ~1.0 px acceptable. Above 1.0 px,
  recapture before you touch the code.
- **Per-image error** (`flag_outliers`) points at the specific bad frame. Remove
  it and recalibrate.
- **Bootstrap stability** (`--bootstrap 50`) tells you whether you have *enough*
  images. If `fx`'s standard deviation is tens of pixels, add more views.

The rule to internalise: **if reprojection error is high, the problem is almost
always the images, not the code.** This guide exists so it usually isn't the
images either.
