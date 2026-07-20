from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera_model import CameraIntrinsics
from .rotations import geodesic_angle, rodrigues_to_matrix
from .solvers import compare_solvers


@dataclass
class CountTrial:
    image_count: int
    overall_rms: float
    fx: float
    fy: float


@dataclass
class DistanceTrial:
    true_distance: float
    estimated_distance: float
    translation_error: float
    rotation_error: float

    @property
    def distance_error(self) -> float:
        return abs(self.estimated_distance - self.true_distance)


@dataclass
class SolverTrial:
    solver: str
    translation_error: float
    rotation_error: float
    reprojection_rms: float
    solve_time_ms: float


@dataclass
class RobustnessTrial:
    condition: str
    level: float
    detected: bool
    translation_error: float | None = None
    rotation_error: float | None = None


def calibration_vs_count(calibrator, counts: list[int], seed: int = 0) -> list[CountTrial]:
    from .calibrator import DetectionError

    rng = np.random.default_rng(seed)
    order = rng.permutation(calibrator.view_count)
    trials = []
    for count in counts:
        if count > calibrator.view_count:
            continue
        subset = _subset_calibrator(calibrator, order[:count])
        try:
            result = subset.calibrate()
        except DetectionError:
            continue
        trials.append(CountTrial(count, result.overall_rms, result.intrinsics.fx, result.intrinsics.fy))
    return trials


def pose_vs_distance(rendered_markers, intrinsics: CameraIntrinsics, marker_size: float, solver: str = "ippe_square") -> list[DistanceTrial]:
    from .pose_estimator import PoseEstimator

    estimator = PoseEstimator(intrinsics)
    trials = []
    for rendered in rendered_markers:
        poses = estimator.estimate_aruco(rendered.image, marker_size)
        if not poses:
            continue
        pose = poses[0]
        true_distance = float(np.linalg.norm(rendered.tvec))
        trials.append(DistanceTrial(
            true_distance=true_distance,
            estimated_distance=pose.distance,
            translation_error=float(np.linalg.norm(pose.translation - rendered.tvec)),
            rotation_error=geodesic_angle(rodrigues_to_matrix(rendered.rvec), pose.rotation_matrix),
        ))
    return trials


def solver_comparison(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    true_rvec: np.ndarray,
    true_tvec: np.ndarray,
) -> list[SolverTrial]:
    truth_rotation = rodrigues_to_matrix(true_rvec)
    true_translation = np.asarray(true_tvec, float).reshape(3)
    trials = []
    for solution in compare_solvers(object_points, image_points, intrinsics):
        trials.append(SolverTrial(
            solver=solution.solver,
            translation_error=float(np.linalg.norm(solution.translation - true_translation)),
            rotation_error=geodesic_angle(truth_rotation, rodrigues_to_matrix(solution.rvec)),
            reprojection_rms=solution.reprojection_rms,
            solve_time_ms=solution.solve_time_ms,
        ))
    return sorted(trials, key=lambda t: t.translation_error)


def robustness_sweep(base_image, intrinsics, marker_size, degradations) -> list[RobustnessTrial]:
    from .pose_estimator import PoseEstimator

    estimator = PoseEstimator(intrinsics)
    reference = estimator.estimate_aruco(base_image, marker_size)
    baseline = reference[0] if reference else None
    trials = []
    for condition, level, transform in degradations:
        degraded = transform(base_image)
        poses = estimator.estimate_aruco(degraded, marker_size)
        if not poses:
            trials.append(RobustnessTrial(condition, level, detected=False))
            continue
        pose = poses[0]
        trans_err = None if baseline is None else float(np.linalg.norm(pose.translation - baseline.translation))
        rot_err = None if baseline is None else geodesic_angle(baseline.rotation_matrix, pose.rotation_matrix)
        trials.append(RobustnessTrial(condition, level, True, trans_err, rot_err))
    return trials


def blur(kernel: int):
    def apply(image):
        k = max(1, kernel | 1)
        return cv2.GaussianBlur(image, (k, k), 0)
    return apply


def occlude(fraction: float):
    def apply(image):
        out = image.copy()
        height = out.shape[0]
        rows = int(height * fraction)
        out[:rows] = 0
        return out
    return apply


def darken(factor: float):
    def apply(image):
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return apply


def _subset_calibrator(calibrator, indices):
    clone = calibrator.__class__(calibrator.spec, refine=calibrator.refine)
    clone._image_size = calibrator._image_size
    clone._image_points = [calibrator._image_points[i] for i in indices]
    clone._sources = [calibrator._sources[i] for i in indices]
    clone._sharpness = [calibrator._sharpness[i] for i in indices] if calibrator._sharpness else []
    return clone
