from __future__ import annotations

from .board import CheckerboardSpec, detect_corners
from .calibrator import CameraCalibrator
from .camera_model import (
    CameraIntrinsics,
    apply_distortion,
    project_points,
    reprojection_errors,
    rms,
)
from .pose_estimator import ObjectPose, PoseEstimator
from .results import BoardPose, CalibrationResult
from .rotations import (
    geodesic_angle,
    matrix_to_euler_zyx,
    matrix_to_rodrigues,
    rodrigues_to_euler_zyx,
    rodrigues_to_matrix,
)
from .solvers import PoseSolution, available_solvers, compare_solvers, solve_pnp

__version__ = "0.1.0"

__all__ = [
    "CameraIntrinsics",
    "apply_distortion",
    "project_points",
    "reprojection_errors",
    "rms",
    "rodrigues_to_matrix",
    "matrix_to_rodrigues",
    "matrix_to_euler_zyx",
    "rodrigues_to_euler_zyx",
    "geodesic_angle",
    "CheckerboardSpec",
    "detect_corners",
    "CameraCalibrator",
    "CalibrationResult",
    "BoardPose",
    "PoseEstimator",
    "ObjectPose",
    "solve_pnp",
    "compare_solvers",
    "available_solvers",
    "PoseSolution",
    "__version__",
]
