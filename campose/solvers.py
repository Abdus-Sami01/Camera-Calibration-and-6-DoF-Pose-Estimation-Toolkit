from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .camera_model import CameraIntrinsics, project_points, reprojection_errors, rms

_SOLVERS = {
    "iterative": cv2.SOLVEPNP_ITERATIVE,
    "epnp": cv2.SOLVEPNP_EPNP,
    "p3p": cv2.SOLVEPNP_P3P,
    "ap3p": cv2.SOLVEPNP_AP3P,
    "ippe": cv2.SOLVEPNP_IPPE,
    "ippe_square": cv2.SOLVEPNP_IPPE_SQUARE,
    "sqpnp": cv2.SOLVEPNP_SQPNP,
}

_PLANAR_ONLY = {"ippe", "ippe_square"}
_EXACTLY_FOUR = {"ippe_square"}
_MINIMUM_POINTS = {"p3p": 4, "ap3p": 4, "ippe": 4, "ippe_square": 4, "epnp": 4}


@dataclass
class PoseSolution:
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_rms: float
    solver: str
    solve_time_ms: float
    success: bool = True

    @property
    def translation(self) -> np.ndarray:
        return self.tvec.reshape(-1)


def available_solvers() -> list[str]:
    return list(_SOLVERS)


def _is_planar(object_points: np.ndarray, tolerance: float = 1e-6) -> bool:
    pts = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    centered = pts - pts.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    return singular[-1] < tolerance * max(singular[0], 1e-12)


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    solver: str = "iterative",
    refine: bool = True,
) -> PoseSolution:
    name = solver.lower()
    if name not in _SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}; choose from {available_solvers()}")
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
    _validate_points(name, obj)

    started = time.perf_counter()
    ok, rvec, tvec = cv2.solvePnP(obj, img, intrinsics.matrix, intrinsics.distortion, flags=_SOLVERS[name])
    if ok and refine and name not in _PLANAR_ONLY:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, intrinsics.matrix, intrinsics.distortion, rvec, tvec)
    elapsed = (time.perf_counter() - started) * 1000.0

    if not ok:
        return PoseSolution(np.zeros(3), np.zeros(3), float("inf"), name, elapsed, success=False)
    projected = project_points(obj.reshape(-1, 3), rvec, tvec, intrinsics)
    error = rms(reprojection_errors(img.reshape(-1, 2), projected))
    return PoseSolution(rvec.reshape(-1), tvec.reshape(-1), error, name, elapsed)


def _validate_points(name: str, obj: np.ndarray) -> None:
    count = obj.shape[0]
    minimum = _MINIMUM_POINTS.get(name, 4)
    if count < minimum:
        raise ValueError(f"Solver {name!r} needs at least {minimum} points, got {count}")
    if name in _EXACTLY_FOUR and count != 4:
        raise ValueError(f"Solver {name!r} expects exactly 4 points, got {count}")
    if name in _PLANAR_ONLY and not _is_planar(obj.reshape(-1, 3)):
        raise ValueError(f"Solver {name!r} requires a planar object; these points are not coplanar")


def compare_solvers(
    object_points: np.ndarray,
    image_points: np.ndarray,
    intrinsics: CameraIntrinsics,
    solvers: list[str] | None = None,
) -> list[PoseSolution]:
    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    planar = _is_planar(obj)
    chosen = solvers or available_solvers()
    results = []
    for name in chosen:
        if name in _PLANAR_ONLY and not planar:
            continue
        if name in _EXACTLY_FOUR and obj.shape[0] != 4:
            continue
        try:
            results.append(solve_pnp(object_points, image_points, intrinsics, solver=name))
        except (cv2.error, ValueError):
            continue
    return results
