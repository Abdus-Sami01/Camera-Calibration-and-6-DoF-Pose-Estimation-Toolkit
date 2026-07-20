from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rotations import rodrigues_to_matrix


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self) -> None:
        self.distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
        if self.distortion.size < 5:
            self.distortion = np.pad(self.distortion, (0, 5 - self.distortion.size))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, distortion: np.ndarray | None = None) -> "CameraIntrinsics":
        mat = np.asarray(matrix, dtype=np.float64)
        return cls(
            fx=float(mat[0, 0]),
            fy=float(mat[1, 1]),
            cx=float(mat[0, 2]),
            cy=float(mat[1, 2]),
            distortion=np.zeros(5) if distortion is None else distortion,
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ])

    @property
    def radial(self) -> np.ndarray:
        d = self.distortion
        return np.array([d[0], d[1], d[4]])

    @property
    def tangential(self) -> np.ndarray:
        d = self.distortion
        return np.array([d[2], d[3]])


def apply_distortion(normalized: np.ndarray, distortion: np.ndarray) -> np.ndarray:
    pts = np.asarray(normalized, dtype=np.float64).reshape(-1, 2)
    k1, k2, p1, p2, k3 = np.asarray(distortion, dtype=np.float64).reshape(-1)[:5]
    x, y = pts[:, 0], pts[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_tang = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_tang = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([x * radial + x_tang, y * radial + y_tang], axis=1)


def project_points(
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    pts = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    rotation = rodrigues_to_matrix(rvec)
    translation = np.asarray(tvec, dtype=np.float64).reshape(3)
    camera = pts @ rotation.T + translation
    depth = camera[:, 2:3]
    if np.any(np.abs(depth) < 1e-9):
        raise ValueError("A point sits on the camera plane (Z=0) and cannot be projected")
    normalized = camera[:, :2] / depth
    distorted = apply_distortion(normalized, intrinsics.distortion)
    u = intrinsics.fx * distorted[:, 0] + intrinsics.cx
    v = intrinsics.fy * distorted[:, 1] + intrinsics.cy
    return np.stack([u, v], axis=1)


def reprojection_errors(observed: np.ndarray, projected: np.ndarray) -> np.ndarray:
    obs = np.asarray(observed, dtype=np.float64).reshape(-1, 2)
    proj = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
    return np.linalg.norm(obs - proj, axis=1)


def rms(errors: np.ndarray) -> float:
    err = np.asarray(errors, dtype=np.float64).reshape(-1)
    if err.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(err * err)))
