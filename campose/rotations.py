from __future__ import annotations

import numpy as np


def _as_vector(rvec: np.ndarray) -> np.ndarray:
    vec = np.asarray(rvec, dtype=np.float64).reshape(-1)
    if vec.size != 3:
        raise ValueError(f"Expected a 3-element rotation vector, got shape {np.asarray(rvec).shape}")
    return vec


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    vec = _as_vector(rvec)
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3)
    axis = vec / theta
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def matrix_to_rodrigues(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got shape {mat.shape}")
    angle = np.arccos(np.clip((np.trace(mat) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-12:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        return _rodrigues_near_pi(mat, angle)
    axis = np.array([
        mat[2, 1] - mat[1, 2],
        mat[0, 2] - mat[2, 0],
        mat[1, 0] - mat[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


def _rodrigues_near_pi(mat: np.ndarray, angle: float) -> np.ndarray:
    diag = (np.diag(mat) + 1.0) / 2.0
    axis = np.sqrt(np.clip(diag, 0.0, None))
    largest = int(np.argmax(axis))
    for other in range(3):
        if other == largest:
            continue
        axis[other] = np.copysign(axis[other], mat[largest, other])
    return axis / np.linalg.norm(axis) * angle


def matrix_to_euler_zyx(matrix: np.ndarray, degrees: bool = True) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    pitch = np.arcsin(np.clip(-mat[2, 0], -1.0, 1.0))
    if np.cos(pitch) > 1e-6:
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        roll = np.arctan2(mat[2, 1], mat[2, 2])
    else:
        yaw = np.arctan2(-mat[0, 1], mat[1, 1])
        roll = 0.0
    angles = np.array([roll, pitch, yaw])
    return np.degrees(angles) if degrees else angles


def rodrigues_to_euler_zyx(rvec: np.ndarray, degrees: bool = True) -> np.ndarray:
    return matrix_to_euler_zyx(rodrigues_to_matrix(rvec), degrees=degrees)


def geodesic_angle(matrix_a: np.ndarray, matrix_b: np.ndarray, degrees: bool = True) -> float:
    relative = np.asarray(matrix_a, dtype=np.float64).T @ np.asarray(matrix_b, dtype=np.float64)
    angle = np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(angle)) if degrees else float(angle)


def is_rotation_matrix(matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        return False
    orthonormal = np.allclose(mat.T @ mat, np.eye(3), atol=tolerance)
    return orthonormal and abs(np.linalg.det(mat) - 1.0) < tolerance
