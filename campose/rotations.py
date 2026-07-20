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


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(mat)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([w, x, y, z])
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    a = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    b = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        result = a + t * (b - a)
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / sin_theta) * a + (np.sin(t * theta) / sin_theta) * b
