from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import (
    matrix_to_quaternion,
    quaternion_slerp,
    quaternion_to_matrix,
    matrix_to_rodrigues,
    rodrigues_to_matrix,
)


class ConstantVelocityKalman:
    def __init__(self, dim: int = 3, process_noise: float = 1.0, measurement_noise: float = 1e-3):
        self.dim = dim
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state = np.zeros(2 * dim)
        self.covariance = np.eye(2 * dim) * 1.0
        self.initialized = False

    def _transition(self, dt: float) -> np.ndarray:
        F = np.eye(2 * self.dim)
        F[:self.dim, self.dim:] = np.eye(self.dim) * dt
        return F

    def _process_covariance(self, dt: float) -> np.ndarray:
        q = self.process_noise
        upper = (dt ** 3) / 3.0
        cross = (dt ** 2) / 2.0
        Q = np.zeros((2 * self.dim, 2 * self.dim))
        eye = np.eye(self.dim)
        Q[:self.dim, :self.dim] = eye * upper
        Q[:self.dim, self.dim:] = eye * cross
        Q[self.dim:, :self.dim] = eye * cross
        Q[self.dim:, self.dim:] = eye * dt
        return Q * q

    def update(self, measurement: np.ndarray, dt: float) -> np.ndarray:
        z = np.asarray(measurement, dtype=np.float64).reshape(self.dim)
        if not self.initialized:
            self.state[:self.dim] = z
            self.initialized = True
            return self.position
        F = self._transition(dt)
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self._process_covariance(dt)
        H = np.zeros((self.dim, 2 * self.dim))
        H[:, :self.dim] = np.eye(self.dim)
        R = np.eye(self.dim) * self.measurement_noise
        innovation = z - H @ self.state
        S = H @ self.covariance @ H.T + R
        gain = self.covariance @ H.T @ np.linalg.inv(S)
        self.state = self.state + gain @ innovation
        self.covariance = (np.eye(2 * self.dim) - gain @ H) @ self.covariance
        return self.position

    @property
    def position(self) -> np.ndarray:
        return self.state[:self.dim].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[self.dim:].copy()


@dataclass
class SmoothedPose:
    rvec: np.ndarray
    tvec: np.ndarray


class PoseSmoother:
    def __init__(self, translation_process_noise: float = 1.0, translation_measurement_noise: float = 1e-3, rotation_gain: float = 0.5):
        self._translation = ConstantVelocityKalman(3, translation_process_noise, translation_measurement_noise)
        self._rotation_gain = float(np.clip(rotation_gain, 0.0, 1.0))
        self._quaternion: np.ndarray | None = None

    def reset(self) -> None:
        self._translation = ConstantVelocityKalman(3, self._translation.process_noise, self._translation.measurement_noise)
        self._quaternion = None

    def update(self, rvec: np.ndarray, tvec: np.ndarray, dt: float = 1.0) -> SmoothedPose:
        position = self._translation.update(np.asarray(tvec, float).reshape(3), dt)
        measured = matrix_to_quaternion(rodrigues_to_matrix(rvec))
        if self._quaternion is None:
            self._quaternion = measured
        else:
            self._quaternion = quaternion_slerp(self._quaternion, measured, self._rotation_gain)
        smoothed_rvec = matrix_to_rodrigues(quaternion_to_matrix(self._quaternion))
        return SmoothedPose(rvec=smoothed_rvec, tvec=position)
