import numpy as np
import pytest

from campose.rotations import (
    geodesic_angle,
    matrix_to_quaternion,
    quaternion_slerp,
    quaternion_to_matrix,
    rodrigues_to_matrix,
)
from campose.smoothing import ConstantVelocityKalman, PoseSmoother


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        matrix = rodrigues_to_matrix(rng.uniform(-1.5, 1.5, 3))
        recovered = quaternion_to_matrix(matrix_to_quaternion(matrix))
        assert np.allclose(recovered, matrix, atol=1e-9)


def test_slerp_endpoints():
    q0 = matrix_to_quaternion(rodrigues_to_matrix(np.array([0.1, 0.0, 0.0])))
    q1 = matrix_to_quaternion(rodrigues_to_matrix(np.array([1.0, 0.2, 0.0])))
    assert np.allclose(quaternion_slerp(q0, q1, 0.0), q0)
    assert np.allclose(np.abs(quaternion_slerp(q0, q1, 1.0)), np.abs(q1))


def test_kalman_tracks_constant_velocity():
    kalman = ConstantVelocityKalman(dim=1, process_noise=0.01, measurement_noise=1e-4)
    for step in range(60):
        kalman.update(np.array([step * 0.1]), dt=1.0)
    assert abs(kalman.position[0] - 5.9) < 0.2
    assert abs(kalman.velocity[0] - 0.1) < 0.05


def test_smoothing_reduces_translation_jitter():
    rng = np.random.default_rng(1)
    true_r = np.array([0.2, -0.3, 0.1])
    true_t = np.array([0.02, -0.01, 0.5])
    smoother = PoseSmoother(translation_process_noise=0.5, translation_measurement_noise=5e-3, rotation_gain=0.3)
    raw, smoothed = [], []
    for i in range(200):
        noisy_t = true_t + rng.normal(0, 0.005, 3)
        out = smoother.update(true_r + rng.normal(0, 0.05, 3), noisy_t, dt=1 / 30)
        if i > 50:
            raw.append(np.linalg.norm(noisy_t - true_t))
            smoothed.append(np.linalg.norm(out.tvec - true_t))
    assert np.mean(smoothed) < np.mean(raw)


def test_smoothing_reduces_rotation_jitter():
    rng = np.random.default_rng(2)
    true_r = np.array([0.2, -0.3, 0.1])
    true_matrix = rodrigues_to_matrix(true_r)
    smoother = PoseSmoother(rotation_gain=0.25)
    raw, smoothed = [], []
    for i in range(200):
        noisy_r = true_r + rng.normal(0, 0.05, 3)
        out = smoother.update(noisy_r, np.array([0.0, 0.0, 0.5]), dt=1 / 30)
        if i > 50:
            raw.append(geodesic_angle(true_matrix, rodrigues_to_matrix(noisy_r)))
            smoothed.append(geodesic_angle(true_matrix, rodrigues_to_matrix(out.rvec)))
    assert np.mean(smoothed) < np.mean(raw)


def test_reset_clears_state():
    smoother = PoseSmoother()
    smoother.update(np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.5]))
    smoother.reset()
    assert smoother._quaternion is None
    assert not smoother._translation.initialized
