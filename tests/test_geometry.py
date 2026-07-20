import cv2
import numpy as np
import pytest

from campose.camera_model import CameraIntrinsics, apply_distortion, project_points
from campose.rotations import (
    geodesic_angle,
    is_rotation_matrix,
    matrix_to_rodrigues,
    rodrigues_to_euler_zyx,
    rodrigues_to_matrix,
)


@pytest.fixture
def intrinsics():
    K = np.array([[800.0, 0, 320], [0, 810.0, 240], [0, 0, 1]])
    dist = np.array([-0.28, 0.12, 0.001, -0.0009, 0.02])
    return CameraIntrinsics.from_matrix(K, dist)


def test_rodrigues_matches_opencv():
    rvec = np.array([0.2, -0.3, 0.1])
    mine = rodrigues_to_matrix(rvec)
    reference, _ = cv2.Rodrigues(rvec)
    assert np.allclose(mine, reference)
    assert is_rotation_matrix(mine)


def test_rodrigues_round_trip():
    rvec = np.array([-0.7, 0.4, 1.1])
    recovered = matrix_to_rodrigues(rodrigues_to_matrix(rvec))
    assert np.allclose(recovered, rvec, atol=1e-9)


def test_rodrigues_near_pi():
    rvec = np.array([np.pi, 0.0, 0.0])
    matrix = rodrigues_to_matrix(rvec)
    assert geodesic_angle(matrix, rodrigues_to_matrix(matrix_to_rodrigues(matrix))) < 1e-6


def test_euler_identity_is_zero():
    assert np.allclose(rodrigues_to_euler_zyx(np.zeros(3)), np.zeros(3))


def test_projection_matches_opencv(intrinsics):
    rng = np.random.default_rng(0)
    obj = rng.uniform(-0.1, 0.1, (40, 3))
    obj[:, 2] += 1.0
    rvec = np.array([0.2, -0.3, 0.1])
    tvec = np.array([0.05, -0.02, 0.8])
    mine = project_points(obj, rvec, tvec, intrinsics)
    reference, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics.matrix, intrinsics.distortion)
    assert np.allclose(mine, reference.reshape(-1, 2), atol=1e-6)


def test_zero_distortion_is_identity():
    normalized = np.array([[0.1, -0.2], [0.3, 0.05]])
    assert np.allclose(apply_distortion(normalized, np.zeros(5)), normalized)


def test_projection_rejects_points_on_camera_plane(intrinsics):
    with pytest.raises(ValueError):
        project_points(np.array([[0.0, 0.0, 0.0]]), np.zeros(3), np.zeros(3), intrinsics)
