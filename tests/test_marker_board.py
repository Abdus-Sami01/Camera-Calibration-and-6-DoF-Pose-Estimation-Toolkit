import numpy as np
import pytest

from campose.camera_model import CameraIntrinsics
from campose.marker_board import grid_board
from campose.pose_estimator import PoseEstimator
from campose.rotations import geodesic_angle, rodrigues_to_matrix
from campose.synthetic import render_marker_board


@pytest.fixture
def intrinsics():
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    return CameraIntrinsics.from_matrix(K, np.array([-0.1, 0.02, 0.0, 0.0, 0.0]))


@pytest.fixture
def board():
    return grid_board(3, 3, marker_length=0.04, marker_separation=0.01)


def test_grid_board_layout(board):
    assert board.all_ids() == list(range(9))
    assert board.markers[0].shape == (4, 3)
    assert np.allclose(np.vstack(list(board.markers.values()))[:, 2], 0.0)


def test_full_board_pose_is_accurate(intrinsics, board):
    rendered = render_marker_board(intrinsics, (640, 480), board, np.array([0.15, -0.2, 0.05]), np.array([0.02, -0.01, 0.6]))
    result = PoseEstimator(intrinsics).estimate_board(rendered.image, board)
    assert result.marker_count == 9
    assert np.linalg.norm(result.pose.translation - rendered.tvec) < 0.01
    assert geodesic_angle(rodrigues_to_matrix(rendered.rvec), result.pose.rotation_matrix) < 3.0
    assert result.pose.reprojection_error < 1.0


def test_pose_survives_heavy_occlusion(intrinsics, board):
    rvec, tvec = np.array([0.15, -0.2, 0.05]), np.array([0.02, -0.01, 0.6])
    rendered = render_marker_board(intrinsics, (640, 480), board, rvec, tvec, only_ids=[0, 4, 8])
    result = PoseEstimator(intrinsics).estimate_board(rendered.image, board)
    assert result is not None
    assert result.marker_count == 3
    assert np.linalg.norm(result.pose.translation - rendered.tvec) < 0.02


def test_single_visible_marker_still_solves(intrinsics, board):
    rvec, tvec = np.array([0.1, -0.15, 0.05]), np.array([0.0, 0.0, 0.6])
    rendered = render_marker_board(intrinsics, (640, 480), board, rvec, tvec, only_ids=[4])
    result = PoseEstimator(intrinsics).estimate_board(rendered.image, board)
    assert result is not None and result.marker_count == 1


def test_no_markers_returns_none(intrinsics, board):
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert PoseEstimator(intrinsics).estimate_board(blank, board) is None


def test_correspondences_ignore_unknown_ids(board):
    detected = {0: np.zeros((4, 2)), 99: np.ones((4, 2))}
    obj, img, used = board.correspondences(detected)
    assert used == [0]
    assert obj.shape == (4, 3) and img.shape == (4, 2)
