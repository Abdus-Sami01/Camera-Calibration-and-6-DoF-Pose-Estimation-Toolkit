import numpy as np
import pytest

from campose.camera_model import CameraIntrinsics, project_points
from campose.io import dict_to_result, load, save
from campose.pose_estimator import PoseEstimator, _marker_object_points
from campose.results import BoardPose, CalibrationResult
from campose.rotations import geodesic_angle, rodrigues_to_matrix
from campose.solvers import available_solvers, compare_solvers, solve_pnp
from campose.synthetic import render_aruco_marker


@pytest.fixture
def intrinsics():
    K = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    return CameraIntrinsics.from_matrix(K, np.array([-0.1, 0.02, 0.0, 0.0, 0.0]))


def _result(intrinsics):
    return CalibrationResult(
        intrinsics=intrinsics,
        image_size=(640, 480),
        overall_rms=0.21,
        board_poses=[BoardPose("a.png", np.array([0.1, 0.2, 0.3]), np.array([0.0, 0.0, 0.5]), 0.19)],
        image_count=1,
    )


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".npz"])
def test_calibration_round_trips(tmp_path, intrinsics, suffix):
    original = _result(intrinsics)
    path = save(original, tmp_path / f"cal{suffix}")
    reloaded = load(path)
    assert np.allclose(reloaded.intrinsics.matrix, original.intrinsics.matrix)
    assert np.allclose(reloaded.intrinsics.distortion, original.intrinsics.distortion)
    assert reloaded.image_size == original.image_size


def test_solvers_agree_on_clean_data(intrinsics):
    rvec, tvec = np.array([0.1, -0.15, 0.05]), np.array([0.0, 0.0, 0.5])
    obj = _marker_object_points(0.08)
    img = project_points(obj, rvec, tvec, intrinsics)
    for solution in compare_solvers(obj, img, intrinsics):
        assert solution.reprojection_rms < 1e-3


def test_all_solver_names_available():
    assert "iterative" in available_solvers()
    assert "sqpnp" in available_solvers()


def test_aruco_pose_accuracy(intrinsics):
    rendered = render_aruco_marker(intrinsics, (640, 480), 23, 0.05,
                                   np.array([0.12, -0.18, 0.05]), np.array([0.02, -0.01, 0.4]))
    poses = PoseEstimator(intrinsics).estimate_aruco(rendered.image, 0.05)
    assert len(poses) == 1
    pose = poses[0]
    assert np.linalg.norm(pose.translation - rendered.tvec) < 0.01
    assert geodesic_angle(rodrigues_to_matrix(rendered.rvec), pose.rotation_matrix) < 5.0


def test_planar_solver_rejects_non_planar(intrinsics):
    obj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]], dtype=np.float64)
    img = project_points(obj, np.zeros(3), np.array([0.0, 0.0, 3.0]), intrinsics)
    with pytest.raises(ValueError):
        solve_pnp(obj, img, intrinsics, solver="ippe")
