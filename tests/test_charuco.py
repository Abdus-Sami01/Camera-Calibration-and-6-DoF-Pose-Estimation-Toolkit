import numpy as np
import pytest

from campose.camera_model import CameraIntrinsics
from campose.charuco import (
    CharucoCalibrator,
    CharucoSpec,
    detect_charuco,
    estimate_charuco_pose,
    render_charuco,
)
from campose.rotations import geodesic_angle, rodrigues_to_matrix, rodrigues_to_matrix as r2m

K = np.array([[820.0, 0, 322], [0, 818.0, 242], [0, 0, 1]])
DIST = np.array([-0.22, 0.09, 0.0006, -0.0005, 0.01])
SPEC = CharucoSpec(5, 7, 0.03, 0.022)


@pytest.fixture
def intrinsics():
    return CameraIntrinsics.from_matrix(K, DIST)


def _centered_tvec(rvec, distance):
    facing = r2m(np.array([np.pi, 0.0, 0.0])) @ r2m(rvec)
    return np.array([0.0, 0.0, distance]) - facing @ np.array([SPEC.width / 2, SPEC.height / 2, 0.0])


def test_spec_validation():
    with pytest.raises(ValueError):
        CharucoSpec(5, 7, 0.03, 0.05)


def test_full_board_detects_all_corners(intrinsics):
    rvec = np.array([0.1, -0.15, 0.05])
    rendered = render_charuco(intrinsics, (640, 480), SPEC, rvec, _centered_tvec(rvec, 0.5))
    detection = detect_charuco(rendered.image, SPEC)
    assert detection.count == SPEC.inner_corners


def test_pose_survives_partial_occlusion(intrinsics):
    rvec = np.array([0.1, -0.15, 0.05])
    tvec = _centered_tvec(rvec, 0.5)
    clean = render_charuco(intrinsics, (640, 480), SPEC, rvec, tvec)
    occluded = render_charuco(intrinsics, (640, 480), SPEC, rvec, tvec, occlude=0.5)
    detection = detect_charuco(occluded.image, SPEC)
    assert 0 < detection.count < SPEC.inner_corners
    solution = estimate_charuco_pose(detection, SPEC, intrinsics)
    assert solution is not None
    assert np.linalg.norm(solution.translation - clean.tvec) < 0.02


def test_calibration_recovers_focal_length(intrinsics):
    rng = np.random.default_rng(4)
    calibrator = CharucoCalibrator(SPEC)
    tries = 0
    while calibrator.view_count < 15 and tries < 400:
        tries += 1
        rvec = rng.uniform(-0.4, 0.4, 3)
        tvec = _centered_tvec(rvec, rng.uniform(0.45, 0.7))
        rendered = render_charuco(intrinsics, (640, 480), SPEC, rvec, tvec)
        if rendered.detected_corners >= 6:
            calibrator.add_image(rendered.image, source=f"v{calibrator.view_count}")
    result = calibrator.calibrate()
    assert abs(result.intrinsics.fx - K[0, 0]) < 15.0
    assert result.overall_rms < 0.5


def test_blank_image_not_detected(intrinsics):
    blank = np.full((480, 640, 3), 210, dtype=np.uint8)
    assert not detect_charuco(blank, SPEC).found
