import numpy as np
import pytest
import cv2

from campose.board import CheckerboardSpec, detect_corners
from campose.camera_model import CameraIntrinsics
from campose.rotations import geodesic_angle, matrix_to_rodrigues, rodrigues_to_matrix
from campose.stereo import StereoCalibrator, rectify_pair
from campose.synthetic import VirtualCamera

SPEC = CheckerboardSpec(9, 6, 0.025)
K1 = np.array([[900.0, 0, 320], [0, 900.0, 240], [0, 0, 1]])
K2 = np.array([[898.0, 0, 321], [0, 899.0, 240], [0, 0, 1]])
T_TRUE = np.array([-0.10, 0.0, 0.0])
R_TRUE = rodrigues_to_matrix(np.array([0.0, -0.03, 0.0]))


def _calibrated():
    left = VirtualCamera(CameraIntrinsics.from_matrix(K1, np.zeros(5)), (640, 480), SPEC)
    right = VirtualCamera(CameraIntrinsics.from_matrix(K2, np.zeros(5)), (640, 480), SPEC)
    cx = -(SPEC.columns - 1) * 0.025 / 2
    cy = -(SPEC.rows - 1) * 0.025 / 2
    rng = np.random.default_rng(2)
    calibrator = StereoCalibrator(SPEC)
    tries = 0
    while calibrator.pair_count < 14 and tries < 2500:
        tries += 1
        rvec_l = rng.uniform(-0.35, 0.35, 3)
        tvec_l = np.array([cx + 0.05 + rng.uniform(-0.04, 0.04), cy + rng.uniform(-0.04, 0.04), rng.uniform(0.6, 0.95)])
        rot_l = rodrigues_to_matrix(rvec_l)
        rvec_r = matrix_to_rodrigues(R_TRUE @ rot_l)
        tvec_r = R_TRUE @ tvec_l + T_TRUE
        lb = left.render(rvec_l, tvec_l)
        rb = right.render(rvec_r, tvec_r)
        if lb.visible and rb.visible:
            calibrator.add_pair(lb.image, rb.image)
    return calibrator, left, right


@pytest.fixture(scope="module")
def calibration():
    calibrator, left, right = _calibrated()
    return calibrator.calibrate(), left, right


def test_recovers_baseline(calibration):
    result = calibration[0]
    assert abs(result.baseline - np.linalg.norm(T_TRUE)) < 0.003


def test_recovers_rotation(calibration):
    result = calibration[0]
    assert geodesic_angle(R_TRUE, result.rotation) < 0.5


def test_low_stereo_rms(calibration):
    assert calibration[0].rms < 0.5


def test_rectification_aligns_epipolar_lines(calibration):
    result, left, right = calibration
    cx = -(SPEC.columns - 1) * 0.025 / 2
    cy = -(SPEC.rows - 1) * 0.025 / 2
    tvec_l = np.array([cx + 0.05, cy, 0.7])
    rvec_r = matrix_to_rodrigues(R_TRUE)
    left_img = left.render(np.zeros(3), tvec_l).image
    right_img = right.render(rvec_r, R_TRUE @ tvec_l + T_TRUE).image
    left_rect, right_rect, Q = rectify_pair(left_img, right_img, result)
    dl = detect_corners(left_rect, SPEC)
    dr = detect_corners(right_rect, SPEC)
    assert dl.found and dr.found
    assert np.abs(dl.corners[:, 1] - dr.corners[:, 1]).mean() < 1.0


def test_depth_from_disparity(calibration):
    result, left, right = calibration
    cx = -(SPEC.columns - 1) * 0.025 / 2
    cy = -(SPEC.rows - 1) * 0.025 / 2
    tvec_l = np.array([cx + 0.05, cy, 0.7])
    left_img = left.render(np.zeros(3), tvec_l).image
    right_img = right.render(matrix_to_rodrigues(R_TRUE), R_TRUE @ tvec_l + T_TRUE).image
    left_rect, right_rect, Q = rectify_pair(left_img, right_img, result)
    dl = detect_corners(left_rect, SPEC)
    dr = detect_corners(right_rect, SPEC)
    disparity = dl.corners[:, 0] - dr.corners[:, 0]
    points = np.stack([dl.corners[:, 0], dl.corners[:, 1], disparity], 1).reshape(-1, 1, 3).astype(np.float32)
    depth = cv2.perspectiveTransform(points, Q).reshape(-1, 3)[:, 2]
    assert abs(depth.mean() - 0.7) < 0.02


def test_too_few_pairs_raises():
    with pytest.raises(RuntimeError):
        StereoCalibrator(SPEC).calibrate()
