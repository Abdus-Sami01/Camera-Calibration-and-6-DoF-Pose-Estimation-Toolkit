import numpy as np
import pytest

from campose.board import CheckerboardSpec
from campose.calibrator import CameraCalibrator, DetectionError
from campose.camera_model import CameraIntrinsics
from campose.synthetic import VirtualCamera


@pytest.fixture(scope="module")
def calibrated():
    spec = CheckerboardSpec(9, 6, 0.025)
    K = np.array([[900.0, 0, 320], [0, 905.0, 240], [0, 0, 1]])
    dist = np.array([-0.25, 0.10, 0.0008, -0.0006, 0.02])
    camera = VirtualCamera(CameraIntrinsics.from_matrix(K, dist), (640, 480), spec)
    rng = np.random.default_rng(7)
    calibrator = CameraCalibrator(spec)
    cx = -(spec.columns - 1) * 0.025 / 2
    cy = -(spec.rows - 1) * 0.025 / 2
    tries = 0
    while calibrator.view_count < 18 and tries < 800:
        tries += 1
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([cx + rng.uniform(-0.05, 0.05), cy + rng.uniform(-0.04, 0.04), rng.uniform(0.55, 1.0)])
        rendered = camera.render(rvec, tvec)
        if rendered.visible:
            calibrator.add_image(rendered.image, source=f"v{calibrator.view_count}")
    return calibrator, (K, dist)


def test_recovers_focal_length(calibrated):
    calibrator, (K, _) = calibrated
    result = calibrator.calibrate()
    assert abs(result.intrinsics.fx - K[0, 0]) < 3.0
    assert abs(result.intrinsics.fy - K[1, 1]) < 3.0


def test_recovers_principal_point(calibrated):
    calibrator, (K, _) = calibrated
    result = calibrator.calibrate()
    assert abs(result.intrinsics.cx - K[0, 2]) < 3.0
    assert abs(result.intrinsics.cy - K[1, 2]) < 3.0


def test_reprojection_error_is_small(calibrated):
    calibrator, _ = calibrated
    result = calibrator.calibrate()
    assert result.overall_rms < 0.5


def test_bootstrap_reports_all_parameters(calibrated):
    calibrator, _ = calibrated
    confidence = calibrator.bootstrap_confidence(trials=8)
    assert set(confidence["std"]) == {"fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"}


def test_too_few_views_raises():
    calibrator = CameraCalibrator(CheckerboardSpec(9, 6, 0.025))
    with pytest.raises(DetectionError):
        calibrator.calibrate()


def test_board_spec_parsing():
    spec = CheckerboardSpec.parse("9x6", 0.025)
    assert spec.pattern_size == (9, 6)
    assert spec.object_points().shape == (54, 3)
