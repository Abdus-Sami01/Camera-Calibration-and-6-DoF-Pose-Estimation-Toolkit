import numpy as np
import pytest

from campose.board import CheckerboardSpec
from campose.calibrator import CameraCalibrator
from campose.camera_model import CameraIntrinsics
from campose.quality import assess_capture
from campose.synthetic import VirtualCamera

SPEC = CheckerboardSpec(9, 6, 0.025)
K = np.array([[900.0, 0, 320], [0, 905.0, 240], [0, 0, 1]])
DIST = np.array([-0.25, 0.10, 0.0008, -0.0006, 0.02])


def _build(mode, n=20, seed=11):
    camera = VirtualCamera(CameraIntrinsics.from_matrix(K, DIST), (640, 480), SPEC)
    cx = -(SPEC.columns - 1) * 0.025 / 2
    cy = -(SPEC.rows - 1) * 0.025 / 2
    rng = np.random.default_rng(seed)
    calibrator = CameraCalibrator(SPEC)
    tries = 0
    while calibrator.view_count < n and tries < 9000:
        tries += 1
        if mode == "good":
            rvec = rng.uniform(-0.6, 0.6, 3)
            tvec = np.array([cx + rng.uniform(-0.18, 0.18), cy + rng.uniform(-0.13, 0.13), rng.uniform(0.5, 1.15)])
        elif mode == "centered":
            rvec = rng.uniform(-0.5, 0.5, 3)
            tvec = np.array([cx + rng.uniform(-0.01, 0.01), cy + rng.uniform(-0.01, 0.01), 0.8])
        elif mode == "frontal":
            rvec = rng.uniform(-0.02, 0.02, 3)
            tvec = np.array([cx + rng.uniform(-0.18, 0.18), cy + rng.uniform(-0.13, 0.13), rng.uniform(0.5, 1.15)])
        elif mode == "samedist":
            rvec = rng.uniform(-0.6, 0.6, 3)
            tvec = np.array([cx + rng.uniform(-0.18, 0.18), cy + rng.uniform(-0.13, 0.13), 0.8])
        rendered = camera.render(rvec, tvec)
        if rendered.visible:
            calibrator.add_image(rendered.image, source=f"v{calibrator.view_count}")
    return calibrator


def test_good_capture_passes():
    assert _build("good").assess_capture().ok


def test_centered_capture_flags_distance_spread():
    checks = {f.check for f in _build("centered").assess_capture().warnings}
    assert "distance spread" in checks


def test_frontal_capture_flags_orientation():
    checks = {f.check for f in _build("frontal").assess_capture().warnings}
    assert "orientation" in checks


def test_same_distance_flags_distance_spread():
    checks = {f.check for f in _build("samedist").assess_capture().warnings}
    assert "distance spread" in checks


def test_too_few_images_flags_count():
    checks = {f.check for f in _build("good", n=6).assess_capture().warnings}
    assert "image count" in checks


def test_empty_capture_reports_count_only():
    report = assess_capture([], (640, 480), SPEC)
    assert len(report.findings) == 1
    assert report.findings[0].check == "image count"
