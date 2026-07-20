from __future__ import annotations

import numpy as np

from campose.board import CheckerboardSpec
from campose.calibrator import CameraCalibrator
from campose.camera_model import CameraIntrinsics
from campose.synthetic import VirtualCamera

SPEC = CheckerboardSpec(9, 6, 0.025)
GROUND_TRUTH_K = np.array([[920.0, 0.0, 328.0], [0.0, 918.0, 244.0], [0.0, 0.0, 1.0]])
GROUND_TRUTH_DIST = np.array([-0.262, 0.121, 0.0009, -0.0007, 0.018])
IMAGE_SIZE = (640, 480)


def ground_truth_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.from_matrix(GROUND_TRUTH_K, GROUND_TRUTH_DIST)


def build_calibrator(views: int = 22, seed: int = 7) -> CameraCalibrator:
    camera = VirtualCamera(ground_truth_intrinsics(), IMAGE_SIZE, SPEC)
    rng = np.random.default_rng(seed)
    calibrator = CameraCalibrator(SPEC)
    cx = -(SPEC.columns - 1) * SPEC.square_size / 2
    cy = -(SPEC.rows - 1) * SPEC.square_size / 2
    tries = 0
    while calibrator.view_count < views and tries < 1500:
        tries += 1
        rvec = rng.uniform(-0.55, 0.55, 3)
        tvec = np.array([cx + rng.uniform(-0.05, 0.05), cy + rng.uniform(-0.04, 0.04), rng.uniform(0.55, 1.05)])
        rendered = camera.render(rvec, tvec)
        if rendered.visible:
            calibrator.add_image(rendered.image, source=f"v{calibrator.view_count:02d}")
    return calibrator
