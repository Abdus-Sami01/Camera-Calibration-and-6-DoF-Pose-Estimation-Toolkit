from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from campose.board import CheckerboardSpec
from campose.camera_model import CameraIntrinsics
from campose.synthetic import VirtualCamera, render_aruco_marker

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

GROUND_TRUTH_K = np.array([[920.0, 0.0, 328.0], [0.0, 918.0, 244.0], [0.0, 0.0, 1.0]])
GROUND_TRUTH_DIST = np.array([-0.262, 0.121, 0.0009, -0.0007, 0.018])
IMAGE_SIZE = (640, 480)
SQUARE_SIZE = 0.025


def _centered_translation(spec: CheckerboardSpec, rng, z_range):
    cx = -(spec.columns - 1) * spec.square_size / 2.0
    cy = -(spec.rows - 1) * spec.square_size / 2.0
    return np.array([
        cx + rng.uniform(-0.05, 0.05),
        cy + rng.uniform(-0.04, 0.04),
        rng.uniform(*z_range),
    ])


def generate_calibration_images(count: int = 22, seed: int = 11) -> dict:
    spec = CheckerboardSpec(9, 6, SQUARE_SIZE)
    camera = VirtualCamera(CameraIntrinsics.from_matrix(GROUND_TRUTH_K, GROUND_TRUTH_DIST), IMAGE_SIZE, spec)
    out_dir = DATA / "sample_calibration_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    saved = []
    while len(saved) < count:
        rvec = rng.uniform(-0.55, 0.55, 3)
        tvec = _centered_translation(spec, rng, (0.55, 1.05))
        rendered = camera.render(rvec, tvec)
        if not rendered.visible:
            continue
        name = f"board_{len(saved):02d}.png"
        cv2.imwrite(str(out_dir / name), rendered.image)
        saved.append({"file": name, "rvec": rvec.tolist(), "tvec": tvec.tolist()})
    return {"board_size": "9x6", "square_size": SQUARE_SIZE, "images": saved}


def generate_aruco_images(seed: int = 5) -> dict:
    intrinsics = CameraIntrinsics.from_matrix(GROUND_TRUTH_K, GROUND_TRUTH_DIST)
    out_dir = DATA / "sample_aruco_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    marker_size = 0.08
    saved = []
    for i, distance in enumerate([0.3, 0.45, 0.6, 0.75, 0.9]):
        rvec = rng.uniform(-0.25, 0.25, 3)
        tvec = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), distance])
        rendered = render_aruco_marker(intrinsics, IMAGE_SIZE, 17, marker_size, rvec, tvec)
        name = f"aruco_{i:02d}.png"
        cv2.imwrite(str(out_dir / name), rendered.image)
        saved.append({
            "file": name,
            "marker_id": 17,
            "marker_size": marker_size,
            "rvec": rendered.rvec.tolist(),
            "tvec": rendered.tvec.tolist(),
        })
    return {"dictionary": "DICT_6X6_250", "markers": saved}


def main() -> None:
    calibration = generate_calibration_images()
    aruco = generate_aruco_images()
    manifest = {
        "note": "Synthetic data rendered by campose.synthetic; ground truth is exact.",
        "camera_matrix": GROUND_TRUTH_K.tolist(),
        "distortion_coefficients": GROUND_TRUTH_DIST.tolist(),
        "image_size": {"width": IMAGE_SIZE[0], "height": IMAGE_SIZE[1]},
        "calibration": calibration,
        "aruco": aruco,
    }
    (DATA / "ground_truth.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(calibration['images'])} calibration images and {len(aruco['markers'])} ArUco images.")


if __name__ == "__main__":
    main()
