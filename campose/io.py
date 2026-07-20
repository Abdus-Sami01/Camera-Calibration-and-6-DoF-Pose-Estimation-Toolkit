from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .camera_model import CameraIntrinsics
from .results import BoardPose, CalibrationResult


def result_to_dict(result: CalibrationResult) -> dict:
    return {
        "image_size": {"width": result.image_size[0], "height": result.image_size[1]},
        "camera_matrix": result.intrinsics.matrix.tolist(),
        "distortion_coefficients": result.intrinsics.distortion.tolist(),
        "overall_rms_error": result.overall_rms,
        "image_count": result.image_count,
        "board_poses": [
            {
                "source": pose.source,
                "rvec": np.asarray(pose.rvec).reshape(-1).tolist(),
                "tvec": np.asarray(pose.tvec).reshape(-1).tolist(),
                "rms_error": pose.rms_error,
            }
            for pose in result.board_poses
        ],
    }


def dict_to_result(data: dict) -> CalibrationResult:
    matrix = np.array(data["camera_matrix"], dtype=np.float64)
    intrinsics = CameraIntrinsics.from_matrix(matrix, np.array(data["distortion_coefficients"], dtype=np.float64))
    size = (int(data["image_size"]["width"]), int(data["image_size"]["height"]))
    poses = [
        BoardPose(
            source=item.get("source"),
            rvec=np.array(item["rvec"], dtype=np.float64),
            tvec=np.array(item["tvec"], dtype=np.float64),
            rms_error=float(item["rms_error"]),
        )
        for item in data.get("board_poses", [])
    ]
    return CalibrationResult(
        intrinsics=intrinsics,
        image_size=size,
        overall_rms=float(data["overall_rms_error"]),
        board_poses=poses,
        image_count=int(data.get("image_count", len(poses))),
    )


def _prepare(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(result: CalibrationResult, path: str | Path) -> Path:
    path = _prepare(path)
    path.write_text(json.dumps(result_to_dict(result), indent=2))
    return path


def save_yaml(result: CalibrationResult, path: str | Path) -> Path:
    import yaml

    path = _prepare(path)
    path.write_text(yaml.safe_dump(result_to_dict(result), sort_keys=False))
    return path


def save_npz(result: CalibrationResult, path: str | Path) -> Path:
    path = _prepare(path)
    np.savez(
        path,
        camera_matrix=result.intrinsics.matrix,
        distortion_coefficients=result.intrinsics.distortion,
        image_size=np.array(result.image_size),
        overall_rms_error=result.overall_rms,
    )
    return path


def save(result: CalibrationResult, path: str | Path) -> Path:
    suffix = Path(path).suffix.lower()
    if suffix == ".json":
        return save_json(result, path)
    if suffix in (".yaml", ".yml"):
        return save_yaml(result, path)
    if suffix == ".npz":
        return save_npz(result, path)
    raise ValueError(f"Unsupported calibration format: {suffix!r} (use .json, .yaml, or .npz)")


def load(path: str | Path) -> CalibrationResult:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return dict_to_result(json.loads(path.read_text()))
    if suffix in (".yaml", ".yml"):
        import yaml

        return dict_to_result(yaml.safe_load(path.read_text()))
    if suffix == ".npz":
        blob = np.load(path)
        intrinsics = CameraIntrinsics.from_matrix(blob["camera_matrix"], blob["distortion_coefficients"])
        size = tuple(int(v) for v in blob["image_size"])
        return CalibrationResult(intrinsics=intrinsics, image_size=size, overall_rms=float(blob["overall_rms_error"]))
    raise ValueError(f"Unsupported calibration format: {suffix!r}")


def load_camera(path: str | Path) -> CameraIntrinsics:
    return load(path).intrinsics
