from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .board import CheckerboardSpec, detect_corners
from .camera_model import CameraIntrinsics, project_points, reprojection_errors, rms
from .quality import CaptureQualityReport, assess_capture
from .results import BoardPose, CalibrationResult

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _sharpness_score(image: np.ndarray) -> float:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class DetectionError(RuntimeError):
    pass


class CameraCalibrator:
    def __init__(self, spec: CheckerboardSpec, refine: bool = True):
        self.spec = spec
        self.refine = refine
        self._image_points: list[np.ndarray] = []
        self._sources: list[str] = []
        self._sharpness: list[float] = []
        self._image_size: tuple[int, int] | None = None

    @property
    def view_count(self) -> int:
        return len(self._image_points)

    def add_image(self, image: np.ndarray, source: str | None = None) -> bool:
        detection = detect_corners(image, self.spec, refine=self.refine, source=source)
        if not detection.found:
            return False
        self._register_size(detection.image_size, source)
        self._image_points.append(detection.corners)
        self._sources.append(source or f"image_{self.view_count}")
        self._sharpness.append(_sharpness_score(image))
        return True

    def add_image_file(self, path: str | Path) -> bool:
        path = Path(path)
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.add_image(image, source=path.name)

    def add_images(self, folder: str | Path) -> int:
        folder = Path(folder)
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
        return sum(int(self.add_image_file(path)) for path in files)

    def _register_size(self, size: tuple[int, int], source: str | None) -> None:
        if self._image_size is None:
            self._image_size = size
        elif self._image_size != size:
            raise ValueError(
                f"Image {source!r} is {size}, but calibration started at {self._image_size}. "
                "Every image must share one resolution and zoom setting."
            )

    def calibrate(self) -> CalibrationResult:
        if self.view_count < 3:
            raise DetectionError(
                f"Only {self.view_count} usable views; need at least 3 (15+ recommended)."
            )
        object_points = [self.spec.object_points() for _ in self._image_points]
        image_points = [pts.astype(np.float32).reshape(-1, 1, 2) for pts in self._image_points]
        overall_rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, self._image_size, None, None
        )
        intrinsics = CameraIntrinsics.from_matrix(matrix, distortion.reshape(-1))
        poses = self._build_poses(intrinsics, rvecs, tvecs)
        return CalibrationResult(
            intrinsics=intrinsics,
            image_size=self._image_size,
            overall_rms=float(overall_rms),
            board_poses=poses,
            image_count=self.view_count,
        )

    def _build_poses(self, intrinsics: CameraIntrinsics, rvecs, tvecs) -> list[BoardPose]:
        object_points = self.spec.object_points()
        poses = []
        for source, observed, rvec, tvec in zip(self._sources, self._image_points, rvecs, tvecs):
            projected = project_points(object_points, rvec, tvec, intrinsics)
            error = rms(reprojection_errors(observed, projected))
            poses.append(BoardPose(source=source, rvec=rvec.reshape(-1), tvec=tvec.reshape(-1), rms_error=error))
        return poses

    def assess_capture(self, min_images: int = 15) -> CaptureQualityReport:
        return assess_capture(self._image_points, self._image_size, self.spec, self._sharpness, min_images=min_images)

    def flag_outliers(self, result: CalibrationResult, sigma: float = 2.0) -> list[BoardPose]:
        errors = result.per_image_rms
        if errors.size == 0:
            return []
        threshold = errors.mean() + sigma * errors.std()
        return [pose for pose in result.board_poses if pose.rms_error > threshold]

    def bootstrap_confidence(self, trials: int = 50, fraction: float = 0.8, seed: int = 0) -> dict:
        if self.view_count < 5:
            raise DetectionError("Bootstrap needs at least 5 views to be meaningful.")
        rng = np.random.default_rng(seed)
        object_points = self.spec.object_points()
        sample_size = max(3, int(round(self.view_count * fraction)))
        samples: list[np.ndarray] = []
        for _ in range(trials):
            idx = rng.choice(self.view_count, size=sample_size, replace=False)
            objs = [object_points for _ in idx]
            imgs = [self._image_points[i].astype(np.float32).reshape(-1, 1, 2) for i in idx]
            _, matrix, distortion, _, _ = cv2.calibrateCamera(objs, imgs, self._image_size, None, None)
            samples.append(self._parameter_vector(matrix, distortion))
        stacked = np.vstack(samples)
        names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
        return {
            "trials": trials,
            "sample_size": sample_size,
            "mean": dict(zip(names, stacked.mean(axis=0))),
            "std": dict(zip(names, stacked.std(axis=0))),
        }

    @staticmethod
    def _parameter_vector(matrix: np.ndarray, distortion: np.ndarray) -> np.ndarray:
        d = np.asarray(distortion, dtype=np.float64).reshape(-1)
        d = np.pad(d, (0, max(0, 5 - d.size)))[:5]
        return np.array([matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2], d[0], d[1], d[2], d[3], d[4]])
