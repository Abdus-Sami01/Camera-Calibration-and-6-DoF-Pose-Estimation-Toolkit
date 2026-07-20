from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .board import CheckerboardSpec, detect_corners
from .camera_model import CameraIntrinsics


@dataclass
class StereoCalibrationResult:
    left: CameraIntrinsics
    right: CameraIntrinsics
    rotation: np.ndarray
    translation: np.ndarray
    essential: np.ndarray
    fundamental: np.ndarray
    image_size: tuple[int, int]
    rms: float

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.translation))

    def rectify(self, alpha: float = 0.0) -> dict:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.left.matrix, self.left.distortion,
            self.right.matrix, self.right.distortion,
            self.image_size, self.rotation, self.translation.reshape(3, 1), alpha=alpha,
        )
        return {"R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q, "roi1": roi1, "roi2": roi2}

    def summary(self) -> str:
        angle = np.degrees(np.arccos(np.clip((np.trace(self.rotation) - 1) / 2, -1, 1)))
        return "\n".join([
            "Stereo calibration summary",
            "--------------------------",
            f"stereo RMS error   : {self.rms:.4f} px",
            f"baseline           : {self.baseline:.4f} (translation magnitude)",
            f"inter-camera angle : {angle:.3f} deg",
            f"left  fx, fy       : {self.left.fx:.2f}, {self.left.fy:.2f}",
            f"right fx, fy       : {self.right.fx:.2f}, {self.right.fy:.2f}",
        ])


class StereoCalibrator:
    def __init__(self, spec: CheckerboardSpec):
        self.spec = spec
        self._object_points: list[np.ndarray] = []
        self._left_points: list[np.ndarray] = []
        self._right_points: list[np.ndarray] = []
        self._image_size: tuple[int, int] | None = None

    @property
    def pair_count(self) -> int:
        return len(self._object_points)

    def add_pair(self, left_image: np.ndarray, right_image: np.ndarray) -> bool:
        left = detect_corners(left_image, self.spec)
        right = detect_corners(right_image, self.spec)
        if not (left.found and right.found):
            return False
        if left.image_size != right.image_size:
            raise ValueError("Left and right images must share one resolution")
        self._image_size = left.image_size
        self._object_points.append(self.spec.object_points())
        self._left_points.append(left.corners.astype(np.float32).reshape(-1, 1, 2))
        self._right_points.append(right.corners.astype(np.float32).reshape(-1, 1, 2))
        return True

    def calibrate(self) -> StereoCalibrationResult:
        if self.pair_count < 5:
            raise RuntimeError(f"Only {self.pair_count} usable pairs; need at least 5")
        left = self._calibrate_single(self._left_points)
        right = self._calibrate_single(self._right_points)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        rms, k1, d1, k2, d2, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
            self._object_points, self._left_points, self._right_points,
            left.matrix, left.distortion, right.matrix, right.distortion,
            self._image_size, flags=cv2.CALIB_FIX_INTRINSIC, criteria=criteria,
        )
        return StereoCalibrationResult(
            left=left,
            right=right,
            rotation=rotation,
            translation=translation.reshape(3),
            essential=essential,
            fundamental=fundamental,
            image_size=self._image_size,
            rms=float(rms),
        )

    def _calibrate_single(self, image_points: list[np.ndarray]) -> CameraIntrinsics:
        _, matrix, distortion, _, _ = cv2.calibrateCamera(self._object_points, image_points, self._image_size, None, None)
        return CameraIntrinsics.from_matrix(matrix, distortion.reshape(-1))


def disparity_map(left_rectified: np.ndarray, right_rectified: np.ndarray, num_disparities: int = 96, block_size: int = 7) -> np.ndarray:
    left_gray = left_rectified if left_rectified.ndim == 2 else cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = right_rectified if right_rectified.ndim == 2 else cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=(num_disparities // 16) * 16,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )
    return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0


def rectify_pair(left_image: np.ndarray, right_image: np.ndarray, result: StereoCalibrationResult, alpha: float = 0.0):
    rect = result.rectify(alpha=alpha)
    left_map = cv2.initUndistortRectifyMap(result.left.matrix, result.left.distortion, rect["R1"], rect["P1"], result.image_size, cv2.CV_32FC1)
    right_map = cv2.initUndistortRectifyMap(result.right.matrix, result.right.distortion, rect["R2"], rect["P2"], result.image_size, cv2.CV_32FC1)
    left_rect = cv2.remap(left_image, left_map[0], left_map[1], cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_image, right_map[0], right_map[1], cv2.INTER_LINEAR)
    return left_rect, right_rect, rect["Q"]
