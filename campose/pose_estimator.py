from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera_model import CameraIntrinsics
from .io import load_camera
from .rotations import matrix_to_euler_zyx, rodrigues_to_matrix
from .solvers import PoseSolution, solve_pnp


@dataclass
class ObjectPose:
    rvec: np.ndarray
    tvec: np.ndarray
    reprojection_error: float
    image_points: np.ndarray
    identifier: int | None = None
    solver: str = "iterative"

    @property
    def translation(self) -> np.ndarray:
        return self.tvec.reshape(-1)

    @property
    def rotation_matrix(self) -> np.ndarray:
        return rodrigues_to_matrix(self.rvec)

    @property
    def rotation_euler(self) -> np.ndarray:
        return matrix_to_euler_zyx(self.rotation_matrix)

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.translation))


@dataclass
class BoardPoseResult:
    pose: ObjectPose
    marker_ids: list[int]
    marker_count: int


def _marker_object_points(marker_size: float) -> np.ndarray:
    half = marker_size / 2.0
    return np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)


class PoseEstimator:
    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics
        self._aruco_detector = None
        self._reference = None

    @classmethod
    def from_calibration(cls, path: str | Path) -> "PoseEstimator":
        return cls(load_camera(path))

    def _ensure_aruco(self, dictionary: int) -> None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def estimate_aruco(
        self,
        image: np.ndarray,
        marker_size: float,
        dictionary: int = cv2.aruco.DICT_6X6_250,
        solver: str = "ippe_square",
    ) -> list[ObjectPose]:
        if self._aruco_detector is None:
            self._ensure_aruco(dictionary)
        corners, ids, _ = self._aruco_detector.detectMarkers(image)
        if ids is None:
            return []
        object_points = _marker_object_points(marker_size)
        poses = []
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            solution = solve_pnp(object_points, marker_corners.reshape(-1, 2), self.intrinsics, solver=solver)
            if solution.success:
                poses.append(self._to_object_pose(solution, marker_corners.reshape(-1, 2), int(marker_id)))
        return sorted(poses, key=lambda p: p.reprojection_error)

    def set_planar_reference(
        self,
        reference_image: np.ndarray,
        width: float,
        height: float,
        detector: str = "orb",
        max_features: int = 1500,
    ) -> None:
        gray = reference_image if reference_image.ndim == 2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        feature = _make_feature(detector, max_features)
        keypoints, descriptors = feature.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 4:
            raise ValueError("Reference image did not yield enough features")
        h, w = gray.shape[:2]
        scale = np.array([width / w, height / h])
        pixel_xy = np.array([kp.pt for kp in keypoints])
        model = np.zeros((len(keypoints), 3), dtype=np.float32)
        model[:, 0] = pixel_xy[:, 0] * scale[0] - width / 2.0
        model[:, 1] = height / 2.0 - pixel_xy[:, 1] * scale[1]
        self._reference = {
            "descriptors": descriptors,
            "model_points": model,
            "detector": detector,
            "matcher": _make_matcher(detector),
            "feature": feature,
        }

    def estimate_planar(
        self,
        image: np.ndarray,
        ratio: float = 0.7,
        min_matches: int = 12,
        solver: str = "ippe",
    ) -> ObjectPose | None:
        if self._reference is None:
            raise RuntimeError("Call set_planar_reference before estimate_planar")
        ref = self._reference
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = ref["feature"].detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 4:
            return None
        matches = self._filter_matches(ref["matcher"].knnMatch(ref["descriptors"], descriptors, k=2), ratio)
        if len(matches) < min_matches:
            return None
        model = np.array([ref["model_points"][m.queryIdx] for m in matches], dtype=np.float32)
        observed = np.array([keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)
        model, observed = self._homography_inliers(model, observed)
        if model is None or len(model) < min_matches:
            return None
        solution = solve_pnp(model, observed, self.intrinsics, solver=solver)
        if not solution.success:
            return None
        return self._to_object_pose(solution, observed, None)

    def estimate_board(self, image: np.ndarray, board, min_markers: int = 1, solver: str = "iterative") -> "BoardPoseResult | None":
        if self._aruco_detector is None:
            self._ensure_aruco(board.dictionary)
        corners, ids, _ = self._aruco_detector.detectMarkers(image)
        if ids is None:
            return None
        detected = {int(marker_id): marker_corners.reshape(4, 2) for marker_id, marker_corners in zip(ids.reshape(-1), corners)}
        object_points, image_points, used = board.correspondences(detected)
        if len(used) < min_markers:
            return None
        solution = solve_pnp(object_points, image_points, self.intrinsics, solver=solver)
        if not solution.success:
            return None
        pose = self._to_object_pose(solution, image_points, None)
        return BoardPoseResult(pose=pose, marker_ids=used, marker_count=len(used))

    def estimate(self, image: np.ndarray, target: str = "aruco", **kwargs):
        if target == "aruco":
            return self.estimate_aruco(image, **kwargs)
        if target == "planar":
            return self.estimate_planar(image, **kwargs)
        if target == "board":
            return self.estimate_board(image, **kwargs)
        raise ValueError(f"Unknown target {target!r}; use 'aruco', 'planar', or 'board'")

    def _to_object_pose(self, solution: PoseSolution, image_points: np.ndarray, identifier: int | None) -> ObjectPose:
        return ObjectPose(
            rvec=solution.rvec,
            tvec=solution.tvec,
            reprojection_error=solution.reprojection_rms,
            image_points=np.asarray(image_points, dtype=np.float64),
            identifier=identifier,
            solver=solution.solver,
        )

    @staticmethod
    def _filter_matches(knn_matches, ratio: float):
        good = []
        for pair in knn_matches:
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                good.append(pair[0])
        return good

    @staticmethod
    def _homography_inliers(model: np.ndarray, observed: np.ndarray):
        _, mask = cv2.findHomography(model[:, :2], observed, cv2.RANSAC, 3.0)
        if mask is None:
            return None, None
        keep = mask.ravel().astype(bool)
        return model[keep], observed[keep]


def _make_feature(detector: str, max_features: int):
    name = detector.lower()
    if name == "orb":
        return cv2.ORB_create(nfeatures=max_features)
    if name == "sift":
        return cv2.SIFT_create(nfeatures=max_features)
    raise ValueError(f"Unknown feature detector {detector!r}; use 'orb' or 'sift'")


def _make_matcher(detector: str):
    if detector.lower() == "sift":
        return cv2.BFMatcher(cv2.NORM_L2)
    return cv2.BFMatcher(cv2.NORM_HAMMING)
