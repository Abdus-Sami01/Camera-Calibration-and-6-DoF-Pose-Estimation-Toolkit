from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera_model import CameraIntrinsics, project_points
from .results import BoardPose, CalibrationResult
from .rotations import matrix_to_rodrigues, rodrigues_to_matrix
from .solvers import solve_pnp

_FACING = rodrigues_to_matrix(np.array([np.pi, 0.0, 0.0]))


@dataclass(frozen=True)
class CharucoSpec:
    squares_x: int
    squares_y: int
    square_length: float
    marker_length: float
    dictionary: int = cv2.aruco.DICT_5X5_100

    def __post_init__(self) -> None:
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("A ChArUco board needs at least 2x2 squares")
        if not 0 < self.marker_length < self.square_length:
            raise ValueError("marker_length must be positive and smaller than square_length")

    @property
    def inner_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def width(self) -> float:
        return self.squares_x * self.square_length

    @property
    def height(self) -> float:
        return self.squares_y * self.square_length

    def build(self) -> cv2.aruco.CharucoBoard:
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary)
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_length, self.marker_length, dictionary)


@dataclass
class CharucoDetection:
    found: bool
    corners: np.ndarray | None
    ids: np.ndarray | None
    image_size: tuple[int, int]

    @property
    def count(self) -> int:
        return 0 if self.ids is None else int(self.ids.shape[0])


def detect_charuco(image: np.ndarray, spec: CharucoSpec, board: cv2.aruco.CharucoBoard | None = None) -> CharucoDetection:
    board = board or spec.build()
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(image)
    height, width = image.shape[:2]
    if ids is None or len(ids) < 4:
        return CharucoDetection(False, None, None, (width, height))
    return CharucoDetection(True, corners.reshape(-1, 2), ids.reshape(-1), (width, height))


def charuco_object_points(spec: CharucoSpec, ids: np.ndarray, board: cv2.aruco.CharucoBoard | None = None) -> np.ndarray:
    board = board or spec.build()
    return board.getChessboardCorners()[np.asarray(ids, dtype=int).reshape(-1)]


def _planar_solve(object_points: np.ndarray, image_points: np.ndarray, intrinsics: CameraIntrinsics):
    solver = "iterative" if len(object_points) >= 6 else "ippe"
    return solve_pnp(object_points, image_points, intrinsics, solver=solver)


def estimate_charuco_pose(detection: CharucoDetection, spec: CharucoSpec, intrinsics: CameraIntrinsics):
    if not detection.found or detection.count < 4:
        return None
    object_points = charuco_object_points(spec, detection.ids)
    solution = _planar_solve(object_points, detection.corners, intrinsics)
    if not solution.success or not np.all(np.isfinite(solution.rvec)) or not np.all(np.isfinite(solution.tvec)):
        return None
    return solution


@dataclass
class RenderedCharuco:
    image: np.ndarray
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    detected_corners: int


def render_charuco(
    intrinsics: CameraIntrinsics,
    image_size: tuple[int, int],
    spec: CharucoSpec,
    rvec: np.ndarray,
    tvec: np.ndarray,
    occlude: float = 0.0,
    facing: bool = True,
    background: int = 210,
    pixels_per_meter: int = 4000,
) -> RenderedCharuco:
    board = spec.build()
    rotation = rodrigues_to_matrix(rvec)
    if facing:
        rotation = _FACING @ rotation
    place_rvec = matrix_to_rodrigues(rotation)
    place_tvec = np.asarray(tvec, dtype=np.float64).reshape(3)

    texture = cv2.cvtColor(board.generateImage((int(spec.width * pixels_per_meter), int(spec.height * pixels_per_meter))), cv2.COLOR_GRAY2BGR)
    th, tw = texture.shape[:2]
    rect = np.array([[0, spec.height, 0], [spec.width, spec.height, 0], [spec.width, 0, 0], [0, 0, 0]], dtype=np.float64)
    projected = project_points(rect, place_rvec, place_tvec, intrinsics).astype(np.float32)
    texture_corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
    homography, _ = cv2.findHomography(texture_corners, projected)
    scene = np.full((image_size[1], image_size[0], 3), background, dtype=np.uint8)
    warped = cv2.warpPerspective(texture, homography, image_size, borderValue=(background, background, background))
    mask = cv2.warpPerspective(np.ones((th, tw), np.uint8), homography, image_size).astype(bool)
    scene[mask] = warped[mask]
    if occlude > 0.0:
        _occlude_board(scene, projected, occlude, background)

    detection = detect_charuco(scene, spec, board)
    if not detection.found:
        return RenderedCharuco(scene, None, None, 0)
    solution = _planar_solve(charuco_object_points(spec, detection.ids, board), detection.corners, intrinsics)
    return RenderedCharuco(scene, solution.rvec, solution.tvec, detection.count)


def _occlude_board(scene: np.ndarray, projected: np.ndarray, fraction: float, background: int) -> None:
    ys = projected[:, 1]
    top, bottom = ys.min(), ys.max()
    cut = int(top + fraction * (bottom - top))
    scene[:max(0, cut)] = background


class CharucoCalibrator:
    def __init__(self, spec: CharucoSpec):
        self.spec = spec
        self._board = spec.build()
        self._object_points: list[np.ndarray] = []
        self._image_points: list[np.ndarray] = []
        self._sources: list[str] = []
        self._image_size: tuple[int, int] | None = None

    @property
    def view_count(self) -> int:
        return len(self._object_points)

    def add_image(self, image: np.ndarray, source: str | None = None) -> bool:
        detection = detect_charuco(image, self.spec, self._board)
        if not detection.found:
            return False
        object_points, image_points = self._board.matchImagePoints(detection.corners.reshape(-1, 1, 2), detection.ids.reshape(-1, 1))
        if object_points is None or len(object_points) < 4:
            return False
        height, width = image.shape[:2]
        self._image_size = (width, height)
        self._object_points.append(object_points)
        self._image_points.append(image_points)
        self._sources.append(source or f"image_{self.view_count}")
        return True

    def calibrate(self) -> CalibrationResult:
        if self.view_count < 3:
            raise RuntimeError(f"Only {self.view_count} usable views; need at least 3")
        overall_rms, matrix, distortion, rvecs, tvecs = cv2.calibrateCamera(
            self._object_points, self._image_points, self._image_size, None, None
        )
        intrinsics = CameraIntrinsics.from_matrix(matrix, distortion.reshape(-1))
        poses = [
            BoardPose(source=src, rvec=rvec.reshape(-1), tvec=tvec.reshape(-1), rms_error=0.0)
            for src, rvec, tvec in zip(self._sources, rvecs, tvecs)
        ]
        return CalibrationResult(intrinsics=intrinsics, image_size=self._image_size, overall_rms=float(overall_rms), board_poses=poses, image_count=self.view_count)
