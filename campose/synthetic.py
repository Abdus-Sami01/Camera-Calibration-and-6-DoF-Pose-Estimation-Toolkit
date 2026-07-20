from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .board import CheckerboardSpec
from .camera_model import CameraIntrinsics, project_points
from .rotations import rodrigues_to_matrix


@dataclass
class RenderedBoard:
    image: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    visible: bool


def _checkerboard_texture(spec: CheckerboardSpec, pixels_per_square: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    squares_x, squares_y = spec.columns + 1, spec.rows + 1
    inner = np.indices((squares_y, squares_x)).sum(axis=0) % 2
    tile = (inner * 255).astype(np.uint8)
    board = cv2.resize(tile, (squares_x * pixels_per_square, squares_y * pixels_per_square), interpolation=cv2.INTER_NEAREST)
    texture = np.full((board.shape[0] + 2 * margin, board.shape[1] + 2 * margin), 255, dtype=np.uint8)
    texture[margin:margin + board.shape[0], margin:margin + board.shape[1]] = board
    scale = spec.square_size / pixels_per_square
    offset = np.array([margin + pixels_per_square, margin + pixels_per_square], dtype=np.float64)
    return texture, np.array([scale, offset[0], offset[1]], dtype=object)


class VirtualCamera:
    def __init__(self, intrinsics: CameraIntrinsics, image_size: tuple[int, int], spec: CheckerboardSpec):
        self.intrinsics = intrinsics
        self.image_size = image_size
        self.spec = spec
        self._pixels_per_square = 40
        self._margin = 60
        self._texture, self._map = _checkerboard_texture(spec, self._pixels_per_square, self._margin)
        self._distort_maps = self._build_distortion_maps()

    def _build_distortion_maps(self) -> tuple[np.ndarray, np.ndarray]:
        width, height = self.image_size
        grid = np.indices((height, width), dtype=np.float32)[::-1].transpose(1, 2, 0).reshape(-1, 1, 2)
        ideal = cv2.undistortPoints(grid, self.intrinsics.matrix, self.intrinsics.distortion, P=self.intrinsics.matrix)
        ideal = ideal.reshape(height, width, 2)
        return ideal[..., 0].copy(), ideal[..., 1].copy()

    def _board_to_pixel_homography(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        rotation = rodrigues_to_matrix(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        plane = np.column_stack([rotation[:, 0], rotation[:, 1], translation.ravel()])
        return self.intrinsics.matrix @ plane

    def render(self, rvec: np.ndarray, tvec: np.ndarray) -> RenderedBoard:
        scale, off_x, off_y = self._map
        texture_to_board = np.array([
            [scale, 0.0, -off_x * scale],
            [0.0, scale, -off_y * scale],
            [0.0, 0.0, 1.0],
        ])
        homography = self._board_to_pixel_homography(rvec, tvec) @ texture_to_board
        ideal = cv2.warpPerspective(
            self._texture, homography, self.image_size,
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=128,
        )
        distorted = cv2.remap(ideal, self._distort_maps[0], self._distort_maps[1], interpolation=cv2.INTER_LINEAR, borderValue=128)
        image = cv2.cvtColor(distorted, cv2.COLOR_GRAY2BGR)
        visible = self._corners_visible(rvec, tvec)
        return RenderedBoard(image=image, rvec=np.asarray(rvec, float), tvec=np.asarray(tvec, float), visible=visible)

    def _corners_visible(self, rvec: np.ndarray, tvec: np.ndarray) -> bool:
        rotation = rodrigues_to_matrix(rvec)
        camera_z = (self.spec.object_points() @ rotation.T + np.asarray(tvec, float))[:, 2]
        if np.any(camera_z <= 1e-3):
            return False
        pts = project_points(self.spec.object_points(), rvec, tvec, self.intrinsics)
        width, height = self.image_size
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height)
        return bool(np.all(inside))


def _marker_corner_points(marker_size: float) -> np.ndarray:
    half = marker_size / 2.0
    return np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)


@dataclass
class RenderedMarker:
    image: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    marker_id: int
    corners: np.ndarray


def render_aruco_marker(
    intrinsics: CameraIntrinsics,
    image_size: tuple[int, int],
    marker_id: int,
    marker_size: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
    dictionary: int = cv2.aruco.DICT_6X6_250,
    background: int = 200,
    quiet_zone: int = 60,
) -> RenderedMarker:
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    glyph = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 300)
    padded = cv2.copyMakeBorder(glyph, quiet_zone, quiet_zone, quiet_zone, quiet_zone, cv2.BORDER_CONSTANT, value=255)
    side = padded.shape[0]
    inner = np.array([[quiet_zone, quiet_zone], [side - quiet_zone, quiet_zone],
                      [side - quiet_zone, side - quiet_zone], [quiet_zone, side - quiet_zone]], dtype=np.float32)
    projected = project_points(_marker_corner_points(marker_size), rvec, tvec, intrinsics).astype(np.float32)
    readable = projected[[3, 2, 1, 0]]
    homography, _ = cv2.findHomography(inner, readable)
    scene = cv2.warpPerspective(
        cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR), homography, image_size,
        borderValue=(background, background, background),
    )
    true_rvec, true_tvec = _pose_from_corners(readable, intrinsics, marker_size)
    return RenderedMarker(image=scene, rvec=true_rvec, tvec=true_tvec, marker_id=marker_id, corners=readable)


def _pose_from_corners(corners: np.ndarray, intrinsics: CameraIntrinsics, marker_size: float):
    _, rvec, tvec = cv2.solvePnP(
        _marker_corner_points(marker_size), corners.astype(np.float32),
        intrinsics.matrix, intrinsics.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    return rvec.reshape(-1), tvec.reshape(-1)
