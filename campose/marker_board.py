from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MarkerBoard:
    dictionary: int
    markers: dict[int, np.ndarray]

    def all_ids(self) -> list[int]:
        return sorted(self.markers)

    def correspondences(self, detected: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        object_points, image_points, used = [], [], []
        for marker_id, corners in detected.items():
            if marker_id in self.markers:
                object_points.append(self.markers[marker_id])
                image_points.append(np.asarray(corners, dtype=np.float64).reshape(4, 2))
                used.append(marker_id)
        if not object_points:
            return np.empty((0, 3)), np.empty((0, 2)), []
        return np.vstack(object_points), np.vstack(image_points), used


def _marker_local_corners(marker_length: float) -> np.ndarray:
    half = marker_length / 2.0
    return np.array([
        [-half, half, 0.0],
        [half, half, 0.0],
        [half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def grid_board(
    markers_x: int,
    markers_y: int,
    marker_length: float,
    marker_separation: float,
    dictionary: int = cv2.aruco.DICT_6X6_250,
    first_id: int = 0,
) -> MarkerBoard:
    pitch = marker_length + marker_separation
    local = _marker_local_corners(marker_length)
    markers: dict[int, np.ndarray] = {}
    for row in range(markers_y):
        for col in range(markers_x):
            center_x = (col - (markers_x - 1) / 2.0) * pitch
            center_y = ((markers_y - 1) / 2.0 - row) * pitch
            offset = np.array([center_x, center_y, 0.0])
            marker_id = first_id + row * markers_x + col
            markers[marker_id] = local + offset
    return MarkerBoard(dictionary=dictionary, markers=markers)
