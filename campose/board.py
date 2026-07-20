from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


@dataclass(frozen=True)
class CheckerboardSpec:
    columns: int
    rows: int
    square_size: float

    def __post_init__(self) -> None:
        if self.columns < 2 or self.rows < 2:
            raise ValueError("A checkerboard needs at least a 2x2 inner-corner grid")
        if self.square_size <= 0:
            raise ValueError("Square size must be positive")

    @property
    def pattern_size(self) -> tuple[int, int]:
        return (self.columns, self.rows)

    @property
    def corner_count(self) -> int:
        return self.columns * self.rows

    @classmethod
    def parse(cls, size: str, square_size: float) -> "CheckerboardSpec":
        try:
            cols, rows = (int(part) for part in size.lower().split("x"))
        except ValueError as exc:
            raise ValueError(f"Board size must look like '9x6', got {size!r}") from exc
        return cls(columns=cols, rows=rows, square_size=square_size)

    def object_points(self) -> np.ndarray:
        grid = np.zeros((self.corner_count, 3), dtype=np.float32)
        grid[:, :2] = np.mgrid[0:self.columns, 0:self.rows].T.reshape(-1, 2)
        return grid * self.square_size


@dataclass
class CornerDetection:
    found: bool
    corners: np.ndarray | None
    image_size: tuple[int, int]
    source: str | None = None


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_corners(
    image: np.ndarray,
    spec: CheckerboardSpec,
    refine: bool = True,
    source: str | None = None,
) -> CornerDetection:
    gray = _to_gray(image)
    height, width = gray.shape[:2]
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, spec.pattern_size, flags)
    if not found:
        return CornerDetection(False, None, (width, height), source)
    if refine:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
    return CornerDetection(True, corners.reshape(-1, 2), (width, height), source)
