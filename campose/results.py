from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .camera_model import CameraIntrinsics


@dataclass
class BoardPose:
    source: str | None
    rvec: np.ndarray
    tvec: np.ndarray
    rms_error: float


@dataclass
class CalibrationResult:
    intrinsics: CameraIntrinsics
    image_size: tuple[int, int]
    overall_rms: float
    board_poses: list[BoardPose] = field(default_factory=list)
    image_count: int = 0

    @property
    def per_image_rms(self) -> np.ndarray:
        return np.array([pose.rms_error for pose in self.board_poses])

    def worst_images(self, limit: int = 3) -> list[BoardPose]:
        return sorted(self.board_poses, key=lambda p: p.rms_error, reverse=True)[:limit]

    def summary(self) -> str:
        k = self.intrinsics
        d = k.distortion
        lines = [
            "Calibration summary",
            "-------------------",
            f"images used        : {self.image_count}",
            f"image size (w x h) : {self.image_size[0]} x {self.image_size[1]}",
            f"overall RMS error  : {self.overall_rms:.4f} px",
            "",
            "Intrinsic matrix K",
            f"  fx, fy : {k.fx:.2f}, {k.fy:.2f}  (focal length in pixels)",
            f"  cx, cy : {k.cx:.2f}, {k.cy:.2f}  (principal point)",
            "",
            "Distortion coefficients",
            f"  radial     k1, k2, k3 : {d[0]:+.5f}, {d[1]:+.5f}, {d[4]:+.5f}",
            f"  tangential p1, p2     : {d[2]:+.5f}, {d[3]:+.5f}",
        ]
        if self.board_poses:
            worst = self.worst_images(1)[0]
            lines += [
                "",
                f"worst image        : {worst.source or '?'} at {worst.rms_error:.4f} px",
            ]
        lines.append("")
        lines.append(self._quality_verdict())
        return "\n".join(lines)

    def _quality_verdict(self) -> str:
        if self.overall_rms < 0.5:
            return "verdict            : good (RMS < 0.5 px)"
        if self.overall_rms < 1.0:
            return "verdict            : acceptable (RMS < 1.0 px)"
        return "verdict            : investigate (RMS > 1.0 px — recapture before trusting)"
