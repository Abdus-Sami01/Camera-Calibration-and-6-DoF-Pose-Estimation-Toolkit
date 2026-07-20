from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import CheckerboardSpec

_OK = "ok"
_WARN = "warning"


@dataclass
class QualityFinding:
    check: str
    severity: str
    message: str


@dataclass
class CaptureQualityReport:
    findings: list[QualityFinding] = field(default_factory=list)
    coverage_fraction: float = 0.0
    filled_cells: int = 0
    scale_spread: float = 0.0
    tilt_fraction: float = 0.0

    @property
    def warnings(self) -> list[QualityFinding]:
        return [f for f in self.findings if f.severity == _WARN]

    @property
    def ok(self) -> bool:
        return not self.warnings

    def summary(self) -> str:
        lines = ["Capture quality assessment", "--------------------------"]
        for finding in self.findings:
            mark = "ok " if finding.severity == _OK else "!! "
            lines.append(f"{mark}{finding.check:<16}: {finding.message}")
        verdict = "capture looks solid" if self.ok else f"{len(self.warnings)} issue(s) to address before trusting calibration"
        lines.append("")
        lines.append(f"verdict: {verdict}")
        return "\n".join(lines)


def _board_quad(corners: np.ndarray, spec: CheckerboardSpec) -> np.ndarray:
    cols, rows = spec.columns, spec.rows
    indices = [0, cols - 1, cols * rows - 1, cols * (rows - 1)]
    return corners[indices]


def _quad_area(quad: np.ndarray) -> float:
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _foreshortening(quad: np.ndarray) -> float:
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    horizontal = abs(np.log(max(top, 1e-6) / max(bottom, 1e-6)))
    vertical = abs(np.log(max(left, 1e-6) / max(right, 1e-6)))
    return max(horizontal, vertical)


def assess_capture(
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    spec: CheckerboardSpec,
    sharpness: list[float] | None = None,
    min_images: int = 15,
    tilt_threshold: float = 0.04,
) -> CaptureQualityReport:
    report = CaptureQualityReport()
    count = len(image_points)
    report.findings.append(_check_count(count, min_images))
    if count == 0:
        return report

    width, height = image_size
    quads = [_board_quad(np.asarray(pts, float), spec) for pts in image_points]

    report.filled_cells, coverage_finding = _check_coverage(image_points, width, height)
    report.findings.append(coverage_finding)
    report.findings.append(_check_periphery(image_points, width, height))

    areas = np.array([_quad_area(quad) for quad in quads])
    report.scale_spread = float(np.std(areas) / max(np.mean(areas), 1e-9))
    report.findings.append(_check_scale(report.scale_spread))

    tilts = np.array([_foreshortening(quad) for quad in quads])
    report.tilt_fraction = float(np.mean(tilts > tilt_threshold))
    report.findings.append(_check_tilt(report.tilt_fraction))

    if sharpness is not None and len(sharpness) == count:
        report.findings.append(_check_sharpness(np.asarray(sharpness, float)))
    return report


def _check_count(count: int, minimum: int) -> QualityFinding:
    if count >= minimum:
        return QualityFinding("image count", _OK, f"{count} views (>= {minimum})")
    return QualityFinding("image count", _WARN, f"only {count} views; capture at least {minimum} for stable estimates")


def _check_coverage(image_points: list[np.ndarray], width: int, height: int) -> tuple[int, QualityFinding]:
    corners = np.vstack([np.asarray(pts, float) for pts in image_points])
    cols = np.clip((corners[:, 0] / width * 3).astype(int), 0, 2)
    rows = np.clip((corners[:, 1] / height * 3).astype(int), 0, 2)
    filled = len({(int(r), int(c)) for r, c in zip(rows, cols)})
    if filled >= 7:
        return filled, QualityFinding("fov coverage", _OK, f"corners reach {filled}/9 frame regions")
    return filled, QualityFinding("fov coverage", _WARN, f"corners reach only {filled}/9 regions; spread boards across the frame")


def _check_periphery(image_points: list[np.ndarray], width: int, height: int) -> QualityFinding:
    all_corners = np.vstack([np.asarray(pts, float) for pts in image_points])
    margin_x, margin_y = 0.18 * width, 0.18 * height
    reaches = {
        "left": all_corners[:, 0].min() < margin_x,
        "right": all_corners[:, 0].max() > width - margin_x,
        "top": all_corners[:, 1].min() < margin_y,
        "bottom": all_corners[:, 1].max() > height - margin_y,
    }
    missing = [edge for edge, hit in reaches.items() if not hit]
    if not missing:
        return QualityFinding("edge coverage", _OK, "board reaches all four image edges")
    return QualityFinding("edge coverage", _WARN, f"board never nears the {', '.join(missing)} edge(s); distortion will be under-constrained there")


def _check_scale(spread: float) -> QualityFinding:
    if spread >= 0.2:
        return QualityFinding("distance spread", _OK, f"apparent-size variation {spread:.2f}")
    return QualityFinding("distance spread", _WARN, f"apparent-size variation only {spread:.2f}; vary the camera-to-board distance")


def _check_tilt(fraction: float) -> QualityFinding:
    if fraction >= 0.4:
        return QualityFinding("orientation", _OK, f"{fraction * 100:.0f}% of views are tilted")
    return QualityFinding("orientation", _WARN, f"only {fraction * 100:.0f}% of views are tilted; add oblique angles to constrain focal length")


def _check_sharpness(scores: np.ndarray) -> QualityFinding:
    median = float(np.median(scores))
    soft = int(np.sum(scores < 0.4 * median))
    if soft == 0:
        return QualityFinding("sharpness", _OK, "no obviously blurred views")
    return QualityFinding("sharpness", _WARN, f"{soft} view(s) look soft relative to the set; check for motion blur")
