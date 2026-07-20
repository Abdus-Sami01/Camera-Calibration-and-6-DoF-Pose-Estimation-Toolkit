from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .board import CheckerboardSpec  # noqa: E402
from .camera_model import CameraIntrinsics, apply_distortion  # noqa: E402
from .results import CalibrationResult  # noqa: E402
from .rotations import rodrigues_to_matrix  # noqa: E402


def draw_axes(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    length: float,
    thickness: int = 3,
) -> np.ndarray:
    canvas = image.copy()
    cv2.drawFrameAxes(canvas, intrinsics.matrix, intrinsics.distortion,
                      np.asarray(rvec, float), np.asarray(tvec, float), length, thickness)
    return canvas


def plot_reprojection_errors(result: CalibrationResult, ax: plt.Axes | None = None) -> plt.Figure:
    errors = result.per_image_rms
    labels = [pose.source or str(i) for i, pose in enumerate(result.board_poses)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, len(errors) * 0.4), 4))
    else:
        fig = ax.figure
    mean = errors.mean() if errors.size else 0.0
    threshold = mean + 2.0 * errors.std() if errors.size else 0.0
    colors = ["#d1495b" if e > threshold else "#4c72b0" for e in errors]
    ax.bar(range(len(errors)), errors, color=colors)
    ax.axhline(mean, color="#2a2a2a", linestyle="--", linewidth=1, label=f"mean {mean:.3f}px")
    ax.axhline(result.overall_rms, color="#e08e0b", linestyle=":", linewidth=1, label=f"overall {result.overall_rms:.3f}px")
    ax.set_xticks(range(len(errors)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("RMS reprojection error (px)")
    ax.set_title("Per-image reprojection error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_board_poses_3d(result: CalibrationResult, spec: CheckerboardSpec) -> plt.Figure:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    _draw_camera(ax)
    corners = _board_outline(spec)
    for pose in result.board_poses:
        rotation = rodrigues_to_matrix(pose.rvec)
        placed = corners @ rotation.T + np.asarray(pose.tvec, float)
        ax.plot(placed[:, 0], placed[:, 1], placed[:, 2], color="#4c72b0", alpha=0.6, linewidth=1)
        centroid = placed[:4].mean(axis=0)
        ax.scatter(*centroid, color="#d1495b", s=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Recovered board poses in the camera frame")
    _equalize_3d(ax)
    return fig


def plot_distortion_map(intrinsics: CameraIntrinsics, image_size: tuple[int, int], step: int = 40) -> plt.Figure:
    width, height = image_size
    xs = np.arange(0, width, step)
    ys = np.arange(0, height, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    displacement = _distortion_displacement(intrinsics, grid_x, grid_y)
    magnitude = np.linalg.norm(displacement, axis=2)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    heat = ax.imshow(magnitude, extent=[0, width, height, 0], cmap="viridis", aspect="auto")
    ax.quiver(grid_x, grid_y, displacement[..., 0], -displacement[..., 1], color="white", alpha=0.7, scale_units="xy")
    fig.colorbar(heat, ax=ax, label="displacement (px)")
    ax.set_title("Lens distortion displacement map")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    fig.tight_layout()
    return fig


def plot_undistortion(image: np.ndarray, intrinsics: CameraIntrinsics) -> plt.Figure:
    undistorted = cv2.undistort(image, intrinsics.matrix, intrinsics.distortion)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, img, title in zip(axes, (image, undistorted), ("original", "undistorted")):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def _distortion_displacement(intrinsics: CameraIntrinsics, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    normalized = np.stack([(grid_x - cx) / fx, (grid_y - cy) / fy], axis=-1).reshape(-1, 2)
    distorted = apply_distortion(normalized, intrinsics.distortion)
    moved = np.stack([distorted[:, 0] * fx + cx, distorted[:, 1] * fy + cy], axis=1)
    original = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    return (moved - original).reshape(*grid_x.shape, 2)


def _board_outline(spec: CheckerboardSpec) -> np.ndarray:
    w = (spec.columns - 1) * spec.square_size
    h = (spec.rows - 1) * spec.square_size
    return np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0], [0, 0, 0]], dtype=np.float64)


def _draw_camera(ax) -> None:
    ax.scatter(0, 0, 0, color="#2a2a2a", s=40, marker="^", label="camera")
    for axis, color in zip(np.eye(3) * 0.05, ("#d1495b", "#2e8b57", "#4c72b0")):
        ax.plot([0, axis[0]], [0, axis[1]], [0, axis[2]], color=color, linewidth=2)
    ax.legend(fontsize=8)


def _equalize_3d(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    span = (limits[:, 1] - limits[:, 0]).max() / 2.0
    centers = limits.mean(axis=1)
    ax.set_xlim3d(centers[0] - span, centers[0] + span)
    ax.set_ylim3d(centers[1] - span, centers[1] + span)
    ax.set_zlim3d(centers[2] - span, centers[2] + span)
