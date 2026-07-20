from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .camera_model import CameraIntrinsics
from .pose_estimator import ObjectPose, PoseEstimator
from .smoothing import PoseSmoother


def annotate_frame(
    frame: np.ndarray,
    estimator: PoseEstimator,
    marker_size: float,
    fps: float | None = None,
    smoothers: dict[int, PoseSmoother] | None = None,
    dt: float = 1.0,
) -> tuple[np.ndarray, list[ObjectPose]]:
    poses = estimator.estimate_aruco(frame, marker_size)
    if smoothers is not None:
        poses = [_smooth_pose(pose, smoothers, dt) for pose in poses]
    canvas = frame.copy()
    axis_length = marker_size * 0.75
    for pose in poses:
        cv2.drawFrameAxes(canvas, estimator.intrinsics.matrix, estimator.intrinsics.distortion,
                          pose.rvec, pose.tvec, axis_length, 2)
    _draw_readout(canvas, poses, fps)
    return canvas, poses


def _smooth_pose(pose: ObjectPose, smoothers: dict[int, PoseSmoother], dt: float) -> ObjectPose:
    if pose.identifier is None:
        return pose
    smoother = smoothers.setdefault(pose.identifier, PoseSmoother())
    result = smoother.update(pose.rvec, pose.tvec, dt)
    return ObjectPose(
        rvec=result.rvec,
        tvec=result.tvec,
        reprojection_error=pose.reprojection_error,
        image_points=pose.image_points,
        identifier=pose.identifier,
        solver=pose.solver,
    )


def _draw_readout(canvas: np.ndarray, poses: list[ObjectPose], fps: float | None) -> None:
    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:4.1f}")
    if not poses:
        lines.append("no marker")
    for pose in poses[:3]:
        t = pose.translation * 100.0
        euler = pose.rotation_euler
        lines.append(
            f"#{pose.identifier} d={pose.distance * 100:5.1f}cm "
            f"xyz=({t[0]:+.1f},{t[1]:+.1f},{t[2]:+.1f}) "
            f"rpy=({euler[0]:+.0f},{euler[1]:+.0f},{euler[2]:+.0f}) "
            f"e={pose.reprojection_error:.2f}px"
        )
    y = 24
    for line in lines:
        color = _quality_color(poses[0].reprojection_error) if poses and line.startswith("#") else (240, 240, 240)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 22


def _quality_color(reprojection_error: float) -> tuple[int, int, int]:
    if reprojection_error < 1.0:
        return (80, 220, 80)
    if reprojection_error < 3.0:
        return (60, 200, 240)
    return (80, 80, 230)


def process_video(
    source: str | int,
    intrinsics: CameraIntrinsics,
    marker_size: float,
    output_path: str | Path | None = None,
    show: bool = True,
    smooth: bool = False,
) -> None:
    estimator = PoseEstimator(intrinsics)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source {source!r}")
    writer = _make_writer(capture, output_path)
    smoothers: dict[int, PoseSmoother] | None = {} if smooth else None
    fps_hint = capture.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps_hint
    smoothed_fps = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            annotated, _ = annotate_frame(frame, estimator, marker_size, smoothed_fps, smoothers, dt)
            instant = 1.0 / max(time.perf_counter() - started, 1e-6)
            smoothed_fps = instant if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * instant
            if writer is not None:
                writer.write(annotated)
            if show:
                cv2.imshow("campose live", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def _make_writer(capture, output_path):
    if output_path is None:
        return None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def run_live(calibration_path: str | Path, marker_size: float, camera_index: int = 0, smooth: bool = False) -> int:
    intrinsics = PoseEstimator.from_calibration(calibration_path).intrinsics
    process_video(camera_index, intrinsics, marker_size, show=True, smooth=smooth)
    return 0
