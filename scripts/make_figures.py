from __future__ import annotations

from pathlib import Path

import cv2  # noqa: E402
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from campose import evaluation as ev  # noqa: E402
from campose import visualization as viz  # noqa: E402
from campose.board import CheckerboardSpec  # noqa: E402
from campose.calibrator import CameraCalibrator  # noqa: E402
from campose.camera_model import CameraIntrinsics  # noqa: E402
from campose.charuco import CharucoSpec, detect_charuco, estimate_charuco_pose, render_charuco  # noqa: E402
from campose.marker_board import grid_board  # noqa: E402
from campose.pose_estimator import PoseEstimator, _marker_object_points  # noqa: E402
from campose.rotations import geodesic_angle, rodrigues_to_matrix  # noqa: E402
from campose.smoothing import PoseSmoother  # noqa: E402
from campose.stereo import StereoCalibrator, rectify_pair  # noqa: E402
from campose.synthetic import VirtualCamera, render_aruco_marker, render_marker_board  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
SPEC = CheckerboardSpec(9, 6, 0.025)
K = np.array([[920.0, 0.0, 328.0], [0.0, 918.0, 244.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.262, 0.121, 0.0009, -0.0007, 0.018])
SIZE = (640, 480)


def _build_calibrator() -> CameraCalibrator:
    camera = VirtualCamera(CameraIntrinsics.from_matrix(K, DIST), SIZE, SPEC)
    rng = np.random.default_rng(7)
    calibrator = CameraCalibrator(SPEC)
    cx = -(SPEC.columns - 1) * SPEC.square_size / 2
    cy = -(SPEC.rows - 1) * SPEC.square_size / 2
    tries = 0
    while calibrator.view_count < 22 and tries < 1000:
        tries += 1
        rvec = rng.uniform(-0.55, 0.55, 3)
        tvec = np.array([cx + rng.uniform(-0.05, 0.05), cy + rng.uniform(-0.04, 0.04), rng.uniform(0.55, 1.05)])
        rendered = camera.render(rvec, tvec)
        if rendered.visible:
            calibrator.add_image(rendered.image, source=f"v{calibrator.view_count:02d}")
    return calibrator


def _calibration_figures(calibrator, result) -> None:
    viz.plot_reprojection_errors(result).savefig(FIGURES / "reprojection_error_per_image.png", dpi=110)
    viz.plot_board_poses_3d(result, SPEC).savefig(FIGURES / "3d_board_poses.png", dpi=110)
    viz.plot_distortion_map(result.intrinsics, SIZE).savefig(FIGURES / "distortion_map.png", dpi=110)
    camera = VirtualCamera(CameraIntrinsics.from_matrix(K, DIST), SIZE, SPEC)
    cx = -(SPEC.columns - 1) * SPEC.square_size / 2
    cy = -(SPEC.rows - 1) * SPEC.square_size / 2
    sample = camera.render(np.array([0.15, -0.1, 0.05]), np.array([cx, cy, 0.7])).image
    viz.plot_undistortion(sample, result.intrinsics).savefig(FIGURES / "undistortion.png", dpi=110)
    plt.close("all")


def _count_figure(calibrator) -> None:
    trials = ev.calibration_vs_count(calibrator, [4, 6, 8, 10, 13, 16, 19, 22])
    counts = [t.image_count for t in trials]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(counts, [t.overall_rms for t in trials], "o-", color="#4c72b0")
    ax1.set_xlabel("images used")
    ax1.set_ylabel("overall RMS (px)")
    ax1.set_title("Calibration error vs image count")
    ax2.plot(counts, [t.fx for t in trials], "o-", color="#d1495b", label="fx")
    ax2.axhline(K[0, 0], color="#2a2a2a", linestyle="--", linewidth=1, label="true fx")
    ax2.set_xlabel("images used")
    ax2.set_ylabel("estimated fx (px)")
    ax2.set_title("Focal length convergence")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration_vs_count.png", dpi=110)
    plt.close(fig)


def _distance_figure(intrinsics) -> None:
    markers = [
        render_aruco_marker(intrinsics, SIZE, 17, 0.08, np.array([0.06, -0.05, 0.03]), np.array([0.0, 0.0, d]))
        for d in [0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
    ]
    trials = ev.pose_vs_distance(markers, intrinsics, 0.08)
    dist = [t.true_distance * 100 for t in trials]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(dist, [t.translation_error * 1000 for t in trials], "o-", color="#4c72b0")
    ax1.set_xlabel("true distance (cm)")
    ax1.set_ylabel("translation error (mm)")
    ax1.set_title("Pose translation error vs distance")
    ax2.plot(dist, [t.rotation_error for t in trials], "o-", color="#d1495b")
    ax2.set_xlabel("true distance (cm)")
    ax2.set_ylabel("rotation error (deg)")
    ax2.set_title("Pose rotation error vs distance")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_vs_distance.png", dpi=110)
    plt.close(fig)


def _solver_figure(intrinsics) -> None:
    rendered = render_aruco_marker(intrinsics, SIZE, 17, 0.08, np.array([0.12, -0.16, 0.05]), np.array([0.0, 0.0, 0.5]))
    pose = PoseEstimator(intrinsics).estimate_aruco(rendered.image, 0.08)[0]
    trials = ev.solver_comparison(_marker_object_points(0.08), pose.image_points, intrinsics, rendered.rvec, rendered.tvec)
    rows = [[t.solver, f"{t.translation_error * 1000:.2f}", f"{t.rotation_error:.2f}",
             f"{t.reprojection_rms:.3f}", f"{t.solve_time_ms:.3f}"] for t in trials]
    fig, ax = plt.subplots(figsize=(8, 0.5 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["solver", "trans err (mm)", "rot err (deg)", "reproj (px)", "time (ms)"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("PnP solver comparison on one marker", pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "solver_comparison_table.png", dpi=110)
    plt.close(fig)


def _robustness_figure(intrinsics) -> None:
    rendered = render_aruco_marker(intrinsics, SIZE, 17, 0.08, np.array([0.1, -0.12, 0.05]), np.array([0.0, 0.0, 0.5]))
    degradations = [
        ("blur", 3, ev.blur(3)), ("blur", 9, ev.blur(9)), ("blur", 15, ev.blur(15)), ("blur", 25, ev.blur(25)),
        ("occlude", 0.1, ev.occlude(0.1)), ("occlude", 0.25, ev.occlude(0.25)), ("occlude", 0.5, ev.occlude(0.5)),
        ("dark", 0.3, ev.darken(0.3)), ("dark", 0.1, ev.darken(0.1)),
    ]
    trials = ev.robustness_sweep(rendered.image, intrinsics, 0.08, degradations)
    labels = [f"{t.condition}\n{t.level}" for t in trials]
    colors = ["#2e8b57" if t.detected else "#d1495b" for t in trials]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(trials)), [1 if t.detected else 0 for t in trials], color=colors)
    ax.set_xticks(range(len(trials)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["failed", "detected"])
    ax.set_title("Marker detection under degradation")
    fig.tight_layout()
    fig.savefig(FIGURES / "robustness.png", dpi=110)
    plt.close(fig)


def _smoothing_figure() -> None:
    rng = np.random.default_rng(3)
    true_t = np.array([0.02, -0.01, 0.5])
    true_r = np.array([0.2, -0.3, 0.1])
    true_matrix = rodrigues_to_matrix(true_r)
    smoother = PoseSmoother(translation_process_noise=0.5, translation_measurement_noise=5e-3, rotation_gain=0.3)
    raw_t, sm_t, raw_a, sm_a = [], [], [], []
    for _ in range(160):
        noisy_t = true_t + rng.normal(0, 0.006, 3)
        noisy_r = true_r + rng.normal(0, 0.06, 3)
        out = smoother.update(noisy_r, noisy_t, dt=1 / 30)
        raw_t.append(np.linalg.norm(noisy_t - true_t) * 1000)
        sm_t.append(np.linalg.norm(out.tvec - true_t) * 1000)
        raw_a.append(geodesic_angle(true_matrix, rodrigues_to_matrix(noisy_r)))
        sm_a.append(geodesic_angle(true_matrix, rodrigues_to_matrix(out.rvec)))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(raw_t, color="#c0c0c0", label="raw")
    a.plot(sm_t, color="#4c72b0", label="smoothed")
    a.set_xlabel("frame")
    a.set_ylabel("translation error (mm)")
    a.set_title("Translation jitter")
    a.legend(fontsize=8)
    b.plot(raw_a, color="#c0c0c0", label="raw")
    b.plot(sm_a, color="#d1495b", label="smoothed")
    b.set_xlabel("frame")
    b.set_ylabel("rotation error (deg)")
    b.set_title("Rotation jitter")
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "temporal_smoothing.png", dpi=110)
    plt.close(fig)


def _board_occlusion_figure(intrinsics) -> None:
    board = grid_board(3, 3, marker_length=0.04, marker_separation=0.01)
    estimator = PoseEstimator(intrinsics)
    rvec, tvec = np.array([0.15, -0.2, 0.05]), np.array([0.02, -0.01, 0.6])
    all_ids = board.all_ids()
    visible_counts, trans_errs, rot_errs = [], [], []
    for shown in range(9, 0, -1):
        rendered = render_marker_board(intrinsics, SIZE, board, rvec, tvec, only_ids=all_ids[:shown])
        result = estimator.estimate_board(rendered.image, board)
        if result is None:
            continue
        visible_counts.append(shown)
        trans_errs.append(np.linalg.norm(result.pose.translation - rendered.tvec) * 1000)
        rot_errs.append(geodesic_angle(rodrigues_to_matrix(rendered.rvec), result.pose.rotation_matrix))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(visible_counts, trans_errs, "o-", color="#4c72b0")
    a.invert_xaxis()
    a.set_xlabel("markers visible (of 9)")
    a.set_ylabel("translation error (mm)")
    a.set_title("Board pose holds under occlusion")
    b.plot(visible_counts, rot_errs, "o-", color="#d1495b")
    b.invert_xaxis()
    b.set_xlabel("markers visible (of 9)")
    b.set_ylabel("rotation error (deg)")
    b.set_title("Rotation stays bounded")
    fig.tight_layout()
    fig.savefig(FIGURES / "board_occlusion.png", dpi=110)
    plt.close(fig)


def _charuco_occlusion_figure(intrinsics) -> None:
    spec = CharucoSpec(5, 7, 0.03, 0.022)
    facing = rodrigues_to_matrix(np.array([np.pi, 0.0, 0.0])) @ rodrigues_to_matrix(np.array([0.1, -0.15, 0.05]))
    tvec = np.array([0.0, 0.0, 0.5]) - facing @ np.array([spec.width / 2, spec.height / 2, 0.0])
    clean = render_charuco(intrinsics, SIZE, spec, np.array([0.1, -0.15, 0.05]), tvec)
    fractions, counts, errors = [], [], []
    for occ in [0.0, 0.15, 0.3, 0.45, 0.6]:
        rendered = render_charuco(intrinsics, SIZE, spec, np.array([0.1, -0.15, 0.05]), tvec, occlude=occ)
        detection = detect_charuco(rendered.image, spec)
        solution = estimate_charuco_pose(detection, spec, intrinsics)
        if solution is None:
            continue
        fractions.append(occ * 100)
        counts.append(detection.count)
        errors.append(np.linalg.norm(solution.translation - clean.tvec) * 1000)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(fractions, counts, "o-", color="#4c72b0")
    a.set_xlabel("board occluded (%)")
    a.set_ylabel(f"corners detected (of {spec.inner_corners})")
    a.set_title("ChArUco degrades gracefully")
    b.plot(fractions, errors, "o-", color="#d1495b")
    b.set_xlabel("board occluded (%)")
    b.set_ylabel("translation error (mm)")
    b.set_title("Pose stays usable while corners remain")
    fig.tight_layout()
    fig.savefig(FIGURES / "charuco_occlusion.png", dpi=110)
    plt.close(fig)


def _stereo_figure() -> None:
    from campose.rotations import matrix_to_rodrigues, rodrigues_to_matrix

    R_true = rodrigues_to_matrix(np.array([0.0, -0.03, 0.0]))
    T_true = np.array([-0.10, 0.0, 0.0])
    left_cam = VirtualCamera(CameraIntrinsics.from_matrix(np.array([[900.0, 0, 320], [0, 900.0, 240], [0, 0, 1]]), np.zeros(5)), SIZE, SPEC)
    right_cam = VirtualCamera(CameraIntrinsics.from_matrix(np.array([[898.0, 0, 321], [0, 899.0, 240], [0, 0, 1]]), np.zeros(5)), SIZE, SPEC)
    cx = -(SPEC.columns - 1) * SPEC.square_size / 2
    cy = -(SPEC.rows - 1) * SPEC.square_size / 2
    rng = np.random.default_rng(2)
    calibrator = StereoCalibrator(SPEC)
    tries = 0
    while calibrator.pair_count < 14 and tries < 3000:
        tries += 1
        rvec_l = rng.uniform(-0.35, 0.35, 3)
        tvec_l = np.array([cx + 0.05 + rng.uniform(-0.04, 0.04), cy + rng.uniform(-0.04, 0.04), rng.uniform(0.6, 0.95)])
        rvec_r = matrix_to_rodrigues(R_true @ rodrigues_to_matrix(rvec_l))
        lb = left_cam.render(rvec_l, tvec_l)
        rb = right_cam.render(rvec_r, R_true @ tvec_l + T_true)
        if lb.visible and rb.visible:
            calibrator.add_pair(lb.image, rb.image)
    result = calibrator.calibrate()
    left_img = left_cam.render(np.array([0.1, -0.1, 0.02]), np.array([cx + 0.05, cy, 0.7])).image
    right_img = right_cam.render(matrix_to_rodrigues(R_true @ rodrigues_to_matrix(np.array([0.1, -0.1, 0.02]))), R_true @ np.array([cx + 0.05, cy, 0.7]) + T_true).image
    left_rect, right_rect, _ = rectify_pair(left_img, right_img, result)
    pair = np.hstack([cv2.cvtColor(left_rect, cv2.COLOR_BGR2RGB), cv2.cvtColor(right_rect, cv2.COLOR_BGR2RGB)])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(pair)
    for y in range(40, SIZE[1], 45):
        ax.axhline(y, color="#2e8b57", linewidth=0.7, alpha=0.8)
    ax.axvline(SIZE[0], color="white", linewidth=2)
    ax.set_title(f"Rectified stereo pair (baseline {result.baseline * 100:.1f} cm) — corners share scanlines")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES / "stereo_rectification.png", dpi=110)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    calibrator = _build_calibrator()
    result = calibrator.calibrate()
    print(result.summary())
    intrinsics = result.intrinsics
    _calibration_figures(calibrator, result)
    _count_figure(calibrator)
    _distance_figure(intrinsics)
    _solver_figure(intrinsics)
    _robustness_figure(intrinsics)
    _smoothing_figure()
    _board_occlusion_figure(intrinsics)
    _charuco_occlusion_figure(intrinsics)
    _stereo_figure()
    print(f"\nWrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
