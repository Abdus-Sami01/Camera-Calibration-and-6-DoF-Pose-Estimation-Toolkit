from __future__ import annotations

import numpy as np

from campose import evaluation as ev
from campose.pose_estimator import PoseEstimator, _marker_object_points
from campose.synthetic import render_aruco_marker

from _common import IMAGE_SIZE, ground_truth_intrinsics


def main() -> None:
    intrinsics = ground_truth_intrinsics()
    marker_size = 0.08
    rendered = render_aruco_marker(intrinsics, IMAGE_SIZE, 17, marker_size,
                                   np.array([0.12, -0.16, 0.05]), np.array([0.0, 0.0, 0.5]))
    pose = PoseEstimator(intrinsics).estimate_aruco(rendered.image, marker_size)[0]
    trials = ev.solver_comparison(_marker_object_points(marker_size), pose.image_points,
                                  intrinsics, rendered.rvec, rendered.tvec)
    print(f"{'solver':>13}  {'trans err (mm)':>14}  {'rot err (deg)':>13}  {'reproj (px)':>11}  {'time (ms)':>9}")
    print("-" * 68)
    for trial in trials:
        print(f"{trial.solver:>13}  {trial.translation_error * 1000:14.2f}  {trial.rotation_error:13.2f}  "
              f"{trial.reprojection_rms:11.3f}  {trial.solve_time_ms:9.3f}")


if __name__ == "__main__":
    main()
