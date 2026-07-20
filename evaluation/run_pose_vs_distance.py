from __future__ import annotations

import numpy as np

from campose import evaluation as ev
from campose.synthetic import render_aruco_marker

from _common import IMAGE_SIZE, ground_truth_intrinsics


def main() -> None:
    intrinsics = ground_truth_intrinsics()
    marker_size = 0.08
    markers = [
        render_aruco_marker(intrinsics, IMAGE_SIZE, 17, marker_size,
                            np.array([0.06, -0.05, 0.03]), np.array([0.0, 0.0, d]))
        for d in [0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2]
    ]
    trials = ev.pose_vs_distance(markers, intrinsics, marker_size)
    print(f"{'true (cm)':>9}  {'est (cm)':>9}  {'trans err (mm)':>14}  {'rot err (deg)':>13}")
    print("-" * 52)
    for trial in trials:
        print(f"{trial.true_distance * 100:9.1f}  {trial.estimated_distance * 100:9.1f}  "
              f"{trial.translation_error * 1000:14.2f}  {trial.rotation_error:13.2f}")


if __name__ == "__main__":
    main()
