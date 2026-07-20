from __future__ import annotations

import numpy as np

from campose import evaluation as ev
from campose.synthetic import render_aruco_marker

from _common import IMAGE_SIZE, ground_truth_intrinsics


def main() -> None:
    intrinsics = ground_truth_intrinsics()
    marker_size = 0.08
    rendered = render_aruco_marker(intrinsics, IMAGE_SIZE, 17, marker_size,
                                   np.array([0.1, -0.12, 0.05]), np.array([0.0, 0.0, 0.5]))
    degradations = [
        ("blur", 3, ev.blur(3)), ("blur", 9, ev.blur(9)), ("blur", 15, ev.blur(15)), ("blur", 25, ev.blur(25)),
        ("occlude", 0.1, ev.occlude(0.1)), ("occlude", 0.25, ev.occlude(0.25)), ("occlude", 0.5, ev.occlude(0.5)),
        ("dark", 0.3, ev.darken(0.3)), ("dark", 0.1, ev.darken(0.1)),
    ]
    trials = ev.robustness_sweep(rendered.image, intrinsics, marker_size, degradations)
    print(f"{'condition':>10}  {'level':>6}  {'detected':>8}  {'trans drift (mm)':>16}  {'rot drift (deg)':>15}")
    print("-" * 62)
    for trial in trials:
        drift_t = "-" if trial.translation_error is None else f"{trial.translation_error * 1000:.2f}"
        drift_r = "-" if trial.rotation_error is None else f"{trial.rotation_error:.2f}"
        print(f"{trial.condition:>10}  {trial.level:>6}  {str(trial.detected):>8}  {drift_t:>16}  {drift_r:>15}")


if __name__ == "__main__":
    main()
