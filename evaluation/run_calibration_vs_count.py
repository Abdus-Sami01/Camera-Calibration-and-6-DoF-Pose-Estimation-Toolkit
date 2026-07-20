from __future__ import annotations

from campose import evaluation as ev

from _common import GROUND_TRUTH_K, build_calibrator


def main() -> None:
    calibrator = build_calibrator(views=24)
    trials = ev.calibration_vs_count(calibrator, [4, 6, 8, 10, 13, 16, 20, 24])
    print(f"{'images':>7}  {'RMS (px)':>9}  {'fx':>8}  {'fx error':>9}")
    print("-" * 40)
    for trial in trials:
        print(f"{trial.image_count:7d}  {trial.overall_rms:9.4f}  {trial.fx:8.2f}  {abs(trial.fx - GROUND_TRUTH_K[0, 0]):9.2f}")


if __name__ == "__main__":
    main()
