from __future__ import annotations

import argparse
from pathlib import Path

from .board import CheckerboardSpec


def _add_calibrate(subparsers) -> None:
    parser = subparsers.add_parser("calibrate", help="Estimate intrinsics from a folder of board images")
    parser.add_argument("--images", required=True, help="Folder of calibration images")
    parser.add_argument("--board-size", required=True, help="Inner corners, e.g. 9x6")
    parser.add_argument("--square-size", type=float, required=True, help="Square size in your chosen unit")
    parser.add_argument("--output", default="calibration.json", help="Where to write the result")
    parser.add_argument("--bootstrap", type=int, default=0, help="Bootstrap trials for parameter stability")
    parser.set_defaults(func=_run_calibrate)


def _add_estimate(subparsers) -> None:
    parser = subparsers.add_parser("estimate", help="Estimate object pose from one image")
    parser.add_argument("--calibration", required=True, help="Calibration file from 'calibrate'")
    parser.add_argument("--image", required=True, help="Image containing the target")
    parser.add_argument("--target", default="aruco", choices=["aruco"], help="Target type")
    parser.add_argument("--marker-size", type=float, required=True, help="Marker side length")
    parser.set_defaults(func=_run_estimate)


def _add_live(subparsers) -> None:
    parser = subparsers.add_parser("live", help="Real-time pose overlay from a webcam")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--target", default="aruco", choices=["aruco"])
    parser.add_argument("--marker-size", type=float, required=True)
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--smooth", action="store_true", help="Apply temporal pose smoothing")
    parser.set_defaults(func=_run_live)


def _run_calibrate(args) -> int:
    from .calibrator import CameraCalibrator
    from . import io

    spec = CheckerboardSpec.parse(args.board_size, args.square_size)
    calibrator = CameraCalibrator(spec)
    found = calibrator.add_images(args.images)
    print(f"Detected the board in {found} image(s).")
    print()
    print(calibrator.assess_capture().summary())
    print()
    result = calibrator.calibrate()
    print(result.summary())
    if args.bootstrap:
        confidence = calibrator.bootstrap_confidence(trials=args.bootstrap)
        print("\nBootstrap parameter stability (std):")
        for name, value in confidence["std"].items():
            print(f"  {name:>3s}: {value:.4f}")
    saved = io.save(result, args.output)
    print(f"\nSaved calibration to {saved}")
    return 0


def _run_estimate(args) -> int:
    import cv2

    from .pose_estimator import PoseEstimator

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    estimator = PoseEstimator.from_calibration(args.calibration)
    poses = estimator.estimate_aruco(image, args.marker_size)
    if not poses:
        print("No markers detected.")
        return 1
    for pose in poses:
        t = pose.translation
        euler = pose.rotation_euler
        print(
            f"marker {pose.identifier}: "
            f"t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) "
            f"euler=({euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f})deg "
            f"reproj={pose.reprojection_error:.3f}px"
        )
    return 0


def _run_live(args) -> int:
    from .live import run_live

    return run_live(args.calibration, args.marker_size, camera_index=args.camera, smooth=args.smooth)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="campose", description="Camera calibration and 6-DoF pose estimation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_calibrate(subparsers)
    _add_estimate(subparsers)
    _add_live(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
