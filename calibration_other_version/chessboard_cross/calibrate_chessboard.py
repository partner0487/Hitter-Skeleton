"""Calibrate cf/cs intrinsics only; run calibrate_chessboard_extrinsic.py for extrinsics."""

# Support both python -m and direct file execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path
import cv2
from Src.calibration.chessboard_cross.chessboard_metric import (DEFAULT_IMAGES, DEFAULT_CALIBRATION,
    calibrate_intrinsics, detect_folder, object_points, read_json, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--pattern", nargs=2, type=int, default=(7, 7), metavar=("COLS", "ROWS"))
    parser.add_argument("--square-mm", type=float, default=100.)
    parser.add_argument("--max-views", type=int, default=60)
    parser.add_argument("--fix-aspect-ratio", type=float, help="Optional known fx/fy ratio, e.g. 1 for confirmed square pixels")
    parser.add_argument("--preview-every", type=int, default=10)
    parser.add_argument("--reuse-detections", action="store_true", help="Reuse saved corner detections (same images/pattern required)")
    args = parser.parse_args()
    if args.max_views < 6 or args.preview_every < 0:
        parser.error("max-views must be >=6; preview-every must be >=0")
    object_points(args.pattern, args.square_mm)
    cv2.setNumThreads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "detections.json"
    if args.reuse_detections:
        data = read_json(cache)
        if data["pattern"] != list(args.pattern) or data["input"] != str(args.input.resolve()):
            parser.error("Cached input/pattern differs; rerun without --reuse-detections")
    else:
        data = {"pattern": list(args.pattern), "input": str(args.input.resolve()), "cameras": {}}
        for camera in ("cf", "cs"):
            data["cameras"][camera] = detect_folder(args.input / camera, tuple(args.pattern),
                args.output / "corners" / camera, args.preview_every)
            write_json(cache, data)
    result = {"schema": "chessboard_intrinsics_v1", "units": "mm", "opencv_version": cv2.__version__,
              "pattern": list(args.pattern), "square_mm": args.square_mm,
              "input": str(args.input.resolve()), "cameras": {},
              "independently_validated_for_measurement": False}
    for camera in ("cf", "cs"):
        print(f"Calibrating {camera}...", flush=True)
        result["cameras"][camera] = calibrate_intrinsics(data["cameras"][camera], tuple(args.pattern), args.square_mm, args.max_views, args.fix_aspect_ratio)
        value = result["cameras"][camera]
        print(f"{camera}: RMS={value['rms_px']:.4f} px, views={len(value['used_images'])}, fx={value['intrinsic_matrix'][0,0]:.2f}", flush=True)
        for warning in value["warnings"]:
            print(f"  REVIEW: {warning}")
        write_json(args.output / "intrinsics.json", result)
    print(f"Saved: {args.output / 'intrinsics.json'}")
    print('Next: python Src/calibration/chessboard_cross/calibrate_chessboard_extrinsic.py --calibration <this output directory>')


if __name__ == "__main__":
    main()
