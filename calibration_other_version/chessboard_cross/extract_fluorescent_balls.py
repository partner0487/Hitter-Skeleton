"""Extract fluorescent pink ball candidates from one image or an image tree.

Only OpenCV/NumPy are needed. No intrinsics, PnP, Qt, camera SDK or paired
cf/cs folders. Candidate IDs are image-local, NOT physical ball1..ball8 IDs.
"""

# Support both python -m and direct file execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from Src.calibration.chessboard_cross.fluorescent_balls import DetectionConfig, detect_balls, read_image, write_image

ROOT = Path(__file__).resolve().parents[3]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "Db/Record/Calibrate_Picture_Auto",
                        help="Single image or folder searched recursively; no paired filenames required")
    parser.add_argument("--output", type=Path, help="New output directory; defaults to Db/Record/Ball_Extraction/<time>")
    parser.add_argument("--expected-count", type=int, default=8,
                        help="Count check only, never drops/adds candidates; 0 disables this check")
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                        help="Search region in original image pixels, applied to every input image")
    parser.add_argument("--h-min", type=int, default=135, help="Minimum hue (0..179)")
    parser.add_argument("--h-max", type=int, default=168, help="Maximum hue; lower than h-min wraps across zero")
    parser.add_argument("--s-min", type=int, default=105, help="Minimum saturation (0..255)")
    parser.add_argument("--v-min", type=int, default=50, help="Minimum brightness (0..255)")
    parser.add_argument("--min-radius-px", type=float, default=3.0, help="Minimum distance-transform peak radius")
    parser.add_argument("--min-area-px", type=float, default=20.0, help="Minimum component/segment pixel area")
    parser.add_argument("--max-area-px", type=float, default=2500.0, help="Maximum connected component pixel area")
    return parser


def collect_images(source):
    source = source.resolve()
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError("Unsupported image extension: {}".format(source.suffix))
        return source.parent, [source]
    if not source.is_dir():
        raise ValueError("Input does not exist: {}".format(source))
    files = sorted((path for path in source.rglob("*")
                    if path.is_file() and path.suffix.lower() in EXTENSIONS),
                   key=lambda path: path.as_posix().casefold())
    if not files:
        raise ValueError("No images found in {}".format(source))
    return source, files


def annotate(image, candidates, expected_count):
    canvas = image.copy()
    for candidate in candidates:
        center = tuple(np.rint(candidate["center_px"]).astype(int))
        radius = max(6, int(round(candidate["radius_px"])))
        cv2.circle(canvas, center, radius, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, center, (0, 255, 0), cv2.MARKER_CROSS, 5, 1)
        cv2.putText(canvas, "c{}".format(candidate["candidate_id"]), (center[0]+radius+2, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    message = "{} candidates (image-local IDs)".format(len(candidates))
    if expected_count:
        message += " | expected {}: {}".format(expected_count,
                                              "count matches" if len(candidates) == expected_count else "CHECK")
    cv2.putText(canvas, message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def run(args):
    config = DetectionConfig(args.h_min, args.h_max, args.s_min, args.v_min,
                             args.min_radius_px, args.min_area_px, args.max_area_px)
    config.validate()
    if args.expected_count < 0:
        raise ValueError("Expected count must be >= 0.")
    source = args.input.resolve()
    base, files = collect_images(source)
    output = (args.output or ROOT / "Db/Record/Ball_Extraction" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    # Prevent a later run from treating generated masks/overlays as source data.
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("Output must be outside the input image folder.")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1, "method": "HSV + opening + distance peaks + segment centroids",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "opencv_version": cv2.__version__, "input": str(source), "output": str(output),
        "detector": asdict(config), "roi_xywh": args.roi, "expected_count": args.expected_count or None,
        "coordinate_convention": "Original distorted image pixels, origin top-left; x right, y down",
        "candidate_id_convention": "1-based image-local IDs sorted by y then x; NOT physical ball identities",
        "count_check_note": "Matching the expected count does not validate physical identities or calibration geometry.",
        "frames": {},
    }
    for path in files:
        relative = path.relative_to(base)
        frame = {"source": str(path), "image_size": None, "candidates": [],
                 "count": 0, "count_matches_expected": None, "error": None, "artifacts": {}}
        try:
            image = read_image(path)
            frame["image_size"] = [image.shape[1], image.shape[0]]
            candidates, mask = detect_balls(image, config, args.roi)
            candidates = [dict(candidate, candidate_id=i+1) for i, candidate in enumerate(candidates)]
            frame.update(candidates=candidates, count=len(candidates),
                         count_matches_expected=(len(candidates) == args.expected_count
                                                 if args.expected_count else None))
            annotated = annotate(image, candidates, args.expected_count)
            visuals = {"mask": ("mask.png", mask),
                       "foreground": ("foreground.png", cv2.bitwise_and(image, image, mask=mask)),
                       "annotated": ("annotated.jpg", annotated)}
            if candidates:
                centers = np.asarray([c["center_px"] for c in candidates])
                x0, y0 = np.maximum(np.floor(centers.min(axis=0)).astype(int)-40, 0)
                x1, y1 = np.minimum(np.ceil(centers.max(axis=0)).astype(int)+60,
                                    [image.shape[1], image.shape[0]])
                detail = cv2.resize(annotated[y0:y1, x0:x1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                visuals["detail"] = ("detail.jpg", detail)
            for name, (suffix, visual) in visuals.items():
                destination = relative.parent / (relative.name+"."+suffix)
                write_image(output / destination, visual)
                frame["artifacts"][name] = destination.as_posix()
        except (ValueError, OSError, cv2.error) as exc:
            frame["error"] = str(exc)
        report["frames"][relative.as_posix()] = frame
        save_json(output / relative.parent / (relative.name+".centers.json"), frame)
        status = frame["error"] or ("count mismatch" if frame["count_matches_expected"] is False else "extracted")
        print("{}: {} candidates, {}".format(relative.as_posix(), frame["count"], status), flush=True)
    report["summary"] = {
        "total_images": len(files),
        "processed_images": sum(frame["error"] is None for frame in report["frames"].values()),
        "count_mismatch_images": sum(frame["count_matches_expected"] is False
                                     for frame in report["frames"].values() if frame["error"] is None),
        "failed_images": sum(frame["error"] is not None for frame in report["frames"].values()),
    }
    report["status"] = "complete" if report["summary"]["failed_images"] == 0 else "partial_failure"
    save_json(output / "detections.json", report)
    print("Output: {}".format(output), flush=True)
    return output, report


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _, report = run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, "Extraction failed: {}\n".format(exc))
    return 2 if report["summary"]["failed_images"] else 0


if __name__ == "__main__":
    sys.exit(main())
