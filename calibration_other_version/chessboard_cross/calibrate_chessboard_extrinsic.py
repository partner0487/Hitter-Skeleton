"""Calibrate chessboard extrinsics from saved intrinsics, without measurement videos."""

# Support both python -m and direct file execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path

import cv2
import numpy as np

from Src.calibration.chessboard_cross.chessboard_metric import (DEFAULT_CALIBRATION, detect_folder,
    read_json, write_json)
from Src.calibration.chessboard_cross.chessboard_extrinsic import fit_chessboard_rig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION,
                        help='Directory containing intrinsics.json and detections.json')
    parser.add_argument('--images', type=Path,
                        help='Optional separate synchronized cf/cs photo folders; same board and camera settings as intrinsics')
    parser.add_argument('--output', type=Path,
                        help='Output root; writes pnp/stereo.json and/or 8point/stereo.json (default: calibration)')
    parser.add_argument('--method', choices=('pnp', '8point', 'both'), default='both')
    parser.add_argument('--max-pairs', type=int, default=40)
    parser.add_argument('--preview-every', type=int, default=10,
                        help='Corner preview interval when using --images; 0 disables previews')
    args = parser.parse_args(argv)
    if args.max_pairs < 5 or args.preview_every < 0:
        parser.error('max-pairs must be >=5; preview-every must be >=0')
    cv2.setNumThreads(4)
    calibration = read_json(args.calibration / 'intrinsics.json')
    if calibration.get('schema') != 'chessboard_intrinsics_v1':
        raise ValueError('Run calibrate_chessboard.py first to create chessboard intrinsics.json')
    output = args.output or args.calibration
    if args.images:
        detections = {'pattern': calibration['pattern'], 'input': str(args.images.resolve()), 'cameras': {}}
        for camera in ('cf', 'cs'):
            detections['cameras'][camera] = detect_folder(
                args.images / camera, tuple(calibration['pattern']),
                output / 'extrinsic_corners' / camera, args.preview_every)
        write_json(output / 'extrinsic_detections.json', detections)
    else:
        detections = read_json(args.calibration / 'detections.json')
        if detections['input'] != calibration['input']:
            raise ValueError('Cached detection source differs from intrinsics; use --images for separate photos')

    methods = ('pnp', '8point') if args.method == 'both' else (args.method,)
    for method in methods:
        result = fit_chessboard_rig(calibration, detections, method, args.max_pairs)
        result['source_intrinsics'] = str((args.calibration / 'intrinsics.json').resolve())
        result['source_images'] = detections['input']
        folder = output / method
        write_json(folder / 'stereo.json', result)
        checks = result['held_out_board_checks']
        summary = [f'Method: {method}', f"Baseline: {result['baseline_mm']:.2f} mm",
                   f"Training pairs: {len(result['training_pairs'])}; held out: {len(checks)}; rejected: {len(result['rejected_pairs'])}",
                   f"Held-out median square edge RMSE: {np.median([r['square_rmse_mm'] for r in checks]):.3f} mm",
                   'Calibration only; no measurement videos or annotations were used.',
                   'Independent physical-distance validation is still required.']
        summary += ['REVIEW: ' + w for w in result['warnings']]
        (folder / 'calibration_summary.txt').write_text('\n'.join(summary) + '\n', encoding='utf-8')
        print('\n'.join(summary))
        print(f"Saved: {folder / 'stereo.json'}")


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, cv2.error) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc
