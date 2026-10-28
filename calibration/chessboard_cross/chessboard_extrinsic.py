"""Chessboard extrinsic fitting and rig loading, independent of video measurement."""
from pathlib import Path

import numpy as np

from Src.calibration.chessboard_cross.chessboard_metric import (align_pairs, camera_arrays, fit_eight_point,
    fit_pnp, object_points, paired_views, read_json, validate_rig)

CONVENTION = "X_cs = R_cs_from_cf @ X_cf + t_cs_from_cf_mm"


def load_rig(path):
    path = Path(path).resolve()
    rig = read_json(path)
    if (rig.get("schema") != "metric_stereo_v1" or rig.get("units") != "mm"
            or rig.get("coordinate_convention") != CONVENTION
            or rig.get("method") not in ("pnp", "8point")):
        raise ValueError("需要棋盤流程產生的 metric_stereo_v1 / mm / CF→CS stereo.json。")
    R = np.asarray(rig["R_cs_from_cf"], float)
    t = np.asarray(rig["t_cs_from_cf_mm"], float)
    if (R.shape != (3, 3) or t.size != 3 or not np.isfinite(R).all()
            or not np.isfinite(t).all() or np.linalg.norm(t) <= 0
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-5)):
        raise ValueError("校正外參不是有效旋轉／毫米平移。")
    for name in ("cf", "cs"):
        camera = rig["intrinsics"][name]
        K, dist = camera_arrays(camera)
        size = np.asarray(camera["image_size"])
        if (K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0
                or not np.allclose(K[2], [0, 0, 1]) or not np.isfinite(dist).all()
                or dist.size not in (4, 5, 8, 12, 14)
                or size.shape != (2,) or not np.isfinite(size).all()
                or np.any(size <= 0) or not np.equal(size, size.astype(int)).all()):
            raise ValueError(f"{name.upper()} 內參／影像尺寸無效。")
    return rig


def fit_chessboard_rig(calibration, detections, method, max_pairs=40):
    """Fit only R/t with fixed intrinsics; no video, annotation or file writes."""
    if method not in ('pnp', '8point'):
        raise ValueError("method must be pnp or 8point")
    if max_pairs < 5:
        raise ValueError("Use >=5 pairs for training and held-out board checks")
    if (calibration.get('schema') != 'chessboard_intrinsics_v1'
            or calibration.get('units') != 'mm'
            or calibration['pattern'] != detections['pattern']):
        raise ValueError("Intrinsics/detection schema, units or pattern mismatch")
    object_points(tuple(calibration['pattern']), calibration['square_mm'])
    for camera in ('cf', 'cs'):
        if list(calibration['cameras'][camera]['image_size']) != list(detections['cameras'][camera]['image_size']):
            raise ValueError(f"{camera}: detection image size differs from intrinsics")
    cameras, pattern, square = calibration["cameras"], tuple(calibration["pattern"]), calibration["square_mm"]
    print("Resolving chessboard corner ordering and checking fixed camera geometry...", flush=True)
    pairs, rejected = align_pairs(paired_views(calibration, detections["cameras"], max_pairs), cameras, pattern, square)
    holdout = pairs[::5]
    training = [p for i, p in enumerate(pairs) if i % 5]
    if len(training) < 3 or not holdout:
        raise ValueError("Insufficient consistent training/held-out pairs")
    solver = fit_pnp if method == "pnp" else fit_eight_point
    print(f"Fitting {method}: {len(training)} pairs; held out {len(holdout)}", flush=True)
    R, t, details = solver(training, cameras, pattern, square)
    train_checks, warnings = validate_rig(training, cameras, R, t, pattern, square)
    held_checks, held_warnings = validate_rig(holdout, cameras, R, t, pattern, square)
    warnings += ["Held-out: " + w for w in held_warnings]
    for c in cameras:
        warnings += [c + ": " + w for w in cameras[c].get("warnings", [])]
    if method == "8point" and details["scale_relative_mad"] > .05:
        warnings.append("Board scale varies by >5% across poses.")
    result = {"schema": "metric_stereo_v1", "method": method, "units": "mm",
              "coordinate_convention": "X_cs = R_cs_from_cf @ X_cf + t_cs_from_cf_mm",
              "R_cs_from_cf": R, "t_cs_from_cf_mm": t, "baseline_mm": float(np.linalg.norm(t)),
              "intrinsics": cameras, "board": {"pattern": pattern, "square_mm": square},
              "fit": details, "training_pairs": [p["name"] for p in training],
              "rejected_pairs": rejected, "training_board_checks": train_checks,
              "held_out_board_checks": held_checks, "warnings": warnings,
              "held_out_note": "Held out from stereo pose/scale fit; may overlap intrinsic calibration photos. Not independent validation.",
              "corner_ordering": "PnP fixed-rig consensus resolves square-board symmetry; 8point pose and scale are then fitted independently.",
              "independently_validated_for_measurement": False}
    return result
