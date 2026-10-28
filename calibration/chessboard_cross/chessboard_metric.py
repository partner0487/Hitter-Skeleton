"""Offline metric stereo geometry. All image coordinates are original pixels.

Camera convention: X_cs = R @ X_cf + t. All exported translations are mm.
No camera SDK, pose network, or Qt dependency is required.
"""
from pathlib import Path
import json
import re

import cv2
import numpy as np
from scipy.optimize import least_squares

from Src.paths import ROOT_DIR as ROOT
DEFAULT_IMAGES = ROOT / "Db/Record/Calibrate_Picture_Chessboard"
DEFAULT_CALIBRATION = ROOT / "Db/Calibration/chessboard"
DEFAULT_VIDEO = ROOT / "Db/Record/20261008_Pitcher02/20261008_1050_P03"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False, default=convert), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def object_points(pattern=(7, 7), square_mm=100.):
    if min(pattern) < 3 or not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("Use at least 3x3 inner corners and a positive square size in mm.")
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2) * square_mm
    return obj


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def grid_variants(points, pattern):
    """Square chessboards have eight possible index orientations."""
    grid = np.asarray(points).reshape(pattern[1], pattern[0], 2)
    variants = [grid, grid[:, ::-1], grid[::-1], grid[::-1, ::-1]]
    if pattern[0] == pattern[1]:
        transposed = grid.transpose(1, 0, 2)
        variants += [transposed, transposed[:, ::-1], transposed[::-1], transposed[::-1, ::-1]]
    return [np.ascontiguousarray(x.reshape(-1, 2), dtype=np.float64) for x in variants]


def canonical_corners(points, pattern):
    # Useful initial orientation for upright boards; stereo ordering is checked
    # independently using all board poses, not assumed from this image rule.
    def score(p):
        g = p.reshape(pattern[1], pattern[0], 2)
        a, b = g[0, -1] - g[0, 0], g[-1, 0] - g[0, 0]
        return a[0] / np.linalg.norm(a) + b[1] / np.linalg.norm(b)
    return max(grid_variants(points, pattern), key=score).astype(np.float32)


def detect_folder(folder, pattern, output, preview_every=10):
    files = sorted((p for p in Path(folder).iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}), key=natural_key)
    if not files:
        raise ValueError(f"No images in {folder}")
    views, failed, size = [], [], None
    for index, path in enumerate(files):
        image = read_image(path)
        current_size = (image.shape[1], image.shape[0])
        if size is not None and current_size != size:
            raise ValueError(f"Mixed image sizes in {folder}: {path.name}")
        size = current_size
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # The third positional argument is the output corners, not flags.
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if not ok:
            flags |= cv2.CALIB_CB_EXHAUSTIVE
            ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if not ok:
            failed.append(path.name)
        else:
            corners = canonical_corners(corners, pattern)
            views.append({"name": path.name, "path": str(path.resolve()), "corners": corners.tolist(),
                          "detector": "findChessboardCornersSB", "detection_flags": int(flags)})
            if preview_every and index % preview_every == 0:
                cv2.drawChessboardCorners(image, pattern, corners.reshape(-1, 1, 2), True)
                for k in (0, pattern[0] - 1, len(corners) - 1):
                    cv2.putText(image, str(k), tuple(corners[k].astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
                save_image(Path(output) / path.name, image)
        if (index + 1) % 10 == 0 or index + 1 == len(files):
            print(f"{Path(folder).name}: {index + 1}/{len(files)}, detected {len(views)}", flush=True)
    return {"image_size": size, "views": views, "failed": failed, "total_images": len(files)}


def calibrate_intrinsics(detections, pattern, square_mm, max_views=60, fixed_aspect_ratio=None):
    views = detections["views"]
    if len(views) < 6:
        raise ValueError(f"Only {len(views)} detected views; need at least six varied board poses.")
    # Uniform sampling avoids cubic runtime on hundreds of near-duplicate frames.
    indices = np.unique(np.linspace(0, len(views) - 1, min(max_views, len(views))).astype(int))
    chosen = [views[i] for i in indices]
    obj = object_points(pattern, square_mm)
    flags = cv2.CALIB_FIX_K3  # k3 is poorly constrained by a small central target.
    if fixed_aspect_ratio is not None:
        if not np.isfinite(fixed_aspect_ratio) or fixed_aspect_ratio <= 0:
            raise ValueError("Fixed fx/fy ratio must be positive and physically justified.")
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    def fit(selected):
        initial = None
        if fixed_aspect_ratio is not None:
            initial = cv2.initCameraMatrix2D([obj] * len(selected),
                [np.array(v["corners"], np.float32) for v in selected],
                tuple(detections["image_size"]), aspectRatio=fixed_aspect_ratio)
        return cv2.calibrateCameraExtended(
            [obj] * len(selected), [np.array(v["corners"], np.float32) for v in selected],
            tuple(detections["image_size"]), initial, None, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8))
    result = fit(chosen)
    errors = result[-1].ravel()
    limit = max(1., float(np.median(errors) + 3 * 1.4826 * np.median(abs(errors - np.median(errors)))))
    kept = [v for v, e in zip(chosen, errors) if e <= limit]
    rejected = [v["name"] for v, e in zip(chosen, errors) if e > limit]
    if len(kept) >= 6 and rejected:
        result = fit(kept)
    else:
        kept, rejected = chosen, []
    rms, K, dist, rvecs, tvecs, std, _, per_view = result
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid intrinsic solution.")
    corners = np.concatenate([np.array(v["corners"]) for v in kept])
    hull = cv2.convexHull(corners.astype(np.float32))
    coverage = cv2.contourArea(hull) / np.prod(detections["image_size"])
    warnings = []
    if coverage < .2:
        warnings.append("Board covers <20% of the image; distortion outside this region is weakly constrained.")
    if max(std[0, 0] / K[0, 0], std[1, 0] / K[1, 1]) > .05:
        warnings.append("Estimated focal-length uncertainty exceeds 5%; more varied tilted views are needed.")
    if rms > 1:
        warnings.append("Intrinsic reprojection RMS exceeds 1 pixel.")
    if abs(K[0, 0] / K[1, 1] - 1) > .03:
        warnings.append("fx/fy differs by >3%; check pixel aspect ratio and calibration conditioning.")
    return {"intrinsic_matrix": K, "distortion_coefficients": dist.ravel(),
            "image_size": detections["image_size"], "rms_px": rms,
            "intrinsic_stddev": std.ravel(), "coverage_fraction": coverage,
            "flags": int(flags), "fixed_aspect_ratio": fixed_aspect_ratio,
            "used_images": [v["name"] for v in kept],
            "rejected_images": rejected, "per_view_rms_px": per_view.ravel(),
            "warnings": warnings}


def camera_arrays(camera):
    return np.array(camera["intrinsic_matrix"], float), np.array(camera["distortion_coefficients"], float)


def undistort(pixels, camera, pixel_units=False):
    K, dist = camera_arrays(camera)
    return cv2.undistortPoints(np.asarray(pixels, float).reshape(-1, 1, 2), K, dist,
                               P=K if pixel_units else None).reshape(-1, 2)


def solve_board_pose(pixels, camera, obj):
    K, dist = camera_arrays(camera)
    ok, rvec, tvec = cv2.solvePnP(obj, np.asarray(pixels, float), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Board PnP failed.")
    R = cv2.Rodrigues(rvec)[0]
    if np.min((obj @ R.T + tvec.reshape(3))[:, 2]) <= 0:
        raise ValueError("Board PnP has negative depth.")
    projected = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
    rms = np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1)))
    return R, tvec.reshape(3), float(rms)


def paired_views(calibration, detections, max_pairs=40):
    maps = {c: {v["name"]: v for v in detections[c]["views"]} for c in ("cf", "cs")}
    names = sorted(maps["cf"].keys() & maps["cs"].keys(), key=natural_key)
    if len(names) < 3:
        raise ValueError("Need >=3 same-name synchronized chessboard pairs at different board positions.")
    names = [names[i] for i in np.unique(np.linspace(0, len(names)-1, min(max_pairs, len(names))).astype(int))]
    return [{"name": n, "cf": np.array(maps["cf"][n]["corners"], float),
             "cs": np.array(maps["cs"][n]["corners"], float)} for n in names]


def align_pairs(pairs, cameras, pattern, square_mm):
    """Resolve grid-index ambiguity using fixed-rig consistency across poses.

    PnP is used only to identify physical corner correspondences in the 8-point
    route. Neither its R nor t nor metric scale is used by that route afterwards.
    """
    obj = object_points(pattern, square_mm)
    candidates = []
    for pair in pairs:
        R1, t1, e1 = solve_board_pose(pair["cf"], cameras["cf"], obj)
        options = []
        for i, pixels in enumerate(grid_variants(pair["cs"], pattern)):
            R2, t2, e2 = solve_board_pose(pixels, cameras["cs"], obj)
            R = R2 @ R1.T
            options.append({"R": R, "t": t2 - R @ t1, "order": i,
                            "cs": pixels, "rms": max(e1, e2)})
        candidates.append(options)
    def distance(a, b):
        # dimensionless consistency score: roughly 3 degrees or 100 mm.
        return np.linalg.norm(a["R"] - b["R"]) / .075 + np.linalg.norm(a["t"] - b["t"]) / 100.
    seeds = [o for options in candidates[::max(1, len(candidates)//10)] for o in options]
    seed = min(seeds, key=lambda s: np.median([min(distance(s, o) for o in opts) for opts in candidates]))
    selected = [min(opts, key=lambda o: distance(seed, o)) for opts in candidates]
    scores = np.array([distance(seed, o) for o in selected])
    limit = max(3., float(np.median(scores) + 3 * 1.4826 * np.median(abs(scores - np.median(scores)))))
    kept, rejected = [], []
    for p, o, score in zip(pairs, selected, scores):
        if score > limit or o["rms"] > 2.5:
            rejected.append(p["name"])
        else:
            kept.append({**p, "cs": o["cs"], "cs_order": o["order"],
                         "pnp_R": o["R"], "pnp_t_mm": o["t"], "pnp_rms_px": o["rms"]})
    if len(kept) < 3:
        raise ValueError("Too few consistent stereo pairs: check synchronization, inner corners, camera settings.")
    return kept, rejected


def triangulate(p1, p2, cameras, R, t):
    a, b = undistort(p1, cameras["cf"]), undistort(p2, cameras["cs"])
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Matching finite Nx2 image points are required.")
    P1, P2 = np.c_[np.eye(3), np.zeros(3)], np.c_[R, np.asarray(t).reshape(3)]
    homogeneous = cv2.triangulatePoints(P1, P2, a.T, b.T)
    if np.any(abs(homogeneous[3]) < 1e-12):
        raise ValueError("Triangulation at infinity (insufficient parallax).")
    xyz = (homogeneous[:3] / homogeneous[3]).T
    positive = (xyz[:, 2] > 0) & ((xyz @ R.T + np.asarray(t).reshape(3))[:, 2] > 0)
    errors = []
    for camera, rotation, trans, pixels in [(cameras["cf"], np.eye(3), np.zeros(3), p1),
                                            (cameras["cs"], R, t, p2)]:
        K, dist = camera_arrays(camera)
        projected = cv2.projectPoints(xyz, cv2.Rodrigues(rotation)[0], np.asarray(trans, float), K, dist)[0].reshape(-1, 2)
        errors.append(np.linalg.norm(projected - pixels, axis=1))
    rays1 = np.c_[a, np.ones(len(a))]
    rays2 = np.c_[b, np.ones(len(b))] @ R
    cosine = np.sum(rays1 * rays2, axis=1) / (np.linalg.norm(rays1, axis=1) * np.linalg.norm(rays2, axis=1))
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return xyz, positive, np.maximum(*errors), angles


def board_edges(xyz, pattern):
    grid = xyz.reshape(pattern[1], pattern[0], 3)
    return np.r_[np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
                 np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel()]


def fit_pnp(pairs, cameras, pattern, square_mm):
    # Both cameras see a common board pose. Refine their fixed relative pose
    # across boards while K and distortion remain fixed.
    obj = object_points(pattern, square_mm)
    K1, d1 = camera_arrays(cameras["cf"])
    K2, d2 = camera_arrays(cameras["cs"])
    result = cv2.stereoCalibrate([obj] * len(pairs),
        [p["cf"].astype(np.float32) for p in pairs], [p["cs"].astype(np.float32) for p in pairs],
        K1, d1, K2, d2, tuple(cameras["cf"]["image_size"]),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8))
    rms, _, _, _, _, R, t, E, F = result
    return R, t.reshape(3), {"stereo_rms_px": rms, "F_undistorted_pixels": F,
                            "E": E, "scale_source": "100% chessboard object coordinates (mm)",
                            "solver": "per-pair solvePnP + fixed-intrinsic joint stereo refinement"}


def estimate_eight_point(p1, p2, cameras, threshold_px=1.5):
    """RANSAC selects matches; final F is explicitly the normalized 8-point fit."""
    a, b = undistort(p1, cameras["cf"], True), undistort(p2, cameras["cs"], True)
    if len(a) < 8 or len(a) != len(b):
        raise ValueError("8-point estimation needs at least eight paired points.")
    cv2.setRNGSeed(20261010)
    _, mask = cv2.findFundamentalMat(a, b, cv2.FM_RANSAC, threshold_px, .999, 10000)
    if mask is None or np.count_nonzero(mask) < 8:
        raise ValueError("Cannot find eight epipolar inliers.")
    good = mask.ravel().astype(bool)
    # One board pose is planar and cannot uniquely determine F.
    H, hm = cv2.findHomography(a[good], b[good], cv2.RANSAC, threshold_px)
    homography_fraction = float(np.mean(hm)) if hm is not None else 0.
    if homography_fraction > .90:
        raise ValueError("Planar/near-planar 8-point degeneracy: move/tilt the board to different depths.")
    F, _ = cv2.findFundamentalMat(a[good], b[good], cv2.FM_8POINT)
    if F is None or F.shape != (3, 3):
        raise ValueError("8-point fit failed.")
    K1, _ = camera_arrays(cameras["cf"])
    K2, _ = camera_arrays(cameras["cs"])
    Eraw = K2.T @ F @ K1
    u, s, vt = np.linalg.svd(Eraw)
    E = u @ np.diag([np.mean(s[:2]), np.mean(s[:2]), 0.]) @ vt
    # Explicit decomposition avoids recoverPose's default 50-baseline depth cutoff.
    Ra, Rb, t = cv2.decomposeEssentialMat(E)
    choices = []
    for R in (Ra, Rb):
        for sign in (1, -1):
            trans = sign * t.reshape(3)
            xyz, positive, errors, angles = triangulate(p1, p2, cameras, R, trans)
            choices.append((int(np.sum(positive & good)), R, trans))
    count, R, t = max(choices, key=lambda x: x[0])
    if count < .9 * np.count_nonzero(good):
        raise ValueError("Essential matrix has poor cheirality; check pairing and intrinsics.")
    # Projecting a noisy F to the essential manifold can move its epipolar lines
    # substantially. Refine only R and unit translation using image matches;
    # this uses neither board dimensions nor any PnP pose.
    ah, bh = np.c_[a[good], np.ones(good.sum())], np.c_[b[good], np.ones(good.sum())]
    inv1, inv2 = np.linalg.inv(K1), np.linalg.inv(K2)
    def matrices(params):
        rot = cv2.Rodrigues(params[:3])[0]
        trans = params[3:] / np.linalg.norm(params[3:])
        x, y, z = trans
        essential = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]]) @ rot
        return rot, trans, essential, inv2.T @ essential @ inv1
    def residual(params):
        f = matrices(params)[3]
        l2, l1 = ah @ f.T, bh @ f
        numerator = np.sum(bh * l2, axis=1)
        return numerator / np.sqrt(np.maximum(np.sum(l1[:, :2]**2 + l2[:, :2]**2, axis=1), 1e-24))
    initial = np.r_[cv2.Rodrigues(R)[0].ravel(), t]
    optimized = least_squares(residual, initial, loss="soft_l1", f_scale=1., max_nfev=500)
    if not optimized.success:
        raise ValueError("8-point essential refinement did not converge.")
    R, t, E, F_refined = matrices(optimized.x)
    _, positive, _, _ = triangulate(p1, p2, cameras, R, t)
    if np.mean(positive[good]) < .9:
        raise ValueError("Essential refinement failed positive-depth validation.")
    return R, t, good, {"F_8point_undistorted_pixels": F, "F_undistorted_pixels": F_refined, "E": E,
                        "essential_raw_singular_values": s, "inlier_count": int(good.sum()),
                        "match_count": len(good), "homography_inlier_fraction": homography_fraction,
                        "refined_sampson_rms_px": float(np.sqrt(np.mean(residual(optimized.x)**2))),
                        "solver": "RANSAC inliers -> FM_8POINT -> E rank constraint -> cheirality -> robust epipolar refinement"}


def fit_eight_point(pairs, cameras, pattern, square_mm):
    p1, p2 = np.concatenate([p["cf"] for p in pairs]), np.concatenate([p["cs"] for p in pairs])
    R, t, good, details = estimate_eight_point(p1, p2, cameras)
    xyz, positive, errors, angles = triangulate(p1, p2, cameras, R, t)
    edges, pair_scales = [], []
    n = pattern[0] * pattern[1]
    for i in range(len(pairs)):
        valid = (good[i*n:(i+1)*n] & positive[i*n:(i+1)*n]
                 & (errors[i*n:(i+1)*n] < 3) & (angles[i*n:(i+1)*n] >= 1))
        grid = valid.reshape(pattern[1], pattern[0])
        edge_valid = np.r_[(grid[1:] & grid[:-1]).ravel(), (grid[:, 1:] & grid[:, :-1]).ravel()]
        lengths = board_edges(xyz[i*n:(i+1)*n], pattern)[edge_valid]
        if len(lengths) >= 20:
            edges.extend(lengths)
            pair_scales.append(square_mm / float(np.median(lengths)))
    if len(pair_scales) < 3:
        raise ValueError("Need three well-triangulated boards to recover a defensible metric scale.")
    scale = square_mm / float(np.median(edges))
    details.update({"mm_per_baseline_unit": scale, "scale_source": "median triangulated adjacent chessboard edges",
                    "per_pair_scale_mm": pair_scales,
                    "scale_relative_mad": float(np.median(abs(np.array(pair_scales) - scale)) / scale)})
    return R, t * scale, details


def validate_rig(pairs, cameras, R, t, pattern, square_mm):
    records = []
    for p in pairs:
        xyz, positive, errors, angles = triangulate(p["cf"], p["cs"], cameras, R, t)
        lengths = board_edges(xyz, pattern)
        records.append({"name": p["name"], "cs_order": p["cs_order"],
                        "median_square_mm": float(np.median(lengths)),
                        "square_rmse_mm": float(np.sqrt(np.mean((lengths - square_mm)**2))),
                        "max_reprojection_px": float(max(errors)),
                        "positive_depth_fraction": float(np.mean(positive)),
                        "min_ray_angle_deg": float(min(angles))})
    warnings = []
    if np.median([r["square_rmse_mm"] for r in records]) > 5:
        warnings.append("Median board edge RMSE exceeds 5 mm; do not treat stride as validated.")
    if np.median([r["max_reprojection_px"] for r in records]) > 3:
        warnings.append("Stereo reprojection exceeds 3 px; check synchronization and lens settings.")
    if any(r["positive_depth_fraction"] < .95 for r in records):
        warnings.append("Some calibration corners have negative depth.")
    if any(r["min_ray_angle_deg"] < 1 for r in records):
        warnings.append("Some calibration corners have <1 degree ray angle; depth is weakly constrained.")
    return records, warnings
