"""Validation and storage for the guided calibration tool (no camera/Qt imports)."""
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime

import cv2
import numpy as np


MIN_PAIRS = 4
POINTS_PER_IMAGE = 6


def validate_roles(front, side):
    if not front or not side or front == side:
        raise ValueError("請指定兩台不同的正面、側面相機。")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", sn) for sn in (front, side)):
        raise ValueError("相機序號只能包含英數字、底線或連字號。")


def validate_intrinsic(entry):
    K = np.asarray(entry["intrinsic_matrix"], dtype=float)
    dist = np.asarray(entry["distortion_coefficients"], dtype=float)
    size = np.asarray(entry["image_size"], dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all()
            or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])
            or dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all()
            or size.shape != (2,) or not np.isfinite(size).all()
            or np.any(size <= 0) or not np.equal(size, size.astype(int)).all()):
        raise ValueError("內參資料無效，請重新校正。")


def match_image_pairs(folder, minimum=MIN_PAIRS):
    """Only pair identical filenames; never silently zip unrelated images."""
    folder = Path(folder)
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    names = []
    for role in ("cf", "cs"):
        directory = folder / role
        if not directory.is_dir():
            raise ValueError("資料夾內必須有 cf 與 cs 兩個子資料夾。")
        names.append({p.name for p in directory.iterdir()
                      if p.is_file() and p.suffix.lower() in allowed})
    if names[0] != names[1]:
        missing = sorted(names[0] ^ names[1])
        raise ValueError("正面／側面照片缺少同名配對：" + ", ".join(missing[:8]))
    if len(names[0]) < minimum:
        raise ValueError("至少需要 {} 組同名的雙機照片。".format(minimum))
    return sorted(names[0], key=lambda n: (0, int(Path(n).stem), n)
                  if Path(n).stem.isdigit() else (1, 0, n))


def epipolar_errors(F, side, front):
    """Maximum of the two point-to-line distances, in raw image pixels."""
    side_h = np.column_stack((side, np.ones(len(side))))
    front_h = np.column_stack((front, np.ones(len(front))))
    lines_f = side_h @ F.T
    lines_s = front_h @ F
    numerator = np.abs(np.sum(front_h * lines_f, axis=1))
    denom = np.minimum(np.linalg.norm(lines_f[:, :2], axis=1),
                       np.linalg.norm(lines_s[:, :2], axis=1))
    return numerator / np.maximum(denom, 1e-12)


def estimate_wand_fundamental(front_points, side_points, threshold=3.0):
    """Keep the existing reconstruction convention: x_front.T F x_side = 0.

    Current reconstruction consumes raw pixels. Do not undistort only the
    calibration points here: that would change the coordinate space of F.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("內點門檻必須是正數。")
    if set(front_points) != set(side_points) or len(front_points) < MIN_PAIRS:
        raise ValueError("至少需要 {} 組完整且同名的雙機標註。".format(MIN_PAIRS))
    names = sorted(front_points)
    front, side = [], []
    for name in names:
        pair = [np.asarray(data[name], dtype=np.float64)
                for data in (front_points, side_points)]
        if any(p.shape != (POINTS_PER_IMAGE, 2) or not np.isfinite(p).all()
               or len(np.unique(p, axis=0)) != POINTS_PER_IMAGE for p in pair):
            raise ValueError("{}：兩張照片都必須標註 6 個不同且順序一致的點。".format(name))
        front.append(pair[0])
        side.append(pair[1])
    front, side = np.vstack(front), np.vstack(side)
    for points in (front, side):
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("標註點全落在同一直線，請把校正桿移到不同位置、角度及深度拍攝。")
    cv2.setRNGSeed(0)
    F, mask = cv2.findFundamentalMat(side, front, cv2.FM_RANSAC, threshold, 0.999)
    if F is None or F.shape != (3, 3) or mask is None or not np.isfinite(F).all():
        raise ValueError("無法估計外參，請檢查左右點位順序並補拍不同深度的照片。")
    mask = mask.ravel().astype(bool)
    # Refit using all inliers, then independently check both epipolar distances.
    for _ in range(2):
        if mask.sum() < 12:
            raise ValueError("有效內點少於 12 點，請修正標註或補拍。")
        fitted, _ = cv2.findFundamentalMat(side[mask], front[mask], cv2.FM_8POINT)
        if fitted is None or fitted.shape != (3, 3) or not np.isfinite(fitted).all():
            raise ValueError("外參重算失敗，請補拍不同位置與深度。")
        F = fitted
        errors = epipolar_errors(F, side, front)
        mask = errors <= threshold
    if mask.sum() < 12 or mask.mean() < 0.5:
        raise ValueError("外參品質不足：內點 {}/{}，至少需 12 點且達 50%。請檢查標註。"
                         .format(mask.sum(), len(mask)))
    if np.linalg.matrix_rank(F) != 2 or np.linalg.norm(F) < 1e-12:
        raise ValueError("外參矩陣退化，請補拍。")
    F = F / np.linalg.norm(F)
    return {
        "F": F.tolist(),
        "convention": "x_front^T F x_side = 0",
        "coordinate_space": "raw_pixels",
        "method": "wand_manual_ransac",
        "inliers": int(mask.sum()),
        "total_points": len(mask),
        "threshold_px": float(threshold),
        "median_error_px": float(np.median(errors)),
        "inlier_rms_px": float(np.sqrt(np.mean(errors[mask] ** 2))),
        "pairs": {name: {"inliers": int(mask[i*6:(i+1)*6].sum()),
                         "median_error_px": float(np.median(errors[i*6:(i+1)*6]))}
                  for i, name in enumerate(names)},
        "estimated_at": datetime.now().isoformat(timespec="seconds"),
    }


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as f:
        temporary = Path(f.name)
        f.write(content)
    try:
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def json_bytes(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def save_json(path, data):
    atomic_write(path, json_bytes(data))


def publish_calibration(calib_dir, session, roles, entries, report, points):
    """Publish only a completed run. Back up replaced files and roll back errors."""
    front, side = roles
    validate_roles(front, side)
    for sn in roles:
        validate_intrinsic(entries[sn])
    matrix = np.asarray(report["F"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 2:
        raise ValueError("外參矩陣無效，未更新校正檔。")
    calib_dir, session = Path(calib_dir), Path(session)
    intrinsics_path = calib_dir / "intrinsic/intrinsics_all.json"
    existing = json.loads(intrinsics_path.read_text(encoding="utf-8")) if intrinsics_path.exists() else {}
    existing.update(entries)
    report = dict(report, front=front, side=side, session=str(session))
    outputs = {
        "camera_roles.json": json_bytes({"front": front, "side": side,
                                         "updated_at": report["estimated_at"]}),
        "intrinsic/intrinsics_all.json": json_bytes(existing),
        "extrinsic/selected_points_cf.json": json_bytes(points["cf"]),
        "extrinsic/selected_points_cs.json": json_bytes(points["cs"]),
        "extrinsic/fundamental.json": json_bytes(report),
    }
    _write_with_backup(calib_dir, session, roles, outputs)
    return report


def publish_intrinsics(calib_dir, session, roles, entries, updated_at):
    """只發布兩台內參與正面／側面角色（外參另用 RoMa 等方法計算）；同樣先備份、出錯還原。"""
    front, side = roles
    validate_roles(front, side)
    for sn in roles:
        validate_intrinsic(entries[sn])
    calib_dir, session = Path(calib_dir), Path(session)
    intrinsics_path = calib_dir / "intrinsic/intrinsics_all.json"
    existing = json.loads(intrinsics_path.read_text(encoding="utf-8")) if intrinsics_path.exists() else {}
    existing.update(entries)
    outputs = {
        "camera_roles.json": json_bytes({"front": front, "side": side, "updated_at": updated_at}),
        "intrinsic/intrinsics_all.json": json_bytes(existing),
    }
    _write_with_backup(calib_dir, session, roles, outputs)


def _write_with_backup(calib_dir, session, roles, outputs):
    """寫入校正檔：先把要被取代的舊檔備份到 session/previous_calibration，任何一檔失敗就全部還原。"""
    for sn in roles:
        name = "intrinsic/calib_SN{}.npz".format(sn)
        if (session / name).exists():
            outputs[name] = (session / name).read_bytes()
    originals = {name: (calib_dir / name).read_bytes() if (calib_dir / name).exists() else None
                 for name in outputs}
    backup = session / "previous_calibration"
    for name, content in originals.items():
        if content is not None:
            atomic_write(backup / name, content)
    changed = []
    try:
        for name, content in outputs.items():
            atomic_write(calib_dir / name, content)
            changed.append(name)
    except Exception:
        for name in reversed(changed):
            if originals[name] is None:
                (calib_dir / name).unlink()
            else:
                atomic_write(calib_dir / name, originals[name])
        raise
