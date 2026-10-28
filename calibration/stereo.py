"""Validation and storage for the guided calibration tool (no camera/Qt imports)."""
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime

import numpy as np

from chessboard_cross.chessboard_extrinsic import fit_chessboard_rig
from chessboard_cross.chessboard_metric import detect_folder


# 棋盤格外參：每 5 組留 1 組做驗證，訓練組至少要 3 組，所以最少 6 組
MIN_PAIRS = 6
RECOMMENDED_PAIRS = 15
MAX_STEREO_RMS = 3.0  # px，超過不發布外參（沿用 PnP.py 門檻）


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


def _plain(value):
    """把 numpy 陣列／純量轉成 JSON 可寫入的型別"""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def estimate_chessboard_stereo(session, roles, entries, pattern, square_mm):
    """偵測 session/cf、cs 的同名棋盤格照片，固定內參求正面→側面外參（R、t，mm）。

    角點偵測用 findChessboardCornersSB；align_pairs 以跨組姿態一致性挑出側面的正確角點排列，
    解決棋盤格旋轉 180° 造成的配對錯誤；每 5 組留 1 組驗證邊長。
    """
    session = Path(session)
    pattern = [int(pattern[0]), int(pattern[1])]
    cameras = {"cf": entries[roles[0]], "cs": entries[roles[1]]}
    calibration = {"schema": "chessboard_intrinsics_v1", "units": "mm",
                   "pattern": pattern, "square_mm": float(square_mm), "cameras": cameras}
    detections = {"pattern": pattern, "cameras": {}}
    for role in ("cf", "cs"):
        # 每 5 張存一張角點預覽（標出 0 號角點），可人工確認兩台的角點方向
        detections["cameras"][role] = detect_folder(session / role, tuple(pattern),
                                                    session / "corners" / role, preview_every=5)
    failed = {role: detections["cameras"][role]["failed"] for role in ("cf", "cs")}
    result = fit_chessboard_rig(calibration, detections, "pnp", max_pairs=60)
    result.update(front=roles[0], side=roles[1], session=str(session), detection_failed=failed,
                  estimated_at=datetime.now().isoformat(timespec="seconds"))
    # 轉成純 JSON 型別，之後存檔與發布都不必再處理 numpy
    return json.loads(json.dumps(result, default=_plain))


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


SESSION_NAME = re.compile(r"\d{8}_\d{6}(_.*)?")


def save_rejected(session, report):
    """不合格的外參報告存成 stereo_failed.json 供檢查，不覆蓋已採用的 stereo.json"""
    save_json(Path(session) / "stereo_failed.json", report)


def _current_session_name(calib_dir):
    """上層 stereo.json 所屬的 session 名稱；不存在或讀不到回傳 None（視同可被取代，舊檔仍會備份）"""
    try:
        data = json.loads((Path(calib_dir) / "stereo.json").read_text(encoding="utf-8"))
        return Path(data["session"]).name if isinstance(data, dict) and data.get("session") else None
    except (OSError, ValueError, KeyError):
        return None


def publish_stereo(calib_dir, session, roles, entries, report):
    """採用一次合格的校正，回傳 (report, 是否成為目前使用中)。

    session/stereo.json 加上 published_at，3D 分析依錄影日期挑校正時只用已採用的 session。
    上層 Db/Calibration/*.json 永遠是日期最新的已採用校正：採用較舊日期的 session
    （例如重算 _legacy）只會標記為可用，不會蓋掉較新的；被取代的檔案先備份、出錯還原，
    上層寫入成功後才標記 session，失敗時 session 維持未採用。
    """
    front, side = roles
    validate_roles(front, side)
    for sn in roles:
        validate_intrinsic(entries[sn])
    R = np.asarray(report["R_cs_from_cf"], dtype=float)
    t = np.asarray(report["t_cs_from_cf_mm"], dtype=float)
    if (R.shape != (3, 3) or t.size != 3 or not np.isfinite(R).all() or not np.isfinite(t).all()
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)):
        raise ValueError("外參無效，未更新校正檔。")
    calib_dir, session = Path(calib_dir), Path(session)
    # 名稱 YYYYMMDD_HHMMSS[_後綴] 的字串順序即時間順序，3D 依日期挑選也靠這個規則
    if not SESSION_NAME.fullmatch(session.name):
        raise ValueError(f"session 名稱須為 YYYYMMDD_HHMMSS[_後綴]：{session.name}")
    report = dict(report, session=str(session), published_at=datetime.now().isoformat(timespec="seconds"))

    current = _current_session_name(calib_dir)
    if current and current > session.name:
        save_json(session / "stereo.json", report)
        return report, False

    intrinsics_path = calib_dir / "intrinsics_all.json"
    existing = json.loads(intrinsics_path.read_text(encoding="utf-8")) if intrinsics_path.exists() else {}
    existing.update(entries)
    outputs = {
        "camera_roles.json": json_bytes({"front": front, "side": side,
                                         "updated_at": report["estimated_at"]}),
        "intrinsics_all.json": json_bytes(existing),
        "stereo.json": json_bytes(report),
    }
    _write_with_backup(calib_dir, session, roles, outputs)
    save_json(session / "stereo.json", report)
    return report, True


def publish_intrinsics(calib_dir, session, roles, entries, updated_at):
    """只發布兩台內參與正面／側面角色，外參沿用目前的 stereo.json；同樣先備份、出錯還原。"""
    front, side = roles
    validate_roles(front, side)
    for sn in roles:
        validate_intrinsic(entries[sn])
    calib_dir, session = Path(calib_dir), Path(session)
    intrinsics_path = calib_dir / "intrinsics_all.json"
    existing = json.loads(intrinsics_path.read_text(encoding="utf-8")) if intrinsics_path.exists() else {}
    existing.update(entries)
    outputs = {
        "camera_roles.json": json_bytes({"front": front, "side": side, "updated_at": updated_at}),
        "intrinsics_all.json": json_bytes(existing),
    }
    _write_with_backup(calib_dir, session, roles, outputs)


def _write_with_backup(calib_dir, session, roles, outputs):
    """寫入校正檔：先把要被取代的舊檔備份到 session/previous_calibration，任何一檔失敗就全部還原。"""
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
