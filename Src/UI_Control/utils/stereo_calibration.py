"""讀取 calibration/calibration.py 產生的雙相機校正，轉成 3D 介面使用的慣例。

校正依日期進行，每次一個 Db/Calibration/sessions/<YYYYMMDD_HHMMSS>/。
載入錄影時，從路徑中的日期（Db/<打者>/2026-10-08/...）挑「當天或之前最新的已發布校正」，
重新校正後舊錄影仍使用當時的參數；路徑沒有日期時使用最新的 Db/Calibration/stereo.json。

校正檔的外參是正面→側面：X_side = R @ X_front + t，t 單位 mm。
3D 介面以側面相機為世界座標（P1 = [I|0] 為側面、P2 = [R|t] 為正面），
因此這裡轉成側面→正面，並把 t 換成公尺，速度才會是 m/s。
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.paths import DB_DIR

CALIB_DIR = DB_DIR / "Calibration"
STEREO_JSON = CALIB_DIR / "stereo.json"
SESSIONS_DIR = CALIB_DIR / "sessions"
_DATE_IN_PATH = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass
class StereoParams:
    K_side: np.ndarray
    dist_side: np.ndarray
    K_front: np.ndarray
    dist_front: np.ndarray
    R: np.ndarray          # 側面→正面旋轉：X_front = R @ X_side + t
    t: np.ndarray          # (3, 1)，公尺
    source: str

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.t))


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_params(d, source):
    cf, cs = d["intrinsics"]["cf"], d["intrinsics"]["cs"]
    R_fs = np.asarray(d["R_cs_from_cf"], dtype=np.float64).reshape(3, 3)
    t_fs = np.asarray(d["t_cs_from_cf_mm"], dtype=np.float64).reshape(3, 1) / 1000.0
    return StereoParams(
        K_side=np.asarray(cs["intrinsic_matrix"], dtype=np.float64),
        dist_side=np.asarray(cs["distortion_coefficients"], dtype=np.float64).ravel(),
        K_front=np.asarray(cf["intrinsic_matrix"], dtype=np.float64),
        dist_front=np.asarray(cf["distortion_coefficients"], dtype=np.float64).ravel(),
        R=R_fs.T,
        t=-R_fs.T @ t_fs,
        source=str(source),
    )


def recording_date(folder):
    """從錄影路徑取出日期（YYYYMMDD）；路徑中沒有 YYYY-MM-DD 時回傳 None"""
    for part in reversed(Path(folder).parts):
        m = _DATE_IN_PATH.fullmatch(part)
        if m:
            return "".join(m.groups())
    return None


def _session_for_date(date):
    """當天或之前最新的已發布 session 的 stereo.json 路徑；沒有回傳 None"""
    if not SESSIONS_DIR.is_dir():
        return None
    # session 名稱以 YYYYMMDD 開頭，由新到舊找第一個已發布的
    for session in sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.name, reverse=True):
        path = session / "stereo.json"
        if session.name[:8] > date or not path.exists():
            continue
        try:
            if _read_json(path).get("published_at"):
                return path
        except (OSError, ValueError):
            continue
    return None


def load_stereo_params(folder=None):
    """依錄影日期挑校正；沒有日期或找不到當天以前的校正時用最新的 stereo.json。都沒有回傳 None。"""
    candidates = []
    date = recording_date(folder) if folder else None
    if date:
        matched = _session_for_date(date)
        if matched is not None:
            candidates.append(matched)
        else:
            print(f"[stereo_calibration] {date} 以前沒有已發布的校正，改用最新的校正")
    candidates.append(STEREO_JSON)
    for path in candidates:
        if not path.exists():
            continue
        try:
            return _to_params(_read_json(path), path)
        except (OSError, ValueError, KeyError) as e:
            print(f"[stereo_calibration] {path} 無法讀取：{e}")
    return None
