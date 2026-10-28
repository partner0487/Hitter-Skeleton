"""讀取 calibration/calibration.py 產生的雙相機校正，轉成 3D 介面使用的慣例。

校正依日期進行，每次一個 Db/Calibration/sessions/<YYYYMMDD_HHMMSS[_後綴]>/。
載入錄影時，從路徑中的日期（Db/<打者>/2026-10-08/... 或 .../20260421/...）挑
「當天或之前最新的已採用校正」，重新校正後舊錄影仍使用當時的參數；
路徑沒有日期或當天以前沒有校正時，使用最新的 Db/Calibration/stereo.json。

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
# 錄影資料夾的日期：2026-10-08 或 20260421
_DATE_IN_PATH = re.compile(r"(20\d{2})-?(\d{2})-?(\d{2})")
# session 名稱 YYYYMMDD_HHMMSS[_後綴]，字串順序即時間順序（與 calibration/stereo.py 相同規則）
_SESSION_NAME = re.compile(r"\d{8}_\d{6}(_.*)?")


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
    """從錄影路徑取出日期（YYYYMMDD）；路徑中沒有 YYYY-MM-DD 或 YYYYMMDD 資料夾時回傳 None"""
    for part in reversed(Path(folder).parts):
        m = _DATE_IN_PATH.fullmatch(part)
        if m:
            return "".join(m.groups())
    return None


def _published_sessions(date):
    """當天或之前已採用的 session 的 stereo.json，由新到舊"""
    if not SESSIONS_DIR.is_dir():
        return []
    sessions = sorted((p for p in SESSIONS_DIR.iterdir()
                       if _SESSION_NAME.fullmatch(p.name) and p.name[:8] <= date),
                      key=lambda p: p.name, reverse=True)
    return [p / "stereo.json" for p in sessions if (p / "stereo.json").exists()]


def _load(path, require_published=False):
    """讀校正檔並轉成 StereoParams；檔案壞掉或（要求時）未採用回傳 None"""
    try:
        data = _read_json(path)
        if not isinstance(data, dict) or (require_published and not data.get("published_at")):
            return None
        return _to_params(data, path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"[stereo_calibration] {path} 無法讀取，略過：{e}")
        return None


def load_stereo_params(folder=None):
    """依錄影日期挑校正：當天或之前最新且可讀的已採用 session；
    沒有日期或找不到時用最新的 stereo.json。都沒有回傳 None。"""
    date = recording_date(folder) if folder else None
    if date:
        for path in _published_sessions(date):
            params = _load(path, require_published=True)
            if params is not None:
                return params
        print(f"[stereo_calibration] {date} 以前沒有可用的校正，改用最新的校正")
    return _load(STEREO_JSON) if STEREO_JSON.exists() else None
