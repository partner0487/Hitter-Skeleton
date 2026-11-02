"""讀取 calibration/calibration.py 產生的雙相機校正，轉成 3D 介面使用的慣例。

校正依日期進行，每次一個 Db/Calibration/sessions/<YYYYMMDD_HHMMSS[_後綴]>/。
右打、左打的側面相機架在不同側，各自校正（後綴 _righthit／_lefthit，或 session.json 的 batter_side）。
載入錄影時，從路徑中的日期（Db/<打者>/2026-10-08/... 或 .../20260421/...）挑
「當天或之前、打擊方向相符的最新已採用校正」，重新校正後舊錄影仍使用當時的參數；
沒標打擊方向的舊校正右打、左打都可用。路徑沒有日期時不限日期；都找不到才用 Db/Calibration/stereo.json（打擊方向不符時不用）。

校正檔的外參是正面→側面：X_side = R @ X_front + t，t 單位 mm。
3D 介面以側面相機為世界座標（P1 = [I|0] 為側面、P2 = [R|t] 為正面），
因此這裡轉成側面→正面，並把 t 換成公尺，速度才會是 m/s。
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.paths import DB_DIR

CALIB_DIR = DB_DIR / "Calibration"
STEREO_JSON = CALIB_DIR / "stereo.json"
SESSIONS_DIR = CALIB_DIR / "sessions"
# 錄影資料夾的日期：2026-10-08 或 20260421（不接受只有一個連字號的 2026-1008）
_DATE_IN_PATH = re.compile(r"(20\d{2})(?:-(\d{2})-|(\d{2}))(\d{2})")
# session 名稱 YYYYMMDD_HHMMSS[_後綴]，字串順序即時間順序（與 calibration/stereo.py 相同規則）
_SESSION_NAME = re.compile(r"\d{8}_\d{6}(_.*)?")
# 打擊方向與 session 資料夾後綴（與 calibration/stereo.py 的 BATTER_SUFFIX 相同）
_SIDE_SUFFIX = {"right": "_righthit", "left": "_lefthit"}
_SIDE_NAME = {"right": "右打", "left": "左打"}


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
            # 20251234 這類數字資料夾不是日期，繼續往上找
            try:
                return datetime.strptime("".join(g for g in m.groups() if g), "%Y%m%d").strftime("%Y%m%d")
            except ValueError:
                continue
    return None


def _published_sessions(date=None):
    """當天或之前（date 為 None 時不限日期）已採用的 session 的 stereo.json，由新到舊"""
    if not SESSIONS_DIR.is_dir():
        return []
    sessions = sorted((p for p in SESSIONS_DIR.iterdir()
                       if _SESSION_NAME.fullmatch(p.name) and (date is None or p.name[:8] <= date)),
                      key=lambda p: p.name, reverse=True)
    return [p / "stereo.json" for p in sessions if (p / "stereo.json").exists()]


def batter_side_of(data, session_dir):
    """校正的打擊方向 right／left，順序與 calibration/stereo.py 的 batter_side_of 相同：
    session.json 的 batter_side、資料夾後綴 _righthit／_lefthit；都沒有才看 stereo.json 的 batter_side
    （上層 Db/Calibration/stereo.json 不在 session 資料夾裡）。都沒有回傳 None（右打、左打都可用）"""
    try:
        side = _read_json(Path(session_dir) / "session.json").get("batter_side")
        if side in _SIDE_SUFFIX:
            return side
    except (OSError, ValueError, AttributeError):
        pass
    side = next((side for side, suffix in _SIDE_SUFFIX.items() if Path(session_dir).name.endswith(suffix)), None)
    return side or (data["batter_side"] if data.get("batter_side") in _SIDE_SUFFIX else None)


def _load(path, require_published=False, batter_side=None):
    """讀校正檔並轉成 StereoParams；檔案壞掉、（要求時）未採用或打擊方向不符回傳 None"""
    try:
        data = _read_json(path)
        if not isinstance(data, dict) or (require_published and not data.get("published_at")):
            return None
        side = batter_side_of(data, Path(path).parent)
        if batter_side and side and side != batter_side:
            return None
        return _to_params(data, path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"[stereo_calibration] {path} 無法讀取，略過：{e}")
        return None


def load_stereo_params(folder=None, batter_side=None):
    """挑校正：錄影日期當天或之前（沒有日期則不限）、打擊方向相符（batter_side 為 right／left，
    None 表示不限）的最新已採用 session；都沒有才用打擊方向相符的 Db/Calibration/stereo.json。都沒有回傳 None。"""
    date = recording_date(folder) if folder else None
    side = _SIDE_NAME.get(batter_side, "")
    # 錄影早於所有校正時，改用同一側最新的校正，避免用到另一側擺位的上層 stereo.json
    for limit in ((date, None) if date else (None,)):
        for path in _published_sessions(limit):
            params = _load(path, require_published=True, batter_side=batter_side)
            if params is not None:
                if limit is None and date:
                    print(f"[stereo_calibration] {date} 以前沒有{side}校正，改用最新的{side}校正")
                return params
    # 上層 stereo.json 是最後採用的那一側，打擊方向不符時不用，免得套到另一側的外參
    params = _load(STEREO_JSON, batter_side=batter_side) if STEREO_JSON.exists() else None
    print(f"[stereo_calibration] 沒有可用的{side}校正，"
          + (f"改用 {STEREO_JSON.name}" if params else "使用程式內寫死的舊參數"))
    return params
