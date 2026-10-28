"""讀取 calibration/calibration.py 產生的雙相機校正，轉成 3D 介面使用的慣例。

Db/Calibration/stereo.json 是目前使用中的校正；錄影時複製一份成錄影資料夾的
calibration.json（快照），日後重新校正也不會影響舊錄影的 3D 重建。

校正檔的外參是正面→側面：X_side = R @ X_front + t，t 單位 mm。
3D 介面以側面相機為世界座標（P1 = [I|0] 為側面、P2 = [R|t] 為正面），
因此這裡轉成側面→正面，並把 t 換成公尺，速度才會是 m/s。
"""
import json
import os
import shutil
from dataclasses import dataclass

import numpy as np

from utils.paths import DB_DIR

STEREO_JSON = DB_DIR / "Calibration" / "stereo.json"
SNAPSHOT_NAME = "calibration.json"


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


def _parse(path):
    """讀 stereo.json 格式的檔案並轉成側面世界座標、公尺"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
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
        source=str(path),
    )


def load_stereo_params(folder=None):
    """依序讀：錄影資料夾的快照 → 上一層資料夾的快照（分析結果放在 CF_* 子資料夾時）
    → 目前的 stereo.json；都沒有回傳 None。"""
    candidates = []
    if folder:
        folder = os.path.abspath(folder)
        candidates += [os.path.join(folder, SNAPSHOT_NAME),
                       os.path.join(os.path.dirname(folder), SNAPSHOT_NAME)]
    candidates.append(str(STEREO_JSON))
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            return _parse(path)
        except (OSError, ValueError, KeyError) as e:
            print(f"[stereo_calibration] {path} 無法讀取：{e}")
    return None


def save_snapshot(folder):
    """把目前的 stereo.json 複製成錄影資料夾的 calibration.json；沒有校正檔回傳 False。"""
    if not STEREO_JSON.exists():
        return False
    shutil.copy2(STEREO_JSON, os.path.join(folder, SNAPSHOT_NAME))
    return True
