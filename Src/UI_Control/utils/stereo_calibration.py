"""讀取 calibration/calibrate.py 產生的雙相機校正，轉成 3D 介面使用的慣例。

校正檔（stereo.json 與錄影資料夾內的 calibration.json）的外參是
正面→側面：X_side = R @ X_front + t，t 單位 mm。
3D 介面以側面相機為世界座標（P1 = [I|0] 為側面、P2 = [R|t] 為正面），
因此這裡轉成側面→正面，並把 t 換成公尺，速度才會是 m/s。
"""
import json
import os
from dataclasses import dataclass

import numpy as np

from utils.paths import DB_DIR

STEREO_JSON = DB_DIR / "Calibration" / "extrinsic" / "stereo.json"
SNAPSHOT_NAME = "calibration.json"   # 錄影當下的校正快照，與 calibration/calib_store.py 相同


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


def _to_side_world(K_F, dist_F, K_S, dist_S, R_fs, t_fs_mm, source):
    R_fs = np.asarray(R_fs, dtype=np.float64).reshape(3, 3)
    t_fs = np.asarray(t_fs_mm, dtype=np.float64).reshape(3, 1) / 1000.0
    return StereoParams(
        K_side=np.asarray(K_S, dtype=np.float64),
        dist_side=np.asarray(dist_S, dtype=np.float64).ravel(),
        K_front=np.asarray(K_F, dtype=np.float64),
        dist_front=np.asarray(dist_F, dtype=np.float64).ravel(),
        R=R_fs.T,
        t=-R_fs.T @ t_fs,
        source=source,
    )


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stereo_params(folder=None):
    """優先讀錄影資料夾內的校正快照，其次讀目前的 stereo.json；都沒有回傳 None。"""
    if folder:
        snapshot = os.path.join(folder, SNAPSHOT_NAME)
        if os.path.exists(snapshot):
            try:
                d = _read_json(snapshot)
                return _to_side_world(d["K_F"], d["dist_F"], d["K_S"], d["dist_S"],
                                      d["R"], d["t"], snapshot)
            except (OSError, ValueError, KeyError) as e:
                print(f"[stereo_calibration] 校正快照無法讀取，改用 stereo.json：{e}")

    if STEREO_JSON.exists():
        try:
            d = _read_json(STEREO_JSON)
            cf, cs = d["intrinsics"]["cf"], d["intrinsics"]["cs"]
            return _to_side_world(cf["intrinsic_matrix"], cf["distortion_coefficients"],
                                  cs["intrinsic_matrix"], cs["distortion_coefficients"],
                                  d["R_cs_from_cf"], d["t_cs_from_cf_mm"], str(STEREO_JSON))
        except (OSError, ValueError, KeyError) as e:
            print(f"[stereo_calibration] stereo.json 無法讀取：{e}")
    return None
