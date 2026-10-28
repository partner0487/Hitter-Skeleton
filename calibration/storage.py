"""
校正資料的統一存放位置與讀寫函式。

所有校正產出都放在 Db/Calibration/ 底下（Db 不進 git）：

    Db/Calibration/
    ├─ camera_roles.json        目前的正面／側面相機序號
    ├─ intrinsics_all.json      目前各相機內參，以序號為鍵
    ├─ stereo.json              目前的外參：正面→側面 R、t（mm），X_cs = R @ X_cf + t，含兩台內參
    └─ sessions/<時間>/         每次校正一個資料夾，內參、外參共用
        ├─ intrinsic/images_SN<序號>/  內參照片（一台一台拍）
        ├─ intrinsic/calib_SN<序號>.npz 內參完整資料（角點、每張誤差、傾斜角）
        ├─ cf/ cs/                   外參照片，同名＝同一時刻
        ├─ corners/                  外參角點預覽
        ├─ intrinsics.json           本次兩台內參
        ├─ stereo.json               本次外參報告
        ├─ session.json              本次設定與進度
        └─ previous_calibration/     發布前被取代的上一版（可還原）

上層三個 json 是「目前使用中」的版本，由最近一次完成的校正發布；
sessions 底下保留每一次的原始資料，可以回頭重算或比較。
路徑以本檔位置推算，不受執行時所在目錄影響。
"""
import os
import json
from datetime import datetime

import numpy as np

from paths import DB_DIR
CALIB_DIR = str(DB_DIR / "Calibration")
SESSIONS_DIR = os.path.join(CALIB_DIR, "sessions")

ROLES_JSON = os.path.join(CALIB_DIR, "camera_roles.json")
INTRINSICS_JSON = os.path.join(CALIB_DIR, "intrinsics_all.json")
STEREO_JSON = os.path.join(CALIB_DIR, "stereo.json")


def new_session_dir(suffix=""):
    """建立一個新的校正資料夾 sessions/<YYYYMMDD_HHMMSS>[_suffix]，回傳路徑字串"""
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{suffix}" if suffix else "")
    path = os.path.join(SESSIONS_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    """讀取 JSON；檔案不存在回傳 None。"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------- 相機角色 ----------

def load_roles():
    """回傳 (front_sn, side_sn)；未設定則回傳 None。"""
    data = read_json(ROLES_JSON)
    if data and data.get("front") and data.get("side"):
        return str(data["front"]), str(data["side"])
    return None


def save_roles(front_sn, side_sn):
    write_json(ROLES_JSON, {
        "front": str(front_sn),
        "side": str(side_sn),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })


# ---------- 內參 ----------

def load_intrinsics():
    """回傳 {序號: 內參資料}；檔案不存在回傳空 dict。"""
    return read_json(INTRINSICS_JSON) or {}


def save_intrinsic(sn, entry):
    """寫入或覆蓋單台相機的內參，回傳目前已校正的相機數。"""
    data = load_intrinsics()
    data[str(sn)] = entry
    write_json(INTRINSICS_JSON, data)
    return len(data)


def get_intrinsic(sn):
    """回傳 (K, dist)；該台尚未校正則回傳 None。"""
    entry = load_intrinsics().get(str(sn))
    if entry is None:
        return None
    K = np.array(entry["intrinsic_matrix"], dtype=np.float64)
    dist = np.array(entry["distortion_coefficients"], dtype=np.float64)
    return K, dist


# 外參 stereo.json 由 stereo.publish_stereo 寫入；
# 3D 分析（Src/UI_Control/utils/stereo_calibration.py）依錄影日期挑當天或之前最新的已發布 session。
