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


# ---------- 外參（棋盤格立體校正） ----------

def load_stereo():
    """回傳 stereo.json 內容（R、t 已轉為 ndarray）；不存在回傳 None。"""
    data = read_json(STEREO_JSON)
    if data is None:
        return None
    data["R_cs_from_cf"] = np.array(data["R_cs_from_cf"], dtype=np.float64)
    data["t_cs_from_cf_mm"] = np.array(data["t_cs_from_cf_mm"], dtype=np.float64).reshape(3, 1)
    return data


def load_stereo_calibration():
    """組合 3D 重建需要的全部參數。

    回傳 dict：K_F, dist_F, K_S, dist_S, R, t, front, side
    （R、t 為正面→側面：X_side = R @ X_front + t，t 單位 mm）
    任一項缺少時回傳 (None, 原因說明)，成功時回傳 (dict, None)。
    """
    stereo = load_stereo()
    if stereo is None:
        return None, f"找不到 {STEREO_JSON}（請先用 calibrate.py 完成棋盤格外參）"
    front, side = str(stereo["front"]), str(stereo["side"])
    roles = load_roles()
    if roles is not None and roles != (front, side):
        return None, (f"外參是用 front={front} side={side} 算的，"
                      f"與目前角色 front={roles[0]} side={roles[1]} 不符，請重新做外參")

    # 外參計算時固定的內參一併存在 stereo.json，兩者必定一致，不另外查 intrinsics_all.json
    cf, cs = stereo["intrinsics"]["cf"], stereo["intrinsics"]["cs"]
    return {
        "K_F": np.array(cf["intrinsic_matrix"], dtype=np.float64),
        "dist_F": np.array(cf["distortion_coefficients"], dtype=np.float64),
        "K_S": np.array(cs["intrinsic_matrix"], dtype=np.float64),
        "dist_S": np.array(cs["distortion_coefficients"], dtype=np.float64),
        "R": stereo["R_cs_from_cf"], "t": stereo["t_cs_from_cf_mm"],
        "front": front, "side": side,
        "estimated_at": stereo.get("estimated_at"),
    }, None


# ---------- 錄影快照 ----------
# 校正值只對「錄影當下的相機擺位」有效，因此每筆錄影資料夾各存一份，
# 日後重新校正也不會影響舊錄影的 3D 重建。

SNAPSHOT_NAME = "calibration.json"
_MATRIX_KEYS = ("K_F", "dist_F", "K_S", "dist_S", "R", "t")


def save_snapshot(folder, front_sn=None, side_sn=None):
    """把目前的校正值存進錄影資料夾。回傳 (是否成功, 說明)。

    front_sn / side_sn 為實際錄影的相機序號，與校正角色不符時不存。
    """
    calib, reason = load_stereo_calibration()
    if calib is None:
        return False, reason
    if front_sn and side_sn and (str(front_sn), str(side_sn)) != (calib["front"], calib["side"]):
        return False, (f"錄影相機 front={front_sn} side={side_sn} 與校正角色 "
                       f"front={calib['front']} side={calib['side']} 不符")

    data = {k: (np.asarray(v).tolist() if k in _MATRIX_KEYS else v) for k, v in calib.items()}
    data["saved_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(os.path.join(folder, SNAPSHOT_NAME), data)
    return True, f"校正快照已存入 {SNAPSHOT_NAME}"


def load_snapshot(folder):
    """讀取錄影資料夾內的校正快照（矩陣已轉為 ndarray）；沒有則回傳 None。"""
    data = read_json(os.path.join(folder, SNAPSHOT_NAME))
    if data is None:
        return None
    for k in _MATRIX_KEYS:
        data[k] = np.array(data[k], dtype=np.float64)
    return data
