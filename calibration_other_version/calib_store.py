"""
校正資料的統一存放位置與讀寫函式。

所有校正產出都放在 Db/Calibration/ 底下（Db 不進 git）：

    Db/Calibration/
    ├─ camera_roles.json            哪台是正面、哪台是側面（calib.py 第 4 區產生）
    ├─ intrinsic/
    │   ├─ intrinsics_all.json      所有相機內參，以序號為鍵（calib.py）
    │   └─ calib_SN<序號>.npz        單台完整校正資料（calib.py）
    └─ extrinsic/
        ├─ selected_points_cf.json  正面標註點（calibrate.py）
        ├─ selected_points_cs.json  側面標註點（calibrate.py）
        └─ fundamental.json         F 矩陣（calibrate.py，或 roma/roma_extrinsic.py --save）

路徑以本檔位置推算，不受執行時所在目錄影響。
"""
import os
import json
from datetime import datetime

import numpy as np

from Src.paths import DB_DIR
CALIB_DIR = str(DB_DIR / "Calibration")
INTRINSIC_DIR = os.path.join(CALIB_DIR, "intrinsic")
EXTRINSIC_DIR = os.path.join(CALIB_DIR, "extrinsic")

ROLES_JSON = os.path.join(CALIB_DIR, "camera_roles.json")
INTRINSICS_JSON = os.path.join(INTRINSIC_DIR, "intrinsics_all.json")
POINTS_JSON = {
    "cf": os.path.join(EXTRINSIC_DIR, "selected_points_cf.json"),
    "cs": os.path.join(EXTRINSIC_DIR, "selected_points_cs.json"),
}
FUNDAMENTAL_JSON = os.path.join(EXTRINSIC_DIR, "fundamental.json")

# 舊設備（2026-02 前人）的校正資料；讀寫它的舊工具已於 2026-10-13 刪除，資料夾保留備查，主流程不使用
OLD_RIG_DIR = os.path.join(CALIB_DIR, "old_rig_20260223")


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


# ---------- 外參（F 矩陣） ----------

def save_fundamental(F, front_sn, side_sn, inliers, total, threshold, extra=None):
    """extra：額外記錄的欄位（例如 RoMa 工具記下方法與來源影片）"""
    write_json(FUNDAMENTAL_JSON, {
        **(extra or {}),
        "F": np.asarray(F).tolist(),
        # 02 以側面為 pts1、正面為 pts2 估計，因此 F 滿足 x_front^T F x_side = 0
        "convention": "x_front^T F x_side = 0",
        "front": front_sn,
        "side": side_sn,
        "inliers": int(inliers),
        "total_points": int(total),
        "threshold_px": float(threshold),
        "estimated_at": datetime.now().isoformat(timespec="seconds"),
    })


def load_fundamental():
    """回傳 fundamental.json 內容（F 已轉為 ndarray）；不存在回傳 None。"""
    data = read_json(FUNDAMENTAL_JSON)
    if data is None:
        return None
    data["F"] = np.array(data["F"], dtype=np.float64)
    return data


def load_stereo_calibration():
    """組合 3D 重建需要的全部參數。

    回傳 dict：K_F, dist_F, K_S, dist_S, F, front, side
    任一項缺少時回傳 (None, 原因說明)，成功時回傳 (dict, None)。
    """
    roles = load_roles()
    if roles is None:
        return None, f"找不到相機角色設定 {ROLES_JSON}（請用 calib.py 第 4 區指定正面/側面）"
    front, side = roles

    intr_f = get_intrinsic(front)
    if intr_f is None:
        return None, f"正面相機 SN{front} 尚未做內參校正"
    intr_s = get_intrinsic(side)
    if intr_s is None:
        return None, f"側面相機 SN{side} 尚未做內參校正"

    fund = load_fundamental()
    if fund is None:
        return None, f"找不到 {FUNDAMENTAL_JSON}（請先用 start_calibrate.bat 完成校正桿外參，或在校正中心用 RoMa 計算外參）"
    if (fund.get("front"), fund.get("side")) != (front, side):
        return None, (f"F 矩陣是用 front={fund.get('front')} side={fund.get('side')} 算的，"
                      f"與目前角色 front={front} side={side} 不符，請重新執行 02")

    return {
        "K_F": intr_f[0], "dist_F": intr_f[1],
        "K_S": intr_s[0], "dist_S": intr_s[1],
        "F": fund["F"],
        "front": front, "side": side,
        "F_estimated_at": fund.get("estimated_at"),
    }, None


# ---------- 錄影快照 ----------
# 校正值只對「錄影當下的相機擺位」有效，因此每筆錄影資料夾各存一份，
# 日後重新校正也不會影響舊錄影的 3D 重建。

SNAPSHOT_NAME = "calibration.json"
_MATRIX_KEYS = ("K_F", "dist_F", "K_S", "dist_S", "F")


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
