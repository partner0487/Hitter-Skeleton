"""錄影用哪幾台相機：正面、側面相機序號與是否使用正面。

- 序號存在 Db/Calibration/camera_roles.json，與校正工具（calibration/storage.py）共用同一份，
  錄影與校正對到同一組相機，3D 才算得對
- 「不使用正面」是錄影自己的設定（camera_config/record_cameras.json），不動校正的檔案：
  只接側面一台時錄成純側面 2D（只寫 CS_ 影片，分析時 3D 才有的指標顯示 —）
- 擊球觸發與 2D 分析都只看側面，所以側面一定要有
"""
import json
import os
from datetime import datetime

from utils.paths import CAMERA_CONFIG_DIR, DB_DIR

ROLES_JSON = DB_DIR / "Calibration" / "camera_roles.json"
RECORD_JSON = CAMERA_CONFIG_DIR / "record_cameras.json"
DEFAULT_ROLES = ("25083472", "21091470")   # 尚無 camera_roles.json 時（正面, 側面）


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_roles():
    """(正面序號, 側面序號)"""
    data = _read(ROLES_JSON) or {}
    if data.get("front") and data.get("side"):
        return str(data["front"]), str(data["side"])
    return DEFAULT_ROLES


def save_roles(front_sn, side_sn):
    """格式與校正工具的 storage.save_roles 相同"""
    _write(ROLES_JSON, {"front": str(front_sn), "side": str(side_sn),
                        "updated_at": datetime.now().isoformat(timespec="seconds")})


def use_front() -> bool:
    return bool((_read(RECORD_JSON) or {}).get("use_front", True))


def set_use_front(value: bool):
    _write(RECORD_JSON, {"use_front": bool(value)})


def list_cameras():
    """目前連接的 FLIR 相機序號（與校正工具 intrinsic_window.list_cameras 相同做法）"""
    import PySpin
    serials = []
    system = PySpin.System.GetInstance()
    try:
        cam_list = system.GetCameras()
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            try:
                serials.append(cam.TLDevice.DeviceSerialNumber.GetValue())
            except Exception:
                pass
            del cam
        cam_list.Clear()
    finally:
        system.ReleaseInstance()
    return sorted(serials)


def resolve_cameras():
    """開相機前決定要開哪幾台：回傳 (正面序號或 None, 側面序號, 提示訊息)。
    側面沒接上時丟 RuntimeError；選了正面但沒接上時改成只用側面，提示訊息說明原因"""
    front, side = load_roles()
    connected = list_cameras()
    if side not in connected:
        found = "、".join(connected) if connected else "沒有"
        raise RuntimeError(f"找不到側面相機 SN{side}（目前連接：{found}）。\n"
                           "擊球觸發與 2D 分析都需要側面相機，請確認連線，或到「相機設定」重新選擇。")
    if not use_front():
        return None, side, ""
    if front not in connected or front == side:
        return None, side, f"找不到正面相機 SN{front}，這次只錄側面（純側面 2D，沒有 3D 指標）。"
    return front, side, ""
