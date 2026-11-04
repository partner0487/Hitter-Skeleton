"""專案共用路徑，皆以本檔案位置為基準，不受執行時工作目錄影響"""
from pathlib import Path

# 本檔位於 <專案根目錄>/Src/UI_Control/utils/paths.py
UI_CONTROL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = UI_CONTROL_DIR.parent.parent

DB_DIR = PROJECT_ROOT / "Db"
RECORD_DIR = DB_DIR / "Record"
ICON_DIR = DB_DIR / "icon"
CAMERA_CONFIG_DIR = UI_CONTROL_DIR / "camera_config"

# Db 底下不是球員的資料夾，列出球員清單時要排除
NON_HITTER_DIRS = {"icon", "pretrain", "Calibration"}


def clip_parts(path):
    """錄影資料夾（或其中的檔案、CF_ 結果子資料夾）相對 Db 的各層名稱，例如 (WeiShu, 2026-10-29, T座, WeiShu_H05)；
    不在 Db 底下回傳 ()"""
    if not path:
        return ()
    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    if p.name.startswith("CF_"):
        p = p.parent
    try:
        return p.relative_to(DB_DIR.resolve()).parts
    except ValueError:
        return ()


def clip_title(path):
    """畫面左上角顯示的錄影名稱：打者｜日期｜模式｜片段；不在 Db 底下時顯示資料夾名稱"""
    parts = clip_parts(path)
    if parts:
        return "｜".join(parts)
    p = Path(path) if path else None
    return p.parent.name if p is not None and p.is_file() else (p.name if p is not None else "")
