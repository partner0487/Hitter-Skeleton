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
