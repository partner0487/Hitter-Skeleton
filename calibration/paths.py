"""校正工具共用路徑，皆以本檔案位置為基準，不受執行時工作目錄影響"""
from pathlib import Path

# 本檔位於 <專案根目錄>/calibration/paths.py
CALIBRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CALIBRATION_DIR.parent

DB_DIR = PROJECT_ROOT / "Db"
# 與主程式共用同一份相機設定，校正時調好的曝光、增益「存成預設」後，主程式開相機也會沿用
CAMERA_CONFIG_DIR = PROJECT_ROOT / "Src" / "UI_Control" / "camera_config"
