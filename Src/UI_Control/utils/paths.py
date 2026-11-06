"""專案共用路徑，皆以本檔案位置為基準，不受執行時工作目錄影響"""
import os
from pathlib import Path

# 本檔位於 <專案根目錄>/Src/UI_Control/utils/paths.py
UI_CONTROL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = UI_CONTROL_DIR.parent.parent

DB_DIR = PROJECT_ROOT / "Db"
RECORD_DIR = DB_DIR / "Record"
ICON_DIR = DB_DIR / "icon"
CAMERA_CONFIG_DIR = UI_CONTROL_DIR / "camera_config"

# Db 底下不是球員的資料夾，列出球員清單時要排除
NON_HITTER_DIRS = {"icon", "pretrain", "Calibration", "full_clip"}  # full_clip：裁剪前的原始長片段（clip_trim）
# 處理結果子資料夾：以正面影片檔名命名（CF_<片段>）；只有側面影片的錄影以側面影片檔名命名（CS_<片段>）
RESULT_PREFIXES = ("CF_", "CS_")


def is_result_dir(name) -> bool:
    return os.path.basename(str(name)).startswith(RESULT_PREFIXES)


def clip_videos(folder):
    """錄影資料夾裡的 (正面 CF*.mp4, 側面 CS*.mp4)，沒有的為 None；只有側面影片的錄影做純側面 2D 分析"""
    try:
        names = sorted(os.listdir(folder))
    except (OSError, TypeError):
        return None, None

    def pick(prefix):
        return next((os.path.join(folder, n) for n in names if n.startswith(prefix) and n.lower().endswith(".mp4")
                     and os.path.isfile(os.path.join(folder, n))), None)
    return pick("CF"), pick("CS")


def source_videos(folder):
    """分析結果資料夾對應的原始 (正面, 側面) 影片，沒有的為 None；骨架由介面依 JSON 即時畫在原始影片上。
    結果子資料夾（CF_／CS_）的影片在上一層錄影資料夾；舊資料沒有原始影片時退回 raw_／skel_ 影片"""
    folder = os.path.normpath(str(folder))
    cf, cs = clip_videos(folder)
    if cf is None and cs is None and is_result_dir(folder):
        cf, cs = clip_videos(os.path.dirname(folder))

    def old(view):
        return next((p for p in (os.path.join(folder, f"raw_{view}.mp4"), os.path.join(folder, f"skel_{view}.mp4"))
                     if os.path.isfile(p)), None)
    return cf or old("front"), cs or old("side")


def clip_parts(path):
    """錄影資料夾（或其中的檔案、CF_／CS_ 結果子資料夾）相對 Db 的各層名稱，例如 (WeiShu, 2026-10-29, T座, WeiShu_H05)；
    不在 Db 底下回傳 ()"""
    if not path:
        return ()
    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    if is_result_dir(p.name):
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
