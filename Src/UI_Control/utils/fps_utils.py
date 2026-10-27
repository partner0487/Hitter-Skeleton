"""影片 fps 取得工具：所有速度、時間換算統一由此取得 fps"""
import csv
import os

# 錄影串流的名目 fps（相機 180 fps ÷ EMIT_EVERY 3），與論文資料集一致
DEFAULT_FPS = 60.0


def read_measured_fps(folder):
    """從錄影時存下的 frame_times.csv 計算實測 fps；沒有檔案或資料不足時回傳 None"""
    if not folder:
        return None
    try:
        with open(os.path.join(folder, "frame_times.csv"), encoding="utf-8") as f:
            times = [float(row["time_sec"]) for row in csv.DictReader(f)]
    except (OSError, ValueError, KeyError):
        return None
    if len(times) > 1 and times[-1] > times[0]:
        return (len(times) - 1) / (times[-1] - times[0])
    return None


def get_video_fps(loader, default=DEFAULT_FPS):
    """依序採用：frame_times.csv 實測值 → 影片標記 fps → 預設值"""
    fps = read_measured_fps(getattr(loader, "folder_path", None))
    if fps:
        return fps
    fps = getattr(loader, "video_fps", None)
    if fps and 0 < fps <= 600:
        return float(fps)
    return default
