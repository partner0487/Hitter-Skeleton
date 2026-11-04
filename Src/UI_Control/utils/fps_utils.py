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


NOMINAL_DURATION_S = 1.5  # 每支錄影的長度（介面標示 x.xx/1.5s）


def frame_seconds(frame: int, total_frames: int, frame_times: dict = None) -> float:
    """畫面上顯示的時間（秒），第一幀 0、最後一幀 NOMINAL_DURATION_S：
    有 frame_times.csv 時把實際擷取時間等比例縮放到 NOMINAL_DURATION_S（保留掉幀造成的不均勻間隔，
    實際最後一幀約 1.48 s），沒有時平均分給每一幀"""
    if frame_times and frame in frame_times and total_frames - 1 in frame_times:
        start = frame_times.get(0, min(frame_times.values()))
        span = frame_times[total_frames - 1] - start
        if span > 0:
            return (frame_times[frame] - start) / span * NOMINAL_DURATION_S
    return frame / (total_frames - 1) * NOMINAL_DURATION_S if total_frames > 1 else 0.0
