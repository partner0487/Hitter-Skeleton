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


NOMINAL_DURATION_S = 1.5  # 標準錄影長度（自動觸發 90 幀 @ 60 fps；長片段裁剪後也是這個長度），介面標示 x.xx/1.5s
# 即時自動觸發存的幀數（觸發前 50 + 後 40）。10-27 以前的錄影 mp4 一律標 180 fps（相機擷取 fps，不是實際送出的），
# 90 幀的舊錄影（如 Player1 2026-04-21）也是標準 1.5 秒
AUTO_CLIP_FRAMES = 90


def clip_duration(total_frames: int, fps: float) -> float:
    """時間標籤的影片長度（秒）：標準長度的片段（AUTO_CLIP_FRAMES 幀，或總幀數與 NOMINAL_DURATION_S × fps 差 1 幀內，
    60 fps 為 90、120 fps 為 180）為 NOMINAL_DURATION_S；找不到擊球點沒裁剪的長片段、剪得比較短的片段用實際長度 總幀數 / fps"""
    if not total_frames or total_frames < 2 or not fps or fps <= 0:
        return NOMINAL_DURATION_S
    if total_frames == AUTO_CLIP_FRAMES or abs(total_frames - NOMINAL_DURATION_S * fps) <= 1:
        return NOMINAL_DURATION_S
    return total_frames / fps


def time_label(seconds: float, duration: float = NOMINAL_DURATION_S) -> str:
    """目前時間/影片長度，如 0.45/1.5s；非標準長度的影片長度顯示到小數兩位（0.45/4.27s）"""
    total = f"{duration:.1f}" if duration == NOMINAL_DURATION_S else f"{duration:.2f}"
    return f"{seconds:.2f}/{total}s"


def frame_seconds(frame: int, total_frames: int, frame_times: dict = None,
                  duration: float = NOMINAL_DURATION_S) -> float:
    """畫面上顯示的時間（秒），第一幀 0、最後一幀 duration（見 clip_duration）：
    有 frame_times.csv 時把實際擷取時間等比例縮放到 duration（保留掉幀造成的不均勻間隔，
    實際最後一幀約 1.48 s），沒有時平均分給每一幀"""
    if frame_times and frame in frame_times and total_frames - 1 in frame_times:
        start = frame_times.get(0, min(frame_times.values()))
        span = frame_times[total_frames - 1] - start
        if span > 0:
            return (frame_times[frame] - start) / span * duration
    return frame / (total_frames - 1) * duration if total_frames and total_frames > 1 else 0.0


def phase_seconds(f_start, f_end, total_frames: int, frame_times: dict = None, duration: float = NOMINAL_DURATION_S):
    """f_start → f_end 的時間（秒），與時間標籤、進度條各段秒數用同一套換算；缺幀回傳 None"""
    if f_start is None or f_end is None or f_end < f_start:
        return None
    return (frame_seconds(f_end, total_frames, frame_times, duration)
            - frame_seconds(f_start, total_frames, frame_times, duration))
