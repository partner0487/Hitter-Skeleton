"""把 ContinuousRecorder 收完的自動錄影片段寫成影片。

- 分析用：每 step 幀取 1（相機 180 fps → 60 fps），寫在錄影資料夾 CF_／CS_<片段>.mp4 與 frame_times.csv，
  與原本完全相同，分析流程與真值驗證不受影響
- 完整 fps：相機每一幀，寫在 full_fps/ 子資料夾（同檔名與 frame_times.csv），在背景執行緒寫，不卡介面；
  之後在擊球點附近用高 fps 細算、或標註 160 fps 真值時使用
相機畫面是原始 Bayer，寫檔時才轉成彩色（與 VideoCaptureThread 送出的畫面同一種轉換）
"""
import os
import threading

import cv2

from utils.paths import FULL_FPS_DIR

BAYER_TO_BGR = cv2.COLOR_BayerBG2BGR


def calc_fps(times):
    """以實際擷取時間計算平均 fps，可反映掉幀"""
    if len(times) > 1 and times[-1] > times[0]:
        return (len(times) - 1) / (times[-1] - times[0])
    return None


def write_video(path, raw_frames, fps):
    h, w = raw_frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for raw in raw_frames:
            out.write(cv2.cvtColor(raw, BAYER_TO_BGR) if raw.ndim == 2 else raw)
    finally:
        out.release()


def write_frame_times(path, times):
    """每幀實際擷取時間（從第一幀起算）：掉幀時 mp4 的固定 fps 不準，分析以此為準"""
    t0 = times[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write("frame,time_sec\n")
        for i, t in enumerate(times):
            f.write(f"{i},{t - t0:.6f}\n")


def _write_set(folder, name, frames1, frames2, times, fps):
    """寫 CF_（有正面時）、CS_ 影片與 frame_times.csv，回傳 (正面路徑或 None, 側面路徑)"""
    os.makedirs(folder, exist_ok=True)
    cf = os.path.join(folder, f"CF_{name}.mp4") if frames1 and frames1[0] is not None else None
    cs = os.path.join(folder, f"CS_{name}.mp4")
    jobs = [(cs, frames2)] + ([(cf, frames1)] if cf else [])
    # 兩台相機各一個執行緒（OpenCV 編碼時會釋放 GIL）
    threads = [threading.Thread(target=write_video, args=(p, fr, fps)) for p, fr in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    write_frame_times(os.path.join(folder, "frame_times.csv"), times)
    return cf, cs


def save_clip(output_dir, name, clip):
    """寫分析用影片（呼叫端執行緒，寫完才回傳），完整 fps 版本交給背景執行緒。
    回傳 (正面路徑或 None, 側面路徑, 分析用 fps, 背景執行緒或 None)"""
    step = clip["step"]
    frames1, frames2, times = clip["frames1"], clip["frames2"], clip["times"]
    sub = slice(0, None, step)
    a_times = times[sub]
    a_fps = calc_fps(a_times) or clip["camera_fps"] / step
    cf, cs = _write_set(output_dir, name, frames1[sub], frames2[sub], a_times, a_fps)

    worker = None
    if step > 1:   # step 為 1 時分析用影片就是完整 fps
        full_fps = calc_fps(times) or clip["camera_fps"]
        worker = threading.Thread(
            target=_write_set, name="full-fps-writer",
            args=(os.path.join(output_dir, FULL_FPS_DIR), name, frames1, frames2, times, full_fps))
        worker.start()
    return cf, cs, a_fps, worker
