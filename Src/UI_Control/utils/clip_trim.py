"""長片段裁成標準錄影長度：擊球前 0.83 秒、擊球後 0.67 秒（60 fps 時為 50、40 幀，與錄影頁 ContinuousRecorder 相同）。

在現場手動錄、事後自己剪的片段長短不一（例如 433 幀），會有兩個問題：
1. 片段裡有多次試揮，離地、著地、擊球容易抓到別次揮棒
2. 兩台相機的影格全部讀進記憶體（433 幀約 5.4 GB），處理與載入都慢

做法：先用整段的偵測結果找擊球點（與分析頁同一套），裁出擊球前後的固定秒數（依影片 fps 換算幀數，
120 fps 為 100、80 幀），覆蓋原本的 CF/CS 影片後重新處理。原始影片與整段的處理結果搬到 full_clip/ 子資料夾保留。
"""
import csv
import os
import shutil

import cv2
import pandas as pd

from utils.phase_detect2d import PhaseDetector

# 錄影頁在 60 fps 保留擊球前 50、後 40 幀；影片可能是 120、160 fps，改用秒數換算成幀數
PRE_IMPACT_S = 50 / 60
POST_IMPACT_S = 40 / 60
DEFAULT_FPS = 60.0
FULL_CLIP_DIR = "full_clip"  # 原始長片段的備份；批量處理、找錄影時略過（paths.NON_HITTER_DIRS）
FRAME_TIMES_FILE = "frame_times.csv"


def window_frames(fps: float):
    """(擊球前幾幀, 擊球後幾幀)：60 fps 為 (50, 40)、120 fps 為 (100, 80)"""
    fps = fps if fps and fps > 0 else DEFAULT_FPS
    return int(round(PRE_IMPACT_S * fps)), int(round(POST_IMPACT_S * fps))


def clip_frames(fps: float) -> int:
    """標準錄影長度的幀數（60 fps 為 90）"""
    return sum(window_frames(fps))


def detect_impact(person_side_df, bat_side_df, batter_index: int):
    """側面骨架與球棒遮罩 → 擊球幀（與分析頁 detectPhases 有遮罩時同一套）；找不到回傳 None"""
    if person_side_df is None or person_side_df.empty or bat_side_df is None or bat_side_df.empty:
        return None
    result = PhaseDetector(person_df=person_side_df, batter_index=batter_index, ball_df=pd.DataFrame(),
                           bat_df=bat_side_df, land_hold=3).detect_phases()
    return None if result is None or result.F_impact is None else int(result.F_impact)


def trim_window(impact: int, total: int, fps: float = DEFAULT_FPS):
    """[起, 迄) 幀範圍：擊球前 PRE_IMPACT_S、後 POST_IMPACT_S 秒；靠近頭尾時整段往內移，長度不變"""
    pre, post = window_frames(fps)
    lo = min(max(impact - pre, 0), max(total - pre - post, 0))
    return lo, min(lo + pre + post, total)


def _unique_dir(path):
    if not os.path.exists(path):
        return path
    k = 2
    while os.path.exists(f"{path}_{k}"):
        k += 1
    return f"{path}_{k}"


def _write_range(src, dst, lo, hi):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    written = 0
    for _ in range(lo, hi):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    return written


def _trim_frame_times(src, dst, lo, hi):
    """frame_times.csv 只留 [lo, hi)，幀號從 0 重新編，時間從第一幀起算"""
    with open(src, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = [r for r in reader if lo <= int(r["frame"]) < hi]
    if not rows:
        return
    t0 = float(rows[0]["time_sec"])
    with open(dst, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "frame": int(r["frame"]) - lo, "time_sec": f"{float(r['time_sec']) - t0:.6f}"})


def trim_clip(cf_path, cs_path, lo, hi, result_folder=None, note=""):
    """把 CF/CS 影片（與 frame_times.csv）裁成 [lo, hi)，覆蓋原檔；原檔與整段的處理結果資料夾搬到 full_clip/。
    回傳裁好的 (cf_path, cs_path)；純側面錄影 cf_path 為 None"""
    folder = os.path.dirname(cf_path or cs_path)
    backup = _unique_dir(os.path.join(folder, FULL_CLIP_DIR))
    os.makedirs(backup)
    for path in filter(None, (cf_path, cs_path)):
        kept = os.path.join(backup, os.path.basename(path))
        shutil.move(path, kept)
        if _write_range(kept, path, lo, hi) != hi - lo:
            print(f"[裁剪] {os.path.basename(path)} 讀到的幀數比預期少")
    times = os.path.join(folder, FRAME_TIMES_FILE)
    if os.path.isfile(times):
        kept = os.path.join(backup, FRAME_TIMES_FILE)
        shutil.move(times, kept)
        _trim_frame_times(kept, times, lo, hi)
    if result_folder and os.path.isdir(result_folder):
        shutil.move(result_folder, os.path.join(backup, os.path.basename(result_folder)))
    with open(os.path.join(backup, "trim.txt"), "w", encoding="utf-8") as f:
        f.write(f"原始片段的第 {lo}～{hi - 1} 幀{note}\n")
    print(f"[裁剪] {folder}：保留第 {lo}～{hi - 1} 幀{note}，原始影片在 {backup}")
    return cf_path, cs_path


def result_folder_of(cf_path, cs_path=None):
    """處理結果子資料夾：影片同一層、以正面影片檔名命名；純側面錄影以側面影片檔名命名（VideoLoaderDual.saveResults）"""
    path = cf_path or cs_path
    return os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0])


def video_info(path):
    """(幀數, fps)"""
    cap = cv2.VideoCapture(path)
    info = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return info


def _read_records(path):
    try:
        return pd.read_json(path, orient="records") if os.path.isfile(path) else pd.DataFrame()
    except ValueError:
        return pd.DataFrame()


def trim_if_long(cf_path, cs_path, batter_index: int, person_side_df=None, bat_side_df=None) -> bool:
    """片段超過標準長度（依 fps 換算，60 fps 為 90 幀）就裁剪。沒給側面偵測結果時讀先前的處理結果（CF_／CS_ 子資料夾）；
    找不到擊球點就不裁。有裁剪回傳 True（之後要重新處理）"""
    infos = [video_info(p) for p in (cf_path, cs_path) if p]
    total = min(n for n, _ in infos)
    fps = infos[-1][1]  # 側面影片的 fps
    if total <= clip_frames(fps):
        return False
    result = result_folder_of(cf_path, cs_path)
    if person_side_df is None or bat_side_df is None:
        person_side_df = _read_records(os.path.join(result, "person_side.json"))
        bat_side_df = _read_records(os.path.join(result, "bat_side.json"))
        if person_side_df.empty:
            return False  # 還沒處理過：整段處理完會再來一次（分析頁 trimLongClip）
    impact = detect_impact(person_side_df, bat_side_df, batter_index)
    if impact is None:
        print(f"[裁剪] {os.path.dirname(cs_path)}：{total} 幀但找不到擊球點，維持整段")
        return False
    lo, hi = trim_window(impact, total, fps)
    pre, post = window_frames(fps)
    try:
        trim_clip(cf_path, cs_path, lo, hi, result,
                  note=f"（{fps:.0f} fps，擊球前 {pre}、後 {post} 幀 = {PRE_IMPACT_S:.2f}、{POST_IMPACT_S:.2f} 秒）")
    except OSError as e:
        print(f"[裁剪] {os.path.dirname(cs_path)} 失敗：{e}")
        return False
    return True
