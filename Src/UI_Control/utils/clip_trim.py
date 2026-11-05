"""長片段裁成標準錄影長度：擊球前 50 幀、擊球後 40 幀，與錄影頁 ContinuousRecorder 觸發前後保留的幀數相同。

在現場手動錄、事後自己剪的片段長短不一（例如 433 幀），會有兩個問題：
1. 片段裡有多次試揮，離地、著地、擊球容易抓到別次揮棒
2. 兩台相機的影格全部讀進記憶體（433 幀約 5.4 GB），處理與載入都慢

做法：先用整段的偵測結果找擊球點（與分析頁同一套），裁出 [擊球-50, 擊球+40) 共 90 幀，
覆蓋原本的 CF/CS 影片後重新處理。原始影片與整段的處理結果搬到 full_clip/ 子資料夾保留。
"""
import csv
import os
import shutil

import cv2
import pandas as pd

from utils.phase_detect2d import PhaseDetector

PRE_IMPACT_FRAMES = 50
POST_IMPACT_FRAMES = 40
CLIP_FRAMES = PRE_IMPACT_FRAMES + POST_IMPACT_FRAMES
FULL_CLIP_DIR = "full_clip"  # 原始長片段的備份；批量處理、找錄影時略過（paths.NON_HITTER_DIRS）
FRAME_TIMES_FILE = "frame_times.csv"


def detect_impact(person_side_df, bat_side_df, batter_index: int):
    """側面骨架與球棒遮罩 → 擊球幀（與分析頁 detectPhases 有遮罩時同一套）；找不到回傳 None"""
    if person_side_df is None or person_side_df.empty or bat_side_df is None or bat_side_df.empty:
        return None
    result = PhaseDetector(person_df=person_side_df, batter_index=batter_index, ball_df=pd.DataFrame(),
                           bat_df=bat_side_df, land_hold=3).detect_phases()
    return None if result is None or result.F_impact is None else int(result.F_impact)


def trim_window(impact: int, total: int):
    """[起, 迄) 幀範圍：擊球前 PRE_IMPACT_FRAMES、後 POST_IMPACT_FRAMES；靠近頭尾時整段往內移，長度不變"""
    lo = min(max(impact - PRE_IMPACT_FRAMES, 0), max(total - CLIP_FRAMES, 0))
    return lo, min(lo + CLIP_FRAMES, total)


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


def trim_clip(cf_path, cs_path, lo, hi, result_folder=None):
    """把 CF/CS 影片（與 frame_times.csv）裁成 [lo, hi)，覆蓋原檔；原檔與整段的處理結果資料夾搬到 full_clip/。
    回傳裁好的 (cf_path, cs_path)"""
    folder = os.path.dirname(cf_path)
    backup = _unique_dir(os.path.join(folder, FULL_CLIP_DIR))
    os.makedirs(backup)
    for path in (cf_path, cs_path):
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
        f.write(f"原始片段的第 {lo}～{hi - 1} 幀（擊球前 {PRE_IMPACT_FRAMES}、後 {POST_IMPACT_FRAMES} 幀）\n")
    print(f"[裁剪] {folder}：保留第 {lo}～{hi - 1} 幀，原始影片在 {backup}")
    return cf_path, cs_path


def result_folder_of(cf_path):
    """處理結果子資料夾：CF 影片同一層、以 CF 檔名命名（VideoLoaderDual.saveDualVideo）"""
    return os.path.join(os.path.dirname(cf_path), os.path.splitext(os.path.basename(cf_path))[0])


def frame_count(path) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _read_records(path):
    try:
        return pd.read_json(path, orient="records") if os.path.isfile(path) else pd.DataFrame()
    except ValueError:
        return pd.DataFrame()


def trim_if_long(cf_path, cs_path, batter_index: int, person_side_df=None, bat_side_df=None) -> bool:
    """片段超過 CLIP_FRAMES 幀就裁剪。沒給側面偵測結果時讀先前的處理結果（CF_ 子資料夾）；
    找不到擊球點就不裁。有裁剪回傳 True（之後要重新處理）"""
    total = min(frame_count(cf_path), frame_count(cs_path))
    if total <= CLIP_FRAMES:
        return False
    result = result_folder_of(cf_path)
    if person_side_df is None or bat_side_df is None:
        person_side_df = _read_records(os.path.join(result, "person_side.json"))
        bat_side_df = _read_records(os.path.join(result, "bat_side.json"))
        if person_side_df.empty:
            return False  # 還沒處理過：整段處理完會再來一次（分析頁 trimLongClip）
    impact = detect_impact(person_side_df, bat_side_df, batter_index)
    if impact is None:
        print(f"[裁剪] {os.path.dirname(cf_path)}：{total} 幀但找不到擊球點，維持整段")
        return False
    lo, hi = trim_window(impact, total)
    try:
        trim_clip(cf_path, cs_path, lo, hi, result)
    except OSError as e:
        print(f"[裁剪] {os.path.dirname(cf_path)} 失敗：{e}")
        return False
    return True
