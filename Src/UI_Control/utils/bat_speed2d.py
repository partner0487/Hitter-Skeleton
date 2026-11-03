"""2D 揮棒速度：前腳著地到擊球點之間，棒頭在側面畫面走過的路徑 ÷ 實際經過時間。

- 棒頭位置：PhaseDetector.bat_head_track()（手腕分棒頭、排除跳動與沿用的遮罩）
- 時間：錄影時存的 frame_times.csv 每幀時間戳（實際 fps 會在 58～62 間浮動）；沒有時用平均 fps
- 內參：有側面相機校正時先去畸變（8mm 鏡頭邊緣畸變明顯），再換算
- 比例尺：單一相機無法得知棒子離相機多遠（外參只描述兩台相機的相對位置，沒有打者位置），
  所以用已知棒長換算：球棒與畫面平行時的像素長度（遮罩兩端距離取 95 百分位）= BAT_LENGTH_M。
  等同針孔模型 m/px = Z / f，深度 Z 由棒長估計。
"""
import csv
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd

BAT_LENGTH_M = 0.84  # 約 33 吋


@dataclass
class BatSpeed2D:
    speed_kmh: float
    path_m: float
    duration_s: float
    first_frame: int
    last_frame: int
    n_points: int
    bat_px: float
    m_per_px: float
    undistorted: bool
    timed_by: str  # "frame_times" 或 "fps"


def read_frame_times(folder) -> dict:
    """{frame: time_sec}；沒有 frame_times.csv 回傳空 dict"""
    if not folder:
        return {}
    try:
        with open(os.path.join(folder, "frame_times.csv"), encoding="utf-8") as f:
            return {int(row["frame"]): float(row["time_sec"]) for row in csv.DictReader(f)}
    except (OSError, ValueError, KeyError):
        return {}


def _undistort(points: np.ndarray, K, dist) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def bat_speed_2d(track: pd.DataFrame, f_landing: Optional[int], f_impact: Optional[int],
                 fps: float, frame_times: dict = None, K=None, dist=None,
                 bat_length_m: float = BAT_LENGTH_M) -> Optional[BatSpeed2D]:
    """track 為 bat_head_track() 的結果；資料不足回傳 None"""
    if track is None or track.empty or f_landing is None or f_impact is None or f_impact <= f_landing:
        return None
    valid = track[(track["source"] == "mask") & ~track["repeated"] & ~track["rejected"]]
    if valid.empty:
        return None

    undistorted = K is not None and dist is not None
    fix = (lambda p: _undistort(p, K, dist)) if undistorted else (lambda p: np.asarray(p, dtype=np.float64))

    # 比例尺：球棒最接近與畫面平行時的長度
    end1 = fix(valid[["end1_x", "end1_y"]].to_numpy())
    end2 = fix(valid[["end2_x", "end2_y"]].to_numpy())
    bat_px = float(np.percentile(np.linalg.norm(end1 - end2, axis=1), 95))
    if bat_px <= 0:
        return None
    m_per_px = bat_length_m / bat_px

    seg = valid[(valid["frame"] >= f_landing) & (valid["frame"] <= f_impact)]
    if len(seg) < 2:
        return None
    heads = fix(seg[["head_x", "head_y"]].to_numpy())
    path_m = float(np.linalg.norm(np.diff(heads, axis=0), axis=1).sum()) * m_per_px

    first, last = int(seg["frame"].iloc[0]), int(seg["frame"].iloc[-1])
    frame_times = frame_times or {}
    if first in frame_times and last in frame_times and frame_times[last] > frame_times[first]:
        duration, timed_by = frame_times[last] - frame_times[first], "frame_times"
    else:
        duration, timed_by = (last - first) / fps, "fps"
    if duration <= 0:
        return None
    return BatSpeed2D(speed_kmh=path_m / duration * 3.6, path_m=path_m, duration_s=duration,
                      first_frame=first, last_frame=last, n_points=len(seg), bat_px=bat_px,
                      m_per_px=m_per_px, undistorted=undistorted, timed_by=timed_by)
