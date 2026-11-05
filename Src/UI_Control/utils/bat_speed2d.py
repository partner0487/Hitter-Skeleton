"""2D 揮棒速度與攻擊角度，採用 Blast Motion 的定義（以 Blast 感測器為基準）：

- Bat Speed：擊球瞬間「甜蜜點」的速度；甜蜜點在距棒頭末端 6 吋（0.1524 m）處
- Attack Angle：擊球瞬間甜蜜點移動方向與水平的夾角，向上為正

做法（側面單一相機）：
- 棒頭／握把端：PhaseDetector.bat_head_track()（手腕分棒頭、排除跳動與沿用的遮罩），
  甜蜜點 = 棒頭往握把端方向，沿畫面上的球棒線段取棒長比例 0.1524 / BAT_LENGTH_M 的位置
- 擊球瞬間（impact_frames）：2D 用擊球點幀（缺時往前最多 MAX_GAP 幀）與它前一個有效幀，即擊球前最後一段。
  3D（bat_speed3d）改用擊球點前後各 IMPACT_WINDOW_3D 幀，比較穩；2D 不這樣做，
  因為擊球後球棒往側面相機揮，畫面上幾乎沒有位移（WeiShu 15 支 2D 速度會只剩 3D 的 0.46～0.93 倍）
- 時間：用 frame_times.csv 在擊球點前後 TIME_WINDOW 幀的平均幀間隔（可反映掉幀）；
  不用單一間隔，因為時間戳是電腦收到影像的時間，相鄰間隔在約 10／24 ms 交錯抖動。沒有檔案時用平均 fps
- 內參：有側面相機校正時先去畸變
- 比例尺：單一相機無法得知棒子離相機多遠（外參只描述兩台相機的相對位置，沒有打者位置），
  所以用已知棒長換算：球棒與畫面平行時的像素長度（遮罩兩端距離取 95 百分位）= BAT_LENGTH_M，
  等同針孔模型 m/px = Z / f、深度 Z 由棒長估計。
  注意：遮罩只有手腕前露出的那段（握把與棒尾被雙手遮住，3D 實測約 0.66 m），所以照幾何應該用 0.66 m；
  但 2D 只量得到球棒在畫面平面上的移動，往相機方向的分量看不到，速度本來就偏低。兩者剛好互相抵銷：
  WeiShu 21 支用 0.84 m 時 2D／3D 速度中位數 0.95，改用 0.66 m 反而變 0.71。所以 BAT_LENGTH_M 保留 0.84，
  視為經驗校正值（打者有設定棒長時依比例換算，見 utils/bat_profile.py）
- 攻擊角度假設側面畫面的水平方向即投手方向（相機水平擺放）
"""
import csv
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd

BAT_LENGTH_M = 0.84             # 約 33 吋
SWEET_SPOT_FROM_TIP_M = 0.1524  # Blast：距棒頭末端 6 吋
MAX_GAP = 3                     # 擊球前一個有效幀最多往前找幾幀
IMPACT_WINDOW_3D = 1            # 3D 擊球瞬間速度取擊球點前後各幾幀（見 bat_speed3d）
TIME_WINDOW = 10                # 幀間隔取擊球點前後幾幀的平均


@dataclass
class BlastMetrics2D:
    speed_kmh: float
    attack_angle_deg: float
    from_frame: int
    to_frame: int
    duration_s: float
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


def impact_frames(frames, f_impact: int, window: int = 0) -> Optional[list]:
    """算擊球瞬間速度用的有效幀（由小到大）：擊球點前後各 window 幀內；window 為 0 或不足兩幀時
    用擊球點（缺時往前最多 MAX_GAP 幀）與它前一個有效幀（間隔 ≤ MAX_GAP），即擊球前最後一段；都沒有回傳 None"""
    frames = sorted(int(f) for f in frames)
    window = [f for f in frames if f_impact - window <= f <= f_impact + window]
    if len(window) >= 2:
        return window
    to = [f for f in frames if f_impact - MAX_GAP <= f <= f_impact]
    if not to:
        return None
    before = [f for f in frames if to[-1] - MAX_GAP <= f < to[-1]]
    return [before[-1], to[-1]] if before else None


def path_and_chord(points: np.ndarray):
    """(沿各點的路徑長, 頭尾連線向量)"""
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()), points[-1] - points[0]


def _undistort(points: np.ndarray, K, dist) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def blast_metrics_2d(track: pd.DataFrame, f_impact: Optional[int], fps: float,
                     frame_times: dict = None, K=None, dist=None,
                     bat_length_m: float = BAT_LENGTH_M) -> Optional[BlastMetrics2D]:
    """track 為 bat_head_track() 的結果；資料不足回傳 None"""
    if track is None or track.empty or f_impact is None:
        return None
    valid = track[(track["source"] == "mask") & ~track["repeated"] & ~track["rejected"]]
    if valid.empty:
        return None

    undistorted = K is not None and dist is not None
    fix = (lambda p: _undistort(p, K, dist)) if undistorted else (lambda p: np.asarray(p, dtype=np.float64))
    head = fix(valid[["head_x", "head_y"]].to_numpy())
    end1 = fix(valid[["end1_x", "end1_y"]].to_numpy())
    end2 = fix(valid[["end2_x", "end2_y"]].to_numpy())
    # 握把端 = 兩端中離棒頭較遠的那端
    grip = np.where((np.linalg.norm(end1 - head, axis=1) > np.linalg.norm(end2 - head, axis=1))[:, None], end1, end2)

    bat_px = float(np.percentile(np.linalg.norm(end1 - end2, axis=1), 95))
    if bat_px <= 0:
        return None
    m_per_px = bat_length_m / bat_px
    sweet = head + (grip - head) * (SWEET_SPOT_FROM_TIP_M / bat_length_m)
    frames = valid["frame"].to_numpy()

    used = impact_frames(frames, f_impact)
    if used is None:
        return None
    index = {int(f): i for i, f in enumerate(frames)}
    f_from, f_to = used[0], used[-1]
    path_px, d = path_and_chord(sweet[[index[f] for f in used]])

    frame_times = frame_times or {}
    near = sorted(f for f in frame_times if f_from - TIME_WINDOW <= f <= f_to + TIME_WINDOW)
    if len(near) >= 2 and frame_times[near[-1]] > frame_times[near[0]]:
        frame_dt = (frame_times[near[-1]] - frame_times[near[0]]) / (near[-1] - near[0])
        duration, timed_by = (f_to - f_from) * frame_dt, "frame_times"
    else:
        duration, timed_by = (f_to - f_from) / fps, "fps"

    speed_kmh = path_px * m_per_px / duration * 3.6
    attack = float(np.degrees(np.arctan2(-d[1], abs(d[0]))))  # 畫面 y 向下，往上為正
    return BlastMetrics2D(speed_kmh=speed_kmh, attack_angle_deg=attack, from_frame=f_from, to_frame=f_to,
                          duration_s=duration, bat_px=bat_px, m_per_px=m_per_px,
                          undistorted=undistorted, timed_by=timed_by)
