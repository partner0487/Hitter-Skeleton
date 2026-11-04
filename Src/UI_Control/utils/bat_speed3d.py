"""3D 揮棒速度、攻擊角度、垂直棒角，採用與 2D 頁相同的 Blast Motion 定義（見 utils/bat_speed2d.py）：

- Bat Speed：擊球瞬間甜蜜點（距棒頭末端 6 吋）的速度
- Attack Angle：甜蜜點移動方向與水平面的夾角，向上為正
- Vertical Bat Angle：擊球時球棒（握把 → 棒頭）與水平面的夾角，棒頭低於手為負

做法：
- 側面、正面各自用 PhaseDetector.bat_head_track() 分棒頭與握把端（手腕分棒頭，排除沿用、跳動、誤抓手臂的遮罩），
  兩邊都有效的幀才三角化；BatAnalyzer 的 batht 沒有這些過濾，收棒時常暴衝
- 三角化的棒長偏離整段中位數 LENGTH_TOLERANCE 以上的幀，表示有一端對錯，不採用
  （手會遮住握把，遮罩兩端的 3D 距離約 0.65～0.72 m，比實際棒長短，但同一支影片應該固定）
- 甜蜜點 = 棒頭沿球棒方向往握把 0.1524 m，不用遮罩長度換算比例
- 擊球瞬間：擊球點幀（缺時往前最多 MAX_GAP 幀）與它前一個有效幀之間的位移，與 2D 相同
- 鉛垂方向用 body_metrics.ground_up()（腳點擬合的地面法向量），沒有時用側面相機的 -Y
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.bat_speed2d import MAX_GAP, SWEET_SPOT_FROM_TIP_M
from utils.body_metrics import CAMERA_UP, frame_interval

LENGTH_TOLERANCE = 0.25


@dataclass
class BlastMetrics3D:
    speed_kmh: float
    attack_angle_deg: float
    vertical_bat_angle_deg: Optional[float]
    from_frame: int
    to_frame: int
    duration_s: float
    bat_m: float          # 遮罩兩端 3D 距離的中位數
    frames_used: int


def head_grip_by_frame(track: Optional[pd.DataFrame]) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """bat_head_track() 的有效遮罩 → {frame: (棒頭, 握把端)}（像素座標）"""
    if track is None or track.empty or "end1_x" not in track:
        return {}
    valid = track[(track["source"] == "mask") & ~track["repeated"] & ~track["rejected"]]
    out = {}
    for row in valid.itertuples():
        head = np.array([row.head_x, row.head_y], dtype=np.float64)
        e1 = np.array([row.end1_x, row.end1_y], dtype=np.float64)
        e2 = np.array([row.end2_x, row.end2_y], dtype=np.float64)
        grip = e1 if np.linalg.norm(e1 - head) > np.linalg.norm(e2 - head) else e2
        out[int(row.frame)] = (head, grip)
    return out


def bat_track_3d(side_track, front_track, triangulate: Callable) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], float]:
    """({frame: (棒頭 3D, 握把端 3D)}, 棒長中位數)；triangulate(pts_side, pts_front) 回傳 (N, 3)"""
    side, front = head_grip_by_frame(side_track), head_grip_by_frame(front_track)
    frames = sorted(set(side) & set(front))
    if not frames:
        return {}, 0.0
    pts = triangulate(np.concatenate([np.stack(side[f]) for f in frames]),
                      np.concatenate([np.stack(front[f]) for f in frames]))
    if pts is None:
        return {}, 0.0
    pts = np.asarray(pts, dtype=np.float64).reshape(len(frames), 2, 3)
    lengths = np.linalg.norm(pts[:, 0] - pts[:, 1], axis=1)
    median = float(np.median(lengths))
    keep = np.abs(lengths - median) <= LENGTH_TOLERANCE * median
    return {f: (p[0], p[1]) for f, p, k in zip(frames, pts, keep) if k}, median


def _sweet_spot(head, grip):
    direction = (grip - head) / np.linalg.norm(grip - head)
    return head + direction * SWEET_SPOT_FROM_TIP_M


def _elevation_deg(v, up):
    """向量與水平面的夾角，朝上為正"""
    return float(np.degrees(np.arcsin(np.clip(v @ up / np.linalg.norm(v), -1.0, 1.0))))


def blast_metrics_3d(track3d: Dict[int, Tuple[np.ndarray, np.ndarray]], bat_m: float, f_impact: Optional[int],
                     fps: float, frame_times: dict = None, up: Optional[np.ndarray] = None) -> Optional[BlastMetrics3D]:
    """資料不足（擊球點附近兩台相機沒有同時偵測到球棒）回傳 None"""
    if not track3d or f_impact is None:
        return None
    frames = np.array(sorted(track3d))
    near = frames[(frames <= f_impact) & (frames >= f_impact - MAX_GAP)]
    if len(near) == 0:
        return None
    f_to = int(near[-1])
    before = frames[(frames < f_to) & (frames >= f_to - MAX_GAP)]
    if len(before) == 0:
        return None
    f_from = int(before[-1])

    d = _sweet_spot(*track3d[f_to]) - _sweet_spot(*track3d[f_from])
    duration = (f_to - f_from) * frame_interval(fps, frame_times, f_to)
    if duration <= 0:
        return None
    up = CAMERA_UP if up is None else up
    head, grip = track3d[f_to]
    return BlastMetrics3D(
        speed_kmh=float(np.linalg.norm(d)) / duration * 3.6,
        attack_angle_deg=_elevation_deg(d, up),
        vertical_bat_angle_deg=_elevation_deg(head - grip, up),
        from_frame=f_from, to_frame=f_to, duration_s=duration, bat_m=bat_m, frames_used=len(track3d),
    )
