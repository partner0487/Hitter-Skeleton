"""側面 2D 指標：2D 頁（main.py、main3d 錄影頁）與分析頁的純側面模式（只有側面影片的錄影）共用。

- 比例尺：有正面骨架時用校正三角化的打者深度；沒有時用球棒像素長度對應實際棒長
- 揮棒速度、攻擊角度：Blast Motion 定義，擊球瞬間甜蜜點的速度與方向（utils/bat_speed2d.py）
- 重心速度（離地、著地）、跨步距離（著地）：側面關鍵點換算成公尺（utils/body_metrics.py）
- 前、後膝角度（擊球）：髖-膝-腳跟在側面畫面上的夾角
"""
from typing import Optional, Tuple

import numpy as np

from utils.bat_speed2d import BlastMetrics2D, blast_metrics_2d
from utils.body_metrics import (SideScale, frame_interval, keypoints_by_frame, pelvis_speed, side_scale_from_stereo,
                                stride_length, undistort_keypoints)


def _K_dist(calib):
    return (calib.K_side, calib.dist_side) if calib is not None else (None, None)


def bat_metrics_2d(bat_track, impact_frame, fps: float, frame_times: Optional[dict], calib, bat) -> Optional[BlastMetrics2D]:
    """揮棒速度與攻擊角度；bat 為打者球棒設定（bat_profile.bat_for_clip），棒長用來把像素換算成公尺"""
    K, dist = _K_dist(calib)
    return blast_metrics_2d(bat_track, impact_frame, fps=fps, frame_times=frame_times, K=K, dist=dist,
                            bat_length_m=bat.length_m)


def side_scale(folder, calib, bat_track, impact_frame, fps: float, bat) -> Optional[SideScale]:
    """側面畫面的公尺換算：有正面骨架時用校正三角化的打者深度，否則用棒長（與揮棒速度相同）"""
    scale = side_scale_from_stereo(folder, calib)
    if scale is not None:
        return scale
    K, dist = _K_dist(calib)
    result = blast_metrics_2d(bat_track, impact_frame, fps=fps, K=K, dist=dist, bat_length_m=bat.length_m)
    if result is None:
        return None
    return SideScale(m_per_px=result.m_per_px, depth_m=None, source=f"{bat.describe()}估計（這支影片沒有正面骨架）")


def body_metrics_2d(person_side_df, phase_result, scale: SideScale, calib, fps: float,
                    frame_times: Optional[dict], is_righty: bool) -> dict:
    """{cog_liftoff, cog_landing: m/s, stride_cm}；量不到的為 None"""
    K, dist = _K_dist(calib)
    kpts = undistort_keypoints(keypoints_by_frame(person_side_df), K, dist)

    def speed(frame):
        return pelvis_speed(kpts, frame, frame_interval(fps, frame_times, frame), scale.m_per_px)

    f_land = phase_result.F_landing
    stride_m = stride_length(kpts.get(f_land), is_righty, scale.m_per_px) if f_land is not None else None
    return {
        "cog_liftoff": speed(phase_result.F_liftoff),
        "cog_landing": speed(f_land),
        "stride_cm": stride_m * 100 if stride_m is not None else None,
    }


def _angle(p1, p2, p3) -> float:
    """p1-p2-p3 夾角（以 p2 為頂點），0～180°"""
    ba = np.asarray(p1, dtype=float) - np.asarray(p2, dtype=float)
    bc = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def leg_joints(is_righty: bool) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """(前腳, 後腳) 的 (髖, 膝, 腳跟) 關鍵點；右打前腳是左腳"""
    left, right = (11, 13, 24), (12, 14, 25)
    return (left, right) if is_righty else (right, left)


def knee_angles_2d(k, is_righty: bool) -> Tuple[Optional[float], Optional[float]]:
    """擊球時前、後膝角度（像素座標取整，與關鍵幀圖上畫的線一致）"""
    if k is None or len(k) == 0:
        return None, None
    k = np.asarray(k)
    return tuple(_angle(*(k[i][:2].astype(int) for i in leg)) for leg in leg_joints(is_righty))
