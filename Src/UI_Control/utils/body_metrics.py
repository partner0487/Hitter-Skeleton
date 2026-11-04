"""打者身體指標（2D 側面、3D 共用）：重心速度、跨步距離、肩髖分離角，以及 2D 的公尺換算。

關節為 Halpe26：肩 5/6、髖 11/12、腳跟 24/25（左/右）。
- 重心：兩髖中點。速度用前後各 SPEED_HALF_WINDOW 幀的位移（中央差分）；
  單幀差分在 60 fps 下骨架抖動 1 px 就差 0.2 m/s
- 跨步距離：著地時前後腳跟的水平距離（2D 取畫面水平方向、3D 取 XZ 平面）
- 肩髖分離角：髖連線與肩連線在水平面（側面相機座標的 XZ，假設相機水平）上的夾角，
  正值 = 髖比肩轉得多（肩落後髖、蓄力），負值 = 肩已轉過髖。
  轉動方向取著地 → 擊球後髖的轉動方向。側面遮擋時三角化的髖寬、肩寬會縮短、連線方向不可信：
  繞垂直軸轉動不會改變水平投影的寬度，所以寬度小於整段中位數 WIDTH_MIN_RATIO 倍的幀不採用，
  再取 SEPARATION_SMOOTH 幀中位數去掉單幀跳動
- 2D 公尺換算：單一相機不知道打者離相機多遠。同一支影片有正面骨架時，用雙相機校正（外參）
  三角化兩髖、取深度中位數 Z，針孔模型 m/px = Z / fx；沒有正面骨架才由呼叫端改用棒長估計
"""
import glob
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_HEEL, R_HEEL = 24, 25

SPEED_HALF_WINDOW = 2
FRAME_TIME_WINDOW = 10       # 有 frame_times.csv 時，幀間隔取前後幾幀的平均（與 bat_speed2d 相同）
WIDTH_MIN_RATIO = 0.7
SEPARATION_SMOOTH = 5
ROTATION_MIN_DEG = 20.0      # 著地 → 擊球後髖轉不到這麼多時，轉動方向改用打擊方向判斷
ROTATION_LOOKAHEAD = 10      # 轉動方向看到擊球後幾幀
HORIZONTAL_3D = (0, 2)       # 側面相機座標 X 右、Y 下、Z 深度


def front_back_heels(is_righty: bool) -> Tuple[int, int]:
    """(前腳跟, 後腳跟)：右打前腳是左腳"""
    return (L_HEEL, R_HEEL) if is_righty else (R_HEEL, L_HEEL)


def keypoints_by_frame(df: Optional[pd.DataFrame], column: str = "keypoints") -> Dict[int, np.ndarray]:
    """{frame_number: (N, D) 座標}；2D 只取 x, y"""
    if df is None or df.empty or column not in df:
        return {}
    out = {}
    for f, k in zip(df["frame_number"], df[column]):
        k = np.asarray(k, dtype=np.float64)
        if k.ndim == 2 and k.shape[0] > R_HEEL:
            out[int(f)] = k[:, :3] if column == "keypoints_3d" else k[:, :2]
    return out


def undistort_keypoints(kpts: Dict[int, np.ndarray], K, dist) -> Dict[int, np.ndarray]:
    """2D 關鍵點去畸變（像素座標），沒有校正時原樣回傳"""
    if K is None or dist is None:
        return kpts
    return {f: cv2.undistortPoints(k.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2) for f, k in kpts.items()}


def frame_interval(fps: float, frame_times: Optional[dict] = None, frame: Optional[int] = None) -> float:
    """frame 附近的平均幀間隔（秒）：有 frame_times.csv 時用實際時間（可反映掉幀），否則 1 / fps"""
    if frame_times and frame is not None:
        near = sorted(f for f in frame_times if abs(f - frame) <= FRAME_TIME_WINDOW)
        if len(near) >= 2 and frame_times[near[-1]] > frame_times[near[0]]:
            return (frame_times[near[-1]] - frame_times[near[0]]) / (near[-1] - near[0])
    return 1.0 / fps if fps else 0.0


def pelvis_speed(kpts: Dict[int, np.ndarray], frame: Optional[int], frame_dt: float,
                 scale: float = 1.0, axes=None) -> Optional[float]:
    """frame 時兩髖中點的速度（座標單位 × scale / 秒）；axes 指定只看哪些軸（例如 3D 水平面）"""
    if frame is None or frame_dt <= 0:
        return None
    before = [f for f in range(frame - SPEED_HALF_WINDOW, frame + 1) if f in kpts]
    after = [f for f in range(frame, frame + SPEED_HALF_WINDOW + 1) if f in kpts]
    if not before or not after or after[-1] <= before[0]:
        return None
    f0, f1 = before[0], after[-1]
    center = lambda k: (k[L_HIP] + k[R_HIP]) / 2
    d = center(kpts[f1]) - center(kpts[f0])
    if axes is not None:
        d = d[list(axes)]
    return float(np.linalg.norm(d)) * scale / ((f1 - f0) * frame_dt)


def stride_length(k: Optional[np.ndarray], is_righty: bool, scale: float = 1.0, axes=(0,)) -> Optional[float]:
    """前後腳跟的水平距離 × scale；2D 預設只取畫面 x（投手方向），3D 傳 HORIZONTAL_3D"""
    if k is None:
        return None
    front, back = front_back_heels(is_righty)
    return float(np.linalg.norm((k[front] - k[back])[list(axes)])) * scale


def _wrap(deg):
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def separation_series(kpts3d: Dict[int, np.ndarray], f_landing: Optional[int], f_impact: Optional[int],
                      is_righty: bool) -> Dict[int, float]:
    """每幀的肩髖分離角（度，正 = 肩落後髖）；遮擋而不可信的幀不列入"""
    frames = np.array(sorted(kpts3d))
    if len(frames) < SEPARATION_SMOOTH:
        return {}
    pts = np.stack([kpts3d[f] for f in frames])
    v_hip = (pts[:, R_HIP] - pts[:, L_HIP])[:, list(HORIZONTAL_3D)]
    v_shd = (pts[:, R_SHOULDER] - pts[:, L_SHOULDER])[:, list(HORIZONTAL_3D)]
    hip_w, shd_w = np.linalg.norm(v_hip, axis=1), np.linalg.norm(v_shd, axis=1)
    reliable = (hip_w >= WIDTH_MIN_RATIO * np.median(hip_w)) & (shd_w >= WIDTH_MIN_RATIO * np.median(shd_w))

    theta_hip = np.degrees(np.unwrap(np.arctan2(v_hip[:, 1], v_hip[:, 0])))
    theta_shd = np.degrees(np.unwrap(np.arctan2(v_shd[:, 1], v_shd[:, 0])))

    # 揮棒時髖往投手方向轉，由此決定正負號；資料不足時右打為正（側面相機座標下實測）、左打相反
    direction = 1.0 if is_righty else -1.0
    if f_landing is not None and f_impact is not None:
        ok = frames[reliable]
        start = ok[ok >= f_landing]
        end = ok[ok <= f_impact + ROTATION_LOOKAHEAD]
        if len(start) and len(end) and end[-1] > start[0]:
            idx = {f: i for i, f in enumerate(frames)}
            turned = theta_hip[idx[end[-1]]] - theta_hip[idx[start[0]]]
            if abs(turned) >= ROTATION_MIN_DEG:
                direction = float(np.sign(turned))

    sep = np.where(reliable, direction * _wrap(theta_hip - theta_shd), np.nan)
    smooth = pd.Series(sep).rolling(SEPARATION_SMOOTH, center=True, min_periods=3).median().to_numpy()
    return {int(f): float(v) for f, v in zip(frames, smooth) if np.isfinite(v)}


def peak_separation(series: Dict[int, float], f_start: Optional[int], f_end: Optional[int]) -> Optional[Tuple[float, int]]:
    """f_start～f_end（離地～擊球）之間最大的分離角與它的幀"""
    if f_start is None or f_end is None:
        return None
    window = {f: v for f, v in series.items() if f_start <= f <= f_end}
    if not window:
        return None
    f = max(window, key=window.get)
    return window[f], f


@dataclass
class SideScale:
    m_per_px: float
    depth_m: Optional[float]   # 由棒長估計時為 None
    source: str


def _front_side_pairs(folder):
    """同一支影片的 (person_side.json, person_front.json)：資料夾本身或 CF_* 子資料夾"""
    for d in [folder] + sorted(glob.glob(os.path.join(folder, "CF_*"))):
        side, front = os.path.join(d, "person_side.json"), os.path.join(d, "person_front.json")
        if os.path.isfile(side) and os.path.isfile(front):
            yield side, front


def side_scale_from_stereo(folder: Optional[str], calib) -> Optional[SideScale]:
    """有正面骨架時，用校正三角化兩髖中點、取深度中位數換算側面畫面的 m/px；沒有資料回傳 None"""
    if not folder or calib is None:
        return None
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([calib.R, calib.t])
    for side_path, front_path in _front_side_pairs(folder):
        try:
            side = keypoints_by_frame(pd.read_json(side_path, orient="records"))
            front = keypoints_by_frame(pd.read_json(front_path, orient="records"))
        except (ValueError, KeyError) as e:
            print(f"[body_metrics] {side_path} 無法讀取：{e}")
            continue
        frames = sorted(set(side) & set(front))
        if not frames:
            continue
        hips = [L_HIP, R_HIP]
        a = np.concatenate([side[f][hips] for f in frames])
        b = np.concatenate([front[f][hips] for f in frames])
        a = cv2.undistortPoints(a.reshape(-1, 1, 2), calib.K_side, calib.dist_side).reshape(-1, 2).T
        b = cv2.undistortPoints(b.reshape(-1, 1, 2), calib.K_front, calib.dist_front).reshape(-1, 2).T
        X = cv2.triangulatePoints(P1, P2, a, b)
        depth = X[2] / X[3]
        depth = depth[np.isfinite(depth) & (depth > 0)]
        if len(depth) == 0:
            continue
        z = float(np.median(depth))
        return SideScale(m_per_px=z / float(calib.K_side[0, 0]), depth_m=z,
                         source=f"雙相機校正三角化的打者深度 {z:.2f} m")
    return None
