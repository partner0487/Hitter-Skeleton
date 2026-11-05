"""打者身體指標（2D 側面、3D 共用）：重心速度、跨步距離、肩髖分離角，以及 2D 的公尺換算。

關節為 Halpe26：肩 5/6、髖 11/12、腳跟 24/25（左/右）。
- 重心：兩髖中點。速度用前後各 SPEED_HALF_WINDOW 幀的位移（中央差分）；
  單幀差分在 60 fps 下骨架抖動 1 px 就差 0.2 m/s
- 跨步距離：著地時前後腳跟的水平距離（2D 取畫面水平方向、3D 取水平面）
- 3D 的水平面：3D 座標是側面相機座標（Y 向下），相機不一定水平（WeiShu 2026-10-29 約往下俯 3～10°），
  所以用整段腳跟、腳趾落地時的點擬合地面，法向量當鉛垂方向（ground_up）；擬合不合理時退回相機的 XZ 平面
- 肩髖分離角：髖連線與肩連線在水平面上的夾角，
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
FOOT_POINTS = list(range(20, 26))  # 腳趾 20～23、腳跟 24～25
FOOT_LOW_PERCENTILE = 60     # 每個腳點取最低的 40% 幀，當作踩在地上
GROUND_MAX_TILT_DEG = 20.0   # 地面法向量與相機 -Y 的夾角超過這麼多視為擬合失敗
GROUND_MIN_SPREAD = 0.12     # 腳點在地面上兩個方向的分布比（第二／第一奇異值），太小表示點幾乎排成一線（WeiShu T座 H04 為 0.20）
GROUND_MAX_RESIDUAL_M = 0.03
CAMERA_UP = np.array([0.0, -1.0, 0.0])  # 側面相機座標 X 右、Y 下、Z 深度


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


def ground_up(kpts3d: Dict[int, np.ndarray]) -> Optional[np.ndarray]:
    """由腳點擬合地面，回傳朝上的單位法向量；擬合不合理時回傳 None（呼叫端用相機座標）"""
    if len(kpts3d) < 10:
        return None
    feet = np.stack([kpts3d[f][FOOT_POINTS] for f in sorted(kpts3d)])  # (F, 6, 3)
    up = CAMERA_UP
    for _ in range(2):  # 先用相機 Y 挑踩地的點，再用第一次擬合的地面重挑一次
        height = feet @ up
        low = [feet[height[:, j] <= np.percentile(height[:, j], 100 - FOOT_LOW_PERCENTILE), j]
               for j in range(feet.shape[1])]
        pts = np.concatenate(low)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 10:
            return None
        center = pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts - center, full_matrices=False)
        normal = vt[2] if vt[2] @ CAMERA_UP > 0 else -vt[2]
        tilt = np.degrees(np.arccos(np.clip(normal @ CAMERA_UP, -1, 1)))
        residual = float(np.median(np.abs((pts - center) @ normal)))
        if tilt > GROUND_MAX_TILT_DEG or sv[1] < GROUND_MIN_SPREAD * sv[0] or residual > GROUND_MAX_RESIDUAL_M:
            return None
        up = normal
    return up


def horizontal_basis(up: Optional[np.ndarray] = None) -> np.ndarray:
    """(3, 2)：水平面上兩個正交單位向量；第一個是相機 X 投影到水平面，up 為 None 時就是相機的 X、Z"""
    up = CAMERA_UP if up is None else up / np.linalg.norm(up)
    e1 = np.array([1.0, 0.0, 0.0]) - up[0] * up
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(up, e1)], axis=1)


def stride_length(k: Optional[np.ndarray], is_righty: bool, scale: float = 1.0, axes=(0,),
                  basis: Optional[np.ndarray] = None) -> Optional[float]:
    """前後腳跟的水平距離 × scale；2D 預設只取畫面 x（投手方向），3D 傳 horizontal_basis()"""
    if k is None:
        return None
    front, back = front_back_heels(is_righty)
    d = k[front] - k[back]
    d = d @ basis if basis is not None else d[list(axes)]
    return float(np.linalg.norm(d)) * scale


def _wrap(deg):
    return (np.asarray(deg) + 180.0) % 360.0 - 180.0


def separation_series(kpts3d: Dict[int, np.ndarray], f_landing: Optional[int], f_impact: Optional[int],
                      is_righty: bool, up: Optional[np.ndarray] = None) -> Dict[int, float]:
    """每幀的肩髖分離角（度，正 = 肩落後髖）；遮擋而不可信的幀不列入。up 為 ground_up() 的鉛垂方向"""
    frames = np.array(sorted(kpts3d))
    if len(frames) < SEPARATION_SMOOTH:
        return {}
    pts = np.stack([kpts3d[f] for f in frames])
    basis = horizontal_basis(up)
    v_hip = (pts[:, R_HIP] - pts[:, L_HIP]) @ basis
    v_shd = (pts[:, R_SHOULDER] - pts[:, L_SHOULDER]) @ basis
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


ROTATION_SMOOTH = 7          # 轉速：Savitzky–Golay 視窗（幀）
ROTATION_BEFORE_LANDING = 10  # 最大轉速的搜尋範圍：著地前幾幀 ～ 擊球後 2 幀
ROTATION_AFTER_IMPACT = 2


def _yaw_rate(pts, a, b, basis, reliable, frame_dt):
    """兩點連線在水平面上的轉速（度/秒）；不可信的幀先以前後內插補上，再平滑微分"""
    from scipy.signal import savgol_filter
    v = (pts[:, b] - pts[:, a]) @ basis
    ang = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    idx = np.arange(len(ang))
    if reliable.sum() >= 2 and not reliable.all():
        ang = np.interp(idx, idx[reliable], ang[reliable])
    return np.degrees(savgol_filter(ang, ROTATION_SMOOTH, 2, deriv=1, delta=frame_dt))


def rotation_speed_peaks(kpts3d: Dict[int, np.ndarray], f_landing: Optional[int], f_impact: Optional[int],
                         up: Optional[np.ndarray], frame_dt: float) -> Dict[str, Tuple[float, int]]:
    """{"pelvis": (最大轉速 °/s, 幀), "torso": (...)}：兩髖、兩肩連線在水平面上的轉速，
    取著地前 ROTATION_BEFORE_LANDING 幀到擊球後 ROTATION_AFTER_IMPACT 幀之間的最大值。
    只回報大小、不判斷峰值先後：60 fps 下骨盆、軀幹峰值常只差 1～2 幀，與三角化及平滑的誤差相當"""
    frames = np.array(sorted(kpts3d))
    if f_landing is None or f_impact is None or len(frames) < ROTATION_SMOOTH or frame_dt <= 0:
        return {}
    # 依幀的順序微分（3D 骨架通常每一幀都有，缺幀時視為相鄰）
    pts = np.stack([kpts3d[f] for f in frames])
    basis = horizontal_basis(up)
    out = {}
    for name, (a, b) in (("pelvis", (L_HIP, R_HIP)), ("torso", (L_SHOULDER, R_SHOULDER))):
        width = np.linalg.norm((pts[:, b] - pts[:, a]) @ basis, axis=1)
        reliable = width >= WIDTH_MIN_RATIO * np.median(width)
        rate = np.abs(_yaw_rate(pts, a, b, basis, reliable, frame_dt))
        window = (frames >= f_landing - ROTATION_BEFORE_LANDING) & (frames <= f_impact + ROTATION_AFTER_IMPACT)
        if window.any():
            i = int(np.argmax(np.where(window, rate, -1.0)))
            out[name] = (float(rate[i]), int(frames[i]))
    return out
