import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class PhaseResult:
    F0: int = 0
    F_liftoff: Optional[int] = None
    F_landing: Optional[int] = None
    F_impact: Optional[int] = None
    Fend: Optional[int] = None


class _SingleViewPhaseDetector:
    def __init__(
        self,
        person_df: pd.DataFrame,
        batter_index: int,              # 0 (RHH) or 1 (LHH)
        ball_df: pd.DataFrame = None,
        batht_df: pd.DataFrame = None,
        bat_df: pd.DataFrame = None,
        baseline_window: int = 10,
        th_lift: float = 5.0,
        th_land: float = 5.0,
        land_hold: int = 3,
        smooth_window: int = 5,
        is_front_view: bool = False,
    ):
        self.person_df = person_df
        self.batter_index = batter_index
        self.ball_df = ball_df
        # 相容兩版命名：3D 版使用 batht_df，2D 版使用 bat_df。
        self.batht_df = batht_df if batht_df is not None else bat_df
        self.bat_df = self.batht_df
        self.baseline_window = baseline_window
        self.th_lift = th_lift
        self.th_land = th_land
        self.land_hold = land_hold
        self.smooth_window = smooth_window
        self.is_front_view = is_front_view

        self.heel_frames, self.heel_vals = self._extract_heel_data(person_df, batter_index)
        self.heel_smooth = self._smooth(self.heel_vals)
        win = min(self.baseline_window, len(self.heel_smooth))
        self.baseline = float(np.median(self.heel_smooth[:win])) if win > 0 else 0.0

    def _extract_heel_data(self, df: pd.DataFrame, batter_index: int):
        if df is None or df.empty:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        target_idx = 24 if batter_index == 0 else 25
        side_str = "Left Heel (idx=24)" if target_idx == 24 else "Right Heel (idx=25)"
        print(f"[_extract_heel_data] Batter is {'RHH' if batter_index == 0 else 'LHH'}. Tracking front leg: {side_str}. df_len: {len(df)}")

        frames = []
        ys = []

        for _, row in df.iterrows():
            f = int(row["frame_number"])
            kpts = np.array(row["keypoints"])
            if kpts.shape[0] <= target_idx:
                continue
            
            y = float(kpts[target_idx][1])
            if y < 1e-3:
                continue

            frames.append(f)
            ys.append(y)

        if not ys:
             return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        frames = np.array(frames, dtype=np.int32)
        ys_arr = np.array(ys, dtype=np.float32)
        ground_y = np.max(ys_arr)
        heights = ground_y - ys_arr

        return frames, heights

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0 or self.smooth_window <= 1:
            return arr.copy()
        # kernel 不可比輸入資料還長，否則 np.convolve(mode="same")
        # 可能回傳比原始序列更長的陣列。
        k = min(int(self.smooth_window), len(arr))
        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(arr, kernel, mode="same")


    def _extract_ball_center_2d(self):
        """從 ball_df 抽出 {frame: [cx, cy]}；支援 bbox=[[x1,y1,x2,y2]]。"""
        centers = {}
        if self.ball_df is None or self.ball_df.empty:
            return centers

        for _, row in self.ball_df.iterrows():
            if "frame_number" not in row or row["bbox"] is None:
                continue
            try:
                bbox = row["bbox"]
                if isinstance(bbox, np.ndarray):
                    bbox = bbox.tolist()
                if not isinstance(bbox, (list, tuple)) or len(bbox) == 0:
                    continue
                box = bbox[0] if isinstance(bbox[0], (list, tuple, np.ndarray)) else bbox
                if len(box) < 4:
                    continue
                x1, y1, x2, y2 = map(float, box[:4])
                centers[int(row["frame_number"])] = np.array(
                    [(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32
                )
            except (TypeError, ValueError, IndexError):
                continue
        return centers

    def _extract_bat_segment_2d(self):
        """從球棒資料抽出 {frame: (head, tail)}，支援 frame_num/frame_number。"""
        segments = {}
        if self.batht_df is None or self.batht_df.empty:
            return segments

        for _, row in self.batht_df.iterrows():
            try:
                frame_col = "frame_num" if "frame_num" in row else "frame_number"
                f = int(row[frame_col])
                head = np.asarray(row["head"], dtype=np.float32).reshape(-1)
                tail = np.asarray(row["tail"], dtype=np.float32).reshape(-1)
                if head.size < 2 or tail.size < 2:
                    continue
                segments[f] = (head[:2], tail[:2])
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        return segments

    @staticmethod
    def _point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """計算 2D 點 p 到線段 a-b 的最短距離。"""
        p = np.asarray(p, dtype=np.float32)[:2]
        a = np.asarray(a, dtype=np.float32)[:2]
        b = np.asarray(b, dtype=np.float32)[:2]
        ab = b - a
        ap = p - a
        ab2 = float(np.dot(ab, ab))
        if ab2 == 0.0:
            return float(np.linalg.norm(ap))
        t = float(np.clip(np.dot(ap, ab) / ab2, 0.0, 1.0))
        return float(np.linalg.norm(p - (a + t * ab)))

    def _detect_impact(self, F_liftoff: Optional[int] = None, ref_front_impact: Optional[int] = None) -> Optional[int]:
        if self.batht_df is None or self.batht_df.empty:
            return None

        # --- A. 正面視角：棒頭 X 軸最大延伸距離法 ---
        if self.is_front_view:
            print("--- 開始偵測 Impact (正面視角：棒頭 X 軸最大延伸距離法) ---")
            if self.person_df is None or self.person_df.empty:
                return None

            max_dist_x = -9999.0
            best_frame = None

            for _, row_bat in self.batht_df.iterrows():
                col = 'frame_num' if 'frame_num' in row_bat else 'frame_number'
                f = int(row_bat[col])

                if 'head' in row_bat and row_bat['head'] is not None and len(row_bat['head']) >= 2:
                    head_x = float(row_bat['head'][0])
                    person_row = self.person_df[self.person_df['frame_number'] == f]
                    if not person_row.empty:
                        kpts = person_row.iloc[0]['keypoints']
                        if len(kpts) > 18:
                            body_x = float(kpts[18][0])
                            dist_x = (body_x - head_x) if self.batter_index == 0 else (head_x - body_x)
                            if dist_x > max_dist_x:
                                max_dist_x = dist_x
                                best_frame = f

            if best_frame is not None:
                print(f"✅ 成功找到正面 Impact！影格: {best_frame}, 橫向延伸距離: {max_dist_x:.2f} px")
                return best_frame

        # --- B. 側面視角：受控時間視窗 + 棒頭極速法 ---
        print("--- 開始偵測 Impact (側面視角：視窗約束極速法) ---")
        bat_heads = {}
        for _, row in self.batht_df.iterrows():
            col = 'frame_num' if 'frame_num' in row else 'frame_number'
            f = int(row[col])
            if 'head' in row and row['head'] is not None and len(row['head']) >= 2:
                bat_heads[f] = np.array(row['head'][:2], dtype=np.float32)

        if len(bat_heads) < 2:
            return ref_front_impact

        sorted_frames = sorted(list(bat_heads.keys()))

        # 鎖定正面 Impact 的微調視窗 [ref - 2, ref + 8]
        if ref_front_impact is not None:
            min_f = max(0, ref_front_impact - 2)
            max_f = ref_front_impact + 8
            sorted_frames = [f for f in sorted_frames if min_f <= f <= max_f]
            print(f"[Impact Constraint] Locking Side Impact search window to: [{min_f}, {max_f}]")

        if len(sorted_frames) < 2:
            sorted_frames = sorted(list(bat_heads.keys()))

        max_spd = -1.0
        max_spd_frame = None

        print("\n🔍 --- 側面視窗內每幀速度計算 DEBUG ---")
        for i in range(1, len(sorted_frames)):
            prev_f = sorted_frames[i-1]
            curr_f = sorted_frames[i]
            
            frame_gap = curr_f - prev_f
            if frame_gap > 2:
                continue

            prev_pos = bat_heads[prev_f]
            curr_pos = bat_heads[curr_f]
            
            dist = np.linalg.norm(curr_pos - prev_pos)
            speed = dist / float(frame_gap)

            print(f"  ├─ Frame {curr_f:3d} (對比 {prev_f:3d}): Pos=({curr_pos[0]:.1f}, {curr_pos[1]:.1f}) | "
                  f"總速度 dist/gap={speed:6.2f} px/f", end="")

            if speed > max_spd:
                max_spd = speed
                max_spd_frame = curr_f
                print(f"  🌟 [NEW MAX! ({max_spd:.2f})]")
            else:
                print("")

        if max_spd_frame is not None:
            # 🔥 關鍵修復：最高速 (105 幀) 是擊球後暴衝點，真正的擊球點是前一幀 (104 幀)
            best_frame = max_spd_frame - 1
            print(f"✅ [DEBUG Result] 最高速影格為 Frame {max_spd_frame} ({max_spd:.2f} px/f)")
            print(f"🎯 [DEBUG Result] 校正擊球點 (前一幀) 為: Frame {best_frame}\n")
            return best_frame

        return ref_front_impact

    def _detect_landing(self, F_impact: Optional[int]) -> Optional[int]:
        """
        以 Impact 為基準往前搜尋 Landing (通常在 Impact 前 5~25 影格之間)
        """
        if len(self.heel_smooth) == 0:
            return None

        # 1. 定義合理搜尋區間：Impact 前 35 影格到 Impact 瞬間
        if F_impact is not None and F_impact in self.heel_frames:
            impact_idx = int(np.where(self.heel_frames == F_impact)[0][0])
            start_idx = max(0, impact_idx - 35)
            end_idx = impact_idx
        else:
            start_idx = 0
            end_idx = len(self.heel_smooth)

        zone_smooth = self.heel_smooth[start_idx:end_idx]
        zone_frames = self.heel_frames[start_idx:end_idx]

        if len(zone_smooth) == 0:
            return None

        # 2. 找到這段搜尋區間內的最高抬腿點 (Peak)
        peak_rel_idx = int(np.argmax(zone_smooth))

        # 3. 從 Peak 往後找第一個回落至閾值 (th_land) 以下的點
        post_peak_zone = zone_smooth[peak_rel_idx:]
        below_th = (post_peak_zone - self.baseline) < self.th_land
        below_idxs = np.where(below_th)[0]

        if len(below_idxs) > 0:
            land_idx = peak_rel_idx + int(below_idxs[0]) + 2 # +2 幀對齊踩穩
            land_idx = min(land_idx, len(zone_frames) - 1)
            return int(zone_frames[land_idx])
        else:
            # 最糟狀況 fallback: Peak 之後的最小值
            min_rel_idx = int(np.argmin(post_peak_zone))
            return int(zone_frames[peak_rel_idx + min_rel_idx])

    def _detect_liftoff(self, F_impact: Optional[int], F_landing: Optional[int]) -> Optional[int]:
        """
        以 Landing 為基準往前尋找 Liftoff：
        從 Landing 往前找到波峰（最高抬腿點），再從波峰往前找到腳跟開始離開地面的起漲點（谷底）。
        """
        if len(self.heel_smooth) == 0:
            return None

        ref_frame = F_landing if F_landing is not None else F_impact
        if ref_frame is None or ref_frame not in self.heel_frames:
            return int(self.heel_frames[0])

        ref_idx = int(np.where(self.heel_frames == ref_frame)[0][0])
        
        # 限制搜尋範圍：Landing 往前推 30 幀內
        max_search_back = 30 
        start_idx = max(0, ref_idx - max_search_back)

        zone_smooth = self.heel_smooth[start_idx:ref_idx]
        zone_frames = self.heel_frames[start_idx:ref_idx]

        if len(zone_smooth) < 3:
            return max(0, ref_frame - 15)

        # 1. 在 Landing 之前的區間找到抬腿最高點 (Peak)
        peak_rel_idx = int(np.argmax(zone_smooth))

        # 2. 從 Peak 往回（往前段影格）搜尋第一個低於基準線 (baseline + 門檻) 的谷底點
        pre_peak_zone = zone_smooth[:peak_rel_idx + 1]
        
        # 計算一階微分（斜率），找開始上升的點 (diff > 0)
        diffs = np.diff(pre_peak_zone)
        
        # 從 Peak 往前倒著找第一個斜率 <= 0 的地方（即谷底開始抬起處）
        liftoff_rel_idx = 0
        for i in range(len(diffs) - 1, -1, -1):
            if pre_peak_zone[i] - self.baseline < self.th_lift or diffs[i] <= 0:
                liftoff_rel_idx = i
                break

        return int(zone_frames[liftoff_rel_idx])

    def detect_phases(self, ref_front_impact: Optional[int] = None) -> PhaseResult:
        print("====== detect_phases START ======")
        print(f"heel_frames length: {len(self.heel_frames)}")
        
        if len(self.heel_frames) == 0:
            print("❌ ERROR: heel_frames is empty!")
            return PhaseResult(F0=0, F_liftoff=None, F_landing=None, F_impact=None, Fend=None)

        F0 = int(self.heel_frames[0])
        Fend = int(self.heel_frames[-1])

        # 1. 偵測 Impact (若有正面 Impact 參考，自動帶入視窗約束)
        F_impact = self._detect_impact(F_liftoff=None, ref_front_impact=ref_front_impact) 
        print(f"_detect_impact 回傳: {F_impact}")

        # 2. 著地 (Landing) - 以 Impact 倒推 35 影格內尋找
        F_landing = self._detect_landing(F_impact=F_impact)
        print(f"F_landing: {F_landing}")

        # 3. 抬腳 (Liftoff) - 以 Landing/Impact 倒推 35 影格內尋找
        F_liftoff = self._detect_liftoff(F_impact=F_impact, F_landing=F_landing)
        print(f"F_liftoff: {F_liftoff}")

        if self.ball_df is not None and not self.ball_df.empty:
             ball_max_f = self.ball_df['frame_number'].max()
             Fend = int(max(Fend, ball_max_f))

        return PhaseResult(
            F0=F0,
            F_liftoff=F_liftoff,
            F_landing=F_landing,
            F_impact=F_impact,
            Fend=Fend,
        )

@dataclass
class PhasePairResult:
    """同一段打擊影片的正面與側面偵測結果。"""
    front: Optional[PhaseResult] = None
    side: Optional[PhaseResult] = None


class PhaseDetector:
    """一次處理正面與側面，不需要呼叫端指定 is_front_view。

    正面欄位使用 *_front_df，側面欄位使用 *_side_df。
    2D/3D 的資料只要整理成 detector 原本使用的 DataFrame schema，皆可共用此介面。

    detect_phases() 回傳 PhasePairResult：
      - result.front：正面 PhaseResult
      - result.side：側面 PhaseResult（若正面 Impact 可用，會用來約束側面搜尋）
    """
    def __init__(
        self,
        batter_index: int,
        person_front_df: pd.DataFrame = None,
        person_side_df: pd.DataFrame = None,
        ball_front_df: pd.DataFrame = None,
        ball_side_df: pd.DataFrame = None,
        batht_front_df: pd.DataFrame = None,
        batht_side_df: pd.DataFrame = None,
        bat_front_df: pd.DataFrame = None,
        bat_side_df: pd.DataFrame = None,
        baseline_window: int = 10,
        th_lift: float = 5.0,
        th_land: float = 5.0,
        land_hold: int = 3,
        smooth_window: int = 5,
        fps: Optional[float] = None,
    ):
        self.batter_index = batter_index
        self.fps = fps  # 保留給呼叫端傳入；目前偵測閾值仍以影格為單位。
        self.person_front_df = person_front_df
        self.person_side_df = person_side_df
        self.ball_front_df = ball_front_df
        self.ball_side_df = ball_side_df
        # 同時相容 3D 專案的 batht_* 與 2D 專案的 bat_* 命名。
        self.bat_front_df = batht_front_df if batht_front_df is not None else bat_front_df
        self.bat_side_df = batht_side_df if batht_side_df is not None else bat_side_df
        self.detector_options = dict(
            baseline_window=baseline_window,
            th_lift=th_lift,
            th_land=th_land,
            land_hold=land_hold,
            smooth_window=smooth_window,
        )
        self.phase_result_front: Optional[PhaseResult] = None
        self.phase_result_side: Optional[PhaseResult] = None

    @staticmethod
    def _has_data(df: Optional[pd.DataFrame]) -> bool:
        return df is not None and not df.empty

    def detect_phases(self) -> PhasePairResult:
        """自動依序偵測正面、側面，並以正面 Impact 約束側面 Impact。"""
        front = None
        side = None

        if self._has_data(self.person_front_df):
            front_detector = _SingleViewPhaseDetector(
                person_df=self.person_front_df,
                batter_index=self.batter_index,
                ball_df=self.ball_front_df,
                batht_df=self.bat_front_df,
                is_front_view=True,
                **self.detector_options,
            )
            front = front_detector.detect_phases()

        if self._has_data(self.person_side_df):
            side_detector = _SingleViewPhaseDetector(
                person_df=self.person_side_df,
                batter_index=self.batter_index,
                ball_df=self.ball_side_df,
                batht_df=self.bat_side_df,
                is_front_view=False,
                **self.detector_options,
            )
            side = side_detector.detect_phases(
                ref_front_impact=front.F_impact if front is not None else None
            )

        self.phase_result_front = front
        self.phase_result_side = side
        self._build_frame_mapping()
        return PhasePairResult(front=front, side=side)

    def detect_and_sync(self):
        """相容舊 2D 呼叫端，回傳 (front_result, side_result)。

        偵測完成後，以 F0、Liftoff、Landing、Impact、Fend 作為時間錨點，
        建立正面影格到側面影格的映射。錨點不足時會退回線性映射。
        """
        results = self.detect_phases()
        return results.front, results.side

    def _build_frame_mapping(self):
        """建立正面 frame -> 側面 frame 的單調時間映射。"""
        self._frame_map_front = None
        self._frame_map_side = None
        front = self.phase_result_front
        side = self.phase_result_side
        if front is None or side is None:
            return

        pairs = [(front.F0, side.F0)]
        for front_f, side_f in (
            (front.F_liftoff, side.F_liftoff),
            (front.F_landing, side.F_landing),
            (front.F_impact, side.F_impact),
            (front.Fend, side.Fend),
        ):
            if front_f is not None and side_f is not None:
                pairs.append((front_f, side_f))

        pairs = sorted((float(x), float(y)) for x, y in pairs)
        # Remove duplicate front-frame anchors and enforce strictly increasing
        # x/y anchors so interpolation cannot reverse time.
        anchors = []
        for x, y in pairs:
            if anchors and x <= anchors[-1][0]:
                if x == anchors[-1][0]:
                    anchors[-1] = (x, y)
                continue
            if anchors and y <= anchors[-1][1]:
                continue
            anchors.append((x, y))

        if len(anchors) < 2:
            f0 = float(front.F0 or 0)
            fend = float(front.Fend if front.Fend is not None else f0 + 1)
            s0 = float(side.F0 or 0)
            send = float(side.Fend if side.Fend is not None else s0 + 1)
            if fend <= f0:
                fend = f0 + 1
            if send <= s0:
                send = s0 + 1
            anchors = [(f0, s0), (fend, send)]

        self._frame_map_front = [p[0] for p in anchors]
        self._frame_map_side = [p[1] for p in anchors]

    def get_warped_side_frame(self, front_frame: float) -> Optional[int]:
        """將正面影格映射至側面影格；錨點間使用 Cubic Spline，否則線性插值。"""
        if self._frame_map_front is None or self._frame_map_side is None:
            self._build_frame_mapping()
        if self._frame_map_front is None or self._frame_map_side is None:
            return None

        x = self._frame_map_front
        y = self._frame_map_side
        value = float(front_frame)

        if len(x) >= 3:
            try:
                from scipy.interpolate import CubicSpline
                mapped = float(CubicSpline(x, y, bc_type="natural", extrapolate=True)(value))
            except ImportError:
                mapped = float(np.interp(value, x, y))
        else:
            mapped = float(np.interp(value, x, y))

        # Clamp to observed side-video frame range when available.
        side = self.phase_result_side
        if side is not None:
            lo = side.F0
            hi = side.Fend
            if lo is not None:
                mapped = max(float(lo), mapped)
            if hi is not None:
                mapped = min(float(hi), mapped)
        return int(round(mapped))


__all__ = ["PhaseResult", "PhasePairResult", "PhaseDetector"]
