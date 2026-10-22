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


class PhaseDetector:
    """
    根據：
      - heel_frames, heel_vals  (左腳腳跟到地面線距離)
      - ball_df (含 frame_number 與 bbox)
    自動偵測：
      - 前腳離地 (liftoff)
      - 前腳著地 (landing)
      - 擊球點   (impact, 以球中心 y 最低為準)
      並回傳四個階段的切點。
    """

    def __init__(
        self,
        person_df: pd.DataFrame,
        batter_index: int,              # 0 or 1
        ball_df: pd.DataFrame = None,
        batht_df: pd.DataFrame = None,         # 球棒 head/tail 資料
        baseline_window: int = 10,
        th_lift: float = 5.0,
        th_land: float = 5.0,
        land_hold: int = 3,
        smooth_window: int = 5,
    ):
        self.ball_df = ball_df
        self.batht_df = batht_df
        self.baseline_window = baseline_window
        self.th_lift = th_lift
        self.th_land = th_land
        self.land_hold = land_hold
        self.smooth_window = smooth_window
        # 從 person_df 抽出 heel_frames, heel_vals
        self.heel_frames, self.heel_vals = self._extract_heel_data(person_df, batter_index)

        # 預先平滑
        self.heel_smooth = self._smooth(self.heel_vals)
        # 基準高度：前幾幀的中位數
        win = min(self.baseline_window, len(self.heel_smooth))
        self.baseline = float(np.median(self.heel_smooth[:win])) if win > 0 else 0.0

    def _extract_heel_data(self, df: pd.DataFrame, batter_index: int):
        """
        根據 batter_index 決定抓哪一腳的腳跟：
          - index=0 (通常對應右打者? 或左打者?): 
            參照 analysis2d_widget: index=0 => kpt_id=10(右腕)
            這通常代表 Right-Handed Batter (RHH) -> 站左打擊區 -> 跨步腳是 Left Leg (左腳跟 idx=24)
          - index=1:
            對應 kpt_id=9(左腕)
            這通常代表 Left-Handed Batter (LHH) -> 站右打擊區 -> 跨步腳是 Right Leg (右腳跟 idx=25)
        
        回傳 (frames, vals)
          - values 代表「腳跟離地高度」
          - 這裡簡單假設：整段影片中，腳跟 Y 的最大值 (畫面最下方) 為地面
            height = (max_y) - current_y
        """
        if df is None or df.empty:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        # 決定腳跟 index
        # 0: RHH -> Left Leg (24)
        # 1: LHH -> Right Leg (25)
        target_idx = 24 if batter_index == 0 else 25
        side_str = "Left Heel (idx=24)" if target_idx == 24 else "Right Heel (idx=25)"
        print(f"[_extract_heel_data] Batter is {'RHH' if batter_index == 0 else 'LHH'}. Tracking front leg: {side_str}. df_len: {len(df)}")

        frames = []
        ys = []

        # 假設 df 有 "frame_number" 與 "keypoints"
        # keypoints shape: (26, 3+)  [x, y, conf, ...]
        for _, row in df.iterrows():
            f = int(row["frame_number"])
            kpts = row["keypoints"]  # list or np.array
            
            # 轉成 numpy 以防萬一
            kpts = np.array(kpts)
            if kpts.shape[0] <= target_idx:
                continue
            
            # 取 (x, y)
            # kpts[target_idx] 可能是 [x, y, conf...]
            y = float(kpts[target_idx][1])
            
            # 簡單過濾掉 0 (未偵測到)
            if y < 1e-3:
                continue

            frames.append(f)
            ys.append(y)

        if not ys:
             return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        frames = np.array(frames, dtype=np.int32)
        ys_arr = np.array(ys, dtype=np.float32)

        # 定義地面 = max(y) 
        # height = ground_y - y
        ground_y = np.max(ys_arr)
        heights = ground_y - ys_arr

        return frames, heights

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        """簡單 moving average smoothing。"""
        if len(arr) == 0 or self.smooth_window <= 1:
            return arr.copy()
        k = self.smooth_window
        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(arr, kernel, mode="same")

    # ----------------- 事件偵測 -----------------
    def _detect_liftoff(self) -> Optional[int]:
        """偵測前腳離地：先用門檻，失敗再用『大跌前最高點』。"""
        if len(self.heel_smooth) == 0:
            return None

        # ---------- 方法一：門檻法（優先） ----------
        above = (self.heel_smooth - self.baseline) > self.th_lift
        idxs = np.where(above)[0]

        if len(idxs) > 0:
            # 找到第一次超過閾值的 index，並取其前一幀代表剛要離地
            liftoff_idx = max(0, int(idxs[0]) - 1)
            return int(self.heel_frames[liftoff_idx])

        # 走到這裡代表門檻法失敗：整條曲線幾乎都貼著 baseline
        # ---------- 方法二：找第一個「大幅下降」 ----------
        diff = np.diff(self.heel_smooth)  # 長度 = len-1

        # drop_thresh：判定「明顯往下」的斜率
        # 通常 -2 ~ -3 pixel 以上就算明顯，你可以視情況調
        drop_thresh = -max(2.0, self.th_lift * 0.4)

        cand = np.where(diff < drop_thresh)[0]
        if len(cand) == 0:
            return None

        # 第一個大幅下降的 index
        drop_idx = int(cand[0])

        # 在 drop_idx 之前找一個局部最大值，當作離地點
        #（避免把一開始就貼平的噪聲當成離地）
        search_end = max(1, drop_idx)
        zone = self.heel_smooth[:search_end]
        peak_idx = int(np.argmax(zone))

        return int(self.heel_frames[peak_idx])

    def _extract_ball_center_2d(self):
        """
        從 ball_df 抽出 {frame: np.array([cx, cy])} 的 dict。
        這裡仍沿用你原本 ball_side 的 bbox 格式：
        bbox: [[x1, y1, x2, y2]]
        """
        centers = {}
        if self.ball_df is None or self.ball_df.empty:
            return centers

        for _, row in self.ball_df.iterrows():
            f = int(row["frame_number"])
            bbox = row["bbox"]
            if not isinstance(bbox, list) or len(bbox) == 0:
                continue
            x1, y1, x2, y2 = bbox[0]
            cx = (float(x1) + float(x2)) / 2.0
            cy = (float(y1) + float(y2)) / 2.0
            centers[f] = np.array([cx, cy], dtype=np.float32)

        return centers

    def _extract_bat_segment_2d(self):
        """
        從 batht_df 抽出 {frame: (head, tail)} 的 dict。
        head, tail 皆為 np.array([x, y]).
        支援欄位名為 frame_num 或 frame_number。
        """
        segments = {}
        if self.batht_df is None or self.batht_df.empty:
            return segments

        for _, row in self.batht_df.iterrows():
            if "frame_num" in row:
                f = int(row["frame_num"])
            else:
                f = int(row["frame_number"])

            head = np.array(row["head"], dtype=np.float32)
            tail = np.array(row["tail"], dtype=np.float32)
            if head.shape[0] < 2 or tail.shape[0] < 2:
                continue
            head = head[:2]
            tail = tail[:2]
            segments[f] = (head, tail)

        return segments

    def _detect_impact(self, F_liftoff: Optional[int]) -> Optional[int]:
        """
        全域搜尋：利用棒頭極速法找出擊球點。
        取代原本計算球心到球棒線段距離找最小值的邏輯。
        """
        print(f"--- 開始偵測 Impact (棒頭極速法 3D) ---")
        if self.batht_df is None or self.batht_df.empty:
            return None

        bat_heads = {}
        for _, row in self.batht_df.iterrows():
            col = 'frame_num' if 'frame_num' in row else 'frame_number'
            f = int(row[col])
            
            if 'head' in row and row['head'] is not None and len(row['head']) >= 2:
                # 保留所有座標維度 (2D 甚至 3D 都可以用歐式距離計算速度)
                bat_heads[f] = np.array(row['head'], dtype=np.float32)

        if len(bat_heads) < 2:
            return None

        sorted_frames = sorted(list(bat_heads.keys()))
        
        # 如果有 liftoff，只在 liftoff 之後尋找
        if F_liftoff is not None:
            sorted_frames = [f for f in sorted_frames if f >= F_liftoff]
            
        if len(sorted_frames) < 2:
            return None

        max_speed = -1
        best_frame = None
        
        for i in range(1, len(sorted_frames)):
            prev_f = sorted_frames[i-1]
            curr_f = sorted_frames[i]
            
            if curr_f - prev_f > 3:  # 容許微小掉幀，斷層太大重新開始
                continue
                
            dist = np.linalg.norm(bat_heads[curr_f] - bat_heads[prev_f])
            speed = dist / (curr_f - prev_f)
            
            if speed > max_speed:
                max_speed = speed
                best_frame = curr_f
                
        if best_frame is not None:
            print(f"✅ 成功找到擊球點(3D)！影格: {best_frame}, 棒頭最高速(px/frame): {max_speed:.2f}")
            return best_frame
            
        return None

    def _point_to_segment_distance(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """
        計算點 p 到線段 a-b 的最短距離（2D 空間）。
        p: 球心座標 [x, y]
        a: 球棒端點 1 (head) [x, y]
        b: 球棒端點 2 (tail) [x, y]
        """
        # 將輸入轉換為 numpy array 確保向量運算正確
        p = np.array(p, dtype=np.float32)
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        # 線段向量 ab 與點到 a 的向量 ap
        ab = b - a
        ap = p - a

        # 計算投影長度比例 t
        # t = (ap ‧ ab) / |ab|^2
        ab2 = np.sum(ab**2)
        
        # 若 a, b 重合為一點，則直接計算 p 到 a 的距離
        if ab2 == 0:
            return float(np.linalg.norm(ap))

        # 計算投影參數 t，並限制在 [0, 1] 之間，代表點落在線段範圍內
        t = np.dot(ap, ab) / ab2
        t = max(0.0, min(1.0, t))

        # 找到線段上距離 p 最近的點 (projection point)
        closest_point = a + t * ab

        # 回傳球心到最近點的歐幾里得距離
        return float(np.linalg.norm(p - closest_point))

    def _detect_landing(self, F_liftoff: Optional[int], F_impact: Optional[int]) -> Optional[int]:
        """
        偵測前腳著地：
        先鎖定抬腿最高點，然後往後尋找腳跟高度第一次跌回閾值以下的瞬間（並加上少許偏移來對齊實際踩穩的時刻）。
        """
        if F_liftoff is None or len(self.heel_smooth) == 0:
            return None

        # 1. 確定搜尋範圍的起始索引 (Liftoff 之後)
        try:
            start_idx = int(np.where(self.heel_frames >= F_liftoff)[0][0])
        except IndexError:
            return None

        # 2. 確定搜尋範圍的結束索引 (Impact 之前，若無 Impact 則搜尋到最後)
        end_idx = len(self.heel_smooth)
        if F_impact is not None:
            try:
                # 找到最後一個小於等於 F_impact 的位置
                potential_end = np.where(self.heel_frames <= F_impact)[0]
                if len(potential_end) > 0:
                    end_idx = int(potential_end[-1]) + 1
            except IndexError:
                pass

        # 如果範圍無效則回傳 None
        if start_idx >= end_idx:
            return None

        # 3. 找抬腿最高點 (在 Liftoff 和 Impact 之間)
        search_zone = self.heel_smooth[start_idx:end_idx]
        if len(search_zone) == 0:
            return None
        rel_peak_idx = int(np.argmax(search_zone))
        peak_idx = start_idx + rel_peak_idx

        # 4. 從最高點往後找第一次跌回閾值以下 (不限於 Impact 之前，以防萬一)
        post_peak_zone = self.heel_smooth[peak_idx:]
        below_th = (post_peak_zone - self.baseline) < self.th_land
        below_idxs = np.where(below_th)[0]

        if len(below_idxs) > 0:
            rel_land_idx = int(below_idxs[0])
            land_idx = peak_idx + rel_land_idx
            
            # 加上少許偏移，對齊實際踩穩
            offset = 2 
            land_idx = min(land_idx + offset, len(self.heel_frames) - 1)
            return int(self.heel_frames[land_idx])
        else:
            # Fallback：直接找最低點
            rel_min_idx = int(np.argmin(post_peak_zone))
            land_idx = peak_idx + rel_min_idx
            return int(self.heel_frames[land_idx])

    # ----------------- 對外介面 -----------------

    def detect_phases(self) -> PhaseResult:
        """主函式：調整偵測順序為 Liftoff -> Impact -> Landing。"""
        print("====== detect_phases START ======")
        print(f"heel_frames length: {len(self.heel_frames)}")
        
        if len(self.heel_frames) == 0:
            print("❌ ERROR: heel_frames is empty! person_df 可能沒有腳跟被偵測到，或是 batter_index 設定錯誤。")
            return PhaseResult(F0=0, F_liftoff=None, F_landing=None, F_impact=None, Fend=None)

        F0 = int(self.heel_frames[0])
        Fend = int(self.heel_frames[-1])

        # 步驟 1: 先找離地
        F_liftoff = self._detect_liftoff()
        print(f"F_liftoff: {F_liftoff}")
        
        # 步驟 2: 先找擊球點 (偵測邏輯內部會判斷是否在 F_liftoff 之後)
        # 註：此時傳入 F_liftoff 給 _detect_impact 確保擊球在離地後
        print("準備呼叫 _detect_impact...")
        F_impact = self._detect_impact(F_liftoff=F_liftoff) 
        print(f"_detect_impact 回傳: {F_impact}")

        # 步驟 3: 在兩者之間找最低位作為著地
        F_landing = self._detect_landing(F_liftoff, F_impact)
        print(f"F_landing: {F_landing}")

        # 若球的 frame 超過腳跟範圍，就把 Fend 往後拉
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
