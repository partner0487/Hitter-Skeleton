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
        bat_df: pd.DataFrame = None,         # 球棒 head/tail 資料
        baseline_window: int = 10,
        th_lift: float = 5.0,
        th_land: float = 5.0,
        land_hold: int = 3,
        smooth_window: int = 5,
    ):
        self.ball_df = ball_df
        self.bat_df = bat_df
        self.baseline_window = baseline_window
        self.th_lift = th_lift
        self.th_land = th_land
        self.land_hold = land_hold
        self.smooth_window = smooth_window

        # 從 person_df 抽出 heel_frames, heel_vals
        self.heel_frames, self.heel_vals = self._extract_heel_data(person_df, batter_index)
        # 前腳腳尖高度（大、小腳趾平均），用來判斷離地：腳跟會先翹起，腳尖離地才算整隻腳離地
        self.toe_frames, self.toe_vals = self._extract_toe_data(person_df, batter_index)
        # 雙手腕中點，用來分辨球棒遮罩哪一端是握把
        self.wrists = self._extract_wrist_centers(person_df)

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

        # 定義地面 = 使用 95 百分位數作為穩定的地面基準，避免噪聲產生的極值位移
        if len(ys_arr) > 10:
            ground_y = np.percentile(ys_arr, 95)
        else:
            ground_y = np.max(ys_arr)
        
        heights = ground_y - ys_arr

        return frames, heights

    @staticmethod
    def _extract_toe_data(df: pd.DataFrame, batter_index: int):
        """前腳腳尖高度：右打左腳（20 大趾、22 小趾），左打右腳（21、23）；地面同腳跟取 95 百分位"""
        if df is None or df.empty or "keypoints" not in df:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        toe_idx = (20, 22) if batter_index == 0 else (21, 23)
        frames, ys = [], []
        for _, row in df.iterrows():
            kpts = np.asarray(row["keypoints"], dtype=np.float32)
            if kpts.ndim != 2 or kpts.shape[0] <= max(toe_idx):
                continue
            pts = [kpts[i, 1] for i in toe_idx if kpts[i, 1] >= 1e-3]
            if pts:
                frames.append(int(row["frame_number"]))
                ys.append(float(np.mean(pts)))
        if not ys:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        ys_arr = np.array(ys, dtype=np.float32)
        ground_y = np.percentile(ys_arr, 95) if len(ys_arr) > 10 else np.max(ys_arr)
        return np.array(frames, dtype=np.int32), ground_y - ys_arr

    @staticmethod
    def _extract_wrist_centers(df: pd.DataFrame, min_conf: float = 0.3) -> dict:
        """{frame: 雙手腕中點 [x, y]}；Halpe26 的 9 左腕、10 右腕，只有一手可信時用那一手"""
        wrists = {}
        if df is None or df.empty or "keypoints" not in df:
            return wrists
        for _, row in df.iterrows():
            kpts = np.asarray(row["keypoints"], dtype=np.float32)
            if kpts.ndim != 2 or kpts.shape[0] <= 10:
                continue
            pts = [kpts[i, :2] for i in (9, 10)
                   if kpts[i, 0] > 0 and (kpts.shape[1] < 3 or kpts[i, 2] >= min_conf)]
            if pts:
                wrists[int(row["frame_number"])] = np.mean(pts, axis=0)
        return wrists

    def _wrist_near(self, frame: int, max_gap: int = 3):
        """該幀的手腕中點；沒有時用前後 max_gap 幀內最近的一幀，都沒有回傳 None"""
        for d in range(max_gap + 1):
            for f in (frame - d, frame + d):
                if f in self.wrists:
                    return self.wrists[f]
        return None

    def _smooth(self, arr: np.ndarray) -> np.ndarray:
        """簡單 moving average smoothing。"""
        if len(arr) == 0 or self.smooth_window <= 1:
            return arr.copy()
        k = self.smooth_window
        kernel = np.ones(k, dtype=np.float32) / float(k)
        return np.convolve(arr, kernel, mode="same")

    # ----------------- 事件偵測 -----------------
    def _lift_peak_frame(self, F_impact: Optional[int]) -> Optional[int]:
        """抬腿最高點：擊球前（無擊球點時整段）前腳腳跟最高的幀"""
        sel = np.ones(len(self.heel_vals), dtype=bool)
        if F_impact is not None:
            sel &= self.heel_frames <= F_impact
        if not sel.any():
            return None
        idx = np.where(sel)[0]
        return int(self.heel_frames[idx[int(np.argmax(self.heel_vals[idx]))]])

    def _detect_liftoff(self, F_impact: Optional[int] = None) -> Optional[int]:
        """
        偵測前腳離地：從抬腿最高點往前找，腳尖高度最後一次還在「最低點 + LIFT_TOL_PX」以內的幀。
        腳跟會先翹起（H06 腳跟第 8 幀就開始升高，腳尖第 16 幀才離地），所以用腳尖判斷。
        影片開始時腳已在空中，會回傳最開頭附近的幀。沒有腳尖資料時退回舊的腳跟門檻法。
        """
        peak = self._lift_peak_frame(F_impact)
        if peak is not None and len(self.toe_vals) > 0:
            sel = self.toe_frames <= peak
            if sel.any():
                frames, heights = self.toe_frames[sel], self.toe_vals[sel]
                on_ground = np.where(heights <= float(np.min(heights)) + self.LIFT_TOL_PX)[0]
                return int(frames[on_ground[-1]])
        return self._detect_liftoff_heel()

    def _detect_liftoff_heel(self) -> Optional[int]:
        """舊方法（腳跟）：先用門檻，失敗再用『大跌前最高點』。"""
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
        if self.bat_df is None or self.bat_df.empty:
            return segments

        for _, row in self.bat_df.iterrows():
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

    # 棒頭離前後各 JUMP_WINDOW 幀的中位數超過 JUMP_MAX_PX 像素視為錯誤偵測。
    # 依 WeiShu 2026-10-29 T座 H01、H02：揮棒時最多偏 36 px，誤抓手臂的幀 144 px 以上
    JUMP_WINDOW = 2
    JUMP_MAX_PX = 100.0
    # 擊球點 = 棒頭最高速幀 + IMPACT_OFFSET（H01、H02 最高速幀比畫面上的擊球早約 1 幀）
    IMPACT_OFFSET = 1
    # 著地：腳跟高度在區間最低點 + LAND_TOL_PX 以內即視為貼地（吸收關節點幾 px 的抖動）
    LAND_TOL_PX = 2.0
    # 離地：腳尖高度在抬腿前最低點 + LIFT_TOL_PX 以內視為還踩在地上（腳跟翹起時腳尖會被帶高 3～4 px）
    LIFT_TOL_PX = 4.0
    # 腳跟資料少於此幀數時不判斷各階段
    MIN_FOOT_FRAMES = 5

    @staticmethod
    def _bat_ends_from_mask(mask_data):
        """球棒遮罩輪廓上相距最遠的兩點 (p1, p2)；讀不到回傳 None"""
        if mask_data is None:
            return None
        try:
            contours = np.asarray(mask_data[1][0], dtype=np.float32).reshape(-1, 2)
            dist_matrix = np.linalg.norm(contours[:, None, :] - contours[None, :, :], axis=-1)
            i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            return contours[i], contours[j]
        except Exception:
            return None

    def bat_head_track(self) -> pd.DataFrame:
        """每一幀的棒頭位置與速度，_detect_impact 依此挑擊球點，也供 debug 輸出。

        欄位：frame、head_x、head_y、source（head 欄位或 mask）、end1_x/y、end2_x/y（mask 的兩端）、
        head_rule（wrist：離雙手腕中點較遠的一端是棒頭；higher：該幀附近沒有手腕，退回用 y 較小的一端）、
        repeated（遮罩與上一幀完全相同，是沒偵測到時沿用的舊資料）、
        jump_px（離前後 JUMP_WINDOW 幀棒頭中位數的距離）、rejected（jump_px > JUMP_MAX_PX，視為錯誤偵測）、
        speed_px_per_frame（與前一個有效幀相比；repeated、rejected 或間隔超過 3 幀為 NaN，不參與比較）
        """
        rows = []
        if self.bat_df is not None and not self.bat_df.empty:
            for _, row in self.bat_df.iterrows():
                col = 'frame_number' if 'frame_number' in row else 'frame_num'
                item = {"frame": int(row[col])}
                if 'head' in row and row['head'] is not None and len(row['head']) > 0:
                    head = np.array(row['head'][:3], dtype=np.float32)
                    item.update(source="head")
                elif 'mask' in row:
                    ends = self._bat_ends_from_mask(row['mask'])
                    if ends is None:
                        continue
                    p1, p2 = ends
                    # 握把在手裡：離手腕較遠的一端是棒頭。只看高低會在球棒接近水平時把握把當棒頭
                    wrist = self._wrist_near(item["frame"])
                    if wrist is not None:
                        head = p1 if np.linalg.norm(p1 - wrist) > np.linalg.norm(p2 - wrist) else p2
                        item.update(head_rule="wrist", wrist_x=float(wrist[0]), wrist_y=float(wrist[1]))
                    else:
                        head = p1 if p1[1] < p2[1] else p2
                        item.update(head_rule="higher")
                    item.update(source="mask", end1_x=float(p1[0]), end1_y=float(p1[1]),
                                end2_x=float(p2[0]), end2_y=float(p2[1]))
                else:
                    continue
                item.update(head_x=float(head[0]), head_y=float(head[1]), _head=head)
                rows.append(item)
        # 同一幀有多筆時保留最後一筆（與原本以 dict 覆蓋相同）
        track = pd.DataFrame(rows)
        if track.empty:
            return track
        track = track.drop_duplicates("frame", keep="last").sort_values("frame").reset_index(drop=True)

        # 沒偵測到球棒時 mergeBatData 會沿用上一幀的遮罩，兩端完全相同的幀不是新的偵測：
        # 不參與跳動判斷與速度，否則棒頭像是停住幾幀再一次追上，產生假的高速
        ends_cols = [c for c in ("end1_x", "end1_y", "end2_x", "end2_y") if c in track] or ["head_x", "head_y"]
        ends = track[ends_cols].to_numpy()
        track["repeated"] = np.r_[False, np.all(ends[1:] == ends[:-1], axis=1)]

        # 排除不合理跳動：遮罩誤抓到手臂等物體時，棒頭會跳到別處再跳回來，離前後幾幀的中位數很遠；
        # 真正揮棒時棒頭連續移動，中位數幾乎就是該幀，再快也不會被排除
        frames = track["frame"].to_numpy()
        heads_xy = track[["head_x", "head_y"]].to_numpy()
        detected = ~track["repeated"].to_numpy()
        jumps = np.full(len(track), np.nan)
        for i, f in enumerate(frames):
            if not detected[i]:
                continue
            window = detected & (np.abs(frames - f) <= self.JUMP_WINDOW)
            if window.sum() >= 3:  # 前後資料太少時無法判斷，不排除
                jumps[i] = float(np.linalg.norm(heads_xy[i] - np.median(heads_xy[window], axis=0)))
        track["jump_px"] = jumps
        track["rejected"] = track["jump_px"] > self.JUMP_MAX_PX

        speeds = [np.nan] * len(track)
        prev = None
        for i in range(len(track)):
            if track.at[i, "rejected"] or track.at[i, "repeated"]:
                continue
            if prev is not None:
                gap = track.at[i, "frame"] - track.at[prev, "frame"]
                if gap <= 3:  # 避免因為斷片掉幀太久造成的計算誤差
                    speeds[i] = float(np.linalg.norm(track.at[i, "_head"] - track.at[prev, "_head"]) / gap)
            prev = i
        track["speed_px_per_frame"] = speeds
        return track.drop(columns="_head")

    def _detect_impact(self, side: str = "right") -> Optional[int]:
        """
        全域搜尋：利用棒頭極速法找出擊球點。取代原本的距離公式。
        """
        print(f"--- 開始偵測 Impact (棒頭極速法) ---")
        if self.bat_df is None or self.bat_df.empty:
            print("偵測失敗：缺乏球棒資料")
            return None

        track = self.bat_head_track()
        if len(track) < 2:
            print("偵測失敗：球棒資料不足以計算速度")
            return None

        speeds = track["speed_px_per_frame"]
        if speeds.notna().any():
            best = int(speeds.idxmax())
            best_frame = int(track.at[best, "frame"])
            print(f"✅ 成功找到擊球點！影格: {best_frame}, 棒頭最高速: {speeds[best]:.2f}")
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
        偵測前腳著地：在「抬腿最高點 ～ 擊球幀」之間，腳跟高度（未平滑）回到最低點附近
        （最低點 + LAND_TOL_PX 以內）且連續 land_hold 幀的第一幀。
        找不到連續幀時退回區間內的最低點。

        從抬腿最高點（離地～擊球間腳跟最高的幀）開始找，而不是從離地幀：
        跨步後腳落在不同位置，畫面上的地面高度會變（H06 落地後腳跟仍比起始高約 18 px），
        若從離地幀開始找，最低點會落在還沒抬腳的幀上。
        """
        if F_liftoff is None or len(self.heel_vals) == 0:
            return None
        in_range = self.heel_frames >= F_liftoff
        if F_impact is not None:
            in_range &= self.heel_frames <= F_impact
        idx = np.where(in_range)[0]
        if len(idx) == 0:
            return None
        idx = idx[int(np.argmax(self.heel_vals[idx])):]

        frames = self.heel_frames[idx]
        heights = self.heel_vals[idx]
        low = heights <= float(np.min(heights)) + self.LAND_TOL_PX
        hold = max(1, self.land_hold)
        for i in range(len(idx) - hold + 1):
            # 連續 hold 幀都貼地，且中間沒有缺幀
            if low[i:i + hold].all() and frames[i + hold - 1] - frames[i] == hold - 1:
                return int(frames[i])
        return int(frames[int(np.argmin(heights))])

    # ----------------- 對外介面 -----------------

    def detect_phases(self) -> PhaseResult:
        """主函式：偵測順序為 Impact -> Liftoff -> Landing。"""
        print("====== detect_phases START ======")
        print(f"heel_frames length: {len(self.heel_frames)}")
        
        if len(self.heel_frames) < self.MIN_FOOT_FRAMES:
            print(f"❌ ERROR: 腳跟資料只有 {len(self.heel_frames)} 幀！person_df 可能沒有腳跟被偵測到，或是 batter_index 設定錯誤。")
            return PhaseResult(F0=0, F_liftoff=None, F_landing=None, F_impact=None, Fend=None)

        F0 = int(self.heel_frames[0])
        Fend = int(self.heel_frames[-1])

        # 步驟 1: 先找擊球點（棒頭最高速，不依賴腳的資料）
        print("準備呼叫 _detect_impact...")
        F_impact = self._detect_impact(side="right") 
        print(f"_detect_impact 回傳: {F_impact}")

        # 步驟 2: 離地（擊球前抬腿最高點往前找腳尖離地）
        F_liftoff = self._detect_liftoff(F_impact)
        print(f"F_liftoff: {F_liftoff}")

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
            F_impact=F_impact + self.IMPACT_OFFSET if F_impact is not None else None,
            Fend=Fend,
        )
