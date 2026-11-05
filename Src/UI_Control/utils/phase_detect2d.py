import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# 肩–肘、肘–腕（Halpe26：5/6 肩、7/8 肘、9/10 腕）
ARM_LINKS = ((5, 7), (7, 9), (6, 8), (8, 10))


def _segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """各點到線段 a-b 的距離"""
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-6), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


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
        self.is_lefty = batter_index == 1
        # 雙手的上臂、前臂骨架線段，用來排除把手臂誤判成球棒的遮罩
        self.arm_segments = self._extract_arm_segments(person_df)

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

    @staticmethod
    def _extract_arm_segments(df: pd.DataFrame, min_conf: float = 0.3) -> dict:
        """{frame: [(a, b), ...]} 肩–肘、肘–腕線段；Halpe26 的 5/6 肩、7/8 肘、9/10 腕"""
        arms = {}
        if df is None or df.empty or "keypoints" not in df:
            return arms
        for _, row in df.iterrows():
            kpts = np.asarray(row["keypoints"], dtype=np.float32)
            if kpts.ndim != 2 or kpts.shape[0] <= 10:
                continue
            ok = lambda i: kpts[i, 0] > 0 and (kpts.shape[1] < 3 or kpts[i, 2] >= min_conf)
            segs = [(kpts[a, :2], kpts[b, :2]) for a, b in ARM_LINKS if ok(a) and ok(b)]
            if segs:
                arms[int(row["frame_number"])] = segs
        return arms

    def _mask_on_arm(self, frame: int, mask_data, ends, bat_px: float) -> bool:
        """遮罩是不是誤抓到手臂：輪廓大多貼著手臂骨架、兩端都在手臂上，而且比整支球棒短很多。
        揮棒時球棒常疊在前臂後方（重疊約 5～6 成），但會延伸超過手肘、長度接近整支球棒，不會被排除"""
        segs = self.arm_segments.get(frame)
        if not segs or bat_px <= 0:
            return False
        try:
            poly = np.asarray(mask_data[1][0], dtype=np.float32).reshape(-1, 2)
        except Exception:
            return False
        pts = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):  # 沿輪廓每 3 px 取一點
            n = max(1, int(np.linalg.norm(b - a) / 3))
            pts += [a + (b - a) * k / n for k in range(n)]
        pts = np.asarray(pts)
        dist = lambda q: np.min([_segment_distances(q, a, b) for a, b in segs], axis=0)
        on_arm = float(np.mean(dist(pts) < self.ARM_NEAR_PX))
        end_far = float(np.max(dist(np.asarray(ends))))
        length = float(np.linalg.norm(ends[0] - ends[1]))
        return (on_arm >= self.ARM_OVERLAP_MIN and end_far <= self.ARM_END_RATIO * bat_px
                and length <= self.ARM_LENGTH_RATIO * bat_px)

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


    # 棒頭離前後各 JUMP_WINDOW 幀的中位數超過 JUMP_MAX_PX 像素視為錯誤偵測。
    # 依 WeiShu 2026-10-29 T座 H01、H02：揮棒時最多偏 36 px，誤抓手臂的幀 144 px 以上
    JUMP_WINDOW = 2
    JUMP_MAX_PX = 100.0
    # 擊球點細修（_refine_impact）：在最高速幀前後 IMPACT_REFINE_SPAN 幀內，把有效棒頭位置內插到每一幀，
    # 用中央差分算速度（前後兩幀的位移 ÷ 2，本身有平滑效果），在最高速幀 ± IMPACT_REFINE_SEARCH 幀內
    # 找峰值並用拋物線估出一幀以下的位置，擊球點 = round(峰值 + IMPACT_PEAK_LAG)。
    # 只取單一幀最高速時，擊球附近常兩幀幾乎同速（146 vs 148 px/幀），結果在 ±1 幀間跳動。
    # 依 tools/phase_truth.csv 的 19 支 WeiShu（60 fps）：原本遮罩與 ROI 遮罩都 17/19 正確、其餘差 1 幀，
    # PEAK_LAG 1.65～1.75 都成立（ROI 遮罩用單取最高速只有 16/19）。PEAK_LAG 以幀為單位，其他 fps 需重新校正
    IMPACT_REFINE_SPAN = 6
    IMPACT_REFINE_SEARCH = 3
    IMPACT_PEAK_LAG = 1.7
    # 無法細修時（有效棒頭太少）：擊球點 = 棒頭最高速幀 + IMPACT_OFFSET（H01、H02 最高速幀比畫面上的擊球早約 1 幀）
    IMPACT_OFFSET = 1
    # 無法細修時，最高速幀之後若接著幾幀遮罩無效（揮棒最快時動態模糊，沒偵測到或被判為跳動），
    # 擊球點改為無效段的最後一幀，最多往後 BLUR_MAX_FRAMES 幀。
    # 依 WeiShu 側拋 H04：最高速 38、39～41 無效，畫面上 41 接觸；側拋 H03：113、114 無效，114 接觸
    BLUR_MAX_FRAMES = 3
    # 著地：腳跟高度在區間最低點 + LAND_TOL_PX 以內即視為貼地（吸收關節點幾 px 的抖動）
    LAND_TOL_PX = 2.0
    # 著地容許值至少為抬腿高度（最高點 − 最低點）的此比例：落地後重心轉移，腳跟常再慢慢降幾 px（H05 落地 5 px → 之後 0 px）
    LAND_TOL_RATIO = 0.05
    # 擊球點：棒頭最高速的前後幀都低於它的此比例時視為孤立尖峰（遮罩誤判）
    SPIKE_RATIO = 0.35
    # 左打的擊球點：取最高速之前「第一個達到最高速 FIRST_PEAK_RATIO 倍」的段落，往後最多 FIRST_PEAK_CLIMB 幀爬到峰值。
    # 左打者面向側面相機，收棒時球棒在身前繞過、畫面上移動很快，收棒最高速常是擊球的 1～2.3 倍
    # （Orangetallboy 2026-10-29 16 支；右打 WeiShu 19 支收棒只有擊球的 0.30～0.64 倍，揮棒前不到 0.35 倍）。
    # 剪成 90 幀時 0.5 倍 15/16 在 ±2 幀內（原本只取最高速 9/16，錯的都是抓到收棒）；右打不套用
    FIRST_PEAK_RATIO = 0.5
    FIRST_PEAK_CLIMB = 4
    # 離地：腳尖高度在抬腿前最低點 + LIFT_TOL_PX 以內視為還踩在地上（腳跟翹起時腳尖會被帶高 3～4 px）
    LIFT_TOL_PX = 4.0
    # 腳跟資料少於此幀數時不判斷各階段
    MIN_FOOT_FRAMES = 5
    # 手臂誤判成球棒：輪廓有 ARM_OVERLAP_MIN 以上在手臂骨架 ARM_NEAR_PX 內、兩端離手臂都不超過
    # 棒長 × ARM_END_RATIO、長度不到棒長 × ARM_LENGTH_RATIO（棒長 = 該片段遮罩兩端距離的 95 百分位）。
    # 依 WeiShu 側拋 H03 f114、H04 f41、T座 H01 f60、正投 H04 f72、handsomeboy 正投 H01 f677/678 等
    ARM_NEAR_PX = 20.0
    ARM_OVERLAP_MIN = 0.6
    ARM_END_RATIO = 0.2
    ARM_LENGTH_RATIO = 0.65

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
                                end2_x=float(p2[0]), end2_y=float(p2[1]), _mask=row['mask'])
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

        # 誤抓到手臂的遮罩（見 _mask_on_arm）：不參與跳動判斷與速度
        on_arm = np.zeros(len(track), dtype=bool)
        if "_mask" in track:
            lengths = np.hypot(track["end1_x"] - track["end2_x"], track["end1_y"] - track["end2_y"]).dropna()
            bat_px = float(np.percentile(lengths, 95)) if len(lengths) else 0.0
            for i in range(len(track)):
                if track.at[i, "source"] == "mask" and not track.at[i, "repeated"]:
                    ends_i = np.array([[track.at[i, "end1_x"], track.at[i, "end1_y"]],
                                       [track.at[i, "end2_x"], track.at[i, "end2_y"]]], dtype=np.float32)
                    on_arm[i] = self._mask_on_arm(int(track.at[i, "frame"]), track.at[i, "_mask"], ends_i, bat_px)
        track["on_arm"] = on_arm

        # 排除不合理跳動：遮罩誤抓到手臂等物體時，棒頭會跳到別處再跳回來，離前後幾幀的中位數很遠；
        # 真正揮棒時棒頭連續移動，中位數幾乎就是該幀，再快也不會被排除
        frames = track["frame"].to_numpy()
        heads_xy = track[["head_x", "head_y"]].to_numpy()
        detected = ~track["repeated"].to_numpy() & ~on_arm
        jumps = np.full(len(track), np.nan)
        for i, f in enumerate(frames):
            if not detected[i]:
                continue
            window = detected & (np.abs(frames - f) <= self.JUMP_WINDOW)
            if window.sum() >= 3:  # 前後資料太少時無法判斷，不排除
                jumps[i] = float(np.linalg.norm(heads_xy[i] - np.median(heads_xy[window], axis=0)))
        track["jump_px"] = jumps
        track["rejected"] = (track["jump_px"] > self.JUMP_MAX_PX) | track["on_arm"]

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
        return track.drop(columns=[c for c in ("_head", "_mask") if c in track])

    def _fastest_non_spike(self, speeds: pd.Series):
        """
        棒頭最高速的那一列；跳過孤立尖峰：前後兩幀都有速度（非 0、非缺）且都低於它的 SPIKE_RATIO。
        真正的揮棒前後幀也很快（H06：161／201／130）；收棒時遮罩換位置只會單幀暴衝
        （H05 第 72 幀 176、前後 6 和 27，真正擊球在第 42 幀 168）。
        速度 0 多半是影片重複幀（Layia 的片段速度 0／有值交錯），視為沒有資料、不拿來判斷。
        """
        valid = speeds.dropna()
        for i in valid.sort_values(ascending=False).index:
            if self._is_spike(speeds, i):
                print(f"略過孤立尖峰：第 {i} 列 {speeds[i]:.1f} px/幀（前後 {speeds.get(i - 1):.1f}、{speeds.get(i + 1):.1f}）")
                continue
            return int(i)
        return int(valid.idxmax()) if not valid.empty else None

    def _is_spike(self, speeds: pd.Series, i) -> bool:
        nb = [speeds.get(j) for j in (i - 1, i + 1)]
        return all(v is not None and not np.isnan(v) and v > 0 for v in nb) and max(nb) < self.SPIKE_RATIO * speeds[i]

    def _first_strong_peak(self, speeds: pd.Series, best: int) -> int:
        """左打：最高速之前第一個達到 FIRST_PEAK_RATIO 倍最高速的段落（非孤立尖峰），往後爬到該段峰值（見 FIRST_PEAK_RATIO）"""
        valid = speeds.dropna()
        for i in valid.index:
            if i >= best:
                break
            if valid[i] < self.FIRST_PEAK_RATIO * speeds[best] or self._is_spike(speeds, i):
                continue
            j = i
            while (j + 1 - i <= self.FIRST_PEAK_CLIMB and j + 1 in speeds.index
                   and not np.isnan(speeds[j + 1]) and speeds[j + 1] >= speeds[j]):
                j += 1
            print(f"左打：最高速在第 {best} 列（收棒），改用較早的峰值第 {j} 列 {speeds[j]:.1f} px/幀")
            return int(j)
        return best

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
        best = self._fastest_non_spike(speeds)
        if best is not None and self.is_lefty:
            best = self._first_strong_peak(speeds, best)
        if best is not None:
            best_frame = int(track.at[best, "frame"])
            print(f"✅ 成功找到擊球點！影格: {best_frame}, 棒頭最高速: {speeds[best]:.2f}")
            return best_frame

        return None

    def _refine_impact(self, raw_impact: int) -> Optional[int]:
        """以中央差分速度的一幀以下峰值細修擊球點（見 IMPACT_REFINE_*）；有效棒頭不足時回傳 None"""
        if self.bat_df is None or self.bat_df.empty:
            return None
        track = self.bat_head_track()
        valid = track[(track["source"] == "mask") & ~track["repeated"] & ~track["rejected"]]
        valid = valid[(valid["frame"] - raw_impact).abs() <= self.IMPACT_REFINE_SPAN]
        if len(valid) < 4:
            return None
        frames = np.arange(int(valid["frame"].min()), int(valid["frame"].max()) + 1)
        # 沒偵測到或被排除的幀用前後有效幀內插（揮棒最快時常模糊）
        head_x = np.interp(frames, valid["frame"], valid["head_x"])
        head_y = np.interp(frames, valid["frame"], valid["head_y"])
        speed = np.hypot(np.gradient(head_x), np.gradient(head_y))
        near = np.where(np.abs(frames - raw_impact) <= self.IMPACT_REFINE_SEARCH)[0]
        if len(near) == 0:
            return None
        i = int(near[np.argmax(speed[near])])
        shift = 0.0
        if 0 < i < len(speed) - 1:
            curve = speed[i - 1] - 2 * speed[i] + speed[i + 1]
            if curve < 0:
                shift = float(np.clip(0.5 * (speed[i - 1] - speed[i + 1]) / curve, -0.5, 0.5))
        self.impact_peak = float(frames[i]) + shift
        impact = int(np.floor(self.impact_peak + self.IMPACT_PEAK_LAG + 0.5))
        print(f"擊球點細修：速度峰值 {self.impact_peak:.2f} 幀 → 擊球點 {impact}（最高速幀 {raw_impact}）")
        return impact

    def _impact_offset(self, raw_impact: int) -> int:
        """最高速幀到擊球點的幀數：IMPACT_OFFSET，或其後連續無效遮罩的幀數（取大者，上限 BLUR_MAX_FRAMES）"""
        if self.bat_df is None or self.bat_df.empty:
            return self.IMPACT_OFFSET
        track = self.bat_head_track().set_index("frame")
        invalid = 0
        frame = raw_impact + 1
        while invalid < self.BLUR_MAX_FRAMES and frame in track.index:
            row = track.loc[frame]
            if row["source"] == "mask" and not row["repeated"] and not row["rejected"]:
                break
            invalid += 1
            frame += 1
        if invalid > self.IMPACT_OFFSET:
            print(f"最高速幀 {raw_impact} 之後 {invalid} 幀遮罩無效（動態模糊），擊球點改為 {raw_impact + invalid}")
        return max(self.IMPACT_OFFSET, invalid)


    def _detect_landing(self, F_liftoff: Optional[int], F_impact: Optional[int]) -> Optional[int]:
        """
        偵測前腳著地：在「抬腿最高點 ～ 擊球幀」之間，腳跟高度（未平滑）回到最低點附近
        （最低點 + max(LAND_TOL_PX, 抬腿高度 × LAND_TOL_RATIO) 以內）且連續 land_hold 幀的第一幀。
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
        floor = float(np.min(heights))
        tol = max(self.LAND_TOL_PX, (float(heights[0]) - floor) * self.LAND_TOL_RATIO)
        low = heights <= floor + tol
        hold = max(1, self.land_hold)
        for i in range(len(idx) - hold + 1):
            # 連續 hold 幀都貼地，且中間沒有缺幀
            if low[i:i + hold].all() and frames[i + hold - 1] - frames[i] == hold - 1:
                return int(frames[i])
        return int(frames[int(np.argmin(heights))])

    # ----------------- 對外介面 -----------------

    def detect_foot_phases(self, F_impact: Optional[int]):
        """給定擊球幀，回傳 (離地, 著地)；3D 用 3D 棒頭最高速改了擊球幀後也用這個重算"""
        if len(self.heel_frames) < self.MIN_FOOT_FRAMES:
            return None, None
        F_liftoff = self._detect_liftoff(F_impact)
        return F_liftoff, self._detect_landing(F_liftoff, F_impact)

    def detect_phases(self) -> PhaseResult:
        """主函式：偵測順序為 Impact -> Liftoff -> Landing。"""
        
        if len(self.heel_frames) < self.MIN_FOOT_FRAMES:
            print(f"❌ ERROR: 腳跟資料只有 {len(self.heel_frames)} 幀！person_df 可能沒有腳跟被偵測到，或是 batter_index 設定錯誤。")
            return PhaseResult(F0=0, F_liftoff=None, F_landing=None, F_impact=None, Fend=None)

        F0 = int(self.heel_frames[0])
        Fend = int(self.heel_frames[-1])

        # 步驟 1: 先找擊球點（棒頭最高速，不依賴腳的資料）
        F_impact = self._detect_impact(side="right") 

        # 步驟 2、3: 離地、著地（以擊球前抬腿最高點為準）
        F_liftoff, F_landing = self.detect_foot_phases(F_impact)

        # 若球的 frame 超過腳跟範圍，就把 Fend 往後拉
        if self.ball_df is not None and not self.ball_df.empty:
             ball_max_f = self.ball_df['frame_number'].max()
             Fend = int(max(Fend, ball_max_f))

        self.raw_impact = F_impact
        if F_impact is not None:
            refined = self._refine_impact(F_impact)
            F_impact = refined if refined is not None else F_impact + self._impact_offset(F_impact)
        return PhaseResult(
            F0=F0,
            F_liftoff=F_liftoff,
            F_landing=F_landing,
            F_impact=F_impact,
            Fend=Fend,
        )
