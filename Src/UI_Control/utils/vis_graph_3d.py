import pyqtgraph as pg
import numpy as np
import pandas as pd

class HeelCurvePlotter:
    """
    地板基準：
        只用「第一幀兩個大腳趾」決定一條地面線，之後固定不變。

    曲線：
        每一幀只看「左腳腳跟」，高度 = 腳跟到地面線的垂直距離（像素單位）。
    """

    def __init__(self, person_df, frame_total: int,
                 left_heel_idx: int, left_toe_idx: int, right_toe_idx: int):

        # 全域顏色設定：白底、黑軸
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(antialias=True)
        self._phase_regions = []
        self.graph = pg.PlotWidget(title="左腳腳跟到地面的距離")
        self.graph.setLabel('bottom', '幀 (frame)')
        self.graph.setLabel('left',   '距離 (pixel)')
        self.frame_total = frame_total
        if person_df is None or person_df.empty:
            self.frames = np.array([], dtype=np.int32)
            self.values = np.array([], dtype=np.float32)
            return

        # ---------- ① 用第一幀兩個大腳趾定義地面線 ----------
        first_row = person_df.iloc[0]
        k0 = np.array(first_row["keypoints"], dtype=np.float32)

        x1, y1 = k0[left_toe_idx, 0],  k0[left_toe_idx, 1]
        x2, y2 = k0[right_toe_idx, 0], k0[right_toe_idx, 1]

        # 線的一般式 ax + by + c = 0
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        norm = np.sqrt(a * a + b * b) + 1e-8  # 避免除以 0

        self._line_a = a
        self._line_b = b
        self._line_c = c
        self._line_norm = norm

        # ---------- ② 每幀計算：左腳腳跟到這條線的距離 ----------
        frames = []
        vals   = []

        for _, row in person_df.iterrows():
            f = int(row["frame_number"])
            k = np.array(row["keypoints"], dtype=np.float32)

            hx, hy = k[left_heel_idx, 0], k[left_heel_idx, 1]
            dist = abs(a * hx + b * hy + c) / norm  # 垂直距離，永遠 >= 0

            frames.append(f)
            vals.append(dist)

        self.frames = np.array(frames, dtype=np.int32)
        self.values = np.array(vals,   dtype=np.float32)

        if len(self.values) == 0:
            return

        # ---------- ③ 設定軸範圍 ----------
        # X 軸：0 ~ frame_total
        self.graph.setXRange(0, max(frame_total, 1))

        # Y 軸：最小值含 0，往上預留 10%
        vmin = 0.0
        vmax = float(self.values.max())
        pad  = vmax * 0.1 if vmax > 0 else 5.0
        self.graph.setYRange(vmin - pad * 0.1, vmax + pad)

        # 畫整條曲線（藍綠色）
        self.graph.plot(
            self.frames,
            self.values,
            pen=pg.mkPen((0, 150, 255), width=2),
        )

        self.frames = np.array(frames, dtype=np.int32)
        self.values = np.array(vals,   dtype=np.float32)

        if len(self.values) == 0:
            return

        # ---------- ③ 設定軸範圍 ----------
        self.graph.setXRange(0, max(frame_total, 1))
        vmin = 0.0
        vmax = float(self.values.max())
        pad  = vmax * 0.1 if vmax > 0 else 5.0
        self.graph.setYRange(vmin - pad * 0.1, vmax + pad)

        # 紀錄 phase regions
        self._phase_regions = []

        # 畫整條曲線（藍綠色），並保留 item
        self.curve_item = self.graph.plot(
            self.frames,
            self.values,
            pen=pg.mkPen((0, 150, 255), width=2),
        )

        # 預先建立一個 marker item，之後只改座標就好
        self.marker_item = self.graph.plot(
            [], [],
            pen=None,
            symbol='o',
            symbolSize=12,
            symbolBrush=pg.mkBrush(255, 200, 0),
            symbolPen=pg.mkPen('k', width=2),
        )


    # ==========================
    #  PhaseResult → 色塊區域
    # ==========================
    def set_phase_result(self, phase_result):
        """
        phase_result: PhaseResult 物件，包含：
            F0, F_liftoff, F_landing, F_impact, Fend (皆為 frame index 或 None)
        呼叫後會在曲線底下畫出對應區塊。
        """
        # 先清掉舊的區塊
        self._clear_phase_regions()
        if phase_result is None:
            print("[Phase] phase_result is None")
            return

        print("[Phase] phase_result =", phase_result)

        # 轉成一段一段的 (start, end)
        ranges = self._phase_to_ranges(phase_result)
        print("[Phase] ranges =", ranges)
        if not ranges:
            print("[Phase] no valid ranges, nothing to draw")
            return

        # 顏色先用非常明顯的，方便肉眼確認
        colors = [
            (255, 230, 180, 60),  # Phase 1 淺橘
            (200, 255, 200, 60),  # Phase 2 淺綠
            (200, 220, 255, 60),  # Phase 3 淺藍
            (255, 200, 220, 60),  # Phase 4 淺粉
        ]

        for i, (s, e) in enumerate(ranges):
            brush = pg.mkBrush(*colors[i % len(colors)])
            region = pg.LinearRegionItem(values=(s, e))
            region.setBrush(brush)
            region.setMovable(False)
            # 邊框也畫出來，避免只靠透明色看不清楚
            # region.setPen(pg.mkPen('k', width=2))
            # 放在曲線下方或上方都可以，先放上方保證看得到
            region.setZValue(-10)

            print(f"[Phase] add region {i}: {s}~{e}")
            self.graph.addItem(region)
            self._phase_regions.append(region)

    def _phase_to_ranges(self, P):
        """把 PhaseResult 轉成 [(start,end), ...]，自動跳過 None 或反向區間。"""
        pairs = [
            ("Start→LiftOff",  P.F0,        P.F_liftoff),
            ("LiftOff→Landing",P.F_liftoff, P.F_landing),
            ("Landing→Impact", P.F_landing, P.F_impact),
            ("Impact→Finish",  P.F_impact,  P.Fend),
        ]
        ranges = []
        for name, s, e in pairs:
            if s is not None and e is not None and e > s:
                s_i, e_i = int(s), int(e)
                print(f"[Phase] {name}: {s_i} → {e_i}")
                ranges.append((s_i, e_i))
            else:
                print(f"[Phase] skip {name}: s={s}, e={e}")
        return ranges

    def _clear_phase_regions(self):
        """把既有的 LinearRegionItem 從 graph 上移除。"""
        for r in getattr(self, "_phase_regions", []):
            self.graph.removeItem(r)
        self._phase_regions = []

    def update(self, frame_num: int):
        """更新當前幀的高亮點（slider / 播放時呼叫）。"""
        if len(self.frames) == 0:
            return

        idx = int(np.argmin(np.abs(self.frames - frame_num)))
        x = int(self.frames[idx])
        y = float(self.values[idx])

        # 只更新 marker 的位置，不清空整個 graph
        self.marker_item.setData([x], [y])

class KneeAnglePlotter:
    """
    使用三角測量後的 3D 關節點計算「膝關節角度」的曲線圖。

    DataFrame 格式要求：
        - column "frame_number": int
        - column "keypoints_3d": shape = (N, 3) 的 np.ndarray

    角度定義：
        以膝為頂點，計算：
            v_thigh = hip - knee
            v_shank = ankle - knee
        之間的夾角（degree），範圍約 0 ~ 180。
    """

    def __init__(self, person_3d_df, frame_total: int,
                 hip_idx: int, knee_idx: int, ankle_idx: int):

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(antialias=True)

        self.graph = pg.PlotWidget(title="膝關節角度 (3D)")
        self.graph.setLabel('bottom', '幀 (frame)')
        self.graph.setLabel('left',   '角度 (degree)')
        self.frame_total = frame_total
        self._phase_regions = []

        if person_3d_df is None or person_3d_df.empty:
            self.frames = np.array([], dtype=np.int32)
            self.values = np.array([], dtype=np.float32)
            return

        frames = []
        vals   = []

        for _, row in person_3d_df.iterrows():
            f = int(row["frame_number"])
            k3d = np.asarray(row["keypoints_3d"], dtype=np.float32)

            # 防呆：index 與維度檢查
            if k3d.ndim != 2 or k3d.shape[0] <= max(hip_idx, knee_idx, ankle_idx):
                continue

            hip   = k3d[hip_idx]    # (3,)
            knee  = k3d[knee_idx]   # (3,)
            ankle = k3d[ankle_idx]  # (3,)

            v_thigh = hip - knee
            v_shank = ankle - knee

            n1 = np.linalg.norm(v_thigh)
            n2 = np.linalg.norm(v_shank)
            if n1 < 1e-6 or n2 < 1e-6:
                angle_deg = 0.0
            else:
                cos_ang = float(np.dot(v_thigh, v_shank) / (n1 * n2 + 1e-8))
                cos_ang = np.clip(cos_ang, -1.0, 1.0)
                angle_deg = float(np.degrees(np.arccos(cos_ang)))
                # 如果你之後要 signed angle，可以在這裡用 cross product + 固定平面再加符號

            frames.append(f)
            vals.append(angle_deg)

        self.frames = np.array(frames, dtype=np.int32)
        self.values = np.array(vals,   dtype=np.float32)

        if len(self.values) == 0:
            return

        # X 軸範圍
        self.graph.setXRange(0, max(frame_total, 1))

        # Y 軸範圍：多留一點空間
        vmin = float(self.values.min())
        vmax = float(self.values.max())
        if abs(vmax - vmin) < 1e-6:
            pad = 5.0
        else:
            pad = (vmax - vmin) * 0.1
        self.graph.setYRange(vmin - pad, vmax + pad)

        # 曲線
        self.curve_item = self.graph.plot(
            self.frames,
            self.values,
            pen=pg.mkPen((0, 150, 255), width=2),
        )

        # Marker
        self.marker_item = self.graph.plot(
            [], [],
            pen=None,
            symbol='o',
            symbolSize=12,
            symbolBrush=pg.mkBrush(255, 200, 0),
            symbolPen=pg.mkPen('k', width=2),
        )

    # ==========================
    #  PhaseResult → 色塊區域
    # ==========================
    def set_phase_result(self, phase_result):
        """
        phase_result: PhaseResult 物件
            F0, F_liftoff, F_landing, F_impact, Fend
        在曲線底下畫出四個淺色區域。
        """
        self._clear_phase_regions()
        if phase_result is None:
            print("[Knee3DPhase] phase_result is None")
            return

        ranges = self._phase_to_ranges(phase_result)
        print("[Knee3DPhase] ranges =", ranges)
        if not ranges:
            print("[Knee3DPhase] no valid ranges")
            return

        colors = [
            (255, 230, 180, 60),  # Phase 1 淺橘
            (200, 255, 200, 60),  # Phase 2 淺綠
            (200, 220, 255, 60),  # Phase 3 淺藍
            (255, 200, 220, 60),  # Phase 4 淺粉
        ]

        for i, (s, e) in enumerate(ranges):
            brush = pg.mkBrush(*colors[i % len(colors)])
            region = pg.LinearRegionItem(values=(s, e))
            region.setBrush(brush)
            region.setMovable(False)
            region.setZValue(-10)
            print(f"[Knee3DPhase] add region {i}: {s}~{e}")
            self.graph.addItem(region)
            self._phase_regions.append(region)

    def _phase_to_ranges(self, P):
        """
        把 PhaseResult 轉成 4 段 (start, end)，
        最後一段若沒有 Fend，補到最後一幀。
        """
        pairs = [
            ("Start→LiftOff",   P.F0,        P.F_liftoff),
            ("LiftOff→Landing", P.F_liftoff, P.F_landing),
            ("Landing→Impact",  P.F_landing, P.F_impact),
            ("Impact→Finish",   P.F_impact,  P.Fend),
        ]

        ranges = []
        last_frame = int(self.frames[-1]) if len(self.frames) > 0 else int(self.frame_total - 1)

        for idx, (name, s, e) in enumerate(pairs):
            if s is None:
                print(f"[Knee3DPhase] skip {name}: start is None")
                continue
            s_i = int(s)

            if idx == 3:
                # 第 4 段：若沒有 Fend，就補到最後一幀
                if e is None or int(e) <= s_i:
                    e_i = last_frame
                else:
                    e_i = int(e)
            else:
                if e is None or int(e) <= s_i:
                    print(f"[Knee3DPhase] skip {name}: invalid range s={s}, e={e}")
                    continue
                e_i = int(e)

            if e_i <= s_i:
                print(f"[Knee3DPhase] skip {name}: e_i <= s_i after adjust s={s_i}, e={e_i}")
                continue

            print(f"[Knee3DPhase] {name}: {s_i} → {e_i}")
            ranges.append((s_i, e_i))

        return ranges

    def _clear_phase_regions(self):
        for r in self._phase_regions:
            self.graph.removeItem(r)
        self._phase_regions = []

    def update(self, frame_num: int):
        """更新當前幀的 marker。"""
        if len(self.frames) == 0:
            return

        idx = int(np.argmin(np.abs(self.frames - frame_num)))
        x = int(self.frames[idx])
        y = float(self.values[idx])
        self.marker_item.setData([x], [y])

class HipShoulderAngleGraph:
    """
    使用每一幀的骨架 keypoints 計算髖、肩向量投影到 X 軸的夾角，並畫出兩條角度曲線。

    假設 person_df 內容：
        - 一列代表一幀
        - 欄位 "frame_number": 幀索引 (int)
        - 欄位 "keypoints": shape = (N, 2) 或 (N, 3) 的座標
            * 若為 3D，使用 (x, z) 平面投影計算旋轉角度
            * 若為 2D，使用 (x, y) 平面計算旋轉角度

    角度定義：
        v_hip = r_hip - l_hip
        v_shd = r_shoulder - l_shoulder
        θ_hip = atan2(v_hip_proj_y, v_hip_proj_x)
        θ_shd = atan2(v_shd_proj_y, v_shd_proj_x)
    """

    def __init__(self, person_df, frame_total: int, is_righty: bool = True,
                 phase_result=None,
                 l_hip_idx: int = 11, r_hip_idx: int = 12,
                 l_shoulder_idx: int = 5, r_shoulder_idx: int = 6,
                 l_knee_idx: int = 13, r_knee_idx: int = 14,
                 l_heel_idx: int = 24, r_heel_idx: int = 25):

        # 顏色與樣式（可視情況保留或移除）
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(antialias=True)
        self._phase_regions = []   # 新增：記錄目前有哪些區塊
        self.graph = pg.PlotWidget(title="肩/髖/前膝 夾角變化與峰值 (度)")
        self.graph.setLabel('bottom', '幀 (frame)')
        self.graph.setLabel('left',   '角度 Angle (degree)')
        self.graph.addLegend()

        if person_df is None or person_df.empty:
            self.frames = np.array([], dtype=np.int32)
            self.hip_vals = np.array([], dtype=np.float32)
            self.shd_vals = np.array([], dtype=np.float32)
            self.knee_vals = np.array([], dtype=np.float32)
            return


        frames = []
        hip_angles = []
        shd_angles = []
        knee_angles = []

        for _, row in person_df.iterrows():
            f = int(row["frame_number"])
            is_3d_data = False
            if "keypoints_3d" in row:
                k = np.array(row["keypoints_3d"], dtype=np.float32)
                is_3d_data = True
            else:
                k = np.array(row["keypoints"], dtype=np.float32)

            # 防呆：至少要有足夠維度
            max_idx = max(l_hip_idx, r_hip_idx, l_shoulder_idx, r_shoulder_idx, l_knee_idx, r_knee_idx, l_heel_idx, r_heel_idx)
            if k.ndim != 2 or k.shape[0] <= max_idx:
                continue

            # 取出關節點
            l_hip = k[l_hip_idx]
            r_hip = k[r_hip_idx]
            l_shd = k[l_shoulder_idx]
            r_shd = k[r_shoulder_idx]
            l_knee = k[l_knee_idx]
            r_knee = k[r_knee_idx]
            l_heel = k[l_heel_idx]
            r_heel = k[r_heel_idx]

            # 1) 肩、髖的左右連線向量
            # 依據公式，統一採用 右減左 (Right - Left) 向量方向
            v_hip = r_hip - l_hip
            v_shd = r_shd - l_shd

            # 肩/髖的投影
            if is_3d_data and k.shape[1] >= 3:
                v_hip_proj = np.array([v_hip[0], v_hip[2]], dtype=np.float32)
                v_shd_proj = np.array([v_shd[0], v_shd[2]], dtype=np.float32)
            else:
                v_hip_proj = np.array([v_hip[0], v_hip[1]], dtype=np.float32)
                v_shd_proj = np.array([v_shd[0], v_shd[1]], dtype=np.float32)

            if np.linalg.norm(v_hip_proj) < 1e-6 or np.linalg.norm(v_shd_proj) < 1e-6:
                hip_deg = 0.0
                shd_deg = 0.0
            else:
                # 使用 arctan 求出嚴格落在 -90 ~ 90 度的旋轉角度 (完全符合圖片公式)
                if abs(v_hip_proj[0]) < 1e-6:
                    hip_deg = 90.0 * np.sign(v_hip_proj[1])
                else:
                    hip_deg = np.degrees(np.arctan(v_hip_proj[1] / v_hip_proj[0]))
                    
                if abs(v_shd_proj[0]) < 1e-6:
                    shd_deg = 90.0 * np.sign(v_shd_proj[1])
                else:
                    shd_deg = np.degrees(np.arctan(v_shd_proj[1] / v_shd_proj[0]))

            # 2) 前腳膝蓋的彎曲角度：Hip-Knee-Heel 夾角
            # 右打留左膝，左打留右膝
            if is_righty:
                v_knee_hip = l_hip - l_knee
                v_knee_heel = l_heel - l_knee
            else:
                v_knee_hip = r_hip - r_knee
                v_knee_heel = r_heel - r_knee

            def angle_between(v1, v2):
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6: return 0.0
                cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                return np.degrees(np.arccos(cos_val))

            knee_deg = angle_between(v_knee_hip, v_knee_heel)

            frames.append(f)
            hip_angles.append(hip_deg)
            shd_angles.append(shd_deg)
            knee_angles.append(knee_deg)

        self.frames = np.array(frames, dtype=np.int32)
        # 取消 unwrap，讓數值嚴格保持在 -90 ~ 90 度之間
        self.hip_vals = np.array(hip_angles, dtype=np.float32)
        self.shd_vals = np.array(shd_angles, dtype=np.float32)
        
        # 真正分離角：肩部角度減去髖部角度，並強制映射到 -90 ~ 90 度區間
        sep = self.shd_vals - self.hip_vals
        self.sep_vals = (sep + 90) % 180 - 90
        self.knee_vals = np.array(knee_angles, dtype=np.float32)

        if len(self.frames) == 0:
            return

        # X 軸範圍：0 ~ frame_total
        self.graph.setXRange(0, max(frame_total, 1))

        # Y 軸：依角度自動抓範圍，多留一點空間
        vmin = min(float(self.hip_vals.min()), float(self.shd_vals.min()), float(self.knee_vals.min()), float(self.sep_vals.min()))
        vmax = max(float(self.hip_vals.max()), float(self.shd_vals.max()), float(self.knee_vals.max()), float(self.sep_vals.max()))
        if abs(vmax - vmin) < 1e-6:
            pad = 5.0
        else:
            pad = (vmax - vmin) * 0.1

        self.graph.setYRange(vmin - pad, vmax + pad)

        # 畫整條曲線
        self.hip_curve = self.graph.plot(
            self.frames, self.hip_vals,
            pen=pg.mkPen('#FF8C00', width=2), name="Hip Angle"
        )
        self.shd_curve = self.graph.plot(
            self.frames, self.shd_vals,
            pen=pg.mkPen('#00BFFF', width=2), name="Shoulder Angle"
        )
        self.knee_curve = self.graph.plot(
            self.frames, self.knee_vals,
            pen=pg.mkPen('#32CD32', width=2), name="Front Knee Angle"
        )
        self.sep_curve = self.graph.plot(
            self.frames, self.sep_vals,
            pen=pg.mkPen('#9370DB', width=2), name="Separation Angle (\u0394\u03B8)"
        )
        
        # 使用 scipy 尋找峰值並標記亮色星星!
        import scipy.signal
        
        def plot_peaks(vals, color, name_prefix, mode='both', phase=None, find_first=False, prominence=2.0):
             # 鎖定搜尋區間：著地 (Landing) 與 擊球 (Impact) 之間
             s_idx = 0
             e_idx = len(vals)
             if phase is not None and phase.F_liftoff is not None and phase.F_impact is not None:
                 s_idx = np.argmin(np.abs(self.frames - phase.F_liftoff))
                 e_idx = np.argmin(np.abs(self.frames - phase.F_impact))
                 
             if s_idx >= e_idx:
                 s_idx, e_idx = 0, len(vals)

             # 提取特定區間內的數值
             window_vals = vals[s_idx:e_idx+1]
             if len(window_vals) == 0: return None

             # 補上 baseline
             baseline = np.mean(window_vals[:5]) if len(window_vals) >= 5 else window_vals[0]
             padded_vals = np.concatenate([window_vals, [baseline]])

             best_idx = None
             
             # 使用傳入的靈敏度門檻
             peaks_pos, props_pos = scipy.signal.find_peaks(padded_vals, prominence=prominence)
             valid_pos = peaks_pos < len(window_vals)
             peaks_pos = peaks_pos[valid_pos]
             proms_pos = props_pos["prominences"][valid_pos]
             
             peaks_neg, props_neg = scipy.signal.find_peaks(-padded_vals, prominence=prominence)
             valid_neg = peaks_neg < len(window_vals)
             peaks_neg = peaks_neg[valid_neg]
             proms_neg = props_neg["prominences"][valid_neg]

             if mode == 'min' and len(peaks_neg) > 0:
                 if find_first:
                     best_idx = peaks_neg[0]
                 else:
                     depths = baseline - window_vals[peaks_neg]
                     best_idx = peaks_neg[np.argmax(depths)]
             
             elif (len(peaks_pos) > 0 or len(peaks_neg) > 0):
                 if find_first:
                     # 優先選第一個出現的峰值 (捕捉動力鏈啟動點)
                     if len(peaks_pos) > 0:
                         best_idx = peaks_pos[0]
                     else:
                         best_idx = peaks_neg[0]
                 else:
                     max_p = np.max(proms_pos) if len(proms_pos) > 0 else 0
                     max_n = np.max(proms_neg) if len(proms_neg) > 0 else 0
                     
                     if max_p >= max_n and len(peaks_pos) > 0:
                         best_idx = peaks_pos[np.argmax(proms_pos)]
                     elif len(peaks_neg) > 0:
                         best_idx = peaks_neg[np.argmax(proms_neg)]
             else:
                 # Fallback: 在區間內找極值
                 best_idx = np.argmax(window_vals)

             if best_idx is not None:
                 global_idx = s_idx + best_idx
                 if global_idx < len(self.frames):
                     self.graph.plot([self.frames[global_idx]], [vals[global_idx]], pen=None, 
                                     symbol='star', symbolSize=20, 
                                     symbolBrush=pg.mkBrush(color), symbolPen='k', name=f"{name_prefix} Peak")
                     return int(self.frames[global_idx])
             return None

        # 髖部使用較低的門檻 (0.5) 並取第一個峰值
        self.hip_peak_frame = plot_peaks(self.hip_vals, '#FFFF00', "Hip", mode='both', phase=phase_result, find_first=True, prominence=0.5)
        self.shd_peak_frame = plot_peaks(self.shd_vals, '#00FFFF', "Shd", mode='min', phase=phase_result, find_first=False, prominence=2.0)
        self.knee_peak_frame = plot_peaks(self.knee_vals, '#00FF00', "Knee", mode='both', find_first=False, prominence=2.0)

        # 指示器
        self.hip_marker = self.graph.plot(
            [], [], pen=None,
            symbol='o', symbolSize=10, symbolBrush=pg.mkBrush('#FF8C00'), symbolPen=pg.mkPen('k', width=1)
        )
        self.shd_marker = self.graph.plot(
            [], [], pen=None,
            symbol='o', symbolSize=10, symbolBrush=pg.mkBrush('#00BFFF'), symbolPen=pg.mkPen('k', width=1)
        )
        self.knee_marker = self.graph.plot(
            [], [], pen=None,
            symbol='o', symbolSize=10, symbolBrush=pg.mkBrush('#32CD32'), symbolPen=pg.mkPen('k', width=1)
        )
        self.sep_marker = self.graph.plot(
            [], [], pen=None,
            symbol='o', symbolSize=10, symbolBrush=pg.mkBrush('#9370DB'), symbolPen=pg.mkPen('k', width=1)
        )

    def update(self, frame_num: int):
        """更新當前幀的高亮點（slider / 播放時呼叫）。"""
        if len(self.frames) == 0:
            return

        idx = int(np.argmin(np.abs(self.frames - frame_num)))
        x = int(self.frames[idx])
        y_hip = float(self.hip_vals[idx])
        y_shd = float(self.shd_vals[idx])
        y_knee = float(self.knee_vals[idx])
        y_sep = float(self.sep_vals[idx])

        self.hip_marker.setData([x], [y_hip])
        self.shd_marker.setData([x], [y_shd])
        self.knee_marker.setData([x], [y_knee])
        self.sep_marker.setData([x], [y_sep])

    def set_phase_result(self, phase_result):
        """
        phase_result: PhaseResult 物件
        在曲線底下畫出四個淺色區域。
        """
        self._clear_phase_regions()
        if phase_result is None:
            return

        ranges = self._phase_to_ranges(phase_result)
        if not ranges:
            return

        colors = [
            (255, 230, 180, 60),  # Phase 1 淺橘
            (200, 255, 200, 60),  # Phase 2 淺綠
            (200, 220, 255, 60),  # Phase 3 淺藍
            (255, 200, 220, 60),  # Phase 4 淺粉
        ]

        for i, (s, e) in enumerate(ranges):
            brush = pg.mkBrush(*colors[i % len(colors)])
            region = pg.LinearRegionItem(values=(s, e))
            region.setBrush(brush)
            region.setMovable(False)
            region.setZValue(-10)
            self.graph.addItem(region)
            self._phase_regions.append(region)

    def _phase_to_ranges(self, P):
        pairs = [
            ("Start→LiftOff",   P.F0,        P.F_liftoff),
            ("LiftOff→Landing", P.F_liftoff, P.F_landing),
            ("Landing→Impact",  P.F_landing, P.F_impact),
            ("Impact→Finish",   P.F_impact,  P.Fend),
        ]

        ranges = []
        last_frame = int(self.frames[-1]) if len(self.frames) > 0 else 0

        for idx, (name, s, e) in enumerate(pairs):
            if s is None:
                continue
            s_i = int(s)

            if idx == 3:
                # 第 4 段：若沒有 Fend，就補到最後一幀
                if e is None or int(e) <= s_i:
                    e_i = last_frame
                else:
                    e_i = int(e)
            else:
                if e is None or int(e) <= s_i:
                    continue
                e_i = int(e)

            if e_i <= s_i:
                continue

            ranges.append((s_i, e_i))
        return ranges

    def _clear_phase_regions(self):
        for r in self._phase_regions:
            self.graph.removeItem(r)
        self._phase_regions = []



class BallCurvePlotter:
    def __init__(self, ball_df, frame_total):

        self.graph = pg.PlotWidget(title="球")
        self.graph.setLabel('bottom','幀 (frame)')
        self.graph.setLabel('left','球 X 座標 (pixel)')

        frames, xs = [], []
        for _, row in ball_df.iterrows():
            f = int(row["frame_number"])
            bbox = row["bbox"][0]
            cx = (bbox[0]+bbox[2])/2
            frames.append(f)
            xs.append(cx)

        self.frames = np.array(frames)
        self.values = np.array(xs)

        # ✔ 自動設定範圍
        self.graph.setXRange(0, frame_total)
        self.graph.setYRange(min(xs)*0.9, max(xs)*1.1)

        self.graph.plot(self.frames, self.values, pen=pg.mkPen('blue', width=2))

    def _clear_phase_regions(self):
        """把既有的 LinearRegionItem 從 graph 上移除。"""
        for r in getattr(self, "_phase_regions", []):
            self.graph.removeItem(r)
        self._phase_regions = []

    def set_phase_result(self, phase_result):
        """phase_result: PhaseResult 物件，包含：
            F0, F_liftoff, F_landing, F_impact, Fend (皆為 frame index 或 None)
        呼叫後會在曲線底下畫出對應區塊。
        """
        # 先清掉舊的區塊
        self._clear_phase_regions()
        if phase_result is None:
            print("[Phase] phase_result is None")
            return

        print("[Phase] phase_result =", phase_result)

        # 轉成一段一段的 (start, end)
        ranges = self._phase_to_ranges(phase_result)
        print("[Phase] ranges =", ranges)
        if not ranges:
            print("[Phase] no valid ranges, nothing to draw")
            return

        # 顏色先用非常明顯的，方便肉眼確認
        colors = [
            (255, 230, 180, 60),  # Phase 1 淺橘
            (200, 255, 200, 60),  # Phase 2 淺綠
            (200, 220, 255, 60),  # Phase 3 淺藍
            (255, 200, 220, 60),  # Phase 4 淺粉
        ]

        for i, (s, e) in enumerate(ranges):
            brush = pg.mkBrush(*colors[i % len(colors)])
            region = pg.LinearRegionItem(values=(s, e))
            region.setBrush(brush)
            region.setMovable(False)
            # 邊框也畫出來，避免只靠透明色看不清楚
            # region.setPen(pg.mkPen('k', width=2))
            # 放在曲線下方或上方都可以，先放上方保證看得到
            region.setZValue(-10)

            print(f"[Phase] add region {i}: {s}~{e}")
            self.graph.addItem(region)
            self._phase_regions.append(region)

    def _phase_to_ranges(self, P):
        """把 PhaseResult 轉成 [(start,end), ...]，自動跳過 None 或反向區間。"""
        pairs = [
            ("Start→LiftOff",  P.F0,        P.F_liftoff),
            ("LiftOff→Landing",P.F_liftoff, P.F_landing),
            ("Landing→Impact", P.F_landing, P.F_impact),
            ("Impact→Finish",  P.F_impact,  P.Fend),
        ]
        ranges = []
        for name, s, e in pairs:
            if s is not None and e is not None and e > s:
                s_i, e_i = int(s), int(e)
                print(f"[Phase] {name}: {s_i} → {e_i}")
                ranges.append((s_i, e_i))
            else:
                print(f"[Phase] skip {name}: s={s}, e={e}")
        return ranges

    def update(self, frame):
        idx = int(np.argmin(np.abs(self.frames-frame)))
        x,y = self.frames[idx], self.values[idx]
        self.graph.clear()
        self.graph.plot(self.frames,self.values,pen=pg.mkPen('blue',width=2))
        self.graph.plot([x],[y],symbol='o',symbolSize=10,symbolBrush='cyan')

class BatKinematicsGraph:
    """
    棒頭速度與加速度變化圖
    接收 frames (int list) 與 head_pts_3d (N, 3 numpy array)。
    """
    def __init__(self, frames, head_pts_3d, frame_total: int, fps: float = 30.0):
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.setConfigOptions(antialias=True)
        
        self.graph = pg.PlotWidget(title="棒頭速度與加速度")
        self.graph.setLabel('bottom', '幀 (frame)')
        self.graph.setLabel('left', '數值')
        self.graph.addLegend()
        self._phase_regions = []
        
        self.frames = np.array(frames, dtype=np.int32)
        
        if len(self.frames) < 2 or len(head_pts_3d) < 2:
            self.speed_vals = np.array([], dtype=np.float32)
            self.acc_vals = np.array([], dtype=np.float32)
            return

        import scipy.signal
        pts = np.array(head_pts_3d)
        
        # 平滑 3D 座標，減少雜訊造成的極端變化
        if len(pts) > 5:
            # 使用 savgol_filter 平滑，window size=5, polyorder=2
            for i in range(3):
                pts[:, i] = scipy.signal.savgol_filter(pts[:, i], min(len(pts)-1 if len(pts)%2==0 else len(pts), 5), 2)
                
        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # 計算速度 (單位長度 / 秒)
        speeds = []
        for i in range(len(self.frames)):
            if i == 0:
                dist = np.linalg.norm(pts[1] - pts[0])
                delta_frame = self.frames[1] - self.frames[0]
            else:
                dist = np.linalg.norm(pts[i] - pts[i-1])
                delta_frame = self.frames[i] - self.frames[i-1]
            if delta_frame <= 0: delta_frame = 1
            speeds.append(dist / (dt * delta_frame))
            
        # 再做一次平滑
        speeds = np.array(speeds, dtype=np.float32)
        if len(speeds) > 5:
            speeds = scipy.signal.savgol_filter(speeds, min(len(speeds)-1 if len(speeds)%2==0 else len(speeds), 5), 2)
            
        # 計算加速度
        accs = []
        for i in range(len(speeds)):
            if i == 0:
                delta_v = speeds[1] - speeds[0]
                delta_frame = self.frames[1] - self.frames[0]
            else:
                delta_v = speeds[i] - speeds[i-1]
                delta_frame = self.frames[i] - self.frames[i-1]
            if delta_frame <= 0: delta_frame = 1
            accs.append(delta_v / (dt * delta_frame))
            
        accs = np.array(accs, dtype=np.float32)
        if len(accs) > 5:
            accs = scipy.signal.savgol_filter(accs, min(len(accs)-1 if len(accs)%2==0 else len(accs), 5), 2)

        self.speed_vals = speeds
        self.acc_vals = accs

        # X 軸範圍
        self.graph.setXRange(0, max(frame_total, 1))

        # 創造第二 Y 軸給加速度
        self.p1 = self.graph.plotItem
        self.p2 = pg.ViewBox()
        self.p1.showAxis('right')
        self.p1.scene().addItem(self.p2)
        self.p1.getAxis('right').linkToView(self.p2)
        self.p2.setXLink(self.p1)
        self.p1.getAxis('right').setLabel('加速度', color='#FF0000')
        self.p1.getAxis('left').setLabel('速度', color='#0000FF')

        def updateViews():
            self.p2.setGeometry(self.p1.vb.sceneBoundingRect())
            self.p2.linkedViewChanged(self.p1.vb, self.p2.XAxis)

        updateViews()
        self.p1.vb.sigResized.connect(updateViews)

        # 繪製曲線
        self.speed_curve = self.p1.plot(self.frames, self.speed_vals, pen=pg.mkPen('b', width=2), name="Bat Speed")
        self.acc_curve = pg.PlotCurveItem(self.frames, self.acc_vals, pen=pg.mkPen('r', width=2), name="Bat Accel")
        self.p2.addItem(self.acc_curve)

        # 標記最高點
        self.max_speed_frame = None
        if len(speeds) > 0:
            max_s_idx = int(np.argmax(speeds))
            self.max_speed_frame = int(self.frames[max_s_idx])
            self.p1.plot([self.frames[max_s_idx]], [speeds[max_s_idx]], pen=None,
                         symbol='star', symbolSize=15, symbolBrush=pg.mkBrush('yellow'), name="Max Speed")

        # 指示器
        self.speed_marker = self.p1.plot([], [], pen=None, symbol='o', symbolSize=10, symbolBrush='b')
        self.acc_marker = pg.PlotDataItem([], [], pen=None, symbol='o', symbolSize=10, symbolBrush='r')
        self.p2.addItem(self.acc_marker)

    def set_phase_result(self, phase_result):
        self._clear_phase_regions()
        if phase_result is None:
            return

        ranges = self._phase_to_ranges(phase_result)
        if not ranges:
            return

        colors = [
            (255, 230, 180, 60),
            (200, 255, 200, 60),
            (200, 220, 255, 60),
            (255, 200, 220, 60),
        ]

        for i, (s, e) in enumerate(ranges):
            brush = pg.mkBrush(*colors[i % len(colors)])
            region = pg.LinearRegionItem(values=(s, e))
            region.setBrush(brush)
            region.setMovable(False)
            region.setZValue(-10)
            self.graph.addItem(region)
            self._phase_regions.append(region)

    def _phase_to_ranges(self, P):
        pairs = [
            ("Start→LiftOff",   P.F0,        P.F_liftoff),
            ("LiftOff→Landing", P.F_liftoff, P.F_landing),
            ("Landing→Impact",  P.F_landing, P.F_impact),
            ("Impact→Finish",   P.F_impact,  P.Fend),
        ]
        ranges = []
        last_frame = int(self.frames[-1]) if len(self.frames) > 0 else 0
        for idx, (name, s, e) in enumerate(pairs):
            if s is None: continue
            s_i = int(s)
            if idx == 3:
                if e is None or int(e) <= s_i: e_i = last_frame
                else: e_i = int(e)
            else:
                if e is None or int(e) <= s_i: continue
                e_i = int(e)
            if e_i <= s_i: continue
            ranges.append((s_i, e_i))
        return ranges

    def _clear_phase_regions(self):
        for r in self._phase_regions:
            self.graph.removeItem(r)
        self._phase_regions = []

    def update(self, frame_num: int):
        if len(self.frames) == 0:
            return
        idx = int(np.argmin(np.abs(self.frames - frame_num)))
        x = int(self.frames[idx])
        ys = float(self.speed_vals[idx])
        ya = float(self.acc_vals[idx])
        
        self.speed_marker.setData([x], [ys])
        self.acc_marker.setData([x], [ya])
