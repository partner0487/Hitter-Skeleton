"""main3d.py 的比較頁：兩支錄影並排播放，指標與分析頁完全相同。

比較頁原本自己算指標（寫死的相機參數、揮棒速度直接加 80、跨步距離用像素乘 0.45、固定右打），
與分析頁的結果對不上。這裡改成交給背景 3D 程序（tools/process3d_worker.py，與錄影頁共用）
用分析頁同一套流程計算，讀回 metrics_3d.json 顯示；關鍵幀也用它的結果。

- 每位打者各自選右打／左打（影響校正與前後腳），改了就重算
- 還沒處理過的錄影交給背景程序做雙相機偵測，處理完自動載入骨架影片
- 指標表與分析頁同樣的項目：一列一個指標，兩支錄影各一欄
- 不提供手動選檔（從打者清單點選錄影）
main.py 的比較頁（CompareWidget）不變。
"""
import json
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel

from compare_widget import CompareWidget
from ui_utils.metric_grid import MISSING, VALUE_SAMPLE, rapsodo_texts, show_spray_chart
from ui_utils.radio_choice import batter_radio_buttons
from ui_utils.worker3d import METRICS_FILE, Worker3D
from utils.paths import clip_parts, clip_title
from utils.phase_detect2d import PhaseResult

PENDING = "計算中…"
SIDES = ("右打", "左打")
# (區塊, [(標題, metrics_3d.json 的鍵), ...])，順序與分析頁相同（鍵名見 Analysis3DWidget.metricTexts）
METRIC_ROWS = (
    ("前腳離地", (("重心初速", "centerGravityLabel"), ("肩髖分離角", "hipShoulderEditLabel1"),
                  ("跨步時間", "strideTime"), ("擊球時間", "swingTime"))),
    ("前腳著地", (("重心速度", "centerGravityLabel2"), ("跨步距離", "strideDisLabel"),
                  ("肩髖分離角", "hipShoulderEditLabel2"), ("最大分離角", "maxSep"),
                  ("骨盆轉速", "pelvisRot"), ("軀幹轉速", "torsoRot"))),
    ("擊球點", (("揮棒速度", "batSpeedLabel"), ("攻擊角度", "horizontalAngleEditLabel"),
                ("垂直棒角", "verticalAngle"), ("前腳膝蓋", "frontKnee"), ("後腳膝蓋", "backKnee"),
                ("肩髖分離角", "hipShoulderEditLabel3"))),
)


def session_folder(folder):
    """CF_ 結果子資料夾或錄影資料夾 → 有 CF*.mp4 與 CS*.mp4 的錄影資料夾；沒有成對影片回傳 None"""
    if not folder:
        return None
    folder = os.path.normpath(folder)
    for d in (folder, os.path.dirname(folder)):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        if any(n.startswith("CF") and n.endswith(".mp4") for n in names) and \
                any(n.startswith("CS") and n.endswith(".mp4") for n in names):
            return d
    return None


class Compare3DWidget(CompareWidget):
    def __init__(self, model=None, worker3d=None, parent=None):
        super().__init__(model, parent)
        self.worker3d = worker3d or Worker3D(self)
        self.owns_worker = worker3d is None
        self.worker3d.done.connect(self.on3DDone)
        self.worker3d.failed.connect(self.on3DFailed)
        self.pending = {1: None, 2: None}   # 等背景程序的 (錄影資料夾, 打擊方向)
        self.fresh = {}                     # 本次開程式後算過的 metrics_3d.json → 打擊方向（舊檔可能是舊算法）
        self.simplifyControls()
        self.buildMetricTable()

    def player(self, idx):
        return self.p1 if idx == 1 else self.p2

    # ---------- 右側控制區 ----------
    def simplifyControls(self):
        # 錄影從打者清單點選，不提供手動選檔
        self.fileGroupBox_2.hide()
        self.fileGroupBox_3.hide()
        self.batterCombos = {}
        for idx, layout, template in ((1, self.verticalLayout_16, self.label_2), (2, self.verticalLayout_17, self.label)):
            combo = QComboBox()
            combo.addItems(SIDES)
            combo.setFont(self.hitterNameInput.font())
            row = QHBoxLayout()
            caption = QLabel("打擊方向:")
            caption.setFont(template.font())
            row.addWidget(caption)
            row.addWidget(combo, 1)
            layout.insertLayout(1, row)
            batter_radio_buttons(combo)
            combo.currentIndexChanged.connect(lambda _, i=idx: self.onBatterSideChanged(i))
            self.batterCombos[idx] = combo

    def batterSide(self, idx):
        return "right" if self.batterCombos[idx].currentText() == "右打" else "left"

    def onBatterSideChanged(self, idx):
        if getattr(self.player(idx), "last_folder_path", None):
            self.run_analysis(idx)

    # ---------- 指標表 ----------
    def buildMetricTable(self):
        """分析分頁：原本三個區塊（欄位與分析頁不同）換成同樣項目的表格，一列一個指標、兩支錄影各一欄"""
        for box in (self.groupBox_7, self.groupBox_8, self.groupBox_2):
            box.hide()
        font = self.centerGravity1.font()
        self.metricLabels = {1: {}, 2: {}}
        groups = []
        for title, rows in METRIC_ROWS:
            box = QGroupBox(title, self.tab_2)
            box.setFont(self.groupBox_7.font())
            grid = QGridLayout(box)
            for r, (caption, key) in enumerate(rows):
                label = QLabel(caption + ":", box)
                label.setFont(font)
                grid.addWidget(label, r, 0)
                for idx in (1, 2):
                    value = QLabel(MISSING, box)
                    value.setFont(font)
                    value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    grid.addWidget(value, r, idx)
                    self.metricLabels[idx][key] = value
            groups.append(grid)
        # 各區塊欄寬一致，數值上下對齊；表頭（1.／2. 錄影名稱）也對齊數值欄
        value_w = max(v.fontMetrics().horizontalAdvance(VALUE_SAMPLE + "0") for v in self.metricLabels[1].values())
        for grid in groups:
            grid.setColumnStretch(0, 1)
            for col in (1, 2):
                grid.setColumnMinimumWidth(col, value_w)
        for i, grid in enumerate(groups):
            self.gridLayout_26.addWidget(grid.parentWidget(), i + 1, 0, 1, 1)
        for label in (self.videoMode1Label, self.videoMode2Label):
            label.setFixedWidth(value_w)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # Rapsodo：數值已含單位，原本的單位欄隱藏
        for unit in (self.label_13, self.label_19, self.label_14, self.label_16, self.label_15, self.label_20):
            unit.hide()

    def update_mode_labels(self, player_idx):
        """表頭：1./2. 加上錄影名稱（完整名稱在提示裡）"""
        p = self.player(player_idx)
        label = self.videoMode1Label if player_idx == 1 else self.videoMode2Label
        folder = getattr(p, "last_folder_path", None)
        parts = clip_parts(folder)
        label.setText(f"{player_idx}. {parts[-1].split('_')[-1] if parts else self.mode_dict.get(p.current_mode, 'T座')}")
        label.setToolTip(clip_title(folder) if folder else "")
        box = self.videoName1groupBox if player_idx == 1 else self.videoName2groupBox
        box.setTitle(f"{player_idx}. {clip_title(folder)}" if folder else f"{player_idx}.")

    def setMetrics(self, idx, texts=None, tooltips=None, fill=MISSING):
        texts, tooltips = texts or {}, tooltips or {}
        for key, label in self.metricLabels[idx].items():
            label.setText(texts.get(key, fill))
            label.setToolTip(tooltips.get(key, ""))

    def _loadRapsodoData(self, player_idx, folder_path):
        """Rapsodo 數值與落點圖：與分析頁同樣的換算與顯示（錄影資料夾或上一層）"""
        labels = [getattr(self, f"{name}Label{player_idx}") for name in ("exitV", "lanuchAngle", "attackDis")]
        texts = (MISSING,) * 3
        for d in (folder_path, os.path.dirname(folder_path or "")):
            path = os.path.join(d or "", "data.json")
            if d and os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        texts = rapsodo_texts(json.load(f))
                except (OSError, ValueError) as e:
                    print(f"[比較頁] 讀不到 {path}：{e}")
                break
        for label, text in zip(labels, texts):
            label.setText(text)
        show_spray_chart(self.rapsodoLabel1 if player_idx == 1 else self.rapsodoLabel12, folder_path)

    # ---------- 3D 指標 ----------
    def load_from_folder_path(self, player_idx, folder_path):
        self.pending[player_idx] = None
        super().load_from_folder_path(player_idx, folder_path)

    def load_processed_data(self, player_idx, folder_path):
        self.player(player_idx).last_folder_path = folder_path
        self.update_mode_labels(player_idx)
        super().load_processed_data(player_idx, folder_path)

    def run_analysis(self, player_idx):
        """已處理的錄影：本次算過且打擊方向相同就直接顯示，否則交給背景程序（只做分析，約數秒）"""
        p = self.player(player_idx)
        folder = getattr(p, "last_folder_path", None)
        if not folder:
            return
        p.phase_result = None
        self.update_phase_markers(player_idx)
        session = session_folder(folder)
        if session is None:
            self.setMetrics(player_idx)
            print(f"[比較頁] {folder} 沒有成對的正面、側面影片，無法做 3D 分析")
            return
        side = self.batterSide(player_idx)
        saved = os.path.join(folder, METRICS_FILE)
        if self.fresh.get(os.path.normcase(os.path.normpath(saved))) == side:
            self.apply3D(player_idx, saved)
            return
        self.request3D(player_idx, session, side)

    def trigger_compare_processing(self, player_idx):
        """還沒處理過的錄影：交給背景程序做雙相機偵測與分析，完成後載入結果（取代原本只做側面 2D 偵測）"""
        p = self.player(player_idx)
        p.last_folder_path = None
        self.display_frame(player_idx)
        session = session_folder(os.path.dirname(p.video_loader.video_path or ""))
        if session is None:
            print("[比較頁] 沒有成對的正面、側面影片，無法做 3D 分析")
            self.setMetrics(player_idx)
            return
        self.request3D(player_idx, session, self.batterSide(player_idx))

    def request3D(self, idx, session, side):
        self.setMetrics(idx, fill=PENDING)
        job = (session, side)
        other = 2 if idx == 1 else 1
        self.pending[idx] = job
        if self.pending[other] != job:   # 兩邊選同一支、同打擊方向時只算一次
            self.worker3d.request(session, side)

    def on3DDone(self, session, metrics_path):
        for idx in (1, 2):
            job = self.pending[idx]
            if job is None or job[0] != session:
                continue
            self.pending[idx] = None
            self.fresh[os.path.normcase(os.path.normpath(metrics_path))] = job[1]
            output = os.path.dirname(metrics_path)
            p = self.player(idx)
            if os.path.normcase(os.path.normpath(getattr(p, "last_folder_path", "") or "")) == os.path.normcase(output):
                self.apply3D(idx, metrics_path)
            else:
                self.load_processed_data(idx, output)   # 原始錄影剛處理完：載入骨架影片，run_analysis 會直接用這份結果

    def on3DFailed(self, session, reason):
        for idx in (1, 2):
            job = self.pending[idx]
            if job is not None and job[0] == session:
                self.pending[idx] = None
                self.setMetrics(idx)
                for label in self.metricLabels[idx].values():
                    label.setToolTip(f"3D 無法分析：{reason}")
                print(f"[比較頁] 3D 無法分析：{reason}")

    def apply3D(self, idx, metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[比較頁] 讀不到 {metrics_path}：{e}")
            self.setMetrics(idx)
            return
        self.setMetrics(idx, data.get("texts"), data.get("tooltips"))
        phases = data.get("phases", {})
        pr = PhaseResult()
        for key in ("F_liftoff", "F_landing", "F_impact"):
            if phases.get(key) is not None:
                setattr(pr, key, int(phases[key]))
        p = self.player(idx)
        p.phase_result = pr
        self.update_phase_markers(idx)
        print(f"[比較頁] {idx}. {clip_title(p.last_folder_path)}：{pr}")

    def shutdown(self):
        super().shutdown()
        if self.owns_worker:
            self.worker3d.shutdown()
