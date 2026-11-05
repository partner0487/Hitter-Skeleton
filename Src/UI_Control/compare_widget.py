"""比較頁：兩支錄影並排播放，指標與分析頁（Analysis3DWidget）完全相同。

指標交給背景 3D 程序（tools/process3d_worker.py）用分析頁同一套流程計算，讀回 metrics_3d.json 顯示；
關鍵幀也用它的結果。main3d.py 與錄影頁共用同一個背景程序，main.py 第一次用到時才自己開一個。

- 每位打者各自選右打／左打（影響校正與前後腳），改了就重算
- 還沒處理過的錄影交給背景程序做雙相機偵測，處理完自動載入骨架影片
- 指標表與分析頁同樣的項目：一列一個指標，兩支錄影各一欄
- 從打者清單點選錄影，不提供手動選檔
"""
import glob
import json
import os

import cv2
import pandas as pd
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer, QEvent
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor
from PyQt5.QtWidgets import (QComboBox, QGraphicsScene, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
                             QWidget)

from cv_utils.cv_control_dual import DualJsonLoader, VideoLoader as VideoLoaderDual
from skeleton.detect_skeleton_dual import PoseEstimater
from ui.dual_camera_system_cp import Ui_hitter_ui_dual
from ui_utils.analysis_bar import AnalysisBar
from ui_utils.metric_grid import MISSING, VALUE_SAMPLE, rapsodo_texts, show_spray_chart
from ui_utils.radio_choice import batter_radio_buttons
from ui_utils.worker3d import METRICS_FILE, Worker3D
from utils.analyze import BatAnalyzer, PoseAnalyzer
from utils.paths import DB_DIR, NON_HITTER_DIRS, clip_parts, clip_title
from utils.phase_detect2d import PhaseResult
from utils.vis_image_dual import ImageDrawer as ImageDrawerDual

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


class PlayerContext:
    """用來存放每個獨立影片播放器的狀態、模型及物件"""
    def __init__(self, model):
        self.pose_estimater_front = PoseEstimater(model)
        self.pose_estimater_side = PoseEstimater(model)
        self.bat_analyzer_front = BatAnalyzer(self.pose_estimater_front)
        self.bat_analyzer_side = BatAnalyzer(self.pose_estimater_side)
        self.pose_analyzer_front = PoseAnalyzer(self.pose_estimater_front)
        self.pose_analyzer_side = PoseAnalyzer(self.pose_estimater_side)

        self.drawer_front = ImageDrawerDual(self.pose_estimater_front, self.pose_analyzer_front, self.bat_analyzer_front)
        self.drawer_side = ImageDrawerDual(self.pose_estimater_side, self.pose_analyzer_side, self.bat_analyzer_side)

        self.video_loader = VideoLoaderDual(self.drawer_side, self.drawer_front)  # 依照系統設計 (side, front)

        self.view_scene_front = QGraphicsScene()
        self.view_scene_side = QGraphicsScene()

        self.total_frames = 0
        self.is_processed = False
        self.current_mode = "Tbar"
        self.recent_video_model = QStandardItemModel()
        self.last_folder_path = None   # 載入中的已處理資料夾（CF_ 子資料夾）
        self.phase_result = None
        # 原始錄影載入完後要交給背景程序處理
        self._auto_process_pending = False


class CompareWidget(QWidget, Ui_hitter_ui_dual):
    def __init__(self, model=None, worker3d=None, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.model = model

        # 定義 Top(1) 跟 Bottom(2) 的上下文資訊
        self.p1 = PlayerContext(self.model)
        self.p2 = PlayerContext(self.model)

        # 模式字典
        self.mode_dict = {"Tbar": "T座", "Side": "側拋", "Front": "正投"}

        # 播放參數與狀態
        self.offset = 0  # 紀錄 slider2 與 slider1 相對的幀數差距
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames_timer)

        # 背景 3D 程序：main3d.py 傳入錄影頁那一個；沒傳就自己開（第一次要算時才啟動）
        self.worker3d = worker3d or Worker3D(self)
        self.owns_worker = worker3d is None
        self.worker3d.done.connect(self.on3DDone)
        self.worker3d.failed.connect(self.on3DFailed)
        self.pending = {1: None, 2: None}   # 等背景程序的 (錄影資料夾, 打擊方向)
        self.fresh = {}                     # 本次開程式後算過的 metrics_3d.json → 打擊方向（舊檔可能是舊算法）

        self.init_ui_states()
        self.init_phase_sync_buttons()
        self.init_batter_side()
        self.init_metric_table()
        self.init_signals()

    def player(self, idx):
        return self.p1 if idx == 1 else self.p2

    def shutdown(self):
        """關閉程式前停止計時器、影片載入；背景程序是自己開的才關"""
        self.timer.stop()
        for p in (self.p1, self.p2):
            if p.video_loader.is_loading:
                p.video_loader.abort_loading()
        if self.owns_worker:
            self.worker3d.shutdown()

    # 關鍵幀同步：按鈕 → PhaseResult 欄位
    PHASE_SYNC = (("前腳離地", "F_liftoff"), ("前腳著地", "F_landing"), ("擊球點", "F_impact"))

    def init_phase_sync_buttons(self):
        """在 Player 1 打擊者區與 Player 2 檔案區之間加「關鍵幀同步」：兩支影片同時跳到該關鍵幀並以此對齊"""
        box = QtWidgets.QGroupBox("關鍵幀同步", self.tab)
        font = QtGui.QFont()
        font.setPointSize(12)
        box.setFont(font)
        row = QtWidgets.QHBoxLayout(box)
        self.phaseSyncBtns = {}
        for text, attr in self.PHASE_SYNC:
            btn = QtWidgets.QPushButton(text, box)
            btn.clicked.connect(lambda _, a=attr, t=text: self.sync_to_phase(a, t))
            row.addWidget(btn)
            self.phaseSyncBtns[attr] = btn
        # gridLayout_10：第 0 列 Player 1（檔案＋打擊者），第 1 列 Player 2 → 把 Player 2 往下移一列
        self.gridLayout_10.removeItem(self.gridLayout_14)
        self.gridLayout_10.addWidget(box, 1, 0, 1, 1)
        self.gridLayout_10.addLayout(self.gridLayout_14, 2, 0, 1, 1)

    def sync_to_phase(self, attr, name):
        """兩支影片都跳到同一個關鍵幀，並以此設定偏移（之後拖主拉條、播放都維持對齊）"""
        frames = {}
        for idx, p in ((1, self.p1), (2, self.p2)):
            pr = getattr(p, 'phase_result', None)
            if p.total_frames > 0 and pr is not None and getattr(pr, attr) is not None:
                frames[idx] = int(getattr(pr, attr))
        if not frames:
            QMessageBox.information(self, "關鍵幀同步", f"兩支影片都沒有{name}幀（需讀取已分析的資料夾）")
            return
        if self.is_playing:
            self.toggle_play()

        if len(frames) == 2:
            self.offset = frames[2] - frames[1]
            master = frames[1]
        elif 1 in frames:
            master = frames[1]  # 只有 Player 1 有：沿用目前偏移
        else:
            master = frames[2] - self.offset
        self.update_master_slider()
        self.frameSlider.blockSignals(True)
        self.frameSlider.setValue(master)
        self.frameSlider.blockSignals(False)
        self.sync_scroll()

    def init_batter_side(self):
        """打擊者區加右打／左打；錄影從打者清單點選，不提供手動選檔"""
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

    def init_metric_table(self):
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

    def init_ui_states(self):
        # 視窗初始分配: 左邊(_1)放 front(CF), 右邊(_2)放 side(CS)
        self.FrameView1_1.setScene(self.p1.view_scene_front)
        self.FrameView1_2.setScene(self.p1.view_scene_side)
        self.FrameView2_1.setScene(self.p2.view_scene_front)
        self.FrameView2_2.setScene(self.p2.view_scene_side)

        # 設定拉條範圍預設值
        self.frameSlider1.setMaximum(100)
        self.frameSlider2.setMaximum(100)
        self.frameSlider.setMaximum(100)

        # 隱藏 / 更名部分沒用到的UI
        # 注意：Ui_hitter_ui_dual 中可能只有 label_2 或 label，需視實際狀況
        if hasattr(self, 'label_2'): self.label_2.setText("打擊者:")
        if hasattr(self, 'label'): self.label.setText("打擊者:")

        # 初始化 AnalysisBar 畫在 frameSlider1 和 frameSlider2 上
        if hasattr(self, 'frameSlider1'):
            self.analysis_bar1 = AnalysisBar(self.frameSlider1, show_seconds=False)
            self.frameSlider1.installEventFilter(self)
        if hasattr(self, 'frameSlider2'):
            self.analysis_bar2 = AnalysisBar(self.frameSlider2, show_seconds=False)
            self.frameSlider2.installEventFilter(self)

        # 初始化模式顯示
        self.update_mode_labels(1)
        self.update_mode_labels(2)

    def eventFilter(self, source, event):
        if source in [getattr(self, 'frameSlider1', None), getattr(self, 'frameSlider2', None)] and event.type() == QEvent.Resize:
            opt = QtWidgets.QStyleOptionSlider()
            source.initStyleOption(opt)
            groove_rect = source.style().subControlRect(
                QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, source
            )
            bar_h = 100
            bar = self.analysis_bar1 if source == getattr(self, 'frameSlider1', None) else self.analysis_bar2
            bar.setGeometry(
                groove_rect.left(),
                source.height() // 2 - bar_h // 2,
                groove_rect.width(),
                bar_h
            )
            bar.raise_()
            return False
        return super().eventFilter(source, event)

    def init_signals(self):
        # ============ Player 1 (Top) ============
        if hasattr(self, 'modeTbar'): self.modeTbar.clicked.connect(lambda: self.setMode(1, "Tbar"))
        if hasattr(self, 'modeSide'): self.modeSide.clicked.connect(lambda: self.setMode(1, "Side"))
        if hasattr(self, 'modeFront'): self.modeFront.clicked.connect(lambda: self.setMode(1, "Front"))

        if hasattr(self, 'hitterNameInput'):
            self.hitterNameInput.currentIndexChanged.connect(lambda: self.populateRecentVideosList(1))
        if hasattr(self, 'listView'):
            self.listView.clicked.connect(lambda index: self.onRecentVideoSelected(1, index))

        self.frameSlider1.valueChanged.connect(lambda: self.on_individual_slider_changed(1))

        # ============ Player 2 (Bottom) ============
        if hasattr(self, 'modeTbar_2'): self.modeTbar_2.clicked.connect(lambda: self.setMode(2, "Tbar"))
        if hasattr(self, 'modeSide_2'): self.modeSide_2.clicked.connect(lambda: self.setMode(2, "Side"))
        if hasattr(self, 'modeFront_2'): self.modeFront_2.clicked.connect(lambda: self.setMode(2, "Front"))

        if hasattr(self, 'hitterNameInput_2'):
            self.hitterNameInput_2.currentIndexChanged.connect(lambda: self.populateRecentVideosList(2))
        if hasattr(self, 'listView_2'):
            self.listView_2.clicked.connect(lambda index: self.onRecentVideoSelected(2, index))

        self.frameSlider2.valueChanged.connect(lambda: self.on_individual_slider_changed(2))

        # ============ Master 控制 (同步) ============
        # 偏移只在使用者拖動個別拉條時更新（on_individual_slider_changed）；
        # 按主拉條時不重算，否則播完後兩側都被夾在最後一幀，偏移會被歸零、主拉條跳回
        self.frameSlider.valueChanged.connect(self.sync_scroll)

        self.playBtn.clicked.connect(self.toggle_play)
        self.backKeyBtn.clicked.connect(lambda: self.step_frame(-1))
        self.forwardKeyBtn.clicked.connect(lambda: self.step_frame(1))

        self.slowMotionInput_2.currentIndexChanged.connect(self.update_playback_speed)

        # 初始刷新 Hitter List
        self.updateHitterList()

    # ================= 載入與檔案庫邏輯 =================

    def updateHitterList(self):
        db_path = str(DB_DIR)
        if not os.path.exists(db_path): return

        hitters = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d)) and d not in NON_HITTER_DIRS]
        hitters.sort()

        for combo in [self.hitterNameInput, self.hitterNameInput_2]:
            combo.blockSignals(True)
            combo.clear()
            for hitter in hitters:
                combo.addItem(hitter)
            combo.blockSignals(False)

        self.populateRecentVideosList(1)
        self.populateRecentVideosList(2)

    def setMode(self, player_idx, mode):
        p = self.p1 if player_idx == 1 else self.p2
        p.current_mode = mode
        self.update_mode_labels(player_idx)
        self.populateRecentVideosList(player_idx)

    def populateRecentVideosList(self, player_idx):
        p = self.p1 if player_idx == 1 else self.p2
        combo = self.hitterNameInput if player_idx == 1 else self.hitterNameInput_2
        list_view = self.listView if player_idx == 1 else self.listView_2

        hitter_name = combo.currentText()
        if not hitter_name:
            p.recent_video_model.clear()
            return

        db_path = str(DB_DIR)
        hitter_path = os.path.join(db_path, hitter_name)

        selected_folders = []
        try:
            date_folders = [d for d in os.listdir(hitter_path) if os.path.isdir(os.path.join(hitter_path, d))]
            date_folders.sort(reverse=True)
            for d_folder in date_folders:
                d_path = os.path.join(hitter_path, d_folder)
                mode_path = os.path.join(d_path, self.mode_dict.get(p.current_mode, 'T座'))
                if not os.path.exists(mode_path): continue
                sessions = [s for s in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, s))]
                sessions.sort(reverse=True)
                for s_folder in sessions:
                    s_path = os.path.join(mode_path, s_folder)
                    files = os.listdir(s_path)
                    if any(f.endswith('.mp4') for f in files) or any('person_side.json' in f for f in files):
                        selected_folders.append((d_folder, s_folder, s_path))
                    if len(selected_folders) >= 15: break
                if len(selected_folders) >= 15: break
        except Exception as e:
            print(f"Populate recent videos error: {e}")

        p.recent_video_model.clear()
        for d_f, s_f, full_path in selected_folders:
            display_text = f"{d_f} - {self.mode_dict.get(p.current_mode, 'T座')} - {s_f}"
            item = QStandardItem(display_text)
            item.setData(full_path, Qt.UserRole)
            p.recent_video_model.appendRow(item)

        list_view.setModel(p.recent_video_model)

    def onRecentVideoSelected(self, player_idx, index):
        p = self.p1 if player_idx == 1 else self.p2
        folder_path = p.recent_video_model.itemFromIndex(index).data(Qt.UserRole)
        self.load_from_folder_path(player_idx, folder_path)

    def load_from_folder_path(self, player_idx, folder_path):
        """核心載入邏輯：判斷是 processed 還是 raw (與 ListView 點選共用)"""
        p = self.p1 if player_idx == 1 else self.p2
        self.update_mode_labels(player_idx)

        # 防呆：如果當前正在載入或分析，則中止前一次作業
        if p.video_loader.is_loading:
            print(f"Player {player_idx} 中止載入前一部影片")
            p.video_loader.abort_loading()

        self.pending[player_idx] = None

        # 1. 核心邏輯優化：優先檢查子資料夾 (視為 3D 處理過的會話)
        target_folder = None

        # 遍歷子資料夾
        for sub in os.listdir(folder_path):
            sub_path = os.path.join(folder_path, sub)
            if os.path.isdir(sub_path):
                # 如果子資料夾內有 3D 骨架，這就是我們要的
                if os.path.exists(os.path.join(sub_path, "person_3d.json")):
                    target_folder = sub_path
                    break
                # 或者有任何處理過的側面骨架
                elif os.path.exists(os.path.join(sub_path, "person_side.json")):
                    target_folder = sub_path
                    # 繼續找找看有沒有更好的(3D)，如果沒有就用這個

        # 如果子資料夾沒找到，再看根目錄
        if not target_folder:
            if os.path.exists(os.path.join(folder_path, "person_side.json")):
                target_folder = folder_path

        if target_folder:
            print(f"Loading PROCESSED session to Player {player_idx}: {target_folder}")
            self.load_processed_data(player_idx, target_folder)
            return

        # 2. 如果沒有 json，尋找 raw 影片
        all_mp4 = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mp4')])
        if not all_mp4:
            print(f"資料夾中找不到影片檔：{folder_path}")
            return

        front_mp4 = [f for f in all_mp4 if 'front' in f.lower() or 'c0' in f.lower()]
        side_mp4 = [f for f in all_mp4 if 'side' in f.lower() or 'c1' in f.lower()]

        vp_f = front_mp4[0] if front_mp4 else all_mp4[0]
        vp_s = side_mp4[0] if side_mp4 else (all_mp4[1] if len(all_mp4) > 1 else all_mp4[0])

        print(f"Loading RAW session to Player {player_idx} → Front: {vp_f}, Side: {vp_s}")
        self.trigger_raw_loading(player_idx, vp_f, vp_s)

    # ================= 檔案載入與解析機制 =================

    def trigger_raw_loading(self, player_idx, v_front, v_side):
        """處理未經分析處理的原始 mp4 載入"""
        p = self.p1 if player_idx == 1 else self.p2

        # 0. 視覺重設：清空畫面讓使用者知道在載入中
        p.view_scene_front.clear()
        p.view_scene_side.clear()
        self.offset = 0 # 重設同步差值

        p.last_folder_path = None
        p.phase_result = None
        self.update_mode_labels(player_idx)
        p.is_processed = False
        p._auto_process_pending = True

        # 借用 VideoLoaderDual 來讀取
        p.video_loader.loadVideo(v_side, v_front)

        # 定時檢查這組 loader 是否跑完了
        # 注意這裡傳入 player_idx 確保知道是哪一邊
        self.check_video_load(player_idx)

    def check_video_load(self, player_idx):
        p = self.p1 if player_idx == 1 else self.p2
        if p.video_loader.is_loading or p.video_loader.total_frames is None:
            QTimer.singleShot(100, lambda: self.check_video_load(player_idx))
            return

        p.total_frames = p.video_loader.total_frames

        slider = self.frameSlider1 if player_idx == 1 else self.frameSlider2
        slider.setMaximum(max(0, p.total_frames - 1))
        slider.setValue(0)

        # [NEW] 初始化幀數顯示 (1-based)
        label = self.frameNumLabel1 if player_idx == 1 else self.frameNumLabel2
        label.setText(f"1 / {p.total_frames}")

        self.update_phase_markers(player_idx)

        self.update_master_slider()

        if getattr(p, '_auto_process_pending', False):
            p._auto_process_pending = False
            self.display_frame(player_idx)
            self.process_raw(player_idx)
        else:
            self.display_frame(player_idx)

    def load_processed_data(self, player_idx, folder_path):
        """讀取已處理的 JSON 及骨架影片"""
        p = self.p1 if player_idx == 1 else self.p2
        p.last_folder_path = folder_path
        self.update_mode_labels(player_idx)
        
        # 0. 重設狀態
        p.view_scene_front.clear()
        p.view_scene_side.clear()
        self.offset = 0

        loader = DualJsonLoader(folder_path)
        data = loader.load()

        p.pose_estimater_side.person_df = data.get("person_side")
        p.pose_estimater_front.person_df = data.get("person_front")
        p.bat_analyzer_side.batHeadTail_df = data.get("batht_side") if data.get("batht_side") is not None else pd.DataFrame()
        p.bat_analyzer_front.batHeadTail_df = data.get("batht_front") if data.get("batht_front") is not None else pd.DataFrame()

        self._loadRapsodoData(player_idx, folder_path)
        self.run_analysis(player_idx)

        # 找出影片檔案 (優先識別 CF/CS 命名規則)
        skel_side = os.path.join(folder_path, "skel_side.mp4")
        skel_front = os.path.join(folder_path, "skel_front.mp4")
        raw_side = os.path.join(folder_path, "raw_side.mp4")
        raw_front = os.path.join(folder_path, "raw_front.mp4")

        mp4_files = glob.glob(os.path.join(folder_path, '*.mp4'))

        # 側面影片 (優先找 skel/raw_side, 其次找 CS 開頭)
        cs_files = [f for f in mp4_files if "CS" in os.path.basename(f)]
        v_s = None
        if os.path.exists(skel_side): v_s = skel_side
        elif os.path.exists(raw_side): v_s = raw_side
        elif cs_files: v_s = cs_files[0]
        elif mp4_files: v_s = sorted(mp4_files)[0]

        # 正面影片 (優先找 skel/raw_front, 其次找 CF 開頭)
        cf_files = [f for f in mp4_files if "CF" in os.path.basename(f)]
        v_f = None
        if os.path.exists(skel_front): v_f = skel_front
        elif os.path.exists(raw_front): v_f = raw_front
        elif cf_files: v_f = cf_files[0]
        elif len(mp4_files) > 1: v_f = sorted(mp4_files)[1]
        else: v_f = v_s # 2D 單邊保底

        if v_s and v_f:
            p.is_processed = True

            # 使用 cv2.VideoCapture 開啟
            p.video_loader.video_path = v_s
            p.video_loader.video_path2 = v_f

            # 使用 background thread conversion (如同 loaderDual 的做法)
            p.video_loader.loadVideo(v_s, v_f)
            self.check_video_load(player_idx)


    # ================= 顯示與渲染邏輯 =================

    def display_frame(self, player_idx):
        p = self.p1 if player_idx == 1 else self.p2
        slider = self.frameSlider1 if player_idx == 1 else self.frameSlider2

        view_front = self.FrameView1_1 if player_idx == 1 else self.FrameView2_1
        view_side = self.FrameView1_2 if player_idx == 1 else self.FrameView2_2

        val = slider.value()

        # [NEW] 更新各自的 frame number 標籤 (1-based)
        label = self.frameNumLabel1 if player_idx == 1 else self.frameNumLabel2
        label.setText(f"{val + 1} / {p.total_frames}")

        if p.video_loader.video_frames1 is None:
            return # 尚未準備好

        if val >= p.total_frames:
            return

        frame_side = p.video_loader.getVideoImage(val)
        frame_front = p.video_loader.getVideoImage2(val)

        if frame_side is not None:
            if p.is_processed:
                # 啟用骨架與數據繪製
                p.drawer_side.setShowSkeleton(True)
                p.drawer_side.setShowOverlay(True)
                # 重新獲取影像避免在原圖修改
                frame_side = p.drawer_side.drawInfo(img=frame_side.copy(), frame_num=val)
                frame_side = p.drawer_side.drawOverlay(frame_side, frame_num=val)

                # 動力鏈 (1,2,3,4) 僅顯示在 bar 上，不畫到影片畫面

            self.render_image(frame_side, p.view_scene_side, view_side)

        if frame_front is not None:
            if p.is_processed:
                # 啟用骨架與數據繪製
                p.drawer_front.setShowSkeleton(True)
                p.drawer_front.setShowOverlay(True)
                frame_front = p.drawer_front.drawInfo(img=frame_front.copy(), frame_num=val)
                frame_front = p.drawer_front.drawOverlay(frame_front, frame_num=val)
            self.render_image(frame_front, p.view_scene_front, view_front)

    def render_image(self, frame, scene, view):
        if frame is None: return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        qimg = QtGui.QImage(frame_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scene.clear()
        scene.addPixmap(pixmap)
        view.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def on_individual_slider_changed(self, player_idx):
        self.display_frame(player_idx)

        # 若是透過 sync_scroll 程式控制改變的值，則不重新計算 offset 與 master slider
        if getattr(self, '_is_syncing', False):
            return

        # 使用者手動拖曳或點擊獨立拉條時，更新 offset 並同步主拉條
        self.offset = self.frameSlider2.value() - self.frameSlider1.value()
        self.frameSlider.blockSignals(True)
        self.update_master_slider()
        self.frameSlider.setValue(self.frameSlider1.value())
        self.frameSlider.blockSignals(False)


    # ================= 同步與播放控制邏輯 =================
    def update_master_slider(self):
        """主拉條範圍：涵蓋已載入影片的最後一幀；Player 2 依偏移換算（偏移為負時要多播幾幀）。
        未載入的一側不算（其拉條仍是預設的 100）。"""
        ends = []
        if self.p1.total_frames > 0:
            ends.append(self.p1.total_frames - 1)
        if self.p2.total_frames > 0:
            ends.append(self.p2.total_frames - 1 - self.offset)
        self.frameSlider.setMaximum(max(ends + [0]))

    def sync_scroll(self):
        """ 拖動主拉條時，兩邊維持相對偏移量同步移動 """
        self._is_syncing = True
        try:
            val = self.frameSlider.value()

            # Player 1 跟隨主拉條 (強制 Clamp 在範圍內)
            idx1 = max(0, min(val, self.p1.total_frames - 1)) if self.p1.total_frames > 0 else 0
            self.frameSlider1.setValue(idx1)

            # Player 2 維持相對差距 (強制 Clamp 在範圍內)
            new_side_val = val + self.offset
            idx2 = max(0, min(new_side_val, self.p2.total_frames - 1)) if self.p2.total_frames > 0 else 0
            self.frameSlider2.setValue(idx2)

            # 與兩側標籤一致用 1-based：最後一幀顯示 N / N
            self.frameNumLabel.setText(f"同步幀: {val + 1} / {self.frameSlider.maximum() + 1}")
        finally:
            self._is_syncing = False

    def toggle_play(self):
        if self.p1.total_frames == 0 and self.p2.total_frames == 0:
            return

        self.is_playing = not self.is_playing
        self.playBtn.setText("||" if self.is_playing else "▶︎")

        if self.is_playing:
            # 兩邊都已播到最後一幀時從頭播放（維持兩支影片的相對偏移）
            if self._is_at_end():
                self.frameSlider.setValue(0)
            self.update_playback_speed()
        else:
            self.timer.stop()

    def _is_at_end(self):
        """主拉條已到底（無法再前進）就算播完；主拉條範圍已涵蓋兩支影片的最後一幀"""
        return self.frameSlider.value() >= self.frameSlider.maximum()

    def update_playback_speed(self):
        if not self.is_playing:
            return

        # UI combobox "1.0x", "0.5x", "0.25x"
        speed_text = self.slowMotionInput_2.currentText()
        if "0.25x" in speed_text:
            interval = 132 # 33 * 4
        elif "0.5x" in speed_text:
            interval = 66  # 33 * 2
        else:
            interval = 33  # Normal

        self.timer.start(interval)

    def update_frames_timer(self):
        # 當任何一個播放器還沒到底的時候，繼續播
        if not self._is_at_end():
            self.frameSlider.setValue(self.frameSlider.value() + 1)
        else:
            self.toggle_play()

    def step_frame(self, step):
        curr = self.frameSlider.value()
        new_val = curr + step
        if 0 <= new_val <= self.frameSlider.maximum():
            self.frameSlider.setValue(new_val)

    def update_phase_markers(self, player_idx):
        p = self.p1 if player_idx == 1 else self.p2
        bar = self.analysis_bar1 if player_idx == 1 else self.analysis_bar2
        if not hasattr(p, 'phase_result') or p.phase_result is None or not p.total_frames:
            bar.set_segments([])
            bar.set_markers([])
            return

        pr = p.phase_result
        tf = float(p.total_frames - 1) if p.total_frames > 1 else 1.0

        markers = []
        if pr.F_liftoff is not None:
            markers.append((pr.F_liftoff / tf, 'a'))
        if pr.F_landing is not None:
            markers.append((pr.F_landing / tf, 'b'))
        if pr.F_impact is not None:
            markers.append((pr.F_impact / tf, 'c'))

        segments = []
        color_prepare = QColor(150, 150, 150)
        color_stride  = QColor(255, 69, 0)
        color_swing   = QColor(180, 165, 32)
        color_follow  = QColor(150, 150, 150)

        if pr.F_liftoff is not None:
            segments.append((0, pr.F_liftoff / tf, color_prepare))
        if pr.F_liftoff is not None and pr.F_landing is not None:
            segments.append((pr.F_liftoff / tf, pr.F_landing / tf, color_stride))
        if pr.F_landing is not None and pr.F_impact is not None:
            segments.append((pr.F_landing / tf, pr.F_impact / tf, color_swing))
        if pr.F_impact is not None:
            segments.append((pr.F_impact / tf, 1.0, color_follow))

        bar.set_segments(segments)
        bar.set_markers(markers)

        # 強制設定初始位置與圖層 (避免尚未觸發 ResizeEvent 時不可見)
        slider = self.frameSlider1 if player_idx == 1 else self.frameSlider2
        opt = QtWidgets.QStyleOptionSlider()
        slider.initStyleOption(opt)
        groove_rect = slider.style().subControlRect(
            QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, slider
        )
        bar_h = 100
        bar.setGeometry(
            groove_rect.left(),
            slider.height() // 2 - bar_h // 2,
            max(groove_rect.width(), slider.width()), # 確保寬度
            bar_h
        )
        bar.raise_()
        bar.show()

    # ================= 3D 指標（背景程序） =================

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

    def process_raw(self, player_idx):
        """還沒處理過的錄影：交給背景程序做雙相機偵測與分析，完成後載入骨架影片與指標"""
        p = self.player(player_idx)
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
