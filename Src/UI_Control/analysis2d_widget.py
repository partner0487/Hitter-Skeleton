from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QFont, QPen, QBrush, QStandardItemModel, QStandardItem
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QRect
from PyQt5.QtWidgets import QSizePolicy
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QFileDialog, QGraphicsView, QGraphicsScene, QMessageBox, QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, QComboBox
import numpy as np
import pandas as pd
import sys
import os
import collections
from typing import Optional
from ui.dual_camera_system_record import Ui_hitter_ui_dual
from datetime import datetime
from cv_utils.cv_control_dual import Camera, VideoLoader
from utils.selector import PersonSelector
from utils.analyze import PoseAnalyzer, BatAnalyzer
import cv2
from utils.vis_image_dual import ImageDrawer
from skeleton.detect_skeleton_dual import PoseEstimater
import pyqtgraph as pg
from utils.model import Model
from utils.phase_detect2d import PhaseDetector, PhaseResult
from ocr import ocr_capture_1time as ocr
from utils.paths import DB_DIR, ICON_DIR, NON_HITTER_DIRS, clip_title
from ui_utils.analysis_bar import AnalysisBar
from utils.fps_utils import NOMINAL_DURATION_S, frame_seconds, get_video_fps, phase_seconds
from utils.bat_speed2d import read_frame_times
from utils.metrics2d import bat_metrics_2d, body_metrics_2d, knee_angles_2d, side_scale
from utils.stereo_calibration import load_stereo_params
from utils.body_metrics import SideScale
from utils.bat_profile import bat_for_clip
from ui_utils.metric_grid import MISSING, align_columns, arrange_metrics, fmt, new_label_like, rapsodo_texts, show_spray_chart


class EnlargedFrameDialog(QDialog):
    """Dialog to display an enlarged frame."""
    def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Enlarged Frame")
            self.setMinimumSize(800, 600)  # Set a reasonable size for the dialog
            self.layout = QVBoxLayout(self)
            
            # Create QGraphicsView for the enlarged frame
            self.scene = QGraphicsScene()
            self.view = QGraphicsView(self)
            self.view.setScene(self.scene)
            self.layout.addWidget(self.view)
            
            # Close dialog on click
            self.view.mousePressEvent = self.close
            self.setModal(False)

    def updateFrame(self, image: np.ndarray):
        """Update the displayed image in the QGraphicsView."""
        self.scene.clear()
        if image is None:
            return
        h, w = image.shape[:2]
        qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.scene.addPixmap(pixmap)
        self.view.setScene(self.scene)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


class Analysis2dWidget(QWidget):
    def __init__(self, model:Model, parent=None):
        super().__init__(parent)
        self.ui = Ui_hitter_ui_dual()
        self.ui.setupUi(self)
        self.arrangeMetricLabels()
        self.model = model
        self.setupComponents()
        self.initVar()
        self.bindUI()

    def arrangeMetricLabels(self):
        """關鍵幀指標排成與 3D 頁相同的格線（ui_utils/metric_grid.py）。
        肩髖分離角是繞垂直軸的轉動，單一側面相機看不到；揮棒垂直角度在擊球時球棒朝側面相機伸出、
        2D 投影會失真，這兩項只在 3D 頁顯示"""
        ui = self.ui
        ui.horizontalAngleLabel.setText("攻擊角度:")
        # 跨步時間（離地 → 著地）、擊球時間（著地 → 擊球），放在離地區塊右欄
        self.strideTimeCaption = new_label_like(ui.centerGravity, "跨步時間:")
        self.strideTimeLabel = new_label_like(ui.centerGravityLabel)
        self.swingTimeCaption = new_label_like(ui.centerGravity, "擊球時間:")
        self.swingTimeLabel = new_label_like(ui.centerGravityLabel)
        self.strideTimeLabel.setToolTip("前腳離地到前腳著地")
        self.swingTimeLabel.setToolTip("前腳著地到擊球；Blast 的 Time to Contact 從開始下揮算起，這裡以著地為起點")
        arrange_metrics([
            (ui.gridLayout_6, [[(ui.centerGravity, ui.centerGravityLabel), (self.strideTimeCaption, self.strideTimeLabel)],
                               [None, (self.swingTimeCaption, self.swingTimeLabel)]]),
            (ui.gridLayout_4, [[(ui.centerGravity2, ui.centerGravityLabel2), (ui.strideDis, ui.strideDisLabel)]]),
            (ui.gridLayout_5, [[(ui.batSpeed, ui.batSpeedLabel), (ui.horizontalAngleLabel, ui.horizontalAngleEditLabel)],
                               [(ui.forefoot3, ui.forefootLabel3), (ui.backfoot3, ui.backfootLabel3)]]),
            (ui.gridLayout_8, [[(ui.exitV, ui.exitVLabel), (ui.lanuchAngle, ui.lanuchAngleLabel)],
                               [(ui.attackDis, ui.attackDisLabel)]]),
        ])
        align_columns([ui.verticalLayout_5, ui.verticalLayout_7, ui.verticalLayout_9, ui.verticalLayout_10])
        # 前腳、後腳膝蓋是側面投影角，膝蓋彎曲有一部分朝向相機，會低估彎曲，以 3D 頁為準
        for label in (ui.forefootLabel3, ui.backfootLabel3):
            label.setToolTip("側面 2D 投影角，膝蓋往相機方向彎時會失真；以 3D 頁為準")

    def shutdown(self):
        """關閉程式前停止計時器、影片載入與相機"""
        self.timer.stop()
        self.video_timer.stop()
        if self.video_loader.is_loading:
            self.video_loader.abort_loading()
        # close_camera 會一併停止擷取執行緒並釋放錄影 writer
        self.camera.close_camera()

    def initVar(self):
        """Initialize variables and timer."""
        self.is_video = True if self.camera is None else False
        self.view_scene = QGraphicsScene()
        self.view_scene2 = QGraphicsScene()
        self.view_scene.clear()
        self.view_scene2.clear()
        self.is_hitting = False
        self.enlarged_dialog = None  # Reference to the EnlargedFrameDialog
        self.enlarged_view_is_frame_view = 0  # Track which view (FrameView or FrameView_2) was clicked
        self.initVideoVar()
        self.cached_keyphase_pixmaps = {}
        self.ui.redRatioSlider.setMinimum(25)
        self.ui.redRatioSlider.setMaximum(400)
        self.ui.blueRatioSlider.setMinimum(25)
        self.ui.blueRatioSlider.setMaximum(400)
        self.ui.gainSlider.setMinimum(0)
        self.ui.gainSlider.setMaximum(300)
        self.watchTime = 3000
        self.stability_buffer = []  # 儲存最近 N 幀的手腕位置
        self.is_ready = False
        self.is_process = False
        self.is_manual_recording = False
        self.mode_dict = {"Tbar": "T座", "Side": "側拋", "Front": "正投"}
        self.current_mode = "Tbar"
        self.hitBallNum = 1
        if hasattr(self.ui, 'hitballNum'):
            self.ui.hitballNum.setText(f"球數: {self.hitBallNum:02d}")
        if hasattr(self.ui, 'modeDisplay'):
            self.ui.modeDisplay.setText(self.mode_dict[self.current_mode])
        
        # --- Multi-stage Trigger Settings ---
        # Strike Zone ROI: [x, y, w, h] (Side view default for 1920x1080)
        self.strike_zone_roi = [800, 200, 800, 600] 
        self.ball_pos_history = collections.deque(maxlen=10)
        self.bat_head_history = collections.deque(maxlen=10)
        self.trigger_cooldown = 0
        self.ball_was_in_roi = False
        self.is_roi_drawing = False
        self.roi_start_pos = None
        self.roi_temp_rect = None
        self.roi_drag_mode = None # 'move', 'top', 'bottom', 'left', 'right', 'top-left', etc.
        self.roi_last_mouse_pos = None

    def initVideoVar(self):
        self.is_play = False
        self.is_processed = False
        self.play_times = 1
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

    def setupComponents(self): 
        self.ui.frameSlider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.ui.frameSlider.setMinimumHeight(100)
        self.analysis_bar = AnalysisBar(self.ui.frameSlider)
        self.ui.frameSlider.installEventFilter(self)
        self.camera = Camera()
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyzeFrame)
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.nextFrame)

        # ROI Button (使用您在 UI 檔中新增的按鈕)
        self.roiBtn = self.ui.roiBtn
        self.roiBtn.setCheckable(True)
        self.roiBtn.setStyleSheet("QPushButton:checked { background-color: #ffaa00; color: black; font-weight: bold; }")
        self.roiBtn.clicked.connect(self.toggleROIDrawing)

        self.pose_estimater = PoseEstimater(self.model)
        self.pose_estimater_2 = PoseEstimater(self.model)
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater)
        self.pose_analyzer_2 = PoseAnalyzer(self.pose_estimater_2)
        self.bat_analyzer = BatAnalyzer(self.pose_estimater)
        self.bat_analyzer_2 = BatAnalyzer(self.pose_estimater_2)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer, self.bat_analyzer)
        self.image_drawer_2 = ImageDrawer(self.pose_estimater_2, self.pose_analyzer_2, self.bat_analyzer_2)
        self.video_loader = VideoLoader(self.image_drawer,self.image_drawer_2)
    
    def resizeEvent(self, event):
        new_size = event.size()
        # 在此執行你想要的操作
        if self.video_loader.video_name is not None:
            self.updateFrame(frame_num=self.ui.frameSlider.value())
        super().resizeEvent(event)  

    def bindUI(self):
        """Bind UI element to their corresponding functions."""
        self.ui.batterInput.currentIndexChanged.connect(self.changeHitter)
        self.updateHitterList() # 啟動時先掃描一次資料庫
        self.ui.hitterNameInput.currentIndexChanged.connect(self.populateRecentVideosList)
        self.ui.hitterNameAddInputBtn.clicked.connect(self.addNewHitter)
        self.bindVideoUI()
        self.bindCheckBox()
        if hasattr(self.ui, 'recordBtn'):
            self.ui.recordBtn.clicked.connect(self.toggleManualRecord)
        if hasattr(self.ui, 'listView'):
            self.ui.listView.clicked.connect(self.onRecentVideoSelected)
            self.populateRecentVideosList()
        self.ui.redRatioSlider.valueChanged.connect(self.onRedRatioChanged)
        self.ui.blueRatioSlider.valueChanged.connect(self.onBlueRatioChanged)
        self.ui.gainSlider.valueChanged.connect(self.onGainChanged)
        self.ui.sideCamera.toggled.connect(self.updateCameraSliders)
        self.ui.impactBtn.clicked.connect(lambda: self.ui.frameSlider.setValue(self.phase_result.F_impact))
        self.ui.landingBtn.clicked.connect(lambda: self.ui.frameSlider.setValue(self.phase_result.F_landing))
        self.ui.liftoffBtn.clicked.connect(lambda: self.ui.frameSlider.setValue(self.phase_result.F_liftoff))
        
        if hasattr(self.ui, 'modeTbar'):
            self.ui.modeTbar.clicked.connect(lambda: self.setMode("Tbar"))
        if hasattr(self.ui, 'modeSide'):
            self.ui.modeSide.clicked.connect(lambda: self.setMode("Side"))
        if hasattr(self.ui, 'modeFront'):
            self.ui.modeFront.clicked.connect(lambda: self.setMode("Front"))

        # 安裝 EventFilter 給 keyphaseLabel1~3 以偵測 Resize
        if hasattr(self.ui, 'keyphaseLabel1'):
            self.ui.keyphaseLabel1.installEventFilter(self)
        if hasattr(self.ui, 'keyphaseLabel2'):
            self.ui.keyphaseLabel2.installEventFilter(self)
        if hasattr(self.ui, 'keyphaseLabel3'):
            self.ui.keyphaseLabel3.installEventFilter(self)

    def eventFilter(self, source, event):
        if source == self.ui.frameSlider and event.type() == QEvent.Resize:
            opt = QtWidgets.QStyleOptionSlider()
            source.initStyleOption(opt)
            
            # 獲取滑軌的精確矩形
            groove_rect = source.style().subControlRect(
                QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, source
            )
            
            handle_rect = source.style().subControlRect(
                QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, source
            )
            self.analysis_bar.handle_px = handle_rect.width()
            bar_h = 100 # ❗ 提高控制項容器高度
            # 讓 AnalysisBar 的寬度完全等於滑軌的寬度，位置與滑軌起始點對齊
            self.analysis_bar.setGeometry(
                groove_rect.left(), 
                source.height() // 2 - bar_h // 2,
                groove_rect.width(), 
                bar_h
            )
            self.analysis_bar.raise_()
            return False # 不攔截事件，讓 QSlider 正常處理 Resize 邏輯

        if event.type() == QEvent.Resize:
            if hasattr(self.ui, 'keyphaseLabel1') and source == self.ui.keyphaseLabel1:
                self.update_keyphase_label_image(self.ui.keyphaseLabel1)
            elif hasattr(self.ui, 'keyphaseLabel2') and source == self.ui.keyphaseLabel2:
                self.update_keyphase_label_image(self.ui.keyphaseLabel2)
            elif hasattr(self.ui, 'keyphaseLabel3') and source == self.ui.keyphaseLabel3:
                self.update_keyphase_label_image(self.ui.keyphaseLabel3)
        return super().eventFilter(source, event)

    def setMode(self, mode):
        self.current_mode = mode
        if hasattr(self.ui, 'modeDisplay'):
            self.ui.modeDisplay.setText(self.mode_dict[mode])
        self.populateRecentVideosList()

    def toggleROIDrawing(self, checked):
        """切換 ROI 繪製模式"""
        self.is_roi_drawing = checked
        if checked:
            self.roiBtn.setText("請由左上往右下拖曳繪製")
            # --- 核心改動：進入設定模式時先清空，讓使用者可以「重新拉一個」 ---
            self.strike_zone_roi = None
            
            self.setMouseTracking(True)
            self.ui.FrameView_2.setMouseTracking(True)
        else:
            self.roiBtn.setText("設定打區 (ROI)")
            self.setCursor(Qt.ArrowCursor)
            self.roi_drag_mode = None
            self.roi_start_pos = None
            self.roi_temp_rect = None

    def bindVideoUI(self):
        self.ui.playBtn.clicked.connect(self.playBtnClicked)
        self.ui.backKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1)
        )
        self.ui.forwardKeyBtn.clicked.connect(
            lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1)
        )
        self.ui.frameSlider.valueChanged.connect(self.analyzeFrame)
        
        self.ui.loadVideoBtn.clicked.connect(lambda: self.loadVideo4Process())
        self.ui.FrameView.mousePressEvent = self.mousePressEvent
        self.ui.FrameView_2.mousePressEvent = self.mousePressEvent
        self.ui.FrameView.mouseMoveEvent = self.mouseMoveEvent
        self.ui.FrameView_2.mouseMoveEvent = self.mouseMoveEvent
        self.ui.FrameView.mouseReleaseEvent = self.mouseReleaseEvent
        self.ui.FrameView_2.mouseReleaseEvent = self.mouseReleaseEvent

    def loadVideo4Process(self):
        self.is_process = True
        self.is_video = True
        self.is_dual = True
        self.toggleSelect(0)
        self.toggleShowSkeleton(0)
        self.toggleShowRegion(0)
        self.is_processed_json = False
        self.video_loader.loadVideo()
        self.checkVideoLoad()

    def populateRecentVideosList(self):
        """抓取目前選中選手的最新 5 個含有影片的資料夾"""
        if not hasattr(self.ui, 'listView'): return
        
        # 1. 取得目前選中的選手姓名
        hitter_name = self.ui.hitterNameInput.currentText()
        if not hitter_name:
            # 清空清單並返回
            if hasattr(self, 'recent_video_model'):
                self.recent_video_model.clear()
            return

        db_path = str(DB_DIR)
        hitter_path = os.path.join(db_path, hitter_name)
        
        # 若該選手資料夾不存在，清空清單並返回
        if not os.path.exists(hitter_path):
            self.recent_video_model = QStandardItemModel()
            self.ui.listView.setModel(self.recent_video_model)
            return

        selected_folders = []
        
        # 2. 取得日期資料夾並排序 (由新到舊)
        try:
            date_folders = [d for d in os.listdir(hitter_path) if os.path.isdir(os.path.join(hitter_path, d))]
            date_folders.sort(reverse=True)
            
            for d_folder in date_folders:
                d_path = os.path.join(hitter_path, d_folder)
                
                # 檢查該日期下是否有當前模式的資料夾 (使用中文對應名稱)
                current_mode_zh = self.mode_dict.get(getattr(self, 'current_mode', 'Tbar'), 'T座')
                mode_path = os.path.join(d_path, current_mode_zh)
                if not os.path.exists(mode_path):
                    continue
                
                # 3. 取得該日期與模式下的 Session (時間) 資料夾並排序 (由大到小)
                sessions = [s for s in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, s))]
                sessions.sort(reverse=True)
                
                for s_folder in sessions:
                    s_path = os.path.join(mode_path, s_folder)
                    
                    # 4. 檢查資料夾內是否含有影片 (.mp4)
                    files = os.listdir(s_path)
                    if any(f.lower().endswith('.mp4') for f in files):
                        selected_folders.append((d_folder, s_folder, s_path))
        except Exception as e:
            print(f"Populate recent videos error: {e}")

        # 5. 更新 UI 模型
        self.recent_video_model = QStandardItemModel()
        for d_f, s_f, full_path in selected_folders:
            # 顯示名稱格式：日期 - 模式 - 時間 (例如 2026-04-06 - T座 - 113305)
            # ※ 根據使用者要求，上面已有選手名稱 (Player1)，下面就不用再顯示了
            mode_display_text = self.mode_dict.get(getattr(self, 'current_mode', 'Tbar'), 'T座')
            display_text = f"{d_f} - {mode_display_text} - {s_f}"
            item = QStandardItem(display_text)
            item.setData(full_path, Qt.UserRole)
            self.recent_video_model.appendRow(item)
            
        self.ui.listView.setModel(self.recent_video_model)

    def onRecentVideoSelected(self, index):
        """比照自動化流程：找到路徑包，載入並處理"""
        folder_path = self.recent_video_model.itemFromIndex(index).data(Qt.UserRole)
        
        # 在選擇影片時先清空 UI，等待 Keyphase 切好後再顯示
        if hasattr(self.ui, 'exitVLabel'): self.ui.exitVLabel.setText("")
        if hasattr(self.ui, 'lanuchAngleLabel'): self.ui.lanuchAngleLabel.setText("")
        if hasattr(self.ui, 'attackDisLabel'): self.ui.attackDisLabel.setText("")
        if hasattr(self.ui, 'rapsodoLabel'): self.ui.rapsodoLabel.clear()

        import glob
        mp4_files = sorted(glob.glob(os.path.join(folder_path, '*.mp4')))
        
        if len(mp4_files) < 2: return

        # 固定配對：第一個為 Front(CF), 第二個為 Side(CS)
        self.video_path = mp4_files[0]
        self.video_path2 = mp4_files[1]
        
        # 進入與自動錄影後相同的處理流程
        self.is_process = self.is_video = self.is_dual = True
        self.is_processed_json = True
        self.video_loader.loadVideo(self.video_path, self.video_path2)
        self.checkVideoLoad()

    def bindCheckBox(self):
        """Bind UI CheckBox to their corresponding functions."""
        self.ui.cameraCheckBox.stateChanged.connect(self.toggleCamera)
        self.ui.showRegionCheckBox.stateChanged.connect(self.toggleShowRegion)
        self.ui.startHitCheckBox.stateChanged.connect(self.toggleHitting)
        self.ui.watchTimeCombobox.currentTextChanged.connect(self.setWatchTime)

    def addNewHitter(self):
        """新增選手功能"""
        new_name = self.ui.hitterNameAddInput.text().strip()
        
        if not new_name:
            QMessageBox.warning(self, "錯誤", "請輸入選手姓名")
            return

        # 定義資料庫基礎路徑 (參考 startRecording 中的路徑結構)
        db_base_path = str(DB_DIR)
        hitter_path = os.path.join(db_base_path, new_name)

        try:
            # 1. 建立選手資料夾
            if not os.path.exists(hitter_path):
                os.makedirs(hitter_path)
            else:
                # 如果資料夾已存在，仍允許加入選單，但給予提示
                QMessageBox.information(self, "提示", f"選手 {new_name} 已存在於資料庫中")

            # 2. 更新下拉選單
            # 檢查是否已經在選單中，避免重複
            existing_items = [self.ui.hitterNameInput.itemText(i) for i in range(self.ui.hitterNameInput.count())]
            
            if new_name not in existing_items:
                self.ui.hitterNameInput.addItem(new_name)
                
            # 3. 自動切換到該選手並清空輸入框
            index = self.ui.hitterNameInput.findText(new_name)
            self.ui.hitterNameInput.setCurrentIndex(index)
            self.ui.hitterNameAddInput.clear()
            
            print(f"成功新增選手: {new_name}, 路徑: {hitter_path}")

        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法建立選手資料: {str(e)}")

    def onRedRatioChanged(self, value):
        ratio = value / 100.0
        self.ui.redRatioValue.setText(f"{ratio:.2f}")
        cam = self.get_selected_camera()
        if cam is not None:
            cam.update_white_balance(red_ratio = ratio)
    
    def onBlueRatioChanged(self, value):
        ratio = value / 100.0
        self.ui.blueRatioValue.setText(f"{ratio:.2f}")
        cam = self.get_selected_camera()
        if cam is not None:
            cam.update_white_balance(blue_ratio = ratio)

    def onGainChanged(self, value):
        ratio = value / 10.0
        self.ui.gainValue.setText(f"{ratio:.1f}")
        cam = self.get_selected_camera()
        if cam is not None:
            cam.update_white_balance(gain = ratio)


    def playBtnClicked(self):
        if self.video_loader.video_name == "" or self.video_loader.video_name2 == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
            
        self.is_play = not self.is_play
        self.ui.playBtn.setText("||" if self.is_play else "▶︎")
        
        if self.is_play:
            if self.ui.frameSlider.value() >= self.video_loader.total_frames - 1:
                self.ui.frameSlider.setValue(0)
            self.updatePlaybackRate()
        else:
            self.video_timer.stop()

    def nextFrame(self):
        """Timer slot to advance to the next frame."""
        current_frame = self.ui.frameSlider.value()
        if current_frame < self.video_loader.total_frames - 1:
            self.ui.frameSlider.setValue(current_frame + 1)
        else:
            self.handleVideoEnd()

    def updatePlaybackRate(self):
        """Update timer interval based on playback rate."""
        try:
            current_rate = float(self.ui.slowMotionInput.currentText())
            delay = int(11 / current_rate) 
            if delay < 1: delay = 1
            
            if self.is_play:
                self.video_timer.start(delay)
        except Exception:
            # Fallback if slowMotionInput is not found or invalid
            if self.is_play:
                self.video_timer.start(30)

    def videoSilder(self, visible:bool):
        elements = [
            self.ui.backKeyBtn,
            self.ui.playBtn,
            self.ui.forwardKeyBtn,
            self.ui.frameSlider,
            self.ui.frameNumLabel
        ]
        
        for element in elements:
            element.setVisible(visible)

    def changeHitter(self):
        """Change the pitcher based on input value. 9: "左腕", 10: "右腕","""
        kpt_id = 10 if self.ui.batterInput.currentIndex() == 0 else 9
        self.pose_estimater.setKptId(kpt_id)
        
        # Reset ball count when hitter changes
        self.hitBallNum = 1
        if hasattr(self.ui, 'hitballNum'):
            self.ui.hitballNum.setText(f"球數: {self.hitBallNum:02d}")

    def get_selected_camera(self):
        if self.ui.sideCamera.isChecked():
            return self.camera.video_thread.camera2 # 側面
        else:
            self.ui.frontCamera.setChecked(True)
            return self.camera.video_thread.camera1 # 正面
        
    def updateCameraSliders(self):
        """根據目前選擇的攝影機更新 slider 和 label 的值"""
        cam = self.get_selected_camera()
        try:
            gain_value = cam.full_config['gain_settings']['gain_value']
            redRatio_value = cam.full_config['white_balance_settings']['white_balance_red_ratio']
            blueRatio_value = cam.full_config['white_balance_settings']['white_balance_blue_ratio']
            gain_slider_value = int(gain_value * 10)
            red_slider_value = int(redRatio_value * 100)
            blue_slider_value = int(blueRatio_value * 100)
            self.ui.gainSlider.setValue(gain_slider_value)
            self.ui.gainValue.setText(f"{gain_value:.1f}")
            self.ui.redRatioSlider.setValue(red_slider_value)
            self.ui.redRatioValue.setText(f"{redRatio_value:.2f}")
            self.ui.blueRatioSlider.setValue(blue_slider_value)
            self.ui.blueRatioValue.setText(f"{blueRatio_value:.2f}")
        except Exception as e:
            print(f"讀取 config 設定失敗: {e}")

    def toggleCamera(self, state:int):
        """Toggle the camera on/off based on checkbox state."""
        self.is_process = False
        if state == 2:
            if self.is_play:
                self.ui.cameraCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開啟相機", "請先暫停播放影片")
                return
            self.reset()
            self.toggleSelect(0)
            self.toggleShowSkeleton(0)
            self.trigger_cooldown = 0
            self.ball_was_in_roi = False
            self.is_video = False
            frame_width, frame_height, fps = self.camera.toggle_camera(True)
            self.model.setImageSize((frame_width, frame_height))
            self.ui.ResolutionLabel.setText(f"(0, 0) - ({frame_width} x {frame_height}), FPS: {fps}")
            self.timer.start(1)
            self.videoSilder(visible=False)
            self.updateCameraSliders()
        else: 
            if self.camera is not None:
                self.camera.toggle_camera(False)
            if self.timer is not None:
                self.timer.stop()
            self.view_scene.clear()
            self.view_scene2.clear()

            self.is_video = True
            self.videoSilder(visible=True)

    def toggleHitting(self, state):
        if state == 2:
            if not self.ui.cameraCheckBox.isChecked():
                self.ui.startHitCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法開始打擊模式", "請先開啟相機")
                return
            self.toggleSelect(2)
            self.toggleShowSkeleton(2)
            self.is_hitting = True
        else:
            self.is_hitting = False

    def saveExistingVideo(self):
        """直接儲存已經在記憶體中的 full_video"""
        # 1. 取得影像資料
        full_video = self.camera.continuous_recorder.full_video
        
        if not full_video:
            print("沒有影像資料可以儲存！")
            return

        # 2. 準備路徑 (建議統一抓一次時間，避免誤差)
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H%M%S")
        current_time_hhmm = now.strftime("%H%M")
        
        # 取得 FPS：優先用實測值（反映戶外過熱掉幀），否則用設定值換算
        recorder = self.camera.continuous_recorder
        fps = recorder.measured_fps
        if not fps:
            try:
                fps = self.camera.get_stream_fps()
            except Exception:
                fps = 60

        hitter_name = self.ui.hitterNameInput.currentText()
        current_mode_zh = self.mode_dict.get(getattr(self, 'current_mode', 'Tbar'), 'T座')
        ball_folder = f"{hitter_name}_H{self.hitBallNum:02d}"
        output_dir = os.path.join(DB_DIR, hitter_name, current_date, current_mode_zh, ball_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        # Increment ball count and update UI
        video_filename = os.path.join(output_dir, f'CF_{ball_folder}.mp4')
        video_filename_2 = os.path.join(output_dir, f'CS_{ball_folder}.mp4')
        self.video_path = video_filename
        self.video_path2 = video_filename_2
        frame1, frame2 = full_video[0]
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out1 = cv2.VideoWriter(video_filename, fourcc, fps, (w1, h1))
        out2 = cv2.VideoWriter(video_filename_2, fourcc, fps, (w2, h2))
        
        for f1, f2 in full_video:
            out1.write(f1)
            out2.write(f2)

        out1.release()
        out2.release()
        # 每幀實際擷取時間：掉幀時 mp4 的固定 fps 不準，分析應以此為準
        if recorder.frame_times:
            t0 = recorder.frame_times[0]
            with open(os.path.join(output_dir, "frame_times.csv"), "w", encoding="utf-8") as f:
                f.write("frame,time_sec\n")
                for i, t in enumerate(recorder.frame_times):
                    f.write(f"{i},{t - t0:.6f}\n")

        print(f"Videos saved: {video_filename}, {video_filename_2}")

        # --- 即時模式下也儲存側面 JSON 資料 (即使是空的) ---
        try:
            self.pose_estimater_2.person_df.to_json(os.path.join(output_dir, "person_side.json"), orient='records')
            
            if hasattr(self.pose_estimater_2, 'bat_df') and self.pose_estimater_2.bat_df is not None:
                self.pose_estimater_2.bat_df.to_json(os.path.join(output_dir, "bat_side.json"), orient='records')
            
            if hasattr(self.pose_estimater_2, 'ball_df') and self.pose_estimater_2.ball_df is not None:
                self.pose_estimater_2.ball_df.to_json(os.path.join(output_dir, "ball_side.json"), orient='records')
                
            if hasattr(self.bat_analyzer_2, 'batHeadTail_df') and self.bat_analyzer_2.batHeadTail_df is not None:
                self.bat_analyzer_2.batHeadTail_df.to_json(os.path.join(output_dir, "batht_side.json"), orient='records')
                
            print(f"Side view JSON files saved (realtime) to {output_dir}")
        except Exception as e:
            print(f"Error saving realtime JSON files: {e}")
        # -----------------------------------------------

        # Increment ball count after successful save or recording start
        self.hitBallNum += 1
        if hasattr(self.ui, 'hitballNum'):
            self.ui.hitballNum.setText(f"球數: {self.hitBallNum:02d}")

        # OCR 觸發移至 resetRecordTimer 以支援手動錄影

    def toggleRecord(self, state:int):
        """Start or stop video recording."""
        if state == 2:
            self.startRecording()
        else:
            if self.camera is None:
                return
            self.camera.stop_recording()

    def toggleManualRecord(self):
        """處理手動錄影按鈕的點擊事件"""
        if not self.is_manual_recording:
            # 1. 啟動錄影
            self.startRecording()
            if self.camera is not None and getattr(self.camera, 'is_opened', False):
                self.ui.recordBtn.setText("停止錄影")
                self.is_manual_recording = True
                print("手動錄影開始！")
        else:
            # 2. 停止錄影並進行分析
            if self.camera is not None:
                self.camera.stop_recording()
                
            self.ui.recordBtn.setText("開始錄影")
            self.is_manual_recording = False
            print("手動錄影結束，即將跳到處理2D介面...")
            
            # 使用已儲存的 video_path 觸發讀取與介面轉移 (如同自動錄影)
            self.resetRecordTimer()

    def startRecording(self):  
        """Start recording the video."""
        if self.camera is None:
            self.toggleRecord(0)
            return
        now = datetime.now()
        current_date = now.strftime("%Y-%m-%d")
        current_time = now.strftime("%H%M%S")
        current_time_hhmm = now.strftime("%H%M")
        try:
            fps = int(self.camera.video_thread.camera1.get_fps())
        except Exception:
            fps = 180

        hitter_name = self.ui.hitterNameInput.currentText()
        current_mode_zh = self.mode_dict.get(getattr(self, 'current_mode', 'Tbar'), 'T座')
        ball_folder = f"{hitter_name}_H{self.hitBallNum:02d}"
        output_dir = os.path.join(DB_DIR, hitter_name, current_date, current_mode_zh, ball_folder)
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.join(output_dir, f'CF_{ball_folder}.mp4')
        video_filename_2 = os.path.join(output_dir, f'CS_{ball_folder}.mp4')
        
        # 紀錄已錄製影片路徑供後續自動載入分析
        self.video_path = video_filename  
        self.video_path2 = video_filename_2
        
        self.camera.start_recording(video_filename,video_filename_2)

        # Increment ball count
        self.hitBallNum += 1
        if hasattr(self.ui, 'hitballNum'):
            self.ui.hitballNum.setText(f"球數: {self.hitBallNum:02d}")

    def toggleSelect(self, state:int):
        """Select a person based on checkbox state."""
        if state == 2:
            self.person_selector = PersonSelector()
            self.person_selector_2 = PersonSelector()
            frame_num = self.ui.frameSlider.value() if self.is_video else None
            search_person_df_2 = self.pose_estimater_2.getPersonDf(frame_num=frame_num) if frame_num is not None else self.pose_estimater_2.pre_person_df
            self.person_selector_2.select(search_person_df = search_person_df_2)
            self.pose_estimater_2.setPersonId(self.person_selector_2.selected_id)
            if search_person_df_2 is not None and not search_person_df_2.empty:
                self.person_selector_2.select(search_person_df=search_person_df_2)
                self.pose_estimater_2.setPersonId(self.person_selector_2.selected_id)
                print(f"成功鎖定側面相機人物 ID: {self.pose_estimater_2.person_id}")

        else:
            self.pose_estimater_2.setPersonId(None)
            self.person_selector_2 = None

    def toggleShowSkeleton(self, state:int):
        """Toggle skeleton detection and FPS control."""
        is_checked = state == 2
        if not is_checked:
            self.toggleSelect(0)
            
        self.pose_estimater.setDetect(is_checked)
        self.pose_estimater_2.setDetect(is_checked)
        self.image_drawer.setShowSkeleton(is_checked)
        self.image_drawer_2.setShowSkeleton(is_checked)

    def toggleShowRegion(self, state:int):
        """Toggle region display."""
        is_checked = state == 2
        self.image_drawer.setShowRegion(is_checked)
        self.image_drawer_2.setShowRegion(is_checked)

    def analyzeFrame(self):
        """Analyze and process each frame from the camera or video"""
        if self.is_video:
            frame_num = self.ui.frameSlider.value()
            # 顯示格式： 0.45/1.5s（有 frame_times.csv 時為實際擷取時間）
            self.ui.frameNumLabel.setText(self.frameTimeText(frame_num, self.video_loader.total_frames))
            frame = self.video_loader.getVideoImage(frame_num)
            frame_2 = self.video_loader.getVideoImage2(frame_num)
            
            _, df2, _, self.bat_side_df, self.ball_side_df = self.pose_estimater_2.detectKpt(frame_2, frame_num, is_video=True)
            self.updateFrame(frame_num=frame_num, frame=frame, frame_2=frame_2)
            if frame_num == self.video_loader.total_frames - 1:
                self.handleVideoEnd()

        else:   
            if not self.camera.frame_buffer.empty() and not self.camera.frame_buffer_2.empty():
                frame = self.camera.frame_buffer.get().copy()
                frame_2 = self.camera.frame_buffer_2.get().copy()
                if self.is_hitting:
                    self.hitterAnaylze(frame_2)
                self.updateFrame(frame=frame, frame_2=frame_2)

    def detectPhases(self):
        # 這裡直接做階段偵測
        self.phase_detector = None  # 避免沿用上一支影片的棒頭軌跡
        # 確保獲取到最新的 DataFrame
        self.person_side_df = self.pose_estimater_2.person_df
        # 也要獲取球和球棒的 DataFrame (如果 detectKpt 有更新它們的話)
        self.ball_side_df = self.pose_estimater_2.ball_df
        self.bat_side_df = self.pose_estimater_2.bat_df
        if self.bat_analyzer_2 is not None and self.bat_analyzer_2.batHeadTail_df is not None:
             self.batht_side_df = self.bat_analyzer_2.batHeadTail_df
        if not self.pose_estimater_2.person_df.empty:
            self.person_side_df = self.pose_estimater_2.person_df
            batter_idx = self.ui.batterInput.currentIndex()
            detector = PhaseDetector(
                person_df=self.person_side_df,
                batter_index=batter_idx,
                ball_df=self.ball_side_df,
                bat_df=self.bat_side_df,
                th_lift=5.0,
                th_land=5.0,
                land_hold=3,
            )
            self.phase_detector = detector  # 揮棒速度也用它的棒頭軌跡
            self.phase_result = detector.detect_phases()
            
        if not hasattr(self, 'phase_result') or self.phase_result is None:
            self.phase_result = PhaseResult()
            
        # --- Fallback 找不到 keyphase 時的填補邏輯 ---
        total = self.video_loader.total_frames if hasattr(self, 'video_loader') and self.video_loader else 0
        f1 = self.phase_result.F_liftoff
        f2 = self.phase_result.F_landing
        f3 = self.phase_result.F_impact

        # 如果三個階段都沒偵測到，就依據使用者需求「不畫」，直接跳過填補
        if f1 is None and f2 is None and f3 is None:
            print("[Analysis2D] No phases detected, skipping fallback.")
        else:
            # 1. 若有離地跟擊球，缺著地 -> 著地 = 離地 + 10
            if f1 is not None and f3 is not None and f2 is None:
                f2 = f1 + 10
                if f2 >= f3: # 防呆，不可超過擊球
                    f2 = f1 + max(1, (f3 - f1) // 2)

            # 2. 若都沒有或還缺，依序填補
            if f1 is None:
                if f2 is not None: f1 = max(0, f2 - 10)
                else: f1 = 15 # 預設一開始

            if f2 is None:
                f2 = f1 + 10
                
            if f3 is None:
                f3 = f2 + 5

            # 限幅與順序防呆
            if total > 0:
                f1 = min(f1, total - 1)
                f2 = min(f2, total - 1)
                f3 = min(f3, total - 1)
                if f2 <= f1: f2 = min(f1 + 1, total - 1)
                if f3 <= f2: f3 = min(f2 + 1, total - 1)

                self.phase_result.F_liftoff = int(f1)
                self.phase_result.F_landing = int(f2)
                self.phase_result.F_impact = int(f3)
        # -----------------------------------------------

        self.setPhaseResult(self.phase_result)
        print("PhaseResult =", self.phase_result)

    def updatePhaseTimes(self, phase_result):
        """跨步時間、擊球時間，與進度條上各段秒數一致"""
        total = int(getattr(self.video_loader, "total_frames", 0) or 0)
        times = getattr(self, "frame_times", None)
        stride = phase_seconds(phase_result.F_liftoff, phase_result.F_landing, total, times)
        swing = phase_seconds(phase_result.F_landing, phase_result.F_impact, total, times)
        self.strideTimeLabel.setText(fmt(stride, "s", 2))
        self.swingTimeLabel.setText(fmt(swing, "s", 2))

    def setPhaseResult(self, phase_result: PhaseResult):
        total = self.video_loader.total_frames
        if total > 0:
            f1 = phase_result.F_liftoff
            f2 = phase_result.F_landing
            f3 = phase_result.F_impact
            
            color_prepare = QColor(150, 150, 150)   # 亮天藍 (Deep Sky Blue)
            color_stride  = QColor(255, 69, 0)    # 亮橘紅 (Orange Red)
            color_swing = QColor(180, 165, 32)
            color_follow = QColor(150, 150, 150)

            segs = []
            # 1. 開始 -> Liftoff
            if f1 is not None:
                segs.append((0, f1/(total-1), color_prepare))
            
            # 2. Liftoff -> Landing
            if f1 is not None and f2 is not None:
                segs.append((f1/(total-1), f2/(total-1), color_stride))
            
            # 3. Landing -> Impact
            if f2 is not None and f3 is not None:
                segs.append((f2/(total-1), f3/(total-1), color_swing))
            
            # 4. Impact -> 結束
            if f3 is not None:
                segs.append((f3/(total-1), 1.0, color_follow))
            
            self.analysis_bar.set_segments(segs)

            # 進度條上標 a、b、c（離地、著地、擊球）
            markers = []
            if f1 is not None: markers.append((f1/(total-1), 'a'))
            if f2 is not None: markers.append((f2/(total-1), 'b'))
            if f3 is not None: markers.append((f3/(total-1), 'c'))

            if hasattr(self.analysis_bar, 'set_markers'):
                self.analysis_bar.set_markers(markers)

        self.ui.liftofflabel.setText(str(phase_result.F_liftoff)+"幀")
        self.ui.landinglabel.setText(str(phase_result.F_landing)+"幀")
        self.ui.impactlabel.setText(str(phase_result.F_impact)+"幀")
        self.updatePhaseTimes(phase_result)
        self.calculateKeyphaseMetrics(phase_result)
        self.calculateBatMetrics2D(phase_result.F_impact, phase_result.F_landing)
        self.display_phase_image(self.ui.keyphaseLabel1, phase_result.F_liftoff, 'liftoff')
        self.display_phase_image(self.ui.keyphaseLabel2, phase_result.F_landing, 'landing')
        self.display_phase_image(self.ui.keyphaseLabel3, phase_result.F_impact, 'impact')

        # [NEW] 在 Keyphase 計算完成後，載入 Rapsodo 數據
        if hasattr(self, 'video_loader') and self.video_loader and hasattr(self.video_loader, 'video_path'):
            folder_path = os.path.dirname(self.video_loader.video_path)
            self._loadRapsodoData(folder_path)

    def _loadRapsodoData(self, folder_path: str):
        """讀取 Rapsodo 資料 (data.json 與 sprayChart.jpg) 並更新 UI"""
        import json
        import os
        import cv2
        import numpy as np
        from PyQt5 import QtGui
        from PyQt5.QtCore import Qt

        # 先在當前資料夾找，如果沒有就往外推一層 (相容不同模式的載入路徑)
        json_path = os.path.join(folder_path, 'data.json')
        if not os.path.exists(json_path):
            json_path = os.path.join(os.path.dirname(folder_path), 'data.json')
        for label in (self.ui.exitVLabel, self.ui.lanuchAngleLabel, self.ui.attackDisLabel):
            label.setText(MISSING)

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    texts = rapsodo_texts(data)
                    for label, text in zip((self.ui.exitVLabel, self.ui.lanuchAngleLabel, self.ui.attackDisLabel), texts):
                        label.setText(text)
            except Exception as e:
                print(f"Error loading {json_path}: {e}")

        if hasattr(self.ui, 'rapsodoLabel'):
            show_spray_chart(self.ui.rapsodoLabel, folder_path)

    def get_kpts_by_frame(self, frame_num):
        """從 Dataframe 獲取指定幀的關鍵點陣列"""
        if self.person_side_df is None or self.person_side_df.empty:
            return None
        target = self.person_side_df[self.person_side_df["frame_number"] == frame_num]
        if target.empty:
            return None
        return np.array(target.iloc[0]["keypoints"]) # 回傳 (N, 2) 或 (N, 3)

    def _calibration(self):
        """(資料夾, 側面相機校正)；校正依錄影日期與打擊方向挑選，找不到時為 None"""
        folder = os.path.dirname(self.video_loader.video_path) if self.video_loader.video_path else None
        side = "right" if self.ui.batterInput.currentText() == "右打" else "left"
        return folder, (load_stereo_params(folder, side) if folder else None)

    def _side_scale(self, folder, calib, impact_frame) -> Optional[SideScale]:
        """側面畫面的公尺換算：有正面骨架時用校正三角化的打者深度，否則用棒長（utils/metrics2d.py）"""
        detector = getattr(self, "phase_detector", None)
        return side_scale(folder, calib, detector.bat_head_track() if detector is not None else None, impact_frame,
                          get_video_fps(self.video_loader), bat_for_clip(self.video_loader.video_path))

    def calculateKeyphaseMetrics(self, phase_result: PhaseResult):
        """重心速度（離地、著地）與跨步距離（著地）：側面 2D 關鍵點換算成公尺（utils/body_metrics.py）"""
        labels = (self.ui.centerGravityLabel, self.ui.centerGravityLabel2, self.ui.strideDisLabel)
        for label in labels:
            label.setText(MISSING)
            label.setToolTip("")
        self.stride_cm = None
        if self.person_side_df is None or self.person_side_df.empty:
            return

        folder, calib = self._calibration()
        scale = self._side_scale(folder, calib, phase_result.F_impact)
        if scale is None:
            print("[Keyphase2D] 沒有正面骨架也沒有可用的球棒長度，無法換算公尺")
            return
        body = body_metrics_2d(self.person_side_df, phase_result, scale, calib, get_video_fps(self.video_loader),
                               read_frame_times(folder), self.ui.batterInput.currentText() == "右打")
        self.stride_cm = body["stride_cm"]
        self.ui.centerGravityLabel.setText(fmt(body["cog_liftoff"], "m/s", 2))
        self.ui.centerGravityLabel2.setText(fmt(body["cog_landing"], "m/s", 2))
        self.ui.strideDisLabel.setText(fmt(self.stride_cm, "cm", 0))
        tip = f"比例尺：{scale.source}，{scale.m_per_px * 1000:.2f} mm/px"
        for label in labels:
            label.setToolTip(tip)
        print(f"[Keyphase2D] {tip}")

    def display_phase_image(self, label: QtWidgets.QLabel, frame_num: int, phase_type: str):
        """
        根據階段類型在影像上繪製輔助線並顯示。
        phase_type: 'liftoff', 'landing', 'impact'
        """
        if frame_num is None:
            label.clear()
            label.setText("  ")
            return
        
        # 1. 取得原始影像 (側面)
        img_bgr = self.video_loader.getVideoImage2(frame_num)
        if img_bgr is None:
            return

        # 為了不影響原始數據，複製一份來畫圖
        draw_img = img_bgr.copy()
        
        # 2. 獲取該幀關鍵點
        kpts = self.get_kpts_by_frame(frame_num)
        # --- 防呆檢查：如果完全沒有關鍵點數據，直接顯示原圖 ---
        if kpts is None or len(kpts) == 0:
            self._update_label_pixmap(label, draw_img)
            return

        # 檢查關鍵點是否包含足夠的索引 (例如 MediaPipe 通常有 33 點)
        def is_valid(idx):
            # 檢查索引是否在範圍內，且座標不是 NaN，且具備基本信心值 (假設第3欄是信心值)
            if idx >= len(kpts): return False
            pt = kpts[idx]
            return not np.any(np.isnan(pt[:2])) # 且可視情況加入 pt[2] > 0.5 之類的判斷

        if kpts is not None:
            is_righty = (self.ui.batterInput.currentText() == "右打")
            if is_righty:
                # Front = Left
                L_HIP, L_KNEE, L_ANKLE = 11, 13, 24
                R_HIP, R_KNEE, R_ANKLE = 12, 14, 25
            else:
                # Front = Right
                R_HIP, R_KNEE, R_ANKLE = 12, 14, 25
                L_HIP, L_KNEE, L_ANKLE = 11, 13, 24
            
            # 決定顏色 (BGR)
            YELLOW = (0, 255, 255) # 亮黃色
            CYAN = (255, 255, 0)   # 青色 (用於骨架)
            RED = (0, 0, 255)      # 紅色 (用於標註文字)
            WHITE = (255, 255, 255) # 白色
            BLACK = (0, 0, 0)       # 黑色
            ORANGE = (0, 0, 255)   # 亮橘紅 (Orange Red)

            # --- A. Liftoff / Landing 共同：繪製重心(COG)十字 ---
            if phase_type in ['liftoff', 'landing']:
                p_l_hip = kpts[L_HIP][:2]
                p_r_hip = kpts[R_HIP][:2]
                cog = ((p_l_hip[0] + p_r_hip[0]) / 2, (p_l_hip[1] + p_r_hip[1]) / 2)
                cx, cy = int(cog[0]), int(cog[1])
                # 畫十字
                length = 30
                cv2.line(draw_img, (cx - length, cy), (cx + length, cy), YELLOW, 6)
                cv2.line(draw_img, (cx, cy - length), (cx, cy + length), YELLOW, 6)

            # --- B. Landing 額外：跨步距離輔助線 ---
            if phase_type == 'landing':
                if kpts is not None:
                    p_l_ank = kpts[L_ANKLE][:2]
                    p_r_ank = kpts[R_ANKLE][:2]
                    ax1, ay1 = int(p_l_ank[0]), int(p_l_ank[1])
                    ax2, ay2 = int(p_r_ank[0]), int(p_r_ank[1])
                    
                    # 畫地平連接線與刻度 (OpenCV 畫線沒問題)
                    cv2.line(draw_img, (ax1, ay1), (ax2, ay1), RED, 3, cv2.LINE_AA)
                    cv2.line(draw_img, (ax1, ay1-15), (ax1, ay1+15), RED, 7)
                    cv2.line(draw_img, (ax2, ay1-15), (ax2, ay1+15), RED, 7)

                    # 準備文字（數值由 calculateKeyphaseMetrics 換算成公尺）
                    text_val = fmt(getattr(self, "stride_cm", None), "cm", 0)
                    text_label = "跨步距離"
                    
                    # --- 中文顯示防呆處理 ---
                    try:
                        # 轉為 PIL 格式
                        img_pil = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)
                        
                        # 字體路徑防呆 (Windows 常用路徑)
                        font_paths = ["msjh.ttc", "msjhbd.ttc", "C:\\Windows\\Fonts\\msjh.ttc", "arial.ttf"]
                        font_val = None
                        font_lbl = None
                        
                        for path in font_paths:
                            try:
                                font_val = ImageFont.truetype(path, 65) # 數值大字體
                                font_lbl = ImageFont.truetype(path, 45) # 標籤中字體
                                break
                            except Exception:
                                continue
                        
                        if font_val is None:
                            font_val = ImageFont.load_default()
                            font_lbl = ImageFont.load_default()

                        # 計算文字大小 (PIL 方式)
                        v_bbox = draw.textbbox((0, 0), text_val, font=font_val)
                        v_w, v_h = v_bbox[2] - v_bbox[0], v_bbox[3] - v_bbox[1]
                        l_bbox = draw.textbbox((0, 0), text_label, font=font_lbl)
                        l_w, l_h = l_bbox[2] - l_bbox[0], l_bbox[3] - l_bbox[1]

                        box_w = max(v_w, l_w) + 40
                        box_h = v_h + l_h + 50
                        
                        # 框框起始座標防呆
                        f_x, f_y = (ax1, ay1) if is_righty else (ax2, ay2)
                        rx1 = min(max(0, f_x + 30), draw_img.shape[1] - box_w)
                        ry1 = min(max(0, f_y - box_h - 20), draw_img.shape[0] - box_h)

                        # 繪製半透明黑框 (先在 OpenCV 畫好再轉 PIL)
                        overlay = draw_img.copy()
                        cv2.rectangle(overlay, (rx1, ry1), (rx1 + box_w, ry1 + box_h), BLACK, -1)
                        cv2.addWeighted(overlay, 0.6, draw_img, 0.4, 0, draw_img)
                        
                        # 重新轉換為 PIL 來寫中文
                        img_pil = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(img_pil)
                        
                        # 寫入文字 (白色)
                        draw.text((rx1 + 20, ry1 + 10), text_val, font=font_val, fill=(255, 255, 255))
                        draw.text((rx1 + 20, ry1 + v_h + 25), text_label, font=font_lbl, fill=(255, 255, 255))
                        
                        # 轉回 OpenCV BGR
                        draw_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                    except Exception as e:
                        print(f"PIL 繪圖失敗，改用預設字體: {e}")
                        # 如果 PIL 失敗，至少用 OpenCV 寫英文數值墊背
                        cv2.putText(draw_img, text_val, (f_x + 30, f_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

            # --- C. Impact：繪製腿部骨架 (髖-膝-踝) ---
            if phase_type == 'impact':
                # 定義要畫的線段：[髖, 膝, 踝]
                if not is_righty:
                    front_legs = [[R_HIP, R_KNEE, R_ANKLE]]
                    back_legs = [[L_HIP, L_KNEE, L_ANKLE]]
                else:
                    front_legs = [[L_HIP, L_KNEE, L_ANKLE]]
                    back_legs = [[R_HIP, R_KNEE, R_ANKLE]]
                for leg in front_legs:
                    p1 = tuple(kpts[leg[0]][:2].astype(int))
                    p2 = tuple(kpts[leg[1]][:2].astype(int))
                    p3 = tuple(kpts[leg[2]][:2].astype(int))
                    # 髖到膝
                    cv2.line(draw_img, p1, p2, CYAN, 7)
                    # 膝到踝
                    cv2.line(draw_img, p2, p3, CYAN, 7)
                    # 畫關節點
                    for p in [p1, p2, p3]:
                        cv2.circle(draw_img, p, 8, YELLOW, -1)

                for leg in back_legs:
                    p1 = tuple(kpts[leg[0]][:2].astype(int))
                    p2 = tuple(kpts[leg[1]][:2].astype(int))
                    p3 = tuple(kpts[leg[2]][:2].astype(int))
                    # 髖到膝
                    cv2.line(draw_img, p1, p2, ORANGE, 7)
                    # 膝到踝
                    cv2.line(draw_img, p2, p3, ORANGE, 7)
                    # 畫關節點
                    for p in [p1, p2, p3]:
                        cv2.circle(draw_img, p, 8, YELLOW, -1)
                front_knee_angle, back_knee_angle = knee_angles_2d(kpts, is_righty)
                self.ui.forefootLabel3.setText(fmt(front_knee_angle, "°"))
                self.ui.backfootLabel3.setText(fmt(back_knee_angle, "°"))

        # 3. 轉為 QPixmap 並顯示
        img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(q_img)

        # --- 填滿 Label 並對齊「底部中間」裁切 ---
        label_size = label.size()
        
        if not label_size.isEmpty():
            # 1. 以「填滿」為原則縮放 (會有一邊超出 Label)
            pixmap = pixmap.scaled(
                label_size, 
                QtCore.Qt.KeepAspectRatioByExpanding, 
                QtCore.Qt.SmoothTransformation
            )
            
            # 2. 計算裁切區域 (Rect)
            # x 軸：置中裁切 (左右各切一半)
            # y 軸：對齊底部 (保留最下面，裁掉上面)
            crop_x = (pixmap.width() - label_size.width()) // 2
            crop_y = pixmap.height() - label_size.height() # 關鍵：直接減去高度，保留底部
            
            # 防呆：確保裁切起點不為負數
            crop_x = max(0, crop_x)
            crop_y = max(0, crop_y)

            rect = QtCore.QRect(
                crop_x,
                crop_y,
                label_size.width(),
                label_size.height()
            )
            
            # 3. 執行裁切
            pixmap = pixmap.copy(rect)

        # --- Draw Strike Zone ROI for Debugging ---
        if not self.is_video and hasattr(self, 'strike_zone_roi') and self.strike_zone_roi is not None:
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor(0, 255, 0, 150), 3, Qt.DashLine))
            # Need to map ROI to pixmap coordinates if needed, but here we assume pixmap is scaled.
            # For simplicity, we just draw it if it's the raw camera feed.
            # Actually, pixmap is already scaled to label_size. We need to scale the ROI too.
            if label == self.ui.FrameView_2: # Side View
                orig_w, orig_h = 1920, 1080 # Assuming default
                if hasattr(self.camera, 'frame_size') and self.camera.frame_size:
                    orig_w, orig_h = self.camera.frame_size
                
                scale_x = label_size.width() / orig_w
                scale_y = label_size.height() / orig_h
                rx, ry, rw, rh = self.strike_zone_roi
                painter.drawRect(int(rx * scale_x), int(ry * scale_y), int(rw * scale_x), int(rh * scale_y))
                painter.setFont(QFont("Arial", 10))
                painter.drawText(int(rx * scale_x), int(ry * scale_y) - 5, "Strike Zone ROI")
            painter.end()

        # 4. 顯示與快取
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.cached_keyphase_pixmaps[label] = pixmap
        self.update_keyphase_label_image(label)

    def calculate_angle(self,p1, p2, p3):
        """
        計算 p1-p2-p3 之間夾角 (以 p2 為頂點)
        回傳值永遠為正 (0 ~ 180 度)
        """
        ba = np.array(p1) - np.array(p2)
        bc = np.array(p3) - np.array(p2)

        # 計算內積與模長
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        # 限制範圍避免數值誤差導致 nan
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)


    def calculateBatMetrics2D(self, impact_frame: int, landing_frame: int = None):
        """揮棒速度與攻擊角度，採 Blast Motion 定義：擊球瞬間甜蜜點（距棒頭 6 吋）的速度與移動方向（見 utils/bat_speed2d.py）"""
        detector = getattr(self, "phase_detector", None)
        folder, calib = self._calibration()
        bat = bat_for_clip(self.video_loader.video_path)   # 畫面上的球棒像素長度 = 棒長，換算成公尺
        result = bat_metrics_2d(detector.bat_head_track() if detector is not None else None, impact_frame,
                                get_video_fps(self.video_loader), read_frame_times(folder), calib, bat)
        if result is None:
            print(f"[BatMetrics] 擊球點 {impact_frame} 附近沒有足夠的棒頭資料")
            self.ui.batSpeedLabel.setText(MISSING)
            self.ui.horizontalAngleEditLabel.setText(MISSING)
            return
        self.ui.batSpeedLabel.setText(fmt(result.speed_kmh, "km/h"))
        self.ui.horizontalAngleEditLabel.setText(fmt(result.attack_angle_deg, "°", signed=True))
        self.ui.batSpeedLabel.setToolTip(f"以{bat.describe()}換算")
        print(f"[BatMetrics] {bat.describe()}；揮棒速度 {result.speed_kmh:.1f} km/h、攻擊角度 {result.attack_angle_deg:+.1f}°："
              f"甜蜜點 幀 {result.from_frame}→{result.to_frame}，{result.duration_s * 1000:.1f} ms（{result.timed_by}），"
              f"球棒 {result.bat_px:.0f} px，{'已' if result.undistorted else '未'}去畸變")


    def _update_label_pixmap(self, label, img):
        """封裝轉換與更新 Pixmap 的邏輯"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        q_img = QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(q_img)
        
        self.cached_keyphase_pixmaps[label] = pixmap
        self.update_keyphase_label_image(label)

    def update_keyphase_label_image(self, label: QtWidgets.QLabel):
        """
        根據 Label 目前的大小，從 Cache 拿原始圖縮放後顯示。
        此函式可被 display_phase_image 呼叫，也可被 ResizeEvent 呼叫。
        """
        if label not in self.cached_keyphase_pixmaps:
            return

        pixmap = self.cached_keyphase_pixmaps[label]
        if pixmap is None or pixmap.isNull():
            return
            
        # 檢查 Label 尺寸，如果太小則給預設值
        target_w = label.width() if label.width() > 10 else 300
        target_h = label.height() if label.height() > 10 else 200
        
        scaled_pixmap = pixmap.scaled(
            target_w, 
            target_h, 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        label.setPixmap(scaled_pixmap)
        # 貼齊格子底部：各欄圖片底部與 Rapsodo 落點圖（填滿格子）對齊
        label.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)

    def updateHitterList(self):
        """掃描 Db 資料夾，將現有的選手名稱加入下拉選單"""
        if not hasattr(self.ui, 'hitterNameInput'): return
        
        db_base_path = str(DB_DIR)
        if not os.path.exists(db_base_path): return
        
        # 暫時阻斷訊號避免重複觸發清單更新
        self.ui.hitterNameInput.blockSignals(True)
        self.ui.hitterNameInput.clear()
        
        folders = [f for f in os.listdir(db_base_path) 
                   if os.path.isdir(os.path.join(db_base_path, f)) 
                   and f not in NON_HITTER_DIRS]
        
        folders.sort()
        self.ui.hitterNameInput.addItems(folders)
        self.ui.hitterNameInput.blockSignals(False)
        
    def setWatchTime(self):
        self.watchTime = float(self.ui.watchTimeCombobox.currentText())*1000

    def handleVideoEnd(self):
        """Handle the logic when video reaches its end."""
        self.play_times -= 1
        if self.play_times > 0:
            # Replay the video
            self.playBtnClicked()
            self.ui.frameSlider.setValue(0)
            self.playBtnClicked()
        else:
            self.detectPhases()
            print("播放結束，回到監控模式...")
            
            # 儲存 JSON (如果是首次處理)
            if not getattr(self, 'is_processed_json', False) and self.is_video and hasattr(self.video_loader, 'video_path') and self.video_loader.video_path:
                folder_path = os.path.dirname(self.video_loader.video_path)
                try:
                    # 改為檢查側面相機資料 (分析 2D 主要是使用 side view)
                    if not self.pose_estimater_2.person_df.empty:
                        # 儲存側面資料
                        self.pose_estimater_2.person_df.to_json(os.path.join(folder_path, "person_side.json"), orient='records')
                        
                        if getattr(self.pose_estimater_2, 'bat_df', None) is not None:
                            self.pose_estimater_2.bat_df.to_json(os.path.join(folder_path, "bat_side.json"), orient='records')
                        
                        if getattr(self.pose_estimater_2, 'ball_df', None) is not None:
                            self.pose_estimater_2.ball_df.to_json(os.path.join(folder_path, "ball_side.json"), orient='records')
                            
                        if self.bat_analyzer_2 and getattr(self.bat_analyzer_2, 'batHeadTail_df', None) is not None:
                            self.bat_analyzer_2.batHeadTail_df.to_json(os.path.join(folder_path, "batht_side.json"), orient='records')
                            
                        print(f"Skeleton info (Side View) saved to {folder_path}!")
                        
                except Exception as e:
                    print(f"Error saving skeleton info: {e}")

            if self.is_play:
                self.playBtnClicked()   
            self.play_times = 1 # 重置播放次數  
            
            QTimer.singleShot(self.watchTime, self.returnToMonitorMode)

    def returnToMonitorMode(self):
        """10秒後觸發的邏輯：重設變數並回到監控模式"""
        print("正在切換回監控模式...")
        self.play_times = 1 # 重置播放次數
        
        # 2. 重置階段文字標籤 (如果有需要)
        # 自動勾選相機開啟 (會觸發 toggleCamera)
        if self.is_process == False:
            self.ui.liftofflabel.setText("幀")
            self.ui.landinglabel.setText("幀")
            self.ui.impactlabel.setText("幀")
            print("切換回監控模式")
            self.ui.cameraCheckBox.setCheckState(2) 
            # 開啟骨架偵測與打擊模式
            self.ui.startHitCheckBox.setCheckState(2)
        self.toggleSelect(2)
        self.toggleShowSkeleton(2)

    def clearKeyphaseDisplays(self):
        """將所有關鍵幀 Label 弄成全白並清空緩存"""
        labels = [self.ui.keyphaseLabel1, self.ui.keyphaseLabel2, self.ui.keyphaseLabel3]
        
        # 清空圖片緩存，防止 resizeEvent 重新填入圖片
        self.cached_keyphase_pixmaps.clear()
        
        for label in labels:
            # 創建一個全白的畫布 (也可以直接用 label.clear()，但這會變透明/背景色)
            # 如果要確保是「全白」，可以手動填色：
            white_pixmap = QPixmap(label.size())
            white_pixmap.fill(Qt.white)
            label.setPixmap(white_pixmap)

    def hitterAnaylze(self, frame_2=None):
        # 如果已經在錄製中或儲存中，就跳過，避免重複觸發
        if self.camera.continuous_recorder.is_recording_post or self.camera.continuous_recorder.is_saving:
            return

        if self.is_bat_swing_detected(frame_2):
            print("!!! 偵測到打擊，開始擷取影片片段 !!!")
            # 1. 觸發 ContinuousRecorder 開始計數「後一秒」的幀數
            self.camera.continuous_recorder.trigger()
            
            # 2. 計算剩餘錄製時間並啟動一個單次計時器來執行儲存
            # 假設 post-roll 是 1 秒 (30幀 @ 180fps 或 180幀 @ 180fps，依據你的設定)
            # 這裡建議給予稍微長一點的時間確保 buffer 填滿
            post_record_ms = 1000  # 1.2 秒後執行存檔
            QTimer.singleShot(post_record_ms, self.autoSaveTriggeredVideo)

    def autoSaveTriggeredVideo(self):
        """當 ContinuousRecorder 錄完 post-roll 後由計時器觸發"""
        # 如果還在持續收信中，延後再試
        if self.camera.continuous_recorder.is_recording_post:
            QTimer.singleShot(100, self.autoSaveTriggeredVideo)
            return
            
        print("正在處理自動存檔...")
        
        # 呼叫你原本就寫好的 saveExistingVideo
        self.saveExistingVideo()
        
        # 存完後，重置 recorder 狀態以便下一次偵測
        self.camera.continuous_recorder.reset_trigger()
        self.resetRecordTimer()

    def _extract_center_from_bbox(self, bbox):
        """
        核心處理邏輯：處理 [[x1, y1, x2, y2]] 或 [x1, y1, x2, y2] 格式並回傳中心點
        """
        if isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()

        if not isinstance(bbox, list) or len(bbox) == 0:
            return None
        
        # 判斷是否為巢狀列表
        if isinstance(bbox[0], (list, np.ndarray)):
            inner = bbox[0]
        else:
            inner = bbox
            
        if len(inner) < 4:
            return None
            
        x1, y1, x2, y2 = inner[:4]
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        return np.array([cx, cy], dtype=np.float32)

    def _extract_ball_center_2d(self, ball_df):
        """
        相容 mergeBallData 產出的格式：[[x1, y1, x2, y2]] 或 [x1, y1, x2, y2]
        """
        centers = {}
        if ball_df is None or ball_df.empty:
            return centers

        for _, row in ball_df.iterrows():
            f = int(row["frame_number"]) # 使用統一的 frame_number
            center = self._extract_center_from_bbox(row["bbox"])
            if center is not None:
                centers[f] = center
        return centers


    def is_bat_swing_detected(self, frame_2=None):
        """
        根據當前模式 (current_mode) 判斷觸發條件：
        - Tbar (T座): 球原本在 ROI 內，然後離開 ROI 時觸發。
        - 其他 (側拋/正投): 球原本不在 ROI 內，然後進入 ROI 時觸發。
        """
        if self.trigger_cooldown > 0:
            self.trigger_cooldown -= 1
            return False

        # 1. 確保有 ROI 設定
        if not hasattr(self, 'strike_zone_roi') or self.strike_zone_roi is None:
            return False
        
        # 2. 進行即時偵測 (優先使用傳入的 frame_2)
        if frame_2 is None:
            return False
            
        roi = self.strike_zone_roi # [x, y, w, h]
        ball_bboxes = self.pose_estimater_2.detectBall(frame_2)
        
        # 3. 檢查是否有球在 ROI 內
        ball_is_in_roi = False
        for bbox in ball_bboxes:
            center = self._extract_center_from_bbox(bbox)
            if center is not None:
                rx, ry, rw, rh = roi
                # 判斷中心點是否在 ROI 矩形內
                if rx <= center[0] <= rx + rw and ry <= center[1] <= ry + rh:
                    ball_is_in_roi = True
                    break # 只要有一顆球在裡面就算
        
        # 4. 根據模式判斷是否觸發
        trigger = False
        current_mode = getattr(self, 'current_mode', 'Tbar')
        
        if current_mode == 'Tbar':
            # T座模式: 靜止離開 ROI 觸發 (原本在裡面，現在不在)
            if self.ball_was_in_roi and not ball_is_in_roi:
                print("[Trigger] Tbar mode: Ball left ROI.")
                trigger = True
        else:
            # 其他模式(側拋/正投): 球進入 ROI 作為 trigger (原本不在，現在在)
            if not self.ball_was_in_roi and ball_is_in_roi:
                print("[Trigger] Toss/Front mode: Ball entered ROI.")
                trigger = True

        # 5. 更新狀態
        self.ball_was_in_roi = ball_is_in_roi
        
        if trigger:
            self.trigger_cooldown = 180 # 1秒冷卻 (180fps)
            return True
            
        return False


    def updateFrame(self, frame: np.ndarray = None, frame_2: np.ndarray = None, frame_num:int = None):
        """Update the displayed frame with additional analysis."""
        # 更新當前的frame和frame_num
        if self.is_video and frame_num is not None:
            frame= self.video_loader.getVideoImage(frame_num)
            frame_2 = self.video_loader.getVideoImage2(frame_num)
            if frame is None or frame_2 is None:
                return
        elif frame is None or frame_2 is None:
            return
        drawed_img = self.image_drawer.drawInfo(frame, frame_num, self.pose_estimater.kpt_buffer)
        drawed_img_2 = self.image_drawer_2.drawInfo(frame_2, frame_num, self.pose_estimater_2.kpt_buffer)
        
        # --- Draw ROI on Side View ---
        if drawed_img_2 is not None:
            # 決定是否要顯示 ROI：1. 正在設定中 2. 使用者勾選了「顯示區域/輔助線」
            show_roi = self.is_roi_drawing or (hasattr(self.ui, 'showRegionCheckBox') and self.ui.showRegionCheckBox.isChecked())
            
            if show_roi:
                # Draw Static ROI
                if hasattr(self, 'strike_zone_roi') and self.strike_zone_roi is not None:
                    rx, ry, rw, rh = self.strike_zone_roi
                    cv2.rectangle(drawed_img_2, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(drawed_img_2, "ROI (Strike Zone)", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Draw Temp ROI (while drawing)
                if self.is_roi_drawing and self.roi_temp_rect is not None:
                    rect = self.roi_temp_rect
                    cv2.rectangle(drawed_img_2, (rect.x(), rect.y()), (rect.x() + rect.width(), rect.y() + rect.height()), (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.putText(drawed_img_2, "Drawing ROI...", (rect.x(), rect.y() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        self.showImage(drawed_img, self.view_scene, self.ui.FrameView) # 正面
        self.showImage(drawed_img_2, self.view_scene2, self.ui.FrameView_2) # 側面
        # Update the enlarged dialog if open
        if self.enlarged_dialog is not None and self.enlarged_dialog.isVisible():
            # Send the correct frame based on the left/right state
            if self.enlarged_view_is_frame_view == 1: 
                enlarged_frame = drawed_img
            elif self.enlarged_view_is_frame_view == 2:
                enlarged_frame = drawed_img_2
            self.enlarged_dialog.updateFrame(enlarged_frame)


    def resetRecordTimer(self):
        video_path = getattr(self, 'video_path', None)
        video_path2 = getattr(self, 'video_path2', None)
        if video_path:
            print(f"影片已存檔: {video_path}")
            # [NEW] OCR 觸發：手動與自動錄影皆會執行此處
            try:
                output_dir = os.path.dirname(video_path)
                print(f"啟動 OCR 辨識 (Rapsodo)... 路徑: {output_dir}")
                ocr.run_ocr_and_save(output_dir, camera_index=0)
            except Exception as e:
                print(f"OCR 執行失敗: {e}")
        self.video_loader.reset()
        self.toggleRecord(0)
        self.ui.startHitCheckBox.setCheckState(0)
        self.ui.cameraCheckBox.setCheckState(0)
        self.is_processed_json = False
        self.video_loader.loadVideo(video_path,video_path2)
        self.checkVideoLoad()
        # 更新最近清單
        self.populateRecentVideosList()

    def checkVideoLoad(self):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 檢查是否有影片名稱，若無則不執行後續操作
        if not self.video_loader.video_name and not self.video_loader.video_name2:
            return
        # 若影片正在讀取中，定時檢查讀取狀況
        if self.video_loader.is_loading:
            self.ui.fileNameLabel.setText("讀取影片中...")
            QTimer.singleShot(100, self.checkVideoLoad)  # 每100ms 檢查一次
            return
        if hasattr(self.ui, 'fileNameLabel'):
            # 使用 os.path.basename 只取得檔案名稱，去除長長的路徑
            name1 = os.path.basename(self.video_loader.video_name)
            if name1.startswith("CF_"):
                name1 = name1[3:]
            self.ui.fileNameLabel.setText(f"{name1}")
        # 影片讀取完成後更新 UI 元素
        self.toggleShowSkeleton(2)
        self.updateVideoInfo()

    def updateVideoInfo(self):
        """更新與影片相關的資訊顯示在 UI 上。"""
        self.reset()
        self.setFrameTimes(os.path.dirname(self.video_loader.video_path) if self.video_loader.video_path else None)
        self.initFrameSlider()

        # 自動從 JSON 載入先前的骨架分析
        if getattr(self, 'is_processed_json', False) and hasattr(self.video_loader, 'video_path') and self.video_loader.video_path:
            folder_path = os.path.dirname(self.video_loader.video_path)
            from cv_utils.cv_control_dual import DualJsonLoader
            loader = DualJsonLoader(folder_path)
            data = loader.load()
            if data.get("person_side") is not None and not data.get("person_side").empty:
                print("Loaded JSON data from", folder_path)
                self.pose_estimater.person_df = data.get("person_front")
                self.pose_estimater_2.person_df = data.get("person_side")
                self.pose_estimater.bat_df = data.get("bat_front")
                self.pose_estimater_2.bat_df = data.get("bat_side")
                self.pose_estimater.ball_df = data.get("ball_front")
                self.pose_estimater_2.ball_df = data.get("ball_side")
                if self.bat_analyzer: self.bat_analyzer.batHeadTail_df = data.get("batht_front")
                if self.bat_analyzer_2: self.bat_analyzer_2.batHeadTail_df = data.get("batht_side")
                
                frameset = set(range(self.video_loader.total_frames))
                self.pose_estimater.processed_frames = frameset
                self.pose_estimater_2.processed_frames = frameset
            else:
                self.is_processed_json = False

        # --- 加入以下邏輯：自動開啟偵測與選人 ---
        self.toggleSelect(2)
        self.toggleShowSkeleton(2)
        frame = self.video_loader.getVideoImage(0)
        frame_2 = self.video_loader.getVideoImage2(0)
        if frame is not None and frame_2 is not None:
            _, _, _, _,_= self.pose_estimater_2.detectKpt(frame_2, 0, is_video=True)
        self.updateFrame(frame_num=0)
        self.model.setImageSize(self.video_loader.video_size)
        video_size = self.video_loader.video_size
        self.ui.ResolutionLabel.setText(clip_title(self.video_loader.video_path))
        self.is_play = False
        self.ui.playBtn.click()

    def initAnalyzeFrame(self):
        self.toggleSelect(2)
        self.toggleShowSkeleton(2)
        frame = self.video_loader.getVideoImage(0)
        frame_2 = self.video_loader.getVideoImage2(0)
        _, _, _, _,_= self.pose_estimater_2.detectKpt(frame_2, 0, is_video=True)
        self.ui.playBtn.click()

    def setFrameTimes(self, folder):
        """讀 frame_times.csv（3D 的 CF_ 子資料夾沒有時看上一層），時間標籤與進度條各段秒數用實際擷取時間"""
        self.frame_times = read_frame_times(folder) or read_frame_times(os.path.dirname(folder or ""))
        total = int(getattr(self.video_loader, "total_frames", 0) or 0)
        if hasattr(self, "analysis_bar") and total > 1:
            self.analysis_bar.set_time_of_ratio(
                lambda r: frame_seconds(int(round(r * (total - 1))), total, self.frame_times))

    def frameTimeText(self, frame_num, total_frames):
        """目前時間/影片長度，影片結束時為 1.50/1.5s（見 fps_utils.frame_seconds）"""
        return f"{frame_seconds(frame_num, total_frames, getattr(self, 'frame_times', None)):.2f}/{NOMINAL_DURATION_S:.1f}s"

    def initFrameSlider(self):
        """初始化影片滑桿和相關的標籤。"""
        total_frames = self.video_loader.total_frames
        
        # --- 核心改動：強制鎖定最小高度為 80 ---
        self.ui.frameSlider.setMinimumHeight(80) 
        # 設定為 Expanding 確保它會嘗試撐開可用空間
        self.ui.frameSlider.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        
        self.ui.frameSlider.setMinimum(0)
        self.ui.frameSlider.setMaximum(total_frames - 1)
        self.ui.frameSlider.setValue(0)
        
        total_seconds = 1.5
        self.ui.frameNumLabel.setText(self.frameTimeText(0, int(total_frames)))

    def resetFrameSlider(self):
        self.ui.frameSlider.setValue(0)
        self.ui.frameSlider.setRange(0, 0)

    def reset(self):
        self.pose_estimater.reset()
        self.pose_estimater_2.reset()
        self.pose_analyzer.reset()
        self.pose_analyzer_2.reset()
        if self.bat_analyzer:
            self.bat_analyzer.reset()
        if self.bat_analyzer_2:
            self.bat_analyzer_2.reset()
        self.image_drawer.reset()
        self.image_drawer_2.reset()
        self.resetFrameSlider()
        self.view_scene.clear()
        self.view_scene2.clear()
        if self.enlarged_dialog is not None:
            self.enlarged_dialog.close()
            self.enlarged_dialog = None

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        """Display an image in the QGraphicsView."""
        scene.clear()
        if image is None:
            exit()
        h, w = image.shape[:2]
        qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)
    
    def mousePressEvent(self, event):
        """Handle mouse events for person, keypoint selection, and ROI adjustment."""
        # 使用 globalPos 避開相對座標可能產生的偏移與誤判
        global_pos = event.globalPos()
        frame_view_pos = self.ui.FrameView.mapFromGlobal(global_pos)
        frame_view_2_pos = self.ui.FrameView_2.mapFromGlobal(global_pos)
        
        is_frame_view = self.ui.FrameView.rect().contains(frame_view_pos)
        is_frame_view_2 = self.ui.FrameView_2.rect().contains(frame_view_2_pos)

        # --- ROI Interaction Mode ---
        if self.is_roi_drawing and is_frame_view_2:
            scene_pos = self.ui.FrameView_2.mapToScene(frame_view_2_pos)
            sx, sy = scene_pos.x(), scene_pos.y()
            
            # Check proximity to existing ROI edges/inside
            if hasattr(self, 'strike_zone_roi') and self.strike_zone_roi is not None:
                rx, ry, rw, rh = self.strike_zone_roi
                margin = 20 # Proximity threshold in scene coordinates
                
                # Determine mode
                on_left = abs(sx - rx) < margin
                on_right = abs(sx - (rx + rw)) < margin
                on_top = abs(sy - ry) < margin
                on_bottom = abs(sy - (ry + rh)) < margin
                inside = rx <= sx <= rx + rw and ry <= sy <= ry + rh
                
                if on_left and on_top: self.roi_drag_mode = 'top-left'
                elif on_right and on_top: self.roi_drag_mode = 'top-right'
                elif on_left and on_bottom: self.roi_drag_mode = 'bottom-left'
                elif on_right and on_bottom: self.roi_drag_mode = 'bottom-right'
                elif on_left: self.roi_drag_mode = 'left'
                elif on_right: self.roi_drag_mode = 'right'
                elif on_top: self.roi_drag_mode = 'top'
                elif on_bottom: self.roi_drag_mode = 'bottom'
                elif inside: self.roi_drag_mode = 'move'
                else: self.roi_drag_mode = 'draw'
            else:
                self.roi_drag_mode = 'draw'
                
            if self.roi_drag_mode == 'draw':
                self.roi_start_pos = scene_pos
                self.roi_temp_rect = QRect(int(sx), int(sy), 0, 0)
            else:
                self.roi_last_mouse_pos = scene_pos
            return

        is_frame_view = self.ui.FrameView.rect().contains(frame_view_pos)
        if not (is_frame_view or is_frame_view_2):
            return

        # Debug which view was clicked
        if is_frame_view:
            self.enlarged_view_is_frame_view = 1
        elif is_frame_view_2:
            self.enlarged_view_is_frame_view = 2

        if self.is_video and self.play_times > 0 and event.button() == Qt.LeftButton:
            if self.enlarged_dialog is None or not self.enlarged_dialog.isVisible():
                self.enlarged_dialog = EnlargedFrameDialog(self)
                self.enlarged_dialog.show()
                self.enlarged_dialog.updateFrame(frame_num=self.ui.frameSlider.value())
            self.updateFrame(frame_num=self.ui.frameSlider.value())

        event.accept()

    def mouseMoveEvent(self, event):
        """Handle mouse move for ROI rectangle visualization and cursor change."""
        global_pos = event.globalPos()
        frame_view_2_pos = self.ui.FrameView_2.mapFromGlobal(global_pos)
        is_frame_view_2 = self.ui.FrameView_2.rect().contains(frame_view_2_pos)

        if self.is_roi_drawing and is_frame_view_2:
            scene_pos = self.ui.FrameView_2.mapToScene(frame_view_2_pos)
            sx, sy = scene_pos.x(), scene_pos.y()
            
            # --- 1. Cursor Feedback (Only if not already dragging) ---
            if self.roi_drag_mode is None:
                if hasattr(self, 'strike_zone_roi') and self.strike_zone_roi is not None:
                    rx, ry, rw, rh = self.strike_zone_roi
                    margin = 20
                    on_left = abs(sx - rx) < margin
                    on_right = abs(sx - (rx + rw)) < margin
                    on_top = abs(sy - ry) < margin
                    on_bottom = abs(sy - (ry + rh)) < margin
                    inside = rx <= sx <= rx + rw and ry <= sy <= ry + rh
                    
                    if (on_left and on_top) or (on_right and on_bottom): self.setCursor(Qt.SizeFDiagCursor)
                    elif (on_right and on_top) or (on_left and on_bottom): self.setCursor(Qt.SizeBDiagCursor)
                    elif on_left or on_right: self.setCursor(Qt.SizeHorCursor)
                    elif on_top or on_bottom: self.setCursor(Qt.SizeVerCursor)
                    elif inside: self.setCursor(Qt.SizeAllCursor)
                    else: self.setCursor(Qt.CrossCursor)
                else:
                    self.setCursor(Qt.CrossCursor)

            # --- 2. Update ROI / Temp Rect ---
            if self.roi_drag_mode == 'draw' and self.roi_start_pos is not None:
                x1, y1 = int(self.roi_start_pos.x()), int(self.roi_start_pos.y())
                x2, y2 = int(sx), int(sy)
                rx, ry, rw, rh = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
                self.roi_temp_rect = QRect(rx, ry, rw, rh)
            
            elif self.roi_drag_mode is not None and self.roi_last_mouse_pos is not None:
                dx = sx - self.roi_last_mouse_pos.x()
                dy = sy - self.roi_last_mouse_pos.y()
                self.roi_last_mouse_pos = scene_pos
                
                rx, ry, rw, rh = self.strike_zone_roi
                if self.roi_drag_mode == 'move':
                    rx += dx; ry += dy
                elif self.roi_drag_mode == 'left':
                    rx += dx; rw -= dx
                elif self.roi_drag_mode == 'right':
                    rw += dx
                elif self.roi_drag_mode == 'top':
                    ry += dy; rh -= dy
                elif self.roi_drag_mode == 'bottom':
                    rh += dy
                elif self.roi_drag_mode == 'top-left':
                    rx += dx; ry += dy; rw -= dx; rh -= dy
                elif self.roi_drag_mode == 'top-right':
                    ry += dy; rw += dx; rh -= dy
                elif self.roi_drag_mode == 'bottom-left':
                    rx += dx; rw -= dx; rh += dy
                elif self.roi_drag_mode == 'bottom-right':
                    rw += dx; rh += dy
                
                # Keep minimum size
                if rw < 50: rw = 50
                if rh < 50: rh = 50
                self.strike_zone_roi = [int(rx), int(ry), int(rw), int(rh)]

            # Force redraw to see changes
            if not self.ui.cameraCheckBox.isChecked():
                self.updateFrame(frame_num=self.ui.frameSlider.value())
        
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finalize the ROI adjustment."""
        if self.is_roi_drawing:
            if self.roi_drag_mode == 'draw' and self.roi_start_pos is not None:
                global_pos = event.globalPos()
                frame_view_2_pos = self.ui.FrameView_2.mapFromGlobal(global_pos)
                scene_pos = self.ui.FrameView_2.mapToScene(frame_view_2_pos)
                x1, y1 = self.roi_start_pos.x(), self.roi_start_pos.y()
                x2, y2 = scene_pos.x(), scene_pos.y()
                rx, ry, rw, rh = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
                if rw > 10 and rh > 10:
                    self.strike_zone_roi = [int(rx), int(ry), int(rw), int(rh)]
                    print(f"ROI Draw Finalized: {self.strike_zone_roi}")
            
            self.roi_drag_mode = None
            self.roi_start_pos = None
            self.roi_temp_rect = None
            self.roi_last_mouse_pos = None
            
        super().mouseReleaseEvent(event)

if __name__ == "__main__":
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = PosePitchTabControl()
    window.show()
    sys.exit(app.exec_())

