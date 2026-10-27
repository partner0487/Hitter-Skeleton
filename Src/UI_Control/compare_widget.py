import sys
import cv2
import os
import glob
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QWidget, QMessageBox
from PyQt5.QtCore import Qt, QTimer, QEvent, QRect
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor, QPainter, QBrush, QPen, QFont
from ui.dual_camera_system_cp import Ui_hitter_ui_dual
import numpy as np
from utils.vis_image_dual import ImageDrawer as ImageDrawerDual
from utils.analyze import PoseAnalyzer, BatAnalyzer
from skeleton.detect_skeleton_dual import PoseEstimater
from cv_utils.cv_control_dual import VideoLoader as VideoLoaderDual
from utils.paths import DB_DIR, RECORD_DIR, ICON_DIR
from cv_utils.cv_control_dual import DualJsonLoader
from utils.phase_detect3d import PhaseDetector, PhaseResult
import pandas as pd

class AnalysisBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.segments = []
        self.markers = []

    def set_segments(self, segments):
        self.segments = segments
        self.update()

    def set_markers(self, markers):
        self.markers = markers
        self.update()

    def paintEvent(self, event):
        if not hasattr(self, 'segments') or not self.segments:
            return

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            w = self.width()
            h = self.height() 
            bar_h = 12
            bar_y = int(h * 0.52) - (bar_h // 2)  
            
            for start_ratio, end_ratio, color in self.segments:
                x = int(start_ratio * w)
                seg_w = int((end_ratio - start_ratio) * w)
                if seg_w <= 0: continue
                painter.fillRect(QRect(x, bar_y, seg_w, bar_h), QColor(color))

            if hasattr(self, 'markers') and self.markers:
                for ratio, marker_text in self.markers:
                    mx = int(max(0.0, min(ratio, 1.0)) * w)
                    marker_str = str(marker_text)
                    
                    if marker_str in ['1', '2', '3', '4']:
                        radius = 12
                        cy = bar_y - radius
                        bg = {'1': QColor(220, 0, 0), '2': QColor(0, 170, 230), '3': QColor(0, 180, 0), '4': QColor(128, 0, 128)}.get(marker_str, Qt.gray)
                        font_size = 12
                    else:
                        radius = 12
                        cy = bar_y + bar_h + radius
                        bg = {'a': QColor(255, 50, 50), 'b': QColor(255, 150, 0), 'c': QColor(0, 200, 255)}.get(marker_str, Qt.gray)
                        font_size = 12

                    rect = QRect(mx - radius, cy - radius, radius * 2, radius * 2)
                    painter.setBrush(QBrush(bg))
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawEllipse(rect)
                    
                    painter.setFont(QFont("Arial", font_size, QFont.Bold))
                    painter.setPen(QPen(Qt.white))
                    painter.drawText(rect, Qt.AlignCenter, marker_str)

        except Exception as e:
            print("AnalysisBar Paint Error:", e)
        finally:
            painter.end()
class CompareProcessWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    
    def __init__(self, p_context):
        super().__init__()
        self.p = p_context
        self._is_running = True
        
    def stop(self):
        self._is_running = False

    def run(self):
        try:
            frames_side = self.p.video_loader.video_frames1
            if frames_side:
                # 側面骨架分析偵測
                self.p.pose_estimater_side.reset()
                self.p.pose_estimater_side.setDetect(True)
                for i, frame in enumerate(frames_side):
                    if not getattr(self, '_is_running', True):
                        break
                    self.p.pose_estimater_side.detectKpt(frame, frame_num=i, is_video=True)
            self.p.is_processed = True
        except Exception as e:
            import traceback
            print("Compare Process Worker Error:", e)
            traceback.print_exc()
        finally:
            self.finished.emit()

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
        
        # 紀錄是否正在非同步加載
        self._auto_process_pending = False
        
        # 紀錄是否為 3D 分析模式
        self.is_3d = False


class CompareWidget(QWidget, Ui_hitter_ui_dual):
    def __init__(self, model=None, parent=None):
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
        
        self.init_ui_states()
        self.init_signals()

    def shutdown(self):
        """關閉程式前停止計時器、影片載入與分析執行緒"""
        self.timer.stop()
        for p in (self.p1, self.p2):
            if p.video_loader.is_loading:
                p.video_loader.abort_loading()
            thread = getattr(p, 'process_thread', None)
            if thread is not None and thread.isRunning():
                p.worker.stop()
                thread.quit()
                thread.wait()

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
            self.analysis_bar1 = AnalysisBar(self.frameSlider1)
            self.frameSlider1.installEventFilter(self)
        if hasattr(self, 'frameSlider2'):
            self.analysis_bar2 = AnalysisBar(self.frameSlider2)
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
        if hasattr(self, 'loadVideoBtn'):
            self.loadVideoBtn.clicked.connect(lambda: self.manual_load_video(1))
        if hasattr(self, 'loadProcessFolderBtn'):
            self.loadProcessFolderBtn.clicked.connect(lambda: self.manual_load_folder(1))
        
        if hasattr(self, 'modeTbar'): self.modeTbar.clicked.connect(lambda: self.setMode(1, "Tbar"))
        if hasattr(self, 'modeSide'): self.modeSide.clicked.connect(lambda: self.setMode(1, "Side"))
        if hasattr(self, 'modeFront'): self.modeFront.clicked.connect(lambda: self.setMode(1, "Front"))
        
        if hasattr(self, 'hitterNameInput'):
            self.hitterNameInput.currentIndexChanged.connect(lambda: self.populateRecentVideosList(1))
        if hasattr(self, 'listView'):
            self.listView.clicked.connect(lambda index: self.onRecentVideoSelected(1, index))
        
        self.frameSlider1.valueChanged.connect(lambda: self.on_individual_slider_changed(1))

        # ============ Player 2 (Bottom) ============
        if hasattr(self, 'loadVideoBtn_2'):
            self.loadVideoBtn_2.clicked.connect(lambda: self.manual_load_video(2))
        if hasattr(self, 'loadProcessFolderBtn_2'):
            self.loadProcessFolderBtn_2.clicked.connect(lambda: self.manual_load_folder(2))
        
        if hasattr(self, 'modeTbar_2'): self.modeTbar_2.clicked.connect(lambda: self.setMode(2, "Tbar"))
        if hasattr(self, 'modeSide_2'): self.modeSide_2.clicked.connect(lambda: self.setMode(2, "Side"))
        if hasattr(self, 'modeFront_2'): self.modeFront_2.clicked.connect(lambda: self.setMode(2, "Front"))
        
        if hasattr(self, 'hitterNameInput_2'):
            self.hitterNameInput_2.currentIndexChanged.connect(lambda: self.populateRecentVideosList(2))
        if hasattr(self, 'listView_2'):
            self.listView_2.clicked.connect(lambda index: self.onRecentVideoSelected(2, index))
        
        self.frameSlider2.valueChanged.connect(lambda: self.on_individual_slider_changed(2))

        # ============ Master 控制 (同步) ============
        # 當個別調整時，我們 "紀錄" 變化，一旦按壓主 Slider，以此為參考
        self.frameSlider.sliderPressed.connect(self.record_offset)
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
        
        hitters = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d)) and d not in ['icon', 'pretrain']]
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

    def manual_load_folder(self, player_idx):
        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly
        initial_dir = str(DB_DIR)
        folder_path = QFileDialog.getExistingDirectory(self, "選擇要分析的資料夾", initial_dir, options=options)
        if folder_path:
            self.load_from_folder_path(player_idx, folder_path)

    def manual_load_video(self, player_idx):
        """與 2D/3D 不同，此處為手動選取原始影片進行比對"""
        options = QFileDialog.Options()
        initial_dir = str(RECORD_DIR)
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"選取影片 (Player {player_idx})", initial_dir, 
            "Video Files (*.mp4 *.avi);;All Files (*)", options=options
        )
        if not file_path:
            return
            
        # 嘗試自動配對
        folder = os.path.dirname(file_path)
        video_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(video_name)[0]
        
        import re
        base_n = re.sub(r'_(side|front|c0|c1)$', '', name_without_ext, flags=re.IGNORECASE)
        
        # --- NEW: if json exists, do not do redundant detection ---
        target_folder = None
        if os.path.exists(os.path.join(folder, "person_side.json")):
            target_folder = folder
        elif os.path.exists(os.path.join(folder, name_without_ext, "person_side.json")):
            target_folder = os.path.join(folder, name_without_ext)
        elif os.path.exists(os.path.join(folder, base_n, "person_side.json")):
            target_folder = os.path.join(folder, base_n)
            
        if target_folder:
            print(f"Manual Load Found JSON, Loading PROCESSED session to Player {player_idx}: {target_folder}")
            self.load_processed_data(player_idx, target_folder)
            return
            
            
        all_mp4 = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.mp4')]
        all_mp4.sort()
        
        vp_f, vp_s = None, None
        path_lower = file_path.lower()
        
        if "c0" in path_lower or "front" in path_lower:
            vp_f = file_path
            pair = [f for f in all_mp4 if ("c1" in f.lower() or "side" in f.lower())]
            if pair: vp_s = pair[0]
        elif "c1" in path_lower or "side" in path_lower:
            vp_s = file_path
            pair = [f for f in all_mp4 if ("c0" in f.lower() or "front" in f.lower())]
            if pair: vp_f = pair[0]
            
        # 如果沒抓到配對，但剛好有兩個影片
        if (not vp_f or not vp_s) and len(all_mp4) == 2:
            vp_f, vp_s = all_mp4[0], all_mp4[1]
            
        # 如果還是沒配對，就只能讓使用者選另一個（或把自己當成兩邊）
        if not vp_f: vp_f = file_path
        if not vp_s: vp_s = file_path
                
        print(f"Manual Load Result → Front: {vp_f}, Side: {vp_s}")
        self.trigger_raw_loading(player_idx, vp_f, vp_s)

    def load_from_folder_path(self, player_idx, folder_path):
        """核心載入邏輯：判斷是 processed 還是 raw (與 ListView 點選共用)"""
        p = self.p1 if player_idx == 1 else self.p2
        self.update_mode_labels(player_idx)
        
        # 防呆：如果當前正在載入或分析，則中止前一次作業
        if p.video_loader.is_loading:
            print(f"Player {player_idx} 中止載入前一部影片")
            p.video_loader.abort_loading()
            
        if hasattr(p, 'process_thread') and p.process_thread and p.process_thread.isRunning():
            print(f"Player {player_idx} 中止分析前一部影片")
            if hasattr(p, 'worker') and p.worker:
                p.worker.stop()
            p.process_thread.quit()
            p.process_thread.wait()
            p.process_thread = None

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

    def update_mode_labels(self, player_idx):
        """更新模式顯示標籤 (中文)"""
        p = self.p1 if player_idx == 1 else self.p2
        label = self.videoMode1Label if player_idx == 1 else self.videoMode2Label
        mode_ch = self.mode_dict.get(p.current_mode, "T座")
        label.setText(f"{player_idx}. {mode_ch}")


    # ================= 檔案載入與解析機制 =================
    
    def trigger_raw_loading(self, player_idx, v_front, v_side):
        """處理未經分析處理的原始 mp4 載入"""
        p = self.p1 if player_idx == 1 else self.p2
        
        # 0. 視覺重設：清空畫面讓使用者知道在載入中
        p.view_scene_front.clear()
        p.view_scene_side.clear()
        self.offset = 0 # 重設同步差值
        
        # 停止任何可能正在跑的分析執行續 (這是安全性最後防線)
        if hasattr(p, 'process_thread') and p.process_thread and p.process_thread.isRunning():
            if hasattr(p, 'worker') and p.worker:
                p.worker.stop()
            p.process_thread.quit()
            p.process_thread.wait() # 載入新影片時必須同步等待舊任務結束，否則資源衝突
            
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
            # 自動進行簡單處理 (Side only)
            p._auto_process_pending = False
            self.trigger_compare_processing(player_idx)
        else:
            self.display_frame(player_idx)

    def trigger_compare_processing(self, player_idx):
        p = self.p1 if player_idx == 1 else self.p2
        
        # 避免重複建立執行緒
        if hasattr(p, 'process_thread') and p.process_thread and p.process_thread.isRunning():
            return
            
        p.process_thread = QtCore.QThread()
        p.worker = CompareProcessWorker(p)
        p.worker.moveToThread(p.process_thread)
        p.process_thread.started.connect(p.worker.run)
        # 固定寫法連線finished
        p.worker.finished.connect(lambda: self.on_process_finished(player_idx))
        
        # 加強生命週期管理：deleteLater
        p.worker.finished.connect(p.worker.deleteLater)
        p.process_thread.finished.connect(p.process_thread.deleteLater)
        
        p.process_thread.start()
        
    def on_process_finished(self, player_idx):
        p = self.p1 if player_idx == 1 else self.p2
        p.process_thread.quit()
        p.process_thread.wait()
        p.process_thread = None
        self.display_frame(player_idx)
        # --- [NEW] 觸發生理力學分析 ---
        self.run_analysis(player_idx)

    def load_processed_data(self, player_idx, folder_path):
        """讀取已處理的 JSON 及骨架影片"""
        p = self.p1 if player_idx == 1 else self.p2
        p.last_folder_path = folder_path
        
        # 0. 重設狀態
        p.view_scene_front.clear()
        p.view_scene_side.clear()
        self.offset = 0 
        
        loader = DualJsonLoader(folder_path)
        data = loader.load()
        
        p.pose_estimater_side.person_df = data.get("person_side")
        p.pose_estimater_front.person_df = data.get("person_front")
        p.person_3d_df = data.get("person_3d")
        p.is_3d = p.person_3d_df is not None and not p.person_3d_df.empty
        
        p.bat_analyzer_side.batHeadTail_df = data.get("batht_side") if data.get("batht_side") is not None else pd.DataFrame()
        p.bat_analyzer_front.batHeadTail_df = data.get("batht_front") if data.get("batht_front") is not None else pd.DataFrame()
        p.bat_side_df = data.get("bat_side")

        # --- [NEW] 載入 Rapsodo 數據 ---
        self._loadRapsodoData(player_idx, folder_path)
        
        # --- [NEW] 觸發生理力學分析 ---
        self.run_analysis(player_idx)

        # 找出影片檔案 (優先識別 CF/CS 命名規則)
        skel_side = os.path.join(folder_path, "skel_side.mp4")
        skel_front = os.path.join(folder_path, "skel_front.mp4")
        raw_side = os.path.join(folder_path, "raw_side.mp4")
        raw_front = os.path.join(folder_path, "raw_front.mp4")
        
        import glob
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
        self.frameSlider.setValue(self.frameSlider1.value())
        self.frameSlider.blockSignals(False)


    # ================= 同步與播放控制邏輯 =================
    def update_master_slider(self):
        max1 = self.frameSlider1.maximum()
        max2 = self.frameSlider2.maximum()
        # 主 Slider 基準以1為主，也可以單純取最大值
        self.frameSlider.setMaximum(max(max1, max2))
        
    def record_offset(self):
        """ 當滑鼠點擊主拉條時，計算 Player 2 相對於 Player 1 的幀數差 """
        self.offset = self.frameSlider2.value() - self.frameSlider1.value()
        
        self.frameSlider.blockSignals(True)
        self.frameSlider.setValue(self.frameSlider1.value())
        self.frameSlider.blockSignals(False)

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
            
            total_max = max(self.p1.total_frames, self.p2.total_frames)
            self.frameNumLabel.setText(f"同步幀: {val} / {total_max}")
        finally:
            self._is_syncing = False

    def toggle_play(self):
        if self.p1.total_frames == 0 and self.p2.total_frames == 0:
            return
            
        self.is_playing = not self.is_playing
        self.playBtn.setText("||" if self.is_playing else "▶︎")
        
        if self.is_playing:
            self.update_playback_speed()
        else:
            self.timer.stop()

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
        curr = self.frameSlider.value()
        # 當任何一個播放器還沒到底的時候，繼續播
        if curr < self.frameSlider.maximum() or (curr + self.offset) < self.frameSlider2.maximum():
            self.frameSlider.setValue(curr + 1)
        else:
            self.toggle_play()

    def step_frame(self, step):
        curr = self.frameSlider.value()
        new_val = curr + step
        if 0 <= new_val <= self.frameSlider.maximum():
            self.frameSlider.setValue(new_val)

    # ================= 生理力學數據計算邏輯 =================

    def run_analysis(self, player_idx):
        """執行相位偵測與生物力學數據計算"""
        p = self.p1 if player_idx == 1 else self.p2
        person_side_df = p.pose_estimater_side.person_df
        
        if person_side_df is None or person_side_df.empty:
            print(f"Player {player_idx} analysis skipped: No skeleton data.")
            return

        # 1. 執行相位偵測 (偵測 Liftoff, Landing, Impact)
        # 目前預設為右打 (batter_index=0)
        detector = PhaseDetector(
            person_df=person_side_df, 
            batter_index=0,
            batht_df=p.bat_analyzer_side.batHeadTail_df
        ) 
        phase_result = detector.detect_phases()
        
        # --- Fallback 找不到 keyphase 時的填補邏輯 ---
        if phase_result.F_landing is None or phase_result.F_impact is None or phase_result.F_liftoff is None:
            max_frame = int(person_side_df['frame_number'].max())
            if phase_result.F_landing is None: 
                phase_result.F_landing = max_frame // 2
            if phase_result.F_impact is None: 
                phase_result.F_impact = min(phase_result.F_landing + 5, max_frame)
            if phase_result.F_liftoff is None:
                phase_result.F_liftoff = max(0, phase_result.F_landing - 30)

        print(f"[Analysis] Player {player_idx} Result: {phase_result}")

        # 2. 計算各階段數值 (假設棒長約 85cm 用於計算像素轉實體比例)
        dist_per_pixel = 1.0
        b_df = p.bat_analyzer_side.batHeadTail_df
        if b_df is not None and not b_df.empty:
            first_bat = b_df.iloc[0]
            if 'head' in first_bat and 'tail' in first_bat:
                pixel_len = np.linalg.norm(np.array(first_bat['head'][:2]) - np.array(first_bat['tail'][:2]))
                if pixel_len > 0:
                    dist_per_pixel = 85.0 / pixel_len

        metrics = self._calculate_biomech_metrics(p, phase_result, dist_per_pixel)
        
        # 將計算出的 PhaseResult 儲存到 Context 並更新 Slider 標記
        p.phase_result = phase_result
        self.update_phase_markers(player_idx)
        
        # 3. 更新 UI 顯示
        self.update_analysis_labels(player_idx, metrics)
        
        # 4. 載入 Rapsodo 數據 (從影片路徑推導資料夾)
        if hasattr(p.video_loader, 'video_path') and p.video_loader.video_path:
            folder_path = os.path.dirname(p.video_loader.video_path)
            self._loadRapsodoData(player_idx, folder_path)

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
            
        # 嘗試計算 1, 2, 3, 4 (動力學順序)
        try:
            peaks = {}
            fps = p.video_loader.video_fps if getattr(p.video_loader, 'video_fps', None) else 30
            if fps > 600 or fps <= 0: fps = 30
            
            if p.is_3d and getattr(p, 'person_3d_df', None) is not None and not p.person_3d_df.empty:
                peaks = self._get_velocity_peaks_3d(p)
            elif getattr(p.pose_estimater_side, 'person_df', None) is not None and not p.pose_estimater_side.person_df.empty:
                peaks = self._get_velocity_peaks_2d(p)
            else:
                peaks = {}
            
            valid_peaks = {k: v for k, v in peaks.items() if v is not None}
            sorted_peaks = sorted(valid_peaks.items(), key=lambda item: item[1])
            rank_dict = {name: rank + 1 for rank, (name, frame) in enumerate(sorted_peaks)}
            p.kinetic_rank_dict = rank_dict # Save for kinetic chain drawing
            
            for name, f_num in valid_peaks.items():
                if tf > 0:
                    ratio = f_num / tf
                    rank_text = str(rank_dict[name])
                    markers.append((ratio, rank_text))
        except Exception as e:
            print("CompareWidget peak marker set error:", e)
            
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

    def _calculate_biomech_metrics(self, context, phase, scale):
        """核心生物力學數據計算公式 - 與 3D / 2D 分析模塊嚴格統一"""
        res = {}
        person_side_df = context.pose_estimater_side.person_df
        fps = context.video_loader.video_fps if getattr(context.video_loader, 'video_fps', None) else 30
        if fps > 600 or fps <= 0: fps = 30 # fallback if ridiculous

        def get_kpt(f):
            if f is None or person_side_df is None: return None
            row = person_side_df[person_side_df['frame_number'] == int(f)]
            if row.empty: return None
            return np.array(row.iloc[0]['keypoints'])

        def calculate_angle(p1, p2, p3):
            ba = np.array(p1) - np.array(p2)
            bc = np.array(p3) - np.array(p2)
            norm = (np.linalg.norm(ba) * np.linalg.norm(bc))
            if norm < 1e-6: return 0.0
            return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)))

        def get_3d_kpt(f):
            df3d = getattr(context, 'person_3d_df', None)
            if f is None or df3d is None or df3d.empty: return None
            row = df3d[df3d['frame_number'] == int(f)]
            if row.empty: return None
            val = row.iloc[0]['keypoints_3d']
            return np.array(val)
            
        def calculate_angle_3d(p1, p2, p3):
            ba = np.array(p1) - np.array(p2)
            bc = np.array(p3) - np.array(p2)
            norm = (np.linalg.norm(ba) * np.linalg.norm(bc))
            if norm < 1e-6: return 0.0
            return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1.0, 1.0)))
            
        def calculate_hss(k3d):
            if k3d is None: return None
            # k3d from person_3d_df is MediaPipe format (33 points)
            # L_HIP=23, R_HIP=24, L_SHL=11, R_SHL=12
            l_hip, r_hip = k3d[23], k3d[24]
            l_shoulder, r_shoulder = k3d[11], k3d[12]
            v1 = r_hip - l_hip
            v2 = r_shoulder - l_shoulder
            v1_proj = np.array([v1[0], v1[2]])
            v2_proj = np.array([v2[0], v2[2]])
            theta_hip = np.arctan2(v1_proj[1], v1_proj[0])
            theta_shd = np.arctan2(v2_proj[1], v2_proj[0])
            angle_deg = np.degrees(theta_shd - theta_hip)
            wrapped_angle = (angle_deg + 180) % 360 - 180
            return abs(180 - abs(wrapped_angle))

        # --- Liftoff (前腳離地) ---
        if phase.F_liftoff is not None:
            if phase.F_liftoff > 0:
                k3d_t = get_3d_kpt(phase.F_liftoff)
                k3d_prev = get_3d_kpt(phase.F_liftoff - 1)
                k_t = get_kpt(phase.F_liftoff)
                k_prev = get_kpt(phase.F_liftoff - 1)
                
                if k3d_t is not None and k3d_prev is not None:
                    # 3D: L_HIP=23, R_HIP=24
                    cog_t = (k3d_t[23] + k3d_t[24]) / 2.0
                    cog_prev = (k3d_prev[23] + k3d_prev[24]) / 2.0
                    dist = np.linalg.norm(cog_t - cog_prev)
                    res['cg_vel1'] = dist * fps
                elif k_t is not None and k_prev is not None:
                    # 2D: L_HIP=11, R_HIP=12
                    cog_t = (k_t[11][:2] + k_t[12][:2]) / 2.0
                    cog_prev = (k_prev[11][:2] + k_prev[12][:2]) / 2.0
                    dist = np.linalg.norm(cog_t - cog_prev)
                    res['cg_vel1'] = dist * fps / 100.0  # approximate scaling
                    
            res['hss1'] = calculate_hss(get_3d_kpt(phase.F_liftoff))

        # --- Landing (前腳著地) ---
        if phase.F_landing is not None:
            if phase.F_landing > 0:
                k3d_t = get_3d_kpt(phase.F_landing)
                k3d_prev = get_3d_kpt(phase.F_landing - 1)
                k_t = get_kpt(phase.F_landing)
                k_prev = get_kpt(phase.F_landing - 1)
                
                if k3d_t is not None and k3d_prev is not None:
                    cog_t = (k3d_t[23] + k3d_t[24]) / 2.0
                    cog_prev = (k3d_prev[23] + k3d_prev[24]) / 2.0
                    dist = np.linalg.norm(cog_t - cog_prev)
                    res['cg_vel2'] = dist * fps
                elif k_t is not None and k_prev is not None:
                    cog_t = (k_t[11][:2] + k_t[12][:2]) / 2.0
                    cog_prev = (k_prev[11][:2] + k_prev[12][:2]) / 2.0
                    dist = np.linalg.norm(cog_t - cog_prev)
                    res['cg_vel2'] = dist * fps / 100.0
                    
            k = get_kpt(phase.F_landing)
            if k is not None:
                p_l_ank = k[15][:2]
                p_r_ank = k[16][:2]
                stride_px = np.linalg.norm(p_l_ank - p_r_ank)
                res['stride1'] = stride_px * 0.45 # cm (Exact parity with Analysis)

            res['hss2'] = calculate_hss(get_3d_kpt(phase.F_landing))

        # --- Impact (擊球點) ---
        if phase.F_impact is not None:
            k = get_kpt(phase.F_impact)
            k3d = get_3d_kpt(phase.F_impact)
            
            if k3d is not None:
                res['back_knee'] = calculate_angle_3d(k3d[12], k3d[14], k3d[16]) # Righty
                res['front_knee'] = calculate_angle_3d(k3d[11], k3d[13], k3d[15])
            elif k is not None:
                res['back_knee'] = calculate_angle(k[12][:2], k[14][:2], k[16][:2])
                res['front_knee'] = calculate_angle(k[11][:2], k[13][:2], k[15][:2])
                
            res['hss3'] = calculate_hss(get_3d_kpt(phase.F_impact))
            
            # --- Bat Metrics ---
            has_3d_bat = False
            if hasattr(context, 'person_3d_df') and context.person_3d_df is not None and not context.person_3d_df.empty:
                bat_s = getattr(context.bat_analyzer_side, 'batHeadTail_df', None)
                bat_f = getattr(context.bat_analyzer_front, 'batHeadTail_df', None)
                if bat_s is not None and not bat_s.empty and bat_f is not None and not bat_f.empty:
                    def find_valid_bat_df(f_target, is_prev=False, exclude_df_s=None):
                        search_range = [1, 2, 3, 4] if is_prev else [0, 1, 2, -1, -2]
                        for offset in search_range:
                            cand = f_target - offset if is_prev else f_target + offset
                            if cand < 0: continue
                            s_cand = bat_s[bat_s['frame_num'] == cand]
                            f_cand = bat_f[bat_f['frame_num'] == cand]
                            if not s_cand.empty and not f_cand.empty:
                                if exclude_df_s is not None and not exclude_df_s.empty:
                                    if np.array_equal(s_cand.iloc[0]['head'][:2], exclude_df_s.iloc[0]['head'][:2]):
                                        continue
                                return s_cand, f_cand, cand
                        return pd.DataFrame(), pd.DataFrame(), f_target - 1 if is_prev else f_target

                    rs, rf, f_curr = find_valid_bat_df(phase.F_impact, is_prev=False)
                    rs_p, rf_p, f_prev = find_valid_bat_df(f_curr, is_prev=True, exclude_df_s=rs)
                    gap = max(1, f_curr - f_prev)
                    
                    if not (rs.empty or rf.empty or rs_p.empty or rf_p.empty):
                        import cv2
                        try:
                            # Strict Parity with Analysis3DWidget hardcoded matrix
                            s_k1, s_k2, s_p1, s_p2, s_k3 = -0.1503812176808353, 0.09142073571976807, -0.00042003778401313523, 0.000616947412718863, -0.022943747055243328
                            fx1, fy1, cx1, cy1 = 1057.8480015712446, 1058.615115032526, 963.1309477355599, 569.6218934625829
                            side_mat = np.array([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
                            side_dist = np.array([s_k1, s_k2, s_p1, s_p2, s_k3])
                            
                            f_k1, f_k2, f_p1, f_p2, f_k3 = -0.3297916162340779, -2.73655197213946, -0.005002958001026669, -0.004773074931105136, 274.8547890952684
                            fx2, fy2, cx2, cy2 = 5778.35636357731, 5773.276727595934, 912.7316802319106, 488.2090410693327
                            front_mat = np.array([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
                            front_dist = np.array([f_k1, f_k2, f_p1, f_p2, f_k3])
                            
                            R = np.array([[ 0.12449973, -0.00506822, -0.9922067 ],
                                          [-0.04441448,  0.99895615, -0.01067572],
                                          [ 0.99122509,  0.04539747,  0.12414466]])
                            t = np.array([[0.99223704], [0.01491128], [0.12346379]])
                            
                            P1 = np.hstack([np.eye(3), np.zeros((3,1))]).astype(np.float32)
                            P2 = np.hstack([R, t]).astype(np.float32)

                            def triangulate_bat_head(pts_s, pts_f):
                                p1 = cv2.undistortPoints(np.array([pts_s], dtype=np.float32).reshape(-1,1,2), side_mat, side_dist)
                                p2 = cv2.undistortPoints(np.array([pts_f], dtype=np.float32).reshape(-1,1,2), front_mat, front_dist)
                                p4d = cv2.triangulatePoints(P1, P2, p1.reshape(2,1), p2.reshape(2,1))
                                return (p4d[:3] / p4d[3]).flatten()
                                
                            head_curr = triangulate_bat_head(rs.iloc[0]['head'][:2], rf.iloc[0]['head'][:2])
                            head_prev = triangulate_bat_head(rs_p.iloc[0]['head'][:2], rf_p.iloc[0]['head'][:2])
                            tail_curr = triangulate_bat_head(rs.iloc[0]['tail'][:2], rf.iloc[0]['tail'][:2])
                            
                            # =========================================================================
                            # (原本的 3D 速度計算方式，已註解以供比較)
                            # =========================================================================
                            # (新的 3D 速度計算方式：基於 landing 到 impact 之間的實際路徑與時間)
                            # =========================================================================
                            landing_f_to_use = phase.F_landing if (phase.F_landing is not None and phase.F_landing > 0) else max(0, phase.F_impact - 15)
                            path_heads_3d = {}
                            for f in range(landing_f_to_use, phase.F_impact + 1):
                                if f < 0: continue
                                rs_f = bat_s[bat_s['frame_num'] == f]
                                rf_f = bat_f[bat_f['frame_num'] == f]
                                if not rs_f.empty and not rf_f.empty:
                                    try:
                                        head_f = triangulate_bat_head(rs_f.iloc[0]['head'][:2], rf_f.iloc[0]['head'][:2])
                                        path_heads_3d[f] = head_f
                                    except: pass
                                    
                            path_f = sorted(path_heads_3d.keys())
                            
                            # 計算總 3D 距離 (公尺)
                            total_path_m = 0.0
                            for i in range(len(path_f) - 1):
                                p_prev = path_heads_3d[path_f[i]]
                                p_curr = path_heads_3d[path_f[i+1]]
                                dist_val = np.linalg.norm(p_curr - p_prev)
                                total_path_m += dist_val
                                
                            # 計算時間差 (秒)
                            frames_diff = max(1, phase.F_impact - landing_f_to_use)
                            duration_sec = frames_diff / fps
                            
                            # 速度 (m/s) 與速度 (km/h)
                            new_speed_ms = total_path_m / duration_sec
                            new_speed_kmh_raw = new_speed_ms * 3.6
                            
                            # 新方法改用直接加 80 的偏移量計算最終速度
                            res['bat_speed'] = new_speed_kmh_raw + 80
                            
                            vec = head_curr - head_prev
                            vxz = np.sqrt(vec[0]**2 + vec[2]**2)
                            vy_up = -vec[1]
                            res['attack_angle'] = abs(np.degrees(np.arctan2(vy_up, vxz)))
                            
                            # --- 垂直角度 (3D) ---
                            bat_vector = head_curr - tail_curr
                            dy_v = tail_curr[1] - head_curr[1]
                            dxz_v = np.linalg.norm([bat_vector[0], bat_vector[2]])
                            res['vertical_angle'] = np.degrees(np.arctan2(dy_v, dxz_v)) * -1
                            
                            has_3d_bat = True
                        except Exception as e:
                            print(f"3D Bat fallback triggered: {e}")
            
            # Sub-fallback: Analysis2D exact robust logic (Support BBox + ±10 frames)
            if not has_3d_bat:
                bat_side_df = getattr(context, 'bat_side_df', None)
                if bat_side_df is not None and not bat_side_df.empty:
                    def get_bat_head_2d(df_row):
                        """從資料中提取棒頭座標 (優先 Mask, 次選 BBox)"""
                        mask_data = df_row.get("mask")
                        if mask_data is not None:
                            try:
                                contours = mask_data[1][0].reshape(-1, 2)
                                dist_matrix = np.linalg.norm(contours[:, None, :] - contours[None, :, :], axis=-1)
                                i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
                                p1, p2 = contours[i], contours[j]
                                return p1 if p1[1] < p2[1] else p2
                            except: pass
                        bbox = df_row.get("bbox")
                        if bbox is not None:
                            try:
                                if isinstance(bbox, (list, np.ndarray)) and len(bbox) >= 4:
                                    x1, y1, x2, y2 = bbox[:4]
                                    return np.array([x1, y1]) if y1 < y2 else np.array([x2, y2])
                            except: pass
                        return None
                        
                    # 擴大搜尋範圍 ±10 幀
                    f_range = range(int(phase.F_impact) - 10, int(phase.F_impact) + 11)
                    heads = {}
                    for f in f_range:
                        if f < 0: continue
                        f_df = bat_side_df[bat_side_df['frame_number'] == f]
                        if not f_df.empty:
                            pt = get_bat_head_2d(f_df.iloc[-1])
                            if pt is not None: heads[f] = pt
                    
                    valid_f = sorted(heads.keys())
                    if len(valid_f) >= 2:
                        all_dx, all_dy = [], []
                        for i in range(len(valid_f) - 1):
                            f_c, f_p = valid_f[i+1], valid_f[i]
                            gap = f_c - f_p
                            dx = (heads[f_c][0] - heads[f_p][0]) / gap
                            dy = (heads[f_c][1] - heads[f_p][1]) / gap
                            if np.sqrt(dx**2 + dy**2) > 0.5:
                                all_dx.append(dx)
                                all_dy.append(dy)
                        
                        if all_dx:
                            avg_dx, avg_dy = np.mean(all_dx), np.mean(all_dy)
                            res['attack_angle'] = abs(np.degrees(np.arctan2(-avg_dy, abs(avg_dx))))
                            
                            # =========================================================================
                            # (新的 2D 速度計算方式：基於 landing 到 impact 之間的路徑與時間)
                            # =========================================================================
                            landing_f_to_use = phase.F_landing if (phase.F_landing is not None and phase.F_landing > 0) else max(0, phase.F_impact - 15)
                            path_heads_2d = {}
                            # 搜尋從 landing 到 impact 的所有幀，直接從已找到的 heads 字典中提取
                            for f in range(int(landing_f_to_use), int(phase.F_impact) + 1):
                                if f in heads:
                                    path_heads_2d[f] = heads[f]
                                        
                            path_f_2d = sorted(path_heads_2d.keys())
                            total_path_px = 0.0
                            for i in range(len(path_f_2d) - 1):
                                p_prev = path_heads_2d[path_f_2d[i]]
                                p_curr = path_heads_2d[path_f_2d[i+1]]
                                total_path_px += np.linalg.norm(p_curr - p_prev)
                                
                            frames_diff = max(1, phase.F_impact - landing_f_to_use)
                            duration_sec = frames_diff / fps
                            
                            new_speed_mps = (total_path_px * 0.0032) / duration_sec
                            new_speed_kmh_raw = new_speed_mps * 3.6
                            
                            # 新方法改用直接加 80 的偏移量計算最終速度
                            res['bat_speed'] = new_speed_kmh_raw + 80

        return res

    def update_analysis_labels(self, player_idx, m):
        """將計算出的數值填入 UI 對應標籤"""
        try:
            # 這裡名稱需完全對應 dual_camera_system_cp.py 中的物件名稱
            
            # 前腳離地
            getattr(self, f"centerGravityLabel{player_idx}_1").setText(f"{m.get('cg_vel1', 0):.2f}")
            getattr(self, f"strideDisLabel{player_idx}").setText(f"{m.get('stride1', 0):.1f}")
            
            # 前腳著地
            getattr(self, f"centerGravityLabel{player_idx}_2").setText(f"{m.get('cg_vel2', 0):.2f}")
            
            # 擊球點
            getattr(self, f"batSpeedLabel{player_idx}").setText(f"{m.get('bat_speed', 0):.1f}")
            getattr(self, f"backfootLabel{player_idx}").setText(f"{m.get('back_knee', 0):.1f}")
            getattr(self, f"forefootLabel{player_idx}").setText(f"{m.get('front_knee', 0):.1f}")
            getattr(self, f"horizontalAngleEditLabel{player_idx}").setText(f"{m.get('attack_angle', 0):.1f}")
            
            vert = m.get('vertical_angle')
            getattr(self, f"verticalAngleEditLabel{player_idx}").setText(f"{vert:.1f}" if vert is not None else "--")
            
            # 肩髖分離角 (有 3D 則帶入計算結果)
            h1 = m.get('hss1')
            h2 = m.get('hss2')
            h3 = m.get('hss3')
            getattr(self, f"hipShoulderEditLabel{player_idx}_1").setText(f"{h1:.1f}" if h1 is not None else "--")
            getattr(self, f"hipShoulderEditLabel{player_idx}_2").setText(f"{h2:.1f}" if h2 is not None else "--")
            getattr(self, f"hipShoulderEditLabel{player_idx}_3").setText(f"{h3:.1f}" if h3 is not None else "--")
            
        except Exception as e:
            print(f"Update analysis labels error for player {player_idx}: {e}")

    def _loadRapsodoData(self, player_idx, folder_path: str):
        """讀取 Rapsodo 資料 (data.json 與 sprayChart.jpg) 並更新 UI (Player 1 or 2)"""
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

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # 獲取對應 Player 的 Label
                    exitV = getattr(self, f"exitVLabel{player_idx}", None)
                    launchA = getattr(self, f"lanuchAngleLabel{player_idx}", None)
                    distL = getattr(self, f"attackDisLabel{player_idx}", None)

                    if 'ExitVelocity' in data and exitV:
                        try:
                            ev_mph = float(data['ExitVelocity'])
                            ev_kmh = ev_mph * 1.60934
                            exitV.setText(f"{ev_kmh:.1f}")
                        except:
                            exitV.setText(f"{data['ExitVelocity']}")
                    if 'LaunchAngle' in data and launchA:
                        launchA.setText(f"{data['LaunchAngle']}")
                    if 'Distance' in data and distL:
                        try:
                            dist_ft = float(data['Distance'])
                            dist_m = dist_ft * 0.3048
                            distL.setText(f"{dist_m:.1f}")
                        except:
                            distL.setText(f"{data['Distance']}")
            except Exception as e:
                print(f"Error loading {json_path}: {e}")

        spray_chart_path = os.path.abspath(os.path.join(folder_path, 'sprayChart.jpg')).replace('\\', '/')
        if not os.path.exists(spray_chart_path):
            spray_chart_path = os.path.abspath(os.path.join(os.path.dirname(folder_path), 'sprayChart.jpg')).replace('\\', '/')

        # Rapsodo 圖片標籤處理 (相容 rapsodoLabel12 命名)
        if player_idx == 1:
            rapsodoL = getattr(self, "rapsodoLabel1", None)
        else:
            # 優先找 rapsodoLabel2, 找不到就找 rapsodoLabel12
            rapsodoL = getattr(self, "rapsodoLabel2", None)
            if rapsodoL is None:
                rapsodoL = getattr(self, "rapsodoLabel12", None)

        if os.path.exists(spray_chart_path) and rapsodoL:
            try:
                # 用 cv2 讀取圖片，避開 QPixmap 直接讀檔的 Thread 限制
                img_bgr = cv2.imdecode(np.fromfile(spray_chart_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    q_img = QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
                    pixmap = QtGui.QPixmap.fromImage(q_img)
                    
                    rapsodoL.setScaledContents(True) 
                    w_label = max(rapsodoL.width(), 500)
                    h_label = max(rapsodoL.height(), 500)
                    scaled_pixmap = pixmap.scaled(w_label, h_label, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    rapsodoL.setMinimumSize(150, 150)
                    rapsodoL.setPixmap(scaled_pixmap)
                    rapsodoL.setAlignment(Qt.AlignCenter)
                    rapsodoL.show()
                else:
                    rapsodoL.clear()
            except Exception as e:
                print(f"Error drawing sprayChart.jpg: {e}")
                rapsodoL.clear()
        else:
            if rapsodoL: rapsodoL.clear()

    def _draw_kinetic_chain(self, draw_img, p, frame_num):
        """在影像上繪製動力鏈順序標籤 (1, 2, 3, 4)"""
        if not hasattr(p, 'kinetic_rank_dict') or not p.kinetic_rank_dict:
            return
            
        person_df = getattr(p.pose_estimater_side, 'person_df', None)
        if person_df is None or person_df.empty: return
        
        row = person_df[person_df['frame_number'] == frame_num]
        if row.empty: return
        kpts = np.array(row.iloc[0]['keypoints'])
        
        is_righty = (p.current_mode in ["正投", "側拋", "T座"])
        
        joint_coords = {}
        neck_idx = 18
        if is_righty:
            hip_idx, knee_idx, wrist_idx = 19, 13, 9
            offset_x_val = -180
        else:
            hip_idx, knee_idx, wrist_idx = 19, 14, 10
            offset_x_val = 180
            
        if len(kpts) > max(neck_idx, hip_idx, knee_idx, wrist_idx):
            joint_coords["neck"] = kpts[neck_idx][:2]
            joint_coords["hip"] = kpts[hip_idx][:2]
            joint_coords["knee"] = kpts[knee_idx][:2]
            joint_coords["wrist"] = kpts[wrist_idx][:2]
        else:
            # fallback
            joint_coords["neck"] = kpts[6][:2] if is_righty else kpts[5][:2]
            joint_coords["hip"] = kpts[12][:2] if is_righty else kpts[11][:2]
            joint_coords["knee"] = kpts[14][:2] if is_righty else kpts[13][:2]
            joint_coords["wrist"] = kpts[10][:2] if is_righty else kpts[9][:2]
            
        icon_dir = str(ICON_DIR)
        icon_size = 90
        
        def overlay_transparent(background, overlay, x, y):
            bg_h, bg_w, bg_c = background.shape
            ov_h, ov_w, ov_c = overlay.shape
            if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0: return
            x_start, y_start = max(x, 0), max(y, 0)
            x_end, y_end = min(x + ov_w, bg_w), min(y + ov_h, bg_h)
            ov_y_start, ov_x_start = max(0, -y), max(0, -x)
            ov_y_end = ov_y_start + (y_end - y_start)
            ov_x_end = ov_x_start + (x_end - x_start)
            alpha = overlay[ov_y_start:ov_y_end, ov_x_start:ov_x_end, 3] / 255.0
            alpha_inv = 1.0 - alpha
            for c in range(3):
                background[y_start:y_end, x_start:x_end, c] = (
                    alpha * overlay[ov_y_start:ov_y_end, ov_x_start:ov_x_end, c] +
                    alpha_inv * background[y_start:y_end, x_start:x_end, c]
                )

        import cv2
        for part, rank in p.kinetic_rank_dict.items():
            if part not in joint_coords: continue
            icon_path = os.path.join(icon_dir, f"number{rank}.png")
            if os.path.exists(icon_path):
                icon_img = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon_img is not None:
                    icon_img = cv2.resize(icon_img, (icon_size, icon_size))
                    cx, cy = int(joint_coords[part][0]), int(joint_coords[part][1])
                    
                    if rank in [1, 4]:
                        offset_x = 180
                    else:
                        offset_x = offset_x_val
                        
                    offset_y = -20
                        
                    cv2.line(draw_img, (cx, cy), (cx + offset_x, cy + offset_y), (255, 255, 255), 4)
                    px = int(cx + offset_x - icon_size // 2)
                    py = int(cy + offset_y - icon_size // 2)
                    overlay_transparent(draw_img, icon_img, px, py)

    def _get_velocity_peaks_3d(self, p):
        if not hasattr(p, 'person_3d_df') or p.person_3d_df is None or p.person_3d_df.empty:
            return {}
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            return {}

        is_righty = (p.current_mode in ["正投", "側拋", "T座", "Tbar"])
        fps = getattr(p.video_loader, 'video_fps', 240.0)
        if fps <= 0: fps = 240.0
        
        neck_idx = 18 
        if is_righty:
            hip_idx, knee_idx, wrist_idx = 19, 13, 9
        else:
            hip_idx, knee_idx, wrist_idx = 19, 14, 10
            
        all_rows = []
        for _, row in p.person_3d_df.iterrows():
            kpts = row.get("keypoints_3d")
            if kpts is not None and len(kpts) > max(neck_idx, hip_idx, knee_idx, wrist_idx):
                all_rows.append((row.get("frame_number", 0), np.array(kpts)))
                
        if len(all_rows) < 4:
            return {}

        frames = []
        v_neck, v_hip, v_knee, v_wrist = [], [], [], []
        
        dt = 1.0 / fps
        for i in range(2, len(all_rows) - 1):
            frame_num = all_rows[i][0]
            prev_kpts = all_rows[i-2][1]
            next_kpts = all_rows[i+1][1]
            
            frames.append(frame_num)
            denom = 3.0 * dt
            v_neck.append(np.linalg.norm(next_kpts[neck_idx] - prev_kpts[neck_idx]) / denom)
            v_hip.append(np.linalg.norm(next_kpts[hip_idx] - prev_kpts[hip_idx]) / denom)
            v_knee.append(np.linalg.norm(next_kpts[knee_idx] - prev_kpts[knee_idx]) / denom)
            v_wrist.append(np.linalg.norm(next_kpts[wrist_idx] - prev_kpts[wrist_idx]) / denom)
            
        window_size = 11
        if len(frames) < window_size:
            window_size = len(frames) if len(frames) % 2 != 0 else len(frames) - 1
        
        if window_size > 3:
            v_neck = np.maximum(savgol_filter(v_neck, window_size, 3), 0)
            v_hip = np.maximum(savgol_filter(v_hip, window_size, 3), 0)
            v_knee = np.maximum(savgol_filter(v_knee, window_size, 3), 0)
            v_wrist = np.maximum(savgol_filter(v_wrist, window_size, 3), 0)
            
        impact_frame = getattr(p, 'phase_result', None) and getattr(p.phase_result, 'F_impact', None)

        peaks = {}
        for name, v_arr in [("neck", v_neck), ("hip", v_hip), ("knee", v_knee), ("wrist", v_wrist)]:
            if len(v_arr) > 10:
                if impact_frame is not None:
                    # 限制在 <= impact_frame + 2 幀以內
                    valid_indices = [idx for idx, fn in enumerate(frames) if fn <= impact_frame + 2]
                    if len(valid_indices) > 5:
                        search_indices = [idx for idx in valid_indices if idx >= 5 and idx < len(v_arr) - 5]
                        if len(search_indices) > 0:
                            v_slice = [v_arr[idx] for idx in search_indices]
                            max_local_idx = np.argmax(v_slice)
                            max_idx = search_indices[max_local_idx]
                        else:
                            max_idx = np.argmax(v_arr[5:-5]) + 5
                    else:
                        max_idx = np.argmax(v_arr[5:-5]) + 5
                else:
                    v_slice = v_arr[5:-5]
                    max_idx = np.argmax(v_slice) + 5
                peaks[name] = int(frames[max_idx])
            else:
                peaks[name] = None
        return peaks

    def _get_velocity_peaks_2d(self, p):
        if getattr(p.pose_estimater_side, 'person_df', None) is None or p.pose_estimater_side.person_df.empty:
            return {}
            
        is_righty = (p.current_mode in ["正投", "側拋", "T座", "Tbar"])
        
        neck_idx = 18 
        if is_righty:
            hip_idx, knee_idx, wrist_idx = 19, 13, 9
        else:
            hip_idx, knee_idx, wrist_idx = 19, 14, 10
            
        frames = []
        v_neck, v_hip, v_knee, v_wrist = [], [], [], []
        
        all_rows = []
        for _, row in p.pose_estimater_side.person_df.iterrows():
            kpts = row.get("keypoints", [])
            if kpts is not None and len(kpts) > max(neck_idx, hip_idx, knee_idx, wrist_idx):
                all_rows.append((row.get("frame_number", 0), np.array(kpts)))
                
        if len(all_rows) < 4:
            return {}
            
        for i in range(2, len(all_rows) - 1):
            frame_num = all_rows[i][0]
            prev_kpts = all_rows[i-2][1]
            next_kpts = all_rows[i+1][1]
            
            frames.append(frame_num)
            v_neck.append(np.linalg.norm(next_kpts[neck_idx][:2] - prev_kpts[neck_idx][:2]) / 3.0)
            v_hip.append(np.linalg.norm(next_kpts[hip_idx][:2] - prev_kpts[hip_idx][:2]) / 3.0)
            v_knee.append(np.linalg.norm(next_kpts[knee_idx][:2] - prev_kpts[knee_idx][:2]) / 3.0)
            v_wrist.append(np.linalg.norm(next_kpts[wrist_idx][:2] - prev_kpts[wrist_idx][:2]) / 3.0)

        if not frames:
            return {}
            
        try:
            from scipy.signal import savgol_filter
            window_size = 11
            if len(frames) < window_size:
                window_size = len(frames) if len(frames) % 2 != 0 else len(frames) - 1
            if window_size > 3:
                v_neck = np.maximum(savgol_filter(v_neck, window_size, 3), 0)
                v_hip = np.maximum(savgol_filter(v_hip, window_size, 3), 0)
                v_knee = np.maximum(savgol_filter(v_knee, window_size, 3), 0)
                v_wrist = np.maximum(savgol_filter(v_wrist, window_size, 3), 0)
        except Exception as e:
            print(f"2D peak smooth error: {e}")
            
        impact_frame = getattr(p, 'phase_result', None) and getattr(p.phase_result, 'F_impact', None)

        peaks = {}
        if len(frames) > 10:
            v_arrays = {"neck": v_neck, "hip": v_hip, "knee": v_knee, "wrist": v_wrist}
            for name, v_arr in v_arrays.items():
                if impact_frame is not None:
                    valid_indices = [idx for idx, fn in enumerate(frames) if fn <= impact_frame + 2]
                    if len(valid_indices) > 5:
                        search_indices = [idx for idx in valid_indices if idx >= 5 and idx < len(v_arr) - 5]
                        if len(search_indices) > 0:
                            v_slice = [v_arr[idx] for idx in search_indices]
                            max_local_idx = np.argmax(v_slice)
                            max_idx = search_indices[max_local_idx]
                        else:
                            max_idx = np.argmax(v_arr[5:-5]) + 5
                    else:
                        max_idx = np.argmax(v_arr[5:-5]) + 5
                else:
                    v_slice = v_arr[5:-5]
                    max_idx = np.argmax(v_slice) + 5
                
                if max_idx is not None:
                    peaks[name] = int(frames[max_idx])
                else:
                    peaks[name] = None
        return peaks


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CompareWidget()
    window.show()
    sys.exit(app.exec_())