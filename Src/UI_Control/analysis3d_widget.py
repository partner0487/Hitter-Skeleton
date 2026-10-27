# hitter_video_widget_dual_processed.py
import os
import pandas as pd
import glob
import cv2
import PyQt5.QtCore as QtCore
from PIL import Image, ImageDraw, ImageFont
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget, QFileDialog, QGraphicsView, QGraphicsScene, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush, QStandardItemModel, QStandardItem
from utils.selector import PersonSelector, KptSelector
from utils.analyze import PoseAnalyzer, BatAnalyzer
from utils.vis_graph_3d import HipShoulderAngleGraph, BatKinematicsGraph
from cv_utils.cv_control_3d import VideoLoader 
import numpy as np
from vispy import app as vis_app, visuals, scene, geometry, color
from vispy import app
from vispy.scene import cameras
from utils.vis_image_dual import ImageDrawer as ImageDrawerDual
from cv_utils.cv_control_dual import VideoLoader as VideoLoaderDual
from utils.vis_image_3d import ImageDrawer
from ui.dual_camera_system_video import Ui_hitter_ui_dual 
from utils.phase_detect3d import PhaseDetector, PhaseResult
from utils.model import Model
from skeleton.detect_skeleton_dual import PoseEstimater
from utils.paths import DB_DIR, RECORD_DIR, ICON_DIR
from PyQt5 import QtGui

class AnalysisBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        # 1. 增加高度：設為 80，確保交錯標籤 (a, b, c) 完整顯示
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True) # 支援樣式
        # self.setFixedHeight(80)
        # 確保滑鼠穿透(如果你不需要拖拽功能)
        # self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        
        self.segments = []

    def set_segments(self, segments):
        self.segments = segments
        self.update()

    def set_markers(self, markers):
        """設定在進度條上標註的點位: [(ratio, 'text'), ...]"""
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
            
            # --- 垂直佈局調整 ---
            bar_h = 12
            # 色塊位置：置中偏下一點點，騰出上方給大字體
            bar_y = int(h * 0.52) - (bar_h // 2)  
            total_duration = 1.5

            # 1. 繪製色塊 (底層)
            for start_ratio, end_ratio, color in self.segments:
                x = int(start_ratio * w)
                seg_w = int((end_ratio - start_ratio) * w)
                if seg_w <= 0: continue
                painter.fillRect(QRect(x, bar_y, seg_w, bar_h), QColor(color))

            # 2. 繪製標記 (1,2,3 與 a,b,c) - 緊貼邊緣
            if hasattr(self, 'markers') and self.markers:
                # 紀錄每個 mx 座標上已繪製的動力鏈標記數量
                from collections import defaultdict
                mx_counts = defaultdict(int)
                
                for ratio, marker_text in self.markers:
                    mx = int(max(0.0, min(ratio, 1.0)) * w)
                    marker_str = str(marker_text)
                    
                    if marker_str in ['1', '2', '3', '4']:
                        radius = 12
                        count = mx_counts[mx]
                        mx_counts[mx] += 1
                        cy = bar_y - radius - count * 24 # 全部都放 bar 上面，但高度遞增錯開
                        bg = {'1': QColor(220, 0, 0), '2': QColor(0, 170, 230), '3': QColor(0, 180, 0), '4': QColor(128, 0, 128)}.get(marker_str, Qt.gray)
                        font_size = 12
                    else:
                        radius = 12
                        cy = bar_y + bar_h + radius # 貼齊色塊下緣
                        bg = {'a': QColor(255, 50, 50), 'b': QColor(255, 150, 0), 'c': QColor(0, 200, 255)}.get(marker_str, Qt.gray)
                        font_size = 12

                    rect = QRect(mx - radius, cy - radius, radius * 2, radius * 2)
                    painter.setBrush(QBrush(bg))
                    painter.setPen(QPen(Qt.black, 1))
                    painter.drawEllipse(rect)
                    
                    painter.setFont(QFont("Arial", font_size, QFont.Bold))
                    painter.setPen(QPen(Qt.white))
                    painter.drawText(rect, Qt.AlignCenter, marker_str)

            # 3. 畫秒數文字 (字體變大、向下靠近 Bar)
            for start_ratio, end_ratio, color in self.segments:
                x = int(start_ratio * w)
                seg_w = int((end_ratio - start_ratio) * w)
                if seg_w < 15: continue
                
                seconds = (end_ratio - start_ratio) * total_duration
                t_color = QColor(color)
                t_color.setAlpha(255)
                painter.setPen(t_color)
                
                # --- 修改點：字體放大至 14 ---
                painter.setFont(QFont("Arial", 14, QFont.Bold))
                
                # --- 修改點：位置向下沈，貼近 Bar ---
                text_h = 24 # 增加文字框高度容納大字體
                text_y = bar_y - text_h - 4 # 距離色塊頂部僅 4px，會與標記圓圈部分重疊
                
                painter.drawText(QRect(x, text_y, seg_w, text_h), Qt.AlignCenter, f"{seconds:.2f}s")
                
        finally:
            painter.end()

class Analysis3DWidget(QWidget):
    """
    -----------------------------------------
     🔥 離線版本 (LoadFolder → 直接分析)
    -----------------------------------------
    - 不需要 PoseEstimater
    - 不跑偵測，可直接讀 JSON
    - 可由 init TAB 傳來 P1/P2 (你要求的耦合方式)
    """
    processFinished = pyqtSignal(str)

    def __init__(self, model=Model,parent=None):
        super().__init__(parent)
        self.ui = Ui_hitter_ui_dual()
        self.ui.setupUi(self)
        self.folder = None
        self.frame = 0
        self.total_frames = 0
        self.video_side = None
        self.video_front = None
        self.P1 = None
        self.P2 = None
        self.timer = QTimer()
        self.model = model
        self.setupComponents()
        self.initVar()
        self.bindUI()
        # self.bindCheckBox()
        self.initVispyCanvas()

    def shutdown(self):
        """關閉程式前停止計時器與影片載入執行緒"""
        self.timer.stop()
        # cv_control_3d 的 VideoLoader 沒有 abort_loading，直接停止它的載入執行緒
        for loader in (self.video_loader, self.video_loader_raw):
            v_t = getattr(loader, 'v_t', None)
            if loader.is_loading and v_t is not None:
                v_t.stop()
                v_t.wait()
        if self.video_loader_dual.is_loading:
            self.video_loader_dual.abort_loading()

    def initVar(self):
        """Initialize variables and timer."""
        self.is_play = False
        self.view_scene = QGraphicsScene()
        self.view_scene2 = QGraphicsScene()
        self.view_scene.clear()
        self.view_scene2.clear()
        self.init_view1 = []
        self.init_view2 = []
        self.cached_keyphase_pixmaps = {}  # 儲存 (label_object_id) -> QPixmap (原始圖)
        self.mode_dict = {"Tbar": "T座", "Side": "側拋", "Front": "正投"}
        self.current_mode = "Tbar"
        self.initParameters()

    def setupComponents(self):
        self.ui.frameSlider.setMinimumHeight(80)
        self.analysis_bar = AnalysisBar(self.ui.frameSlider)
        self.ui.frameSlider.installEventFilter(self)
        self.timer = QTimer()
        self.timer.timeout.connect(self.analyzeFrame)
        self.pose_estimater = PoseEstimater(self.model)
        self.pose_estimater_2 = PoseEstimater(self.model)
        self.bat_analyzer = BatAnalyzer(self.pose_estimater)
        self.bat_analyzer_2 = BatAnalyzer(self.pose_estimater_2)
        self.pose_analyzer = PoseAnalyzer(self.pose_estimater)
        self.pose_analyzer_2 = PoseAnalyzer(self.pose_estimater_2)
        self.image_drawer = ImageDrawer(self.pose_estimater, self.pose_analyzer, self.bat_analyzer)
        self.image_drawer_2 = ImageDrawer(self.pose_estimater_2, self.pose_analyzer_2, self.bat_analyzer_2)
        self.video_loader = VideoLoader()
        self.video_loader_raw = VideoLoader()
        self.image_drawer_dual = ImageDrawerDual(self.pose_estimater, self.pose_analyzer, self.bat_analyzer)
        self.image_drawer_2_dual = ImageDrawerDual(self.pose_estimater_2, self.pose_analyzer_2, self.bat_analyzer_2)
        self.video_loader_dual = VideoLoaderDual(self.image_drawer_2_dual, self.image_drawer_dual)
        self.is_dual = True

    def initParameters(self):
        # ---- 256083472 camera ---- front
        # s_k1, s_k2, s_p1, s_p2, s_k3 = -0.1507604495950069, 0.0992797366428844, -0.00020610385938892725, -0.00036900923582035034, -0.02925990774478617
        # fx1, fy1, cx1, cy1 = 1040.5555632535795, 1041.692258926554, 950.0364409793515, 530.3984673949892
        # # ---- 256083471 camera ---- side
        # # ---- side camera ---- 6 mm
        # s_k1, s_k2, s_p1, s_p2, s_k3 = -0.1504238482962934, 0.10046068595508036, 0.0009543802109945303, -0.0019857762932287463, -0.03150633224872767
        # fx1, fy1, cx1, cy1 = 1056.463741369799, 1055.9566674653763, 936.2691880195921, 552.3062783018795
        # self.side_intrinsic_matrix = np.array([[fx2, 0, cx2],
        #                                         [0, fy2, cy2],
        #                                         [0,   0,   1]])
        # self.side_distortion_coeffs = np.array([s_k1, s_k2, s_p1, s_p2, s_k3])
        s_k1, s_k2, s_p1, s_p2, s_k3 = -0.1503812176808353, 0.09142073571976807, -0.00042003778401313523, 0.000616947412718863, -0.022943747055243328
        fx1, fy1, cx1, cy1 = 1057.8480015712446, 1058.615115032526, 963.1309477355599, 569.6218934625829
        self.side_intrinsic_matrix = np.array([[fx1, 0, cx1],
                                            [0, fy1, cy1],
                                            [0,   0,   1]])
        self.side_distortion_coeffs = np.array([s_k1, s_k2, s_p1, s_p2, s_k3])
        # ---- side camera ---- 8mm
        # s_k1, s_k2, s_p1, s_p2, s_k3 = -0.17670068446109766, 0.14033116395467476, 0.0002611678101988233, -0.00042558869821785123, -0.059959390420739854
        # fx1, fy1, cx1, cy1 = 1417.0645150420887, 1417.9407893921314, 955.4524361261298, 565.8145551025882

        # self.side_intrinsic_matrix = np.array([[fx1, 0, cx1],
        #                                     [0, fy1, cy1],
        #                                     [0,   0,   1]])
        # self.side_distortion_coeffs = np.array([s_k1, s_k2, s_p1, s_p2, s_k3])
        # ---- front camera ----
        f_k1, f_k2, f_p1, f_p2, f_k3 = -0.3297916162340779, -2.73655197213946, -0.005002958001026669, -0.004773074931105136, 274.8547890952684
        fx2, fy2, cx2, cy2 = 5778.35636357731, 5773.276727595934, 912.7316802319106, 488.2090410693327
        self.front_intrinsic_matrix = np.array([[fx2, 0, cx2],
                                            [0, fy2, cy2],
                                            [0,   0,   1]])
        self.front_distortion_coeffs = np.array([f_k1, f_k2, f_p1, f_p2, f_k3])
        # ---- 256083471 camera ---- side
        # s_k1, s_k2, s_p1, s_p2, s_k3 = -0.1503812176808353, 0.09142073571976807, -0.00042003778401313523, 0.000616947412718863, -0.022943747055243328
        # fx2, fy2, cx2, cy2 = 1057.8480015712446, 1058.615115032526, 963.1309477355599, 569.6218934625829
        # self.side_intrinsic_matrix = np.array([[fx2, 0, cx2],
        #                                         [0, fy2, cy2],
        #                                         [0,   0,   1]])
        # self.side_distortion_coeffs = np.array([s_k1, s_k2, s_p1, s_p2, s_k3])
        # self.R = np.array([[ 0.56247702, 0.01608364, 0.82665647],
        #                     [ 0.15069805, 0.98106936, -0.12162653],
        #                     [-0.81296353, 0.19298764, 0.54940519]])
        # self.t = np.array([[-0.95885392],
        #                     [ 0.11790162],
        #                     [-0.25826026]])
        # self.R = np.array([[ 0.04639104,  0.02477501, -0.99861608],
        #                     [-0.09558267,  0.99521549,  0.02025032],
        #                     [ 0.99433989,  0.09451096,  0.04853714]])
        # self.t = np.array([[ 0.97078711],
        #                     [-0.03059048],
        #                     [ 0.23798446]])
        self.R = np.array([[ 0.12449973, -0.00506822, -0.9922067 ],
                            [-0.04441448,  0.99895615, -0.01067572],
                            [ 0.99122509,  0.04539747,  0.12414466]])
        self.t = np.array([[0.99223704],
                            [0.01491128],
                            [0.12346379]])
        
        # self.R = np.array([[ 0.06097417,  0.03540566, -0.9975112 ],
        #                     [-0.10767217,  0.99377236,  0.02869135],
        #                     [ 0.9923149,   0.10565477,  0.06440665]])
        # self.t = np.array([[ 0.97458141],
        #                     [-0.03994468],
        #                     [ 0.22044385]])
        # self.R =np.array([[ 0.06633578,  0.04084328, -0.99696108],
        #                     [-0.13090099,  0.9908826,   0.03188437],
        #                     [ 0.98917364,  0.12838812,  0.07107739]])
        # self.t = np.array([[ 0.97618844],
        #                     [-0.04295478],
        #                     [ 0.21262881]])
        # self.R = np.array([[ 0.94887348, -0.0012563, -0.31565415],
        #                     [ 0.02227689,  0.99776522,  0.06299442],
        #                     [ 0.31486959, -0.06680553,  0.94678095]])
        # self.t = np.array([[ 0.99957881],
        #                     [-0.01371136],
        #                     [-0.02557737]])
        # self.R = np.array([[ 0.23179385,  0.13894262, -0.96279102],
        #             [-0.13782324,  0.98445328,  0.10888753],
        #             [ 0.96295190,  0.10745552,  0.24733974]])
        # self.t = np.array([[ 0.99450367], [-0.10454567], [-0.00571446]])
        self.P1 = None
        self.P2 = None
        print("Loading Parameters Success!")
        self.computeEssenMatrix()

    def initVispyCanvas(self):
        app.use_app('PyQt5')
            
        self.canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor='white')
        self.view = self.canvas.central_widget.add_view()

        # 建立一顆 TurntableCamera，預設為「側面視角」
        self.view.camera = cameras.TurntableCamera(
            fov=80,
            distance=0.5,      # 你原本的距離
            azimuth=90.0,      # 90 度 ≈ 從側面看；如果方向反了可以改成 -90 或 270
            elevation=10.0,     # 0 度：水平看
            up='-y',           # y 軸向上
            center=(0.0, -0.10, 0.0),
        )
   
        axis = scene.visuals.XYZAxis(parent=self.view.scene)

        # 嵌到 Qt UI
        if isinstance(self.ui.pose3dView, QtWidgets.QWidget):
            layout = QtWidgets.QVBoxLayout(self.ui.pose3dView)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.canvas.native)
        else:
            self.ui.pose3dView.addWidget(self.canvas.native)

        self.skeleton_link = self.image_drawer.joints['haple']['skeleton_links']
        self.skeleton_node = scene.Node(parent=self.view.scene)

    def change2SideView(self):
        self.view.camera = cameras.TurntableCamera(
            fov=60,
            distance=0.5,      # 你原本的距離
            azimuth=90.0,      # 90 度 ≈ 從側面看；如果方向反了可以改成 -90 或 270
            elevation=10.0,     # 0 度：水平看
            up='-y',           # y 軸向上
            center=(0.0, -0.3, 0.0),
        )

    def change2FrontView(self):
        self.view.camera = cameras.TurntableCamera(
            fov=60,
            distance=0.5,      # 你原本的距離
            azimuth=180.0,      # 90 度 ≈ 從前面看
            elevation=10.0,     # 0 度：水平看
            up='-y',           # y 軸向上
            center=(0.0, -0.3, 0.0),
        )

    def computeEssenMatrix(self):
        """Use normalized projection matrices for triangulation."""
        self.P1 = np.hstack([np.eye(3), np.zeros((3,1))])   # normalized domain # side camera
        self.P2 = np.hstack([self.R, self.t])               # normalized domain # front camera
        print("Compute Essential Matrix Successs!")
        print("Projection matrices (normalized) ready.")


    # ======================================================
    # 三角測量法 >> ball, bat, person
    # ======================================================


    def triangulate_points(self, pts_side_px, pts_front_px):
        """
        通用三角測量：
        - pts_side_px:  側面影像上的點，shape (N, 2) 或 list[[x, y], ...]
        - pts_front_px: 正面影像上的點，shape (N, 2)，順序對應 pts_side_px
        回傳:
        - points_3d: shape (N, 3)，以側面相機座標系為主 (P1 = [I|0])
        """
        if self.P1 is None or self.P2 is None:
            print("[triangulate_points] P1/P2 尚未初始化")
            return None

        pts1 = np.asarray(pts_side_px, dtype=np.float32).reshape(-1, 2)
        pts2 = np.asarray(pts_front_px, dtype=np.float32).reshape(-1, 2)

        # 1) undistort → normalized coordinate
        pts1_norm = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2),
            self.side_intrinsic_matrix,
            self.side_distortion_coeffs
        )
        pts2_norm = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2),
            self.front_intrinsic_matrix,
            self.front_distortion_coeffs
        )

        # 2) 變成 (2, N)
        pts1_h = pts1_norm.squeeze().T.astype(np.float64)   # (2, N)
        pts2_h = pts2_norm.squeeze().T.astype(np.float64)

        # 3) triangulate
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pts1_h, pts2_h)

        # 4) homogeneous → 3D
        points_3d = (points_4d / points_4d[3])[:3].T   # (N, 3)
        return points_3d

    def _get_ball_center_2d(self, frame_num: int):
        """
        從 ball_side_df / ball_front_df 中，取出指定 frame 的球中心 2D 座標。
        回傳:
        (cx_side, cy_side), (cx_front, cy_front)
        若任一邊沒有資料，回傳 (None, None)。
        """
        if self.ball_side_df is None or self.ball_front_df is None:
            return None, None
        if self.ball_side_df.empty or self.ball_front_df.empty:
            return None, None
        if "frame_number" not in self.ball_side_df.columns:
            print(f"Warning: 'frame_number' not found in ball_side_df. Available: {self.ball_side_df.columns}")
            return None, None
        if "frame_number" not in self.ball_front_df.columns:
            print(f"Warning: 'frame_number' not found in ball_front_df. Available: {self.ball_front_df.columns}")
            return None, None
        side_rows = self.ball_side_df[self.ball_side_df["frame_number"] == frame_num]
        front_rows = self.ball_front_df[self.ball_front_df["frame_number"] == frame_num]

        if side_rows.empty or front_rows.empty:
            return None, None

        bbox_side = side_rows.iloc[0]["bbox"]
        bbox_front = front_rows.iloc[0]["bbox"]

        if (not isinstance(bbox_side, list)) or len(bbox_side) == 0:
            return None, None
        if (not isinstance(bbox_front, list)) or len(bbox_front) == 0:
            return None, None

        x1s, y1s, x2s, y2s = bbox_side[0]
        x1f, y1f, x2f, y2f = bbox_front[0]

        cx_side  = (float(x1s) + float(x2s)) / 2.0
        cy_side  = (float(y1s) + float(y2s)) / 2.0
        cx_front = (float(x1f) + float(x2f)) / 2.0
        cy_front = (float(y1f) + float(y2f)) / 2.0

        return (cx_side, cy_side), (cx_front, cy_front)

    def _get_bat_head_tail_2d(self, frame_num: int):
        """
        從 batht_side_df / batht_front_df 取出指定 frame 的棒頭、棒尾 2D 座標。
        回傳:
        (tail_side, head_side), (tail_front, head_front)
        各為 [x, y] 的 list 或 np.array。
        若任一邊無資料則回傳 (None, None)。
        """
        if self.batht_side_df is None or self.batht_front_df is None:
            return None, None
        if self.batht_side_df.empty or self.batht_front_df.empty:
            return None, None
        if "frame_num" not in self.batht_side_df.columns:
            print(f"Warning: 'frame_num' not found in batht_side_df. Available: {self.batht_side_df.columns}")
            return None, None
        if "frame_num" not in self.batht_front_df.columns:
            print(f"Warning: 'frame_num' not found in batht_front_df. Available: {self.batht_front_df.columns}")
            return None, None
        side_rows = self.batht_side_df[self.batht_side_df["frame_num"] == frame_num]
        front_rows = self.batht_front_df[self.batht_front_df["frame_num"] == frame_num]
        if side_rows.empty or front_rows.empty:
            return None, None
        row_side = side_rows.iloc[0]
        row_front = front_rows.iloc[0]

        head_side = row_side["head"]   # [x, y]
        tail_side = row_side["tail"]
        head_front = row_front["head"]
        tail_front = row_front["tail"]

        # 基本型態檢查
        if head_side is None or tail_side is None:
            return None, None
        if head_front is None or tail_front is None:
            return None, None
        return (tail_side, head_side), (tail_front, head_front)

    def triangulateBallPoint(self, frame_num: int):
        """用 side/front 的球中心點做三角測量，回傳 3D（已以 origin 中心化）。"""
        if not hasattr(self, "origin"):
            # 一定要先跑 triangularPersonPoints 才有 origin
            return None

        pt_side, pt_front = self._get_ball_center_2d(frame_num)
        if pt_side is None or pt_front is None:
            return None

        # 1 點 → shape (1,1,2)
        pts1_pix = np.array([[pt_side]], dtype=np.float64)    # (1,1,2)
        pts2_pix = np.array([[pt_front]], dtype=np.float64)   # (1,1,2)

        # undistort → normalized
        pts1_norm = cv2.undistortPoints(
            pts1_pix,
            self.side_intrinsic_matrix,
            self.side_distortion_coeffs
        )
        pts2_norm = cv2.undistortPoints(
            pts2_pix,
            self.front_intrinsic_matrix,
            self.front_distortion_coeffs
        )

        pts1 = pts1_norm.squeeze().reshape(2, 1)  # (2,1)
        pts2 = pts2_norm.squeeze().reshape(2, 1)  # (2,1)

        # triangulate
        X4 = cv2.triangulatePoints(self.P1, self.P2, pts1, pts2)  # (4,1)
        X3 = (X4[:3] / X4[3]).reshape(3)                          # (3,)

        # 以人骨架的 origin 為中心
        X3_centered = X3 - self.origin
        return X3_centered

    def triangulate_bat(self, frame_num: int):
        """
        指定 frame_num，三角測量棒尾／棒頭的 3D 座標。
        回傳:
        bat_tail_3d, bat_head_3d （各為 shape (3,) 的 np.array）
        若失敗則回傳 (None, None)
        """
        side_pts, front_pts = self._get_bat_head_tail_2d(frame_num)
        if side_pts is None or front_pts is None:
            return None, None

        tail_side, head_side = side_pts
        tail_front, head_front = front_pts

        pts_side = [tail_side, head_side]
        pts_front = [tail_front, head_front]

        pts_3d = self.triangulate_points(pts_side, pts_front)
        if pts_3d is None or len(pts_3d) < 2:
            return None, None

        bat_tail_3d = pts_3d[0] - self.origin
        bat_head_3d = pts_3d[1] - self.origin
        return bat_tail_3d, bat_head_3d

    def triangularPersonPoints(self, frame_num=None):
        LEFT_TOE_IDX = 20
        RIGHT_TOE_IDX = 21

        # ---------- Step 1: 取得當幀的 2D keypoints ----------

        # 先選出 side / front 的那一筆 row
        if frame_num is None:
            row_side  = self.person_side_df.iloc[0]
            row_front = self.person_front_df.iloc[0]
        else:
            # 用 frame_number == frame_num 過濾，比直接用 iloc[frame_num] 安全
            side_rows  = self.person_side_df[self.person_side_df["frame_number"] == frame_num]
            front_rows = self.person_front_df[self.person_front_df["frame_number"] == frame_num]
            if side_rows.empty or front_rows.empty:
                # 該幀沒有偵測到骨架，可以視情況回傳 None 或上一幀
                print(f"[triangularPersonPoints] frame {frame_num} 沒有 person keypoints")
                return None

            row_side  = side_rows.iloc[0]
            row_front = front_rows.iloc[0]

        # 這裡的 row_side["keypoints"] / row_front["keypoints"] 是 list-of-list
        kpt1_all = np.array(row_side["keypoints"], dtype=np.float32)   # (N, 3) 或 (N, 2+)
        kpt2_all = np.array(row_front["keypoints"], dtype=np.float32)

        # 只取 x, y
        kpt1 = kpt1_all[:, :2]   # shape (N, 2)
        kpt2 = kpt2_all[:, :2]

        # ---------- Step 2: undistort → normalized coordinates ----------
        pts1_norm = cv2.undistortPoints(
            np.expand_dims(kpt1, axis=1),
            self.side_intrinsic_matrix,
            self.side_distortion_coeffs,
        )
        pts2_norm = cv2.undistortPoints(
            np.expand_dims(kpt2, axis=1),
            self.front_intrinsic_matrix,
            self.front_distortion_coeffs,
        )

        # 轉成 (2, N)
        pts1_h = pts1_norm.squeeze().T.astype(np.float64)  # (2, N)
        pts2_h = pts2_norm.squeeze().T.astype(np.float64)

        # ---------- Step 3: 三角測量 ----------
        points_4d = cv2.triangulatePoints(self.P1, self.P2, pts1_h, pts2_h)
        points_3d = (points_4d / points_4d[3])[:3].T  # (N, 3)，側面相機座標系

        # ---------- Step 4: 以雙腳趾中點做平移 ----------
        # left_toe  = points_3d[LEFT_TOE_IDX]
        # right_toe = points_3d[RIGHT_TOE_IDX]
        # self.origin = (left_toe + right_toe) / 2.0
        # points_3d_centered = points_3d - self.origin
        # return points_3d_centered
        if not hasattr(self, 'origin') or self.origin is None:
            left_toe  = points_3d[LEFT_TOE_IDX]
            right_toe = points_3d[RIGHT_TOE_IDX]
            self.origin = (left_toe + right_toe) / 2.0
            # 如果需要讓地面在 Y=0，可以強制 self.origin[1] = 0
        
        points_3d_centered = points_3d - self.origin
        return points_3d_centered

    def _get_velocity_peaks_3d(self, impact_frame=None):
        """
        計算 3D 動力鏈 (頸部、骨盆、膝蓋、手腕) 的速度峰值幀號。
        回傳: dict, 例如 {"neck": 120, "hip": 125, "knee": 130, "wrist": 140}
        """
        if not hasattr(self, 'person_3d_df') or self.person_3d_df is None or self.person_3d_df.empty:
            self.person_3d_df = self.triangularPersonPointsAllFrames()
            
        if self.person_3d_df is None or self.person_3d_df.empty:
            return {}
            
        import numpy as np
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            return {}

        is_righty = (self.ui.batterInput.currentText() == "右打")
        fps = getattr(self.video_loader, 'video_fps', 240.0)
        if fps <= 0: fps = 240.0
        
        # 定義關節索引 (與 2D 保持一致)
        neck_idx = 18 
        if is_righty:
            hip_idx, knee_idx, wrist_idx = 19, 13, 9
        else:
            hip_idx, knee_idx, wrist_idx = 19, 14, 10
            
        all_rows = []
        for _, row in self.person_3d_df.iterrows():
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
            # 3D 歐氏距離 / (3 * dt)
            denom = 3.0 * dt
            v_neck.append(np.linalg.norm(next_kpts[neck_idx] - prev_kpts[neck_idx]) / denom)
            v_hip.append(np.linalg.norm(next_kpts[hip_idx] - prev_kpts[hip_idx]) / denom)
            v_knee.append(np.linalg.norm(next_kpts[knee_idx] - prev_kpts[knee_idx]) / denom)
            v_wrist.append(np.linalg.norm(next_kpts[wrist_idx] - prev_kpts[wrist_idx]) / denom)
            
        # 平滑
        window_size = 11
        if len(frames) < window_size:
            window_size = len(frames) if len(frames) % 2 != 0 else len(frames) - 1
        
        if window_size > 3:
            v_neck = np.maximum(savgol_filter(v_neck, window_size, 3), 0)
            v_hip = np.maximum(savgol_filter(v_hip, window_size, 3), 0)
            v_knee = np.maximum(savgol_filter(v_knee, window_size, 3), 0)
            v_wrist = np.maximum(savgol_filter(v_wrist, window_size, 3), 0)
            
        # 找峰值 (排除頭尾 5 幀)
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

    def showJointVelocityChart3D(self):
        """顯示指定打擊者的 3D 關節速度變化圖表 (公尺/秒)"""
        # 取得 3D 數據
        if not hasattr(self, 'person_3d_df') or self.person_3d_df is None or self.person_3d_df.empty:
            self.person_3d_df = self.triangularPersonPointsAllFrames()
            
        if self.person_3d_df is None or self.person_3d_df.empty:
            print("無 3D 骨架資料，無法繪製 3D 關節速度圖")
            return
            
        import pyqtgraph as pg
        from PyQt5.QtWidgets import QDialog, QVBoxLayout
        import numpy as np
        
        is_righty = (self.ui.batterInput.currentText() == "右打")
        fps = getattr(self.video_loader, 'video_fps', 240.0)
        if fps <= 0: fps = 240.0
        
        # 定義關節索引 (與 2D 保持一致)
        neck_idx = 18 
        if is_righty:
            hip_idx, knee_idx, wrist_idx = 19, 13, 9
            title_prefix = "右打者 (3D)"
            side_name = "左"
        else:
            hip_idx, knee_idx, wrist_idx = 19, 14, 10
            title_prefix = "左打者 (3D)"
            side_name = "右"
            
        all_rows = []
        for _, row in self.person_3d_df.iterrows():
            kpts = row.get("keypoints_3d")
            if kpts is not None and len(kpts) > max(neck_idx, hip_idx, knee_idx, wrist_idx):
                all_rows.append((row.get("frame_number", 0), np.array(kpts)))
                
        if len(all_rows) < 4:
            print("3D 速度計算失敗：有效幀不足")
            return

        frames = []
        v_neck, v_hip, v_knee, v_wrist = [], [], [], []
        
        dt = 1.0 / fps
        for i in range(2, len(all_rows) - 1):
            frame_num = all_rows[i][0]
            prev_kpts = all_rows[i-2][1]
            next_kpts = all_rows[i+1][1]
            
            frames.append(frame_num)
            # 3D 歐氏距離 / (3 * dt) -> m/s (假設單位是米)
            denom = 3.0 * dt
            v_neck.append(np.linalg.norm(next_kpts[neck_idx] - prev_kpts[neck_idx]) / denom)
            v_hip.append(np.linalg.norm(next_kpts[hip_idx] - prev_kpts[hip_idx]) / denom)
            v_knee.append(np.linalg.norm(next_kpts[knee_idx] - prev_kpts[knee_idx]) / denom)
            v_wrist.append(np.linalg.norm(next_kpts[wrist_idx] - prev_kpts[wrist_idx]) / denom)
            
        # 套用 Savitzky-Golay 平滑濾波
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
            print(f"3D 速度平滑濾波失敗: {e}")
            
        # 建立彈出視窗
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{title_prefix} 3D 關節速度圖表 (m/s)")
        dialog.resize(800, 500)
        
        layout = QVBoxLayout(dialog)
        plot_widget = pg.PlotWidget(title=f"{title_prefix} 各關節 3D 瞬間速度")
        layout.addWidget(plot_widget)
        
        plot_widget.addLegend()
        plot_widget.setLabel('bottom', 'Frame Number')
        plot_widget.setLabel('left', 'Velocity (m/s)')
        plot_widget.showGrid(x=True, y=True)
        
        plot_widget.plot(frames, v_neck, pen=pg.mkPen('m', width=2), name="頸部 (Neck)")
        plot_widget.plot(frames, v_hip, pen=pg.mkPen('g', width=2), name="臀部 (Pelvis)")
        plot_widget.plot(frames, v_knee, pen=pg.mkPen('c', width=2), name=f"{side_name}膝 (Knee)")
        plot_widget.plot(frames, v_wrist, pen=pg.mkPen('y', width=2), name=f"{side_name}腕 (Wrist)")
        
        # 標示最大值
        impact_frame = getattr(self, 'phase_result', None) and self.phase_result.F_impact
        peaks = self._get_velocity_peaks_3d(impact_frame)
        for name, color, v_arr in [("neck", 'm', v_neck), ("hip", 'g', v_hip), ("knee", 'c', v_knee), ("wrist", 'y', v_wrist)]:
            f_num = peaks.get(name)
            if f_num is not None and f_num in frames:
                idx = frames.index(f_num)
                val = v_arr[idx]
                scatter = pg.ScatterPlotItem([f_num], [val], size=12, pen=pg.mkPen(None), brush=pg.mkBrush(color))
                plot_widget.addItem(scatter)
                text = pg.TextItem(f"{val:.2f}", color=color, anchor=(0.5, 1.5))
                text.setPos(f_num, val)
                plot_widget.addItem(text)
        
        self._velocity_chart_dialog_3d = dialog
        dialog.show()

    def triangularPersonPointsAllFrames(self) -> pd.DataFrame:
        """
        跑過整支影片，把每一幀的 3D 關鍵點存成一個 DataFrame：

            columns = ["frame_number", "keypoints_3d"]

        之後畫圖、算角度都用這個 df。
        """
        rows = []
        total = int(self.video_loader.total_frames)

        for f in range(total):
            pts_3d = self.triangularPersonPoints(f)

            # 防呆：None 或 shape 不對就跳過
            if pts_3d is None:
                continue
            pts_3d = np.asarray(pts_3d, dtype=np.float32)
            if pts_3d.ndim != 2 or pts_3d.shape[1] != 3:
                continue

            rows.append({
                "frame_number": f,
                "keypoints_3d": pts_3d,   # DataFrame 這一欄會是 object 型別 (存 ndarray)
            })

        df = pd.DataFrame(rows, columns=["frame_number", "keypoints_3d"])
        return df

    # ======================================================
    # 3D 視圖 >> ball, bat, person
    # ======================================================

    def draw_bat_3d(self, tail_3d, head_3d):
        """
        修改後版本：顏色改為與軌跡一致的藍色
        """
        if self.view is None or tail_3d is None or head_3d is None:
            return

        try:
            bat_pts = np.vstack([tail_3d, head_3d]).astype(np.float32)
        except Exception as e:
            print(f"[draw_bat_3d] 堆疊資料失敗: {e}")
            return

        # 定義與軌跡一致的顏色 (深藍色，不透明)
        # 這裡建議使用 [0.0, 0.4, 1.0, 1.0] 或是與您 swing_lines 一致的色調
        bat_color = (0.0, 0.4, 1.0, 1.0) 

        if not hasattr(self, "bat_line") or self.bat_line is None:
            self.bat_line = scene.visuals.Line(
                pos=bat_pts,
                color=bat_color,  # 使用新的藍色
                width=4,          # 稍微加粗讓它比軌跡線明顯
                parent=self.view.scene,
                method='gl'
            )
        else:
            # 更新位置的同時也更新顏色 (確保顏色同步)
            self.bat_line.set_data(pos=bat_pts, color=bat_color)

    def draw_3d_pose(self, points3d):
        """
        在 3D 視圖裡更新骨架：
        - Markers 畫關節點
        - 一條 Line Visual 畫所有骨頭線段
        只在第一次呼叫時建立 Visual，之後每一幀只更新資料。
        """
        if self.view is None:
            print("[draw_3d_pose] view 尚未初始化，請先呼叫 initVispyCanvas()")
            return

        # 保證是 numpy array (N, 3)
        pts = np.asarray(points3d, dtype=np.float32)

        # ---------- 1. Markers：關節點 ----------
        if not hasattr(self, "skeleton_scatter") or self.skeleton_scatter is None:
            # 第一次建立
            self.skeleton_scatter = scene.visuals.Markers(parent=self.view.scene)
            # 之後每幀只要呼叫 set_data 更新點即可
        self.skeleton_scatter.set_data(
            pts,
            face_color='blue',
            size=5
        )

        # ---------- 2. Line：骨頭線段（全部一個 Visual 處理） ----------
        # 把所有骨頭線段展開成 pos: [p0, p1, p2, p3, ...]，兩兩一組
        segment_points = []
        for idx1, idx2 in self.skeleton_link:
            segment_points.append(pts[idx1])
            segment_points.append(pts[idx2])

        if len(segment_points) == 0:
            return

        pos = np.vstack(segment_points).astype(np.float32)  # shape: (2 * num_bones, 3)

        if not hasattr(self, "skeleton_line") or self.skeleton_line is None:
            # 第一次建立一個 Line Visual
            self.skeleton_line = scene.visuals.Line(
                pos=pos,
                color='red',
                width=3,
                connect='segments',   # 每兩個點當一條線段
                method='gl',
                parent=self.view.scene
            )
        else:
            # 之後每一幀只更新資料，不要重建物件
            self.skeleton_line.set_data(
                pos=pos,
                color='red',
                width=3,
                connect='segments'
            )

    def draw_ground_once(
        self,
        points_3d,
        size_scale: float = 1.5,   # 地板邊長 ≈ 座標範圍的 1.5 倍
        l_big_toe_idx: int = 20,
        r_big_toe_idx: int = 21,
        offset_scale: float = 0.1,  # 地板比腳趾再往下 2% 的座標範圍
    ):
        """
        只呼叫一次：在「兩個大腳趾連線的高度」畫一個地面矩形 (x,z 平面)。

        - size_scale: 以座標範圍為基準決定地板大小
        - offset_scale: 以座標範圍為基準決定地板下移量（負值代表往下）
        """

        if getattr(self, "ground_initialized", False):
            return

        pts = np.asarray(points_3d, dtype=np.float32)

        # 整體座標範圍，用來決定合理的 scale
        coord_min = float(pts.min())
        coord_max = float(pts.max())
        coord_range = coord_max - coord_min if coord_max > coord_min else 1.0

        print(f"[GROUND] coord_range = {coord_range:.4f}")

        l_toe = pts[l_big_toe_idx]
        r_toe = pts[r_big_toe_idx]

        # 這裡假設 y 軸是垂直方向（從你的 toe 資料來看是合理的）
        toe_y = 0.5 * (l_toe[1] + r_toe[1])

        # 地板高度 = 腳趾平均高度 + 一點點向下偏移
        ground_y = float(toe_y + offset_scale * coord_range)

        print(f"[GROUND] toe_y = {toe_y:.4f}, ground_y = {ground_y:.4f}")

        # 地板邊長大小，也用座標範圍決定
        size = coord_range * size_scale
        print(f"[GROUND] size = {size:.4f}")

        ground_pts = np.array(
            [
                [-size, ground_y, -size],
                [-size, ground_y,  size],
                [ size, ground_y,  size],
                [ size, ground_y, -size],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        # 若之前有 ground_node，先從 scene 拿掉
        if hasattr(self, "ground_node") and self.ground_node is not None:
            self.ground_node.parent = None

        self.ground_node = scene.Node(parent=self.view.scene)
        self.ground_mesh = scene.visuals.Mesh(
            vertices=ground_pts,
            faces=faces,
            color=(0.6, 0.6, 0.6, 0.7),
            parent=self.ground_node,
        )

        self.ground_initialized = True
        self.last_ground = ground_pts

    def draw_hip_shoulder_angle(self, points_3d_centered):
        """
        - 在 3D 視圖裡畫「髖部線段」與「肩部線段」
        - 計算二者夾角並顯示在畫面右上角
        - Visual 只建立一次，之後每幀只更新資料
        """
        if self.view is None or self.canvas is None:
            print("[draw_hip_shoulder_angle] view/canvas 尚未初始化")
            return

        # -------------------------
        # 1) 取關節點座標（這裡用你原本的 index）
        # -------------------------
        l_hip = points_3d_centered[11]
        r_hip = points_3d_centered[12]
        l_shoulder = points_3d_centered[5]
        r_shoulder = points_3d_centered[6]

        hip_pts = np.array([l_hip, r_hip], dtype=np.float32)
        shoulder_pts = np.array([l_shoulder, r_shoulder], dtype=np.float32)

        # -------------------------
        # 2) 畫線：只建一次，之後 set_data
        # -------------------------
        # 髖部線（黃色）
        if not hasattr(self, "hip_line") or self.hip_line is None:
            self.hip_line = scene.visuals.Line(
                pos=hip_pts,
                color='yellow',
                width=5,
                parent=self.view.scene,
                method='gl'
            )
        else:
            self.hip_line.set_data(pos=hip_pts, color='yellow', width=5)

        # 肩部線（藍色）
        if not hasattr(self, "shoulder_line") or self.shoulder_line is None:
            self.shoulder_line = scene.visuals.Line(
                pos=shoulder_pts,
                color='blue',
                width=5,
                parent=self.view.scene,
                method='gl'
            )
        else:
            self.shoulder_line.set_data(pos=shoulder_pts, color='blue', width=5)

        # -------------------------
        # 3) 計算夾角
        # -------------------------
        v1 = r_hip - l_hip
        v2 = r_shoulder - l_shoulder

        # 投影到 XZ 平面（MLB 定義的旋轉平面）
        v1_proj = np.array([v1[0], v1[2]])
        v2_proj = np.array([v2[0], v2[2]])

        # atan2 得到旋轉角度（有方向性）
        theta_hip = np.arctan2(v1_proj[1], v1_proj[0])
        theta_shd = np.arctan2(v2_proj[1], v2_proj[0])

        # HSS = 肩 - 髖（MLB 定義）
        angle_rad = theta_shd - theta_hip

        # 將角度包到 -180 ～ +180
        angle_deg = np.degrees(angle_rad)
        angle_deg = (angle_deg + 180) % 360 - 180

        # text = f"Hip–Shoulder Separation: {angle_deg:.1f}°"
        # self.ui.hipShoulderEditLabel.setText(f"{angle_deg:.1f}")

    def draw_swing_trajectory(self, current_f: int):
        """
        在 3D 視圖中畫出揮棒軌跡，且只顯示從 Landing 到 current_f 的部分。
        """
        if not hasattr(self, "phase_result") or self.phase_result is None:
            return
            
        start_f = self.phase_result.F_liftoff+3
        if start_f is None:
            return

        # 1. 如果當前影格還沒到落地點，隱藏軌跡並返回
        if current_f < start_f:
            if hasattr(self, "swing_mesh"): self.swing_mesh.visible = False
            if hasattr(self, "swing_lines"): self.swing_lines.visible = False
            return

        # 2. 提取 Landing 到當前影格之間的 3D 座標
        raw_tails = []
        raw_heads = []
        # 設定顯示終點，避免擊球後軌跡無限延伸
        end_limit = (self.phase_result.F_impact+5) if self.phase_result.F_impact else 999
        display_end = min(current_f, end_limit)

        for f in range(start_f, display_end + 1):
            t_3d, h_3d = self.triangulate_bat(f)
            if t_3d is not None and h_3d is not None:
                raw_tails.append(t_3d)
                raw_heads.append(h_3d)

        if len(raw_tails) < 2:
            return

        # 3. 數據平滑 (Smoothing)
        def smooth_trajectory(data, window=3):
            data = np.array(data)
            if len(data) <= window: return data
            smoothed = np.zeros_like(data)
            for i in range(3):
                smoothed[:, i] = np.convolve(data[:, i], np.ones(window)/window, mode='same')
            smoothed[:window//2] = data[:window//2]
            smoothed[-window//2:] = data[-window//2:]
            return smoothed

        tails = smooth_trajectory(raw_tails)
        heads = smooth_trajectory(raw_heads)

        # 4. 準備繪圖數據
        vertices = []
        faces = []
        v_colors = []
        num_pts = len(tails)

        # 定義統一的藍色調 (與 Curve 顏色一致)
        # 這裡使用 [0.0, 0.4, 1.0] 作為基調
        for i in range(num_pts):
            vertices.append(tails[i])
            vertices.append(heads[i])
            
            # 漸變透明度：越舊的影格越淡
            alpha = (i / num_pts) * 0.4  # 最高 0.4 透明度
            v_colors.append([0.3, 0.6, 1.0, alpha]) # 淺藍色軌跡面
            v_colors.append([0.3, 0.6, 1.0, alpha])

            if i < num_pts - 1:
                v1, v2, v3, v4 = 2*i, 2*i+1, 2*(i+1), 2*(i+1)+1
                faces.append([v1, v2, v3])
                faces.append([v2, v3, v4])

        # 5. 更新 Mesh 節點
        if not hasattr(self, "swing_mesh"):
            self.swing_mesh = scene.visuals.Mesh(parent=self.view.scene, shading=None)
        
        self.swing_mesh.visible = True
        self.swing_mesh.set_data(
            vertices=np.array(vertices, dtype=np.float32),
            faces=np.array(faces, dtype=np.int32),
            vertex_colors=np.array(v_colors, dtype=np.float32)
        )

        # 6. 更新 Line 節點 (百葉窗線條)
        line_pts = []
        line_colors = []
        for i in range(num_pts):
            line_pts.append(tails[i])
            line_pts.append(heads[i])
            # 線條顏色稍深
            l_alpha = (i / num_pts) * 0.5
            line_colors.append([0.0, 0.4, 1.0, l_alpha])
            line_colors.append([0.0, 0.4, 1.0, l_alpha])

        if not hasattr(self, "swing_lines"):
            self.swing_lines = scene.visuals.Line(connect='segments', width=1.0, parent=self.view.scene)
        
        self.swing_lines.visible = True
        self.swing_lines.set_data(
            pos=np.array(line_pts, dtype=np.float32),
            color=np.array(line_colors, dtype=np.float32)
        )

    # ======================================================
    # 🔹 Tab 初始化與 UI 連動
    # ======================================================
    def bindUI(self):
        self.ui.loadProcessFolderBtn.setText("📁 載入資料夾分析")
        self.ui.loadProcessFolderBtn.clicked.connect(self.loadFolder)

        # 安裝 EventFilter 給 keyphaseLabel1~3 以偵測 Resize
        if hasattr(self.ui, 'keyphaseLabel1'):
            self.ui.keyphaseLabel1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.ui.keyphaseLabel1.setMinimumSize(50, 50)
            self.ui.keyphaseLabel1.installEventFilter(self)
        if hasattr(self.ui, 'keyphaseLabel2'):
            self.ui.keyphaseLabel2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.ui.keyphaseLabel2.setMinimumSize(50, 50)
            self.ui.keyphaseLabel2.installEventFilter(self)
        if hasattr(self.ui, 'keyphaseLabel3'):
            self.ui.keyphaseLabel3.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            self.ui.keyphaseLabel3.setMinimumSize(50, 50)
            self.ui.keyphaseLabel3.installEventFilter(self)

        self.ui.playBtn.clicked.connect(self.playBtnClicked)
        self.ui.backKeyBtn.clicked.connect(lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() - 1))
        self.ui.forwardKeyBtn.clicked.connect(lambda: self.ui.frameSlider.setValue(self.ui.frameSlider.value() + 1))
        self.ui.impactBtn.clicked.connect(lambda: self.ui.frameSlider.setValue(self.phase_result.F_impact))
        self.ui.landingBtn.clicked.connect(lambda: self.ui.frameSlider.setValue(self.phase_result.F_landing))
        self.ui.liftoffBtn.clicked.connect(lambda: self.ui.frameSlider.setValue(self.phase_result.F_liftoff))
        self.ui.loadVideoBtn.clicked.connect(lambda: self.loadVideo4Process())
        self.ui.processVideosBtn.clicked.connect(lambda: self.toggleDetect())
        self.ui.sideCameraBtn.setText("📹側向")
        self.ui.frontCameraBtn.setText("📸正向")
        self.ui.sideCameraBtn.clicked.connect(self.change2SideView)
        self.ui.frontCameraBtn.clicked.connect(self.change2FrontView)
        self.ui.frameSlider.valueChanged.connect(self.chooseUpdateFrame)

        self.ui.showSkeletonCheckBox.stateChanged.connect(self.toggleShowSkeleton)
        self.ui.selectCheckBox.stateChanged.connect(self.toggleSelect)

        if hasattr(self.ui, 'listView'):
            self.ui.listView.clicked.connect(self.onRecentVideoSelected)
            self.populateRecentVideosList()
            
        if hasattr(self.ui, 'modeTbar'):
            self.ui.modeTbar.clicked.connect(lambda: self.setMode("Tbar"))
        if hasattr(self.ui, 'modeSide'):
            self.ui.modeSide.clicked.connect(lambda: self.setMode("Side"))
        if hasattr(self.ui, 'modeFront'):
            self.ui.modeFront.clicked.connect(lambda: self.setMode("Front"))

        if hasattr(self.ui, 'hitterNameInput'):
            self.updateHitterList()
            self.ui.hitterNameInput.currentIndexChanged.connect(self.populateRecentVideosList)
            
        if hasattr(self.ui, 'hitterNameAddInputBtn'):
            self.ui.hitterNameAddInputBtn.clicked.connect(self.addNewHitter)

    def setMode(self, mode):
        self.current_mode = mode
        if hasattr(self.ui, 'modeDisplay'):
            self.ui.modeDisplay.setText(self.mode_dict[mode])
        self.populateRecentVideosList()

    def updateHitterList(self):
        """初始化與更新選手選單"""
        if not hasattr(self.ui, 'hitterNameInput'): return
        db_path = str(DB_DIR)
        if not os.path.exists(db_path): return
        current_name = self.ui.hitterNameInput.currentText()
        self.ui.hitterNameInput.blockSignals(True)
        self.ui.hitterNameInput.clear()
        hitters = [d for d in os.listdir(db_path) if os.path.isdir(os.path.join(db_path, d)) and d not in ['icon', 'pretrain']]
        hitters.sort()
        for hitter in hitters:
            self.ui.hitterNameInput.addItem(hitter)
        if current_name and current_name in hitters:
            index = self.ui.hitterNameInput.findText(current_name)
            self.ui.hitterNameInput.setCurrentIndex(index)
        self.ui.hitterNameInput.blockSignals(False)

    def addNewHitter(self):
        """新增選手功能"""
        new_name = self.ui.hitterNameAddInput.text().strip()
        if not new_name:
            QMessageBox.warning(self, "錯誤", "請輸入選手姓名")
            return
        db_base_path = str(DB_DIR)
        hitter_path = os.path.join(db_base_path, new_name)
        try:
            if not os.path.exists(hitter_path):
                os.makedirs(hitter_path)
            else:
                QMessageBox.information(self, "提示", f"選手 {new_name} 已存在於資料庫中")
            existing_items = [self.ui.hitterNameInput.itemText(i) for i in range(self.ui.hitterNameInput.count())]
            if new_name not in existing_items:
                self.ui.hitterNameInput.addItem(new_name)
            index = self.ui.hitterNameInput.findText(new_name)
            self.ui.hitterNameInput.setCurrentIndex(index)
            self.ui.hitterNameAddInput.clear()
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法建立選手資料: {str(e)}")

    def populateRecentVideosList(self):
        """抓取目前選中選手的最新 5 個含有資料夾"""
        if not hasattr(self.ui, 'listView'): return
        hitter_name = self.ui.hitterNameInput.currentText()
        if not hitter_name:
            if hasattr(self, 'recent_video_model'):
                self.recent_video_model.clear()
            return
        db_path = str(DB_DIR)
        hitter_path = os.path.join(db_path, hitter_name)
        if not os.path.exists(hitter_path):
            self.recent_video_model = QStandardItemModel()
            self.ui.listView.setModel(self.recent_video_model)
            return
        selected_folders = []
        try:
            date_folders = [d for d in os.listdir(hitter_path) if os.path.isdir(os.path.join(hitter_path, d))]
            date_folders.sort(reverse=True)
            for d_folder in date_folders:
                d_path = os.path.join(hitter_path, d_folder)
                current_mode_zh = self.mode_dict.get(getattr(self, 'current_mode', 'Tbar'), 'T座')
                mode_path = os.path.join(d_path, current_mode_zh)
                if not os.path.exists(mode_path): continue
                sessions = [s for s in os.listdir(mode_path) if os.path.isdir(os.path.join(mode_path, s))]
                sessions.sort(reverse=True)
                for s_folder in sessions:
                    s_path = os.path.join(mode_path, s_folder)
                    # 4. 檢查資料夾內是否有影片檔 (.mp4) 或者 json (因為3D是直接load offline)
                    files = os.listdir(s_path)
                    if any(f.endswith('.mp4') for f in files) or any('person_side.json' in f for f in files):
                        selected_folders.append((d_folder, s_folder, s_path))
        except Exception as e:
            print(f"Populate recent videos error: {e}")
            
        self.recent_video_model = QStandardItemModel()
        for d_f, s_f, full_path in selected_folders:
            mode_display_text = self.mode_dict.get(getattr(self, 'current_mode', 'Tbar'), 'T座')
            display_text = f"{d_f} - {mode_display_text} - {s_f}"
            item = QStandardItem(display_text)
            item.setData(full_path, Qt.UserRole)
            self.recent_video_model.appendRow(item)
        self.ui.listView.setModel(self.recent_video_model)

    def onRecentVideoSelected(self, index):
        folder_path = self.recent_video_model.itemFromIndex(index).data(Qt.UserRole)
        
        # 1. 如果資料夾裡有資料夾，代表處理完，直接呼叫 self.loadFolder 就可以了
        # 尋找內部的 CF_ 子資料夾 (因為 3D 處理完的完整數據會存在裡面)
        has_subfolder = False
        target_load_path = folder_path
        try:
            for d in os.listdir(folder_path):
                if d.startswith("CF_") and os.path.isdir(os.path.join(folder_path, d)):
                    has_subfolder = True
                    target_load_path = os.path.join(folder_path, d)
                    break
        except Exception:
            pass

        if has_subfolder:
            # 傳遞內部那一包的完整路徑給 loadFolder
            self.loadFolder(target_load_path)
            return

        # 2. 若沒有就是 loadVideo4Process -> toggleDetect
        import glob
        mp4_files = sorted(glob.glob(os.path.join(folder_path, '*.mp4')))
        
        if len(mp4_files) >= 2:
            self.video_loader_dual.video_path = mp4_files[0]
            self.video_loader_dual.video_path2 = mp4_files[1]
            self.loadVideo4Process()
            self._auto_process_pending = True
            # checkVideoLoad4Process 會等待影片載入完畢後，透過 _auto_process_pending 觸發 toggleDetect
            self.checkVideoLoad4Process()
        elif mp4_files:
            # 如果只有一個影片，就單純載入
            self.loadFolder(folder_path)

    # ======================================================
    # Process → 2D 骨架
    # ======================================================
    def toggleDetect(self): 
        # Check if video is loaded
        if self.video_loader_dual.video_frames1 is None:
            QMessageBox.warning(self, "無法處理", "請先讀取影片!")
            return

        # print("skeleton state: "+ str(state))
        # 重置所有分析器狀態以允許重新處理
        self.ui.showSkeletonCheckBox.setChecked(True)
        # self.ui.selectCheckBox.setChecked(True)
        frame = self.video_loader_dual.getVideoImage(0)
        frame_2 = self.video_loader_dual.getVideoImage2(0)
        _, _, _, _, _ = self.pose_estimater.detectKpt(frame, 0, is_video=True)
        _, _, _, _, _ = self.pose_estimater_2.detectKpt(frame_2, 0, is_video=True)
        self.ui.selectCheckBox.setChecked(True)
        self.ui.playBtn.click()
    
    def toggleShowSkeleton(self, state:int):
        """Toggle skeleton detection and FPS control."""
        print("skeleton state: "+ str(state))
        # if state == 2:
        #     self.ui.showSkeletonCheckBox.setCheckState(0)
        #     QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
        #     return
        is_checked = state == 2
        if not is_checked:
            self.ui.selectCheckBox.setCheckState(0)
        self.pose_estimater.setDetect(is_checked)
        self.pose_estimater_2.setDetect(is_checked)
        self.image_drawer_dual.setShowSkeleton(is_checked)
        self.image_drawer_2_dual.setShowSkeleton(is_checked)
        self.image_drawer.setShowSkeleton(is_checked)
        self.image_drawer_2.setShowSkeleton(is_checked)
    
    def toggleSelect(self, state:int):
        """Select a person based on checkbox state."""
        if state == 2:
            if not self.ui.showSkeletonCheckBox.isChecked():
                self.ui.selectCheckBox.setCheckState(0)
                QMessageBox.warning(self, "無法選擇人", "請選擇顯示人體骨架!")
                return
            self.person_selector = PersonSelector()
            self.person_selector_2 = PersonSelector()
            frame_num = self.ui.frameSlider.value()
            search_person_df = self.pose_estimater.getPersonDf(frame_num=frame_num) if frame_num is not None else self.pose_estimater.pre_person_df
            search_person_df_2 = self.pose_estimater_2.getPersonDf(frame_num=frame_num) if frame_num is not None else self.pose_estimater_2.pre_person_df
            self.person_selector.select(search_person_df = search_person_df)
            self.person_selector_2.select(search_person_df = search_person_df_2)
            self.pose_estimater.setPersonId(self.person_selector.selected_id)
            self.pose_estimater_2.setPersonId(self.person_selector_2.selected_id)
        else:
            self.pose_estimater.setPersonId(None)
            self.pose_estimater_2.setPersonId(None)
            self.person_selector = None
            self.person_selector_2 = None

    def loadVideo4Process(self):
        self.reset4Process()
        self.is_dual = True
        self.ui.showSkeletonCheckBox.setChecked(False)
        self.ui.selectCheckBox.setChecked(False)
        if self.is_play:
            # self.ui.playBtn.click()
            self.playBtnClicked4Process()
        self.video_loader_dual.loadVideo()
        self.checkVideoLoad4Process()

    def analyzeFrame4Process(self):
        """Analyze and process each video frame."""
        frame_num = self.ui.frameSlider.value()
        total_duration = 1.5
        current_seconds = (float(frame_num) / float(self.video_loader_dual.total_frames - 1)) * total_duration
        # 顯示格式： 0.45s / 1.50s
        self.ui.frameNumLabel.setText(f'{current_seconds:.2f}/{total_duration:.1f}s')
        # self.ui.frameNumLabel.setText(f'{frame_num}/{self.video_loader_dual.total_frames - 1}')
        frame = self.video_loader_dual.getVideoImage(frame_num)
        frame_2 = self.video_loader_dual.getVideoImage2(frame_num)
        fps = 0
        if self.ui.showSkeletonCheckBox.isChecked():
            _, _, fps, _, _ = self.pose_estimater.detectKpt(frame, frame_num, is_video=True)
            _, _, fps, _, _ = self.pose_estimater_2.detectKpt(frame_2, frame_num, is_video=True)
            if self.pose_estimater.person_id is not None and self.pose_estimater_2.person_id is not None:
                self.pose_analyzer.addAnalyzeInfo(frame_num)
                self.pose_analyzer_2.addAnalyzeInfo(frame_num)
                self.bat_analyzer.appendHeadTail(frame_num)
                self.bat_analyzer_2.appendHeadTail(frame_num)

        if frame_num == self.video_loader_dual.total_frames - 1:
            print("DEBUG: Emission condition met, emitting processFinished")
            self.video_loader_dual.saveDualVideo(False)
            self.video_loader_dual.saveDualVideo(True)
            print(f"DEBUG: Emitting signal with folder: {self.video_loader_dual.output_folder}")
            self.processFinished.emit(self.video_loader_dual.output_folder)
            self.is_dual = False

        self.updateFrame4Process(frame_num=frame_num)

    def checkVideoLoad4Process(self):
        """檢查影片是否讀取完成，並確保幀數資訊已更新。"""
        # 1. 檢查是否有路徑
        if not self.video_loader_dual.video_path and not self.video_loader_dual.video_path2:
            return
        
        # 2. 若還在讀取中，或者 total_frames 尚未被賦值（None），則繼續等待
        if self.video_loader_dual.is_loading or self.video_loader_dual.total_frames is None:
            QTimer.singleShot(100, self.checkVideoLoad4Process)
            return

        # 3. 只有當 total_frames 有值後才進行後續 UI 初始化
        self.updateVideoInfo4Process()
        
    def changeBatter(self):
        """Change the pitcher based on input value. 9: "左腕", 10: "右腕","""
        kpt_id = 10 if self.ui.batterInput.currentIndex() == 0 else 9
        self.pose_estimater.setKptId(kpt_id)
        # self.pose_estimater.setPitchHandId(kpt_id)

    def updateVideoInfo4Process(self):
        """更新與影片相關的資訊顯示在 UI 上。"""
        self.reset()
        self.changeBatter()
        self.initFrameSlider(self.video_loader_dual.total_frames)
        self.updateFrame4Process(frame_num = 0)
        self.model.setImageSize(self.video_loader_dual.video_size)
        # self.ui.videoNameLabel.setText(self.video_loader_dual.video_name)
        video_size = self.video_loader_dual.video_size
        self.ui.ResolutionLabel.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")
        
        if getattr(self, '_auto_process_pending', False):
            self._auto_process_pending = False
            self.toggleDetect()
       
    def playBtnClicked4Process(self):
        """Handle play/pause button click."""
        if self.video_loader_dual.video_name == "" or self.video_loader_dual.video_name2 == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader_dual.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.playBtn.setText("||" if self.is_play else "▶︎")
        if self.is_play:
            self.playFrame4Process(self.ui.frameSlider.value())

    def playFrame4Process(self, start_num: int = 0):
        """Play video frames starting from start_num."""
        for i in range(start_num, self.video_loader_dual.total_frames):
            if not self.is_play:
                break
            self.ui.frameSlider.setValue(i)
            if i == self.video_loader_dual.total_frames - 1 and self.is_play:
                self.playBtnClicked4Process()
            
            # 還原原本系統 1 倍速為最低延遲 1ms 的體驗
            delay = 1
            try:
                if hasattr(self.ui, 'slowMotionInput'):
                    val_str = self.ui.slowMotionInput.currentText().replace('x', '').replace('X', '').strip()
                    current_rate = float(val_str)
                    if current_rate >= 1.0:
                        delay = 1
                    elif current_rate > 0:
                        # 當速度小於 1，成比例加上額外的延遲 (0.5倍加約33ms，0.25倍加約99ms)
                        delay = int((1.0 / current_rate - 1.0) * 33) + 1
            except:
                pass
                
            cv2.waitKey(max(1, delay))

    def updateFrame4Process(self, frame: np.ndarray = None, frame_2: np.ndarray = None, frame_num:int = None):
        """Update the displayed frame with additional analysis."""
        # 若外部沒傳，從 slider 讀（保險用）
        if frame_num is None:
            frame_num = self.ui.frameSlider.value()

        # 更新當前的frame和frame_num
        if frame_num is not None:
            frame= self.video_loader_dual.getVideoImage(frame_num)
            frame_2 = self.video_loader_dual.getVideoImage2(frame_num)
            if frame is None or frame_2 is None:
                return
        drawed_img = self.image_drawer_dual.drawInfo(frame, frame_num, self.pose_estimater.kpt_buffer)
        drawed_img_2 = self.image_drawer_2_dual.drawInfo(frame_2, frame_num, self.pose_estimater_2.kpt_buffer)
        self.showImage(drawed_img, self.view_scene, self.ui.FrameView)
        self.showImage(drawed_img_2, self.view_scene2, self.ui.FrameView_2)
    
    def chooseUpdateFrame(self):
        if self.is_dual:
            self.analyzeFrame4Process()
        else:
            self.updateFrame()

        if hasattr(self, 'hip_shoulder_plot') and self.hip_shoulder_plot is not None:
            self.hip_shoulder_plot.update(self.ui.frameSlider.value())

        if hasattr(self, 'bat_speed_plot') and self.bat_speed_plot is not None:
            self.bat_speed_plot.update(self.ui.frameSlider.value())

    def reset4Process(self):
        self.pose_estimater.reset()
        self.pose_estimater_2.reset()
        self.pose_analyzer.reset()
        self.pose_analyzer_2.reset()
        self.bat_analyzer.reset()
        self.bat_analyzer_2.reset()
        self.image_drawer.reset()
        self.image_drawer_2.reset()
        self.resetFrameSlider()
        self.view_scene.clear()
        self.view_scene2.clear()

    # ======================================================
    # 📁 LoadFolder → 自動分析 JSON + video
    # ======================================================
    def _loadFolderCore(self, folder: str):
        """核心載入流程：讀取 JSON 資料夾並載入影片。"""
        # 統一將路徑中的反斜線轉為正斜線，確保 glob.glob 的行為與 QFileDialog 完全一致
        folder = folder.replace('\\', '/')
        self.folder = folder
        self.is_dual = False
        print(f"📁 Loading Folder: {folder}")

        # 定義輔助函式，支援根目錄或 CF_ 子目錄 (讀取內部那包)
        def get_json_df(fname, sub_prefix="CF_"):
            p = os.path.join(folder, fname)
            if not os.path.exists(p):
                subs = glob.glob(os.path.join(folder, f"{sub_prefix}*"))
                if subs:
                    # 如果是 front 的，子目錄裡的檔名通常會是 side (因為是 2D 分析產出)
                    p = os.path.join(subs[0], fname.replace("front", "side"))
            
            if os.path.exists(p):
                try: return pd.read_json(p, orient='records')
                except: pass
            return pd.DataFrame()

        # 1. 讀取所有資料
        self.person_side_df  = get_json_df("person_side.json")
        self.person_front_df = get_json_df("person_front.json")
        self.batht_side_df   = get_json_df("batht_side.json")
        self.batht_front_df  = get_json_df("batht_front.json")
        self.ball_side_df    = get_json_df("ball_side.json")
        self.ball_front_df   = get_json_df("ball_front.json")

        # 2. 更新 Estimater/Analyzer 數據
        self.pose_estimater.person_df = self.person_front_df
        self.pose_estimater_2.person_df = self.person_side_df
        self.bat_analyzer.batHeadTail_df = self.batht_front_df
        self.bat_analyzer_2.batHeadTail_df = self.batht_side_df
        self.pose_estimater.ball_df = self.ball_front_df
        self.pose_estimater_2.ball_df = self.ball_side_df

        # 3. 影片路徑解析 (優先找 skel_，再找 CS/CF 前綴)
        def find_mp4(prefix, fallback_suffix):
            skel = os.path.join(folder, f"skel_{fallback_suffix}.mp4")
            if os.path.exists(skel): return skel
            vids = glob.glob(os.path.join(folder, f"{prefix}*.mp4"))
            return vids[0] if vids else None

        self.video_side_path = find_mp4("CF_", "front")
        self.video_front_path = find_mp4("CS_", "side")
        
        # Raw 影片路徑
        self.video_side_raw = (glob.glob(os.path.join(folder, "CF*.mp4")) + [os.path.join(folder, "raw_front.mp4")])[0]
        self.video_front_raw = (glob.glob(os.path.join(folder, "CS*.mp4")) + [os.path.join(folder, "raw_side.mp4")])[0]

        # 4. 執行載入
        self.loadVideo()

    def loadFolder(self, folder=None):
        if not folder:
            default_dir = str(RECORD_DIR)

            folder = QFileDialog.getExistingDirectory(
                self,
                "選擇錄製完成的資料夾",
                default_dir
            )
        if not folder:
            return

        self._loadFolderCore(folder)

    def load_from_folder(self, folder: str):
        """給主視窗 / 其他 tab 直接指定資料夾路徑載入用。"""
        if not folder or not os.path.isdir(folder):
            print(f"[Analysis3D] 無效的資料夾路徑：{folder}")
            return
        self._loadFolderCore(folder)

    def initGraph(self):
        LEFT_TOE_IDX = 20
        RIGHT_TOE_IDX = 21
        LEFT_HEEL_IDX = 24

        LEFT_KNEE_IDX = 13
        LEFT_HIP_IDX = 11
        LEFT_ANKLE_IDX = 15
        
        # 2) 建 kneeAnglePlot
        self.person_3d_df = self.triangularPersonPointsAllFrames()
        # [NEW] Save 3D Person DataFrame
        if self.folder and not self.person_3d_df.empty:
            save_path = os.path.join(self.folder, "person_3d.json")
            # Convert numpy arrays to lists for JSON serialization if necessary, 
            # but pandas read_json/to_json might handle list-of-lists. 
            # However, for safety with direct to_json, ensure it's standard lists.
            # safe_df = self.person_3d_df.copy()
            # safe_df["keypoints_3d"] = safe_df["keypoints_3d"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            # safe_df.to_json(save_path, orient='records')
            
            # Simplified approach if we trust the structure or just want to dump it:
            try:
                # Making a copy to avoid modifying the original dataframe used in memory
                export_df = self.person_3d_df.copy()
                export_df['keypoints_3d'] = export_df['keypoints_3d'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
                export_df.to_json(save_path, orient='records')
                print(f"Saved 3D person data to {save_path}")
            except Exception as e:
                print(f"Failed to save person_3d.json: {e}")
        self.detectPhases()
        
        # 建立肩髖夾角變化圖表並顯示
        target_df = getattr(self, 'person_side_df', self.person_3d_df)
        total_frames = int(self.video_loader.total_frames) if hasattr(self.video_loader, 'total_frames') and self.video_loader.total_frames else 0
        if target_df is not None and not target_df.empty:
            is_righty = (self.ui.batterInput.currentText() == "右打")
            self.hip_shoulder_plot = HipShoulderAngleGraph(target_df, total_frames, is_righty=is_righty, phase_result=self.phase_result)
            
            # 加上階段底色 (Phase Region)
            if hasattr(self, 'phase_result') and self.phase_result is not None:
                self.hip_shoulder_plot.set_phase_result(self.phase_result)
            
            # 使用獨立視窗顯示圖表
            self.hip_shoulder_plot.graph.resize(800, 400)
            self.hip_shoulder_plot.graph.show()

        # 棒頭速度與加速度圖表
        bat_frames = []
        bat_head_pts = []
        for f in range(total_frames):
            t_3d, h_3d = self.triangulate_bat(f)
            if h_3d is not None:
                bat_frames.append(f)
                bat_head_pts.append(h_3d)

        fps = getattr(self.video_loader, 'video_fps', 240.0)
        self.bat_speed_plot = BatKinematicsGraph(bat_frames, bat_head_pts, total_frames, fps=fps)
        
        # [NEW] 用 3D 真實速度曲線的最高點覆寫 Impact，保證與圖表完美對齊
        if getattr(self.bat_speed_plot, 'max_speed_frame', None) is not None:
            if hasattr(self, 'phase_result') and self.phase_result is not None:
                print(f"3D Impact override: {self.phase_result.F_impact} -> {self.bat_speed_plot.max_speed_frame}")
                self.phase_result.F_impact = self.bat_speed_plot.max_speed_frame
                self.setPhaseResult(self.phase_result) # 重刷 UI 與計算
                
        if len(bat_head_pts) > 0:
            if hasattr(self, 'phase_result') and self.phase_result is not None:
                self.bat_speed_plot.set_phase_result(self.phase_result)
            self.bat_speed_plot.graph.resize(800, 400)
            self.bat_speed_plot.graph.show()
            
        # [NEW] 顯示 3D 關節速度圖表
        self.showJointVelocityChart3D()

    def detectPhases(self):
        # 這裡直接做階段偵測
        # 如果側面的球棒數據是空的，嘗試使用正面的球棒數據作為替代，確保能算出 Impact
        usable_batht = self.batht_side_df
        if usable_batht is None or usable_batht.empty:
            usable_batht = getattr(self, 'batht_front_df', None)

        batter_idx = self.ui.batterInput.currentIndex()
        detector = PhaseDetector(
            person_df=self.person_side_df,
            batter_index=batter_idx,
            ball_df=self.ball_side_df,
            batht_df=usable_batht,
            th_lift=5.0,
            th_land=5.0,
            land_hold=3,
        )
        self.phase_result = detector.detect_phases()
        
        # --- [NEW] Fallback Mechanism ---
        total_frames = len(self.person_side_df) if self.person_side_df is not None else 0
        if self.phase_result is None:
            self.phase_result = PhaseResult()
            
        print(f"[DEBUG] Original PhaseResult: F_liftoff={self.phase_result.F_liftoff}, F_landing={self.phase_result.F_landing}, F_impact={self.phase_result.F_impact}")

        if self.phase_result.F_liftoff is None and self.phase_result.F_landing is None and self.phase_result.F_impact is None:
            # self.phase_result.F_liftoff = min(15, max(0, total_frames - 3))
            # self.phase_result.F_landing = min(25, max(0, total_frames - 2))
            # self.phase_result.F_impact = min(30, max(0, total_frames - 1))
            print("[Analysis3D] All keyphases missing. Skipping fallback as per user request.")
        else:
            if self.phase_result.F_landing is None:
                if self.phase_result.F_liftoff is not None:
                    fallback_landing = self.phase_result.F_liftoff + 10
                    if self.phase_result.F_impact is not None and fallback_landing >= self.phase_result.F_impact:
                        fallback_landing = self.phase_result.F_impact - 1
                    self.phase_result.F_landing = min(fallback_landing, max(0, total_frames - 1))
                    print(f"[DEBUG] Landing missing. Fallback to Liftoff + 10: {self.phase_result.F_landing}")
                elif self.phase_result.F_impact is not None:
                    self.phase_result.F_landing = max(0, self.phase_result.F_impact - 5)
                    print(f"[DEBUG] Landing & Liftoff missing. Fallback Landing to Impact - 5: {self.phase_result.F_landing}")
        
        print(f"[DEBUG] Final PhaseResult: {self.phase_result}")
        # ---------------------------------
        
        self.setPhaseResult(self.phase_result)
        print("PhaseResult =", self.phase_result)

    def setPhaseResult(self, phase_result: PhaseResult):
        total = self.video_loader.total_frames
        if total > 0:
            f1 = phase_result.F_liftoff
            f2 = phase_result.F_landing
            f3 = phase_result.F_impact
            
            # 深藍 (Dark Blue)
            # color_prepare = QColor(0, 191, 255)   # 亮天藍 (Deep Sky Blue)
            # color_stride  = QColor(255, 69, 0)    # 亮橘紅 (Orange Red)
            # color_swing   = QColor(255, 215, 0)   # 亮金黃 (Gold)
            # color_follow  = QColor(50, 205, 50)   # 亮綠 (Lime Green)
            color_prepare = QColor(150, 150, 150)   # 亮天藍 (Deep Sky Blue)
            color_stride  = QColor(255, 69, 0)    # 亮橘紅 (Orange Red)
            # color_swing   = QColor(255, 215, 0)   # 亮金黃 (Gold)
            color_swing = QColor(180, 165, 32)
            # color_follow  = QColor(50, 205, 50)   # 亮綠 (Lime Green)
            color_follow = QColor(150, 150, 150)

            segs = []
            # 1. 開始 -> Liftoff
            if f1 is not None:
                segs.append((0, f1/total, color_prepare))
            
            # 2. Liftoff -> Landing
            if f1 is not None and f2 is not None:
                segs.append((f1/total, f2/total, color_stride))
            
            # 3. Landing -> Impact
            if f2 is not None and f3 is not None:
                segs.append((f2/total, f3/total, color_swing))
            
            # 4. Impact -> 結束
            if f3 is not None:
                segs.append((f3/total, 1.0, color_follow))
            self.analysis_bar.set_segments(segs)

            # --- 新增：將 A, B, C (Liftoff, Landing, Impact) 與 1, 2, 3 (峰值) 標在 Slider 上 ---
            markers = []
            if f1 is not None: markers.append((f1/total, 'a'))
            if f2 is not None: markers.append((f2/total, 'b'))
            if f3 is not None: markers.append((f3/total, 'c'))

            # 動力啟動順序 (1,2,3,4)
            try:
                peaks = self._get_velocity_peaks_3d(f3)
                valid_peaks = {k: v for k, v in peaks.items() if v is not None}
                sorted_peaks = sorted(valid_peaks.items(), key=lambda item: item[1])
                rank_dict = {name: rank + 1 for rank, (name, frame) in enumerate(sorted_peaks)}
                
                for name, f_num in valid_peaks.items():
                    if total > 1:
                        ratio = f_num / total
                        rank_text = str(rank_dict[name])
                        markers.append((ratio, rank_text))
            except Exception as e:
                print("Slider peak marker set error (3D):", e)

            if hasattr(self.analysis_bar, 'set_markers'):
                self.analysis_bar.set_markers(markers)

        self.ui.liftofflabel.setText(str(phase_result.F_liftoff)+"幀")
        self.ui.landinglabel.setText(str(phase_result.F_landing)+"幀")
        self.ui.impactlabel.setText(str(phase_result.F_impact)+"幀")
        self.display_phase_image(self.ui.keyphaseLabel1, phase_result.F_liftoff, 'liftoff')
        self.display_phase_image(self.ui.keyphaseLabel2, phase_result.F_landing, 'landing')
        self.display_phase_image(self.ui.keyphaseLabel3, phase_result.F_impact, 'impact')

        if phase_result.F_impact is not None:
            self.calculateBatAnglesAtImpact(phase_result.F_impact, phase_result.F_landing)
        self.calculateKeyphaseMetrics(phase_result)

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

        # 先在當前資料夾找，如果沒有就往外推一層 (為了支援 3D 影像序列資料夾)
        data_json_path = os.path.join(folder_path, 'data.json')
        if not os.path.exists(data_json_path):
            data_json_path = os.path.join(os.path.dirname(folder_path), 'data.json')

        if os.path.exists(data_json_path):
            try:
                with open(data_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'ExitVelocity' in data and hasattr(self.ui, 'exitVLabel'):
                        try:
                            ev_mph = float(data['ExitVelocity'])
                            ev_kmh = ev_mph * 1.60934
                            self.ui.exitVLabel.setText(f"{ev_kmh:.1f} kmh")
                        except:
                            self.ui.exitVLabel.setText(f"{data['ExitVelocity']} kmh")
                    if 'LaunchAngle' in data and hasattr(self.ui, 'lanuchAngleLabel'):
                        self.ui.lanuchAngleLabel.setText(f"{data['LaunchAngle']}°")
                    if 'Distance' in data and hasattr(self.ui, 'attackDisLabel'):
                        try:
                            dist_ft = float(data['Distance'])
                            dist_m = dist_ft * 0.3048
                            self.ui.attackDisLabel.setText(f"{dist_m:.1f} m")
                        except:
                            self.ui.attackDisLabel.setText(f"{data['Distance']}'")
            except Exception as e:
                print(f"Error loading {data_json_path}: {e}")

        spray_chart_path = os.path.abspath(os.path.join(folder_path, 'sprayChart.jpg')).replace('\\', '/')
        if not os.path.exists(spray_chart_path):
            spray_chart_path = os.path.abspath(os.path.join(os.path.dirname(folder_path), 'sprayChart.jpg')).replace('\\', '/')

        if os.path.exists(spray_chart_path) and hasattr(self.ui, 'rapsodoLabel'):
            try:
                # 像 keyphase 一樣，用 cv2 讀取圖片，避開 QPixmap 直接讀檔的 Thread 限制
                # cv2.imdecode 可以支援中文路徑
                img_bgr = cv2.imdecode(np.fromfile(spray_chart_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    q_img = QtGui.QImage(img_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
                    pixmap = QtGui.QPixmap.fromImage(q_img)
                    
                    self.ui.rapsodoLabel.setScaledContents(True) 
                    w_label = max(self.ui.rapsodoLabel.width(), 500)
                    h_label = max(self.ui.rapsodoLabel.height(), 500)
                    scaled_pixmap = pixmap.scaled(w_label, h_label, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.ui.rapsodoLabel.setMinimumSize(150, 150)
                    self.ui.rapsodoLabel.setPixmap(scaled_pixmap)
                    self.ui.rapsodoLabel.setAlignment(Qt.AlignCenter)
                    self.ui.rapsodoLabel.show()
                else:
                    self.ui.rapsodoLabel.clear()
            except Exception as e:
                print(f"Error drawing sprayChart.jpg: {e}")
                self.ui.rapsodoLabel.clear()
        else:
            if hasattr(self.ui, 'rapsodoLabel'): self.ui.rapsodoLabel.clear()

    def get_kpts_by_frame(self, frame_num):
        """從 Dataframe 獲取指定幀的關鍵點陣列"""
        if self.person_side_df is None or self.person_side_df.empty:
            return None
        target = self.person_side_df[self.person_side_df["frame_number"] == frame_num]
        if target.empty:
            return None
        return np.array(target.iloc[0]["keypoints"]) # 回傳 (N, 2) 或 (N, 3)

    def calculateKeyphaseMetrics(self, phase_result: PhaseResult):
        """Calculate and update UI for detailed metrics."""
        if self.person_3d_df is None or self.person_3d_df.empty:
            return

        # 1. Determine Front Leg based on Batter Type
        is_righty = (self.ui.batterInput.currentText() == "右打")
        # Indices (MediaPipe)
        # Left: Hip=23, Knee=25, Ankle=27
        # Right: Hip=24, Knee=26, Ankle=28
        if is_righty:
            # Front = Left
            front_hip, front_knee, front_ankle = 11, 13, 24
            back_hip, back_knee, back_ankle = 12, 14, 25
        else:
            # Front = Right
            front_hip, front_knee, front_ankle = 12, 14, 25
            back_hip, back_knee, back_ankle = 11, 13, 24
        
        # Helper to get 3D point (x, y, z)
        def get_pt(frame, idx):
            # 1. 基本檢查：DataFrame 是否存在
            if self.person_3d_df is None or self.person_3d_df.empty:
                return None

            # 2. 取得對應 frame_number 的那一行
            # 注意：這裡使用過濾方式，比 iloc[frame] 更安全（防止有缺幀）
            target_rows = self.person_3d_df[self.person_3d_df["frame_number"] == frame]
            
            if target_rows.empty:
                return None
        
            try:
                # 3. 取得該幀的 3D 陣列 (ndarray)
                # target_rows.iloc[0]["keypoints_3d"] 是一個 shape 為 (N, 3) 的陣列
                pts_array = target_rows.iloc[0]["keypoints_3d"]
        
                # 4. 檢查索引 idx 是否在陣列範圍內 (你的 JSON 顯示有 26 個點，索引 0~25)
                if idx < 0 or idx >= len(pts_array):
                    # print(f"Warning: Index {idx} out of range (max: {len(pts_array)-1})")
                    return None
            
                # 5. 回傳對應索引的 [x, y, z]
                return pts_array[idx]
        
            except Exception as e:
                print(f"Error in get_pt: {e}")
                return None
        # --- A. Liftoff Phase Metrics ---
        f_lift = phase_result.F_liftoff
        if f_lift is not None:
            if f_lift > 0:
                pt_L_t = get_pt(f_lift, 23)
                pt_R_t = get_pt(f_lift, 24)
                pt_L_prev = get_pt(f_lift-1, 23)
                pt_R_prev = get_pt(f_lift-1, 24)
                if all(p is not None for p in [pt_L_t, pt_R_t, pt_L_prev, pt_R_prev]):
                    cog_t = (pt_L_t + pt_R_t) / 2
                    cog_prev = (pt_L_prev + pt_R_prev) / 2
                    dist = np.linalg.norm(cog_t - cog_prev)
                    fps = self.video_loader.video_fps if self.video_loader.video_fps else 30
                    speed = dist * fps
                    self.ui.centerGravityLabel.setText(f"{speed:.1f}") # Check units
                else:
                    self.ui.centerGravityLabel.setText("  ")
            else:
                self.ui.centerGravityLabel.setText("  ")
        else:
            self.ui.forefootLabel.setText("  ")
            self.ui.centerGravityLabel.setText("  ")

        # --- B. Landing Phase Metrics ---
        f_land = phase_result.F_landing
        if f_land is not None:
            # # 1. Stride Distance (L_Ankle to R_Ankle dist)
            # pt_ank_f = get_pt(f_land, front_ankle)
            # pt_ank_b = get_pt(f_land, back_ankle)
            # stride_px = np.linalg.norm(pt_ank_f - pt_ank_b)
            # if pt_ank_f is not None and pt_ank_b is not None:
            #     self.ui.strideDisLabel.setText(f"{stride_px:.1f} cm")
            # else:
            #     self.ui.strideDisLabel.setText("  ")

            # 2. Knee Angle (Hip-Knee-Ankle of front leg)
            if f_land > 0:
                pt_L_t = get_pt(f_land, 23)
                pt_R_t = get_pt(f_land, 24)
                pt_L_prev = get_pt(f_land-1, 23)
                pt_R_prev = get_pt(f_land-1, 24)
                if all(p is not None for p in [pt_L_t, pt_R_t, pt_L_prev, pt_R_prev]):
                    cog_t = (pt_L_t + pt_R_t) / 2
                    cog_prev = (pt_L_prev + pt_R_prev) / 2
                    dist = np.linalg.norm(cog_t - cog_prev)
                    fps = self.video_loader.video_fps if self.video_loader.video_fps else 30
                    speed = dist * fps
                    self.ui.centerGravityLabel2.setText(f"{speed:.1f}") # Check units
                else:
                    self.ui.centerGravityLabel2.setText("  ")
            else:
                self.ui.centerGravityLabel2.setText("  ")
        else:
            self.ui.strideDisLabel.setText("  ")
            self.ui.centerGravityLabel2.setText("  ")

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
        if self.video_loader_raw is None:
            print("[display_phase_image] video_loader_raw is None")
            return
        img_bgr = self.video_loader_raw.getVideoImage2(frame_num)
        if img_bgr is None:
            print(f"[display_phase_image] Cannot get image for frame {frame_num} from video_loader_raw")
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
            ORANGE = (0, 0, 255)

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

                    # 準備文字
                    stride_px = np.linalg.norm(p_l_ank - p_r_ank)
                    stride_px = stride_px * 0.45
                    text_val = f"{stride_px:.1f}"
                    text_label = "跨步距離"
                    self.ui.strideDisLabel.setText(f"{stride_px:.1f}")
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
                            except:
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
                        print(f"PIL 繪圖失敗: {e}")
                        # 修正：將未定義的 font 改為 cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(draw_img, text_val, (f_x + 30, f_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)

            # --- C. Impact：繪製腿部骨架 (髖-膝-踝) ---
            # if phase_type == 'impact':
            #     # 定義要畫的線段：[髖, 膝, 踝]
            #     if not is_righty:
            #         legs = [[L_HIP, L_KNEE, L_ANKLE]]
            #     else:
            #         legs = [[R_HIP, R_KNEE, R_ANKLE]]
            #     for leg in legs:
            #         p1 = tuple(kpts[leg[0]][:2].astype(int))
            #         p2 = tuple(kpts[leg[1]][:2].astype(int))
            #         p3 = tuple(kpts[leg[2]][:2].astype(int))
            #         # 髖到膝
            #         cv2.line(draw_img, p1, p2, CYAN, 7)
            #         # 膝到踝
            #         cv2.line(draw_img, p2, p3, CYAN, 7)
            #         # 畫關節點
            #         for p in [p1, p2, p3]:
            #             cv2.circle(draw_img, p, 8, YELLOW, -1)
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
                    # 計算前導腿角度
                    front_knee_angle = self.calculate_angle(p1, p2, p3)

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
                    # 計算後腿角度
                    back_knee_angle = self.calculate_angle(p1, p2, p3)
                self.ui.forefootLabel.setText(f"{front_knee_angle:.1f}")
                self.ui.backfootLabel.setText(f"{back_knee_angle:.1f}")

                # ---- 新增繪製動力啟動順序標籤 (1, 2, 3, 4) ----
                try:
                    # 擷取 1,2,3,4 的發生順序 (3D 速度峰值)
                    impact_frame = getattr(self, 'phase_result', None) and self.phase_result.F_impact
                    peaks = self._get_velocity_peaks_3d(impact_frame)
                    valid_peaks = {k: v for k, v in peaks.items() if v is not None}
                    # 依照出現的 frame_number 升冪排序
                    sorted_peaks = sorted(valid_peaks.items(), key=lambda item: item[1])
                    
                    # rank_dict: 例如 {"knee": 1, "hip": 2, "neck": 3, "wrist": 4}
                    rank_dict = {name: rank + 1 for rank, (name, frame) in enumerate(sorted_peaks)}
                    
                    # 準備關節的坐標
                    joint_coords = {}
                    neck_idx = 18
                    if is_righty:
                        hip_idx, knee_idx, wrist_idx = 19, 13, 9
                        offset_x_val = -180 # 向後拉 (左方)
                    else:
                        hip_idx, knee_idx, wrist_idx = 19, 14, 10
                        offset_x_val = 180 # 向後拉 (右方)
                        
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
                            
                    # 圖標目錄
                    icon_dir = str(ICON_DIR)
                    icon_size = 90 # 設定 icon 統一大小
                    
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

                    for part, rank in rank_dict.items():
                        icon_path = os.path.join(icon_dir, f"number{rank}.png")
                        if os.path.exists(icon_path):
                            icon_img = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                            if icon_img is not None:
                                icon_img = cv2.resize(icon_img, (icon_size, icon_size))
                                cx, cy = int(joint_coords[part][0]), int(joint_coords[part][1])
                                
                                # 動態位移，避免完全疊在一起或擋住身體
                                # 使用者要求 1, 4 畫在右邊
                                if rank in [1, 4]:
                                    offset_x = 180
                                else:
                                    offset_x = offset_x_val
                                    
                                offset_y = -20
                                    
                                cv2.line(draw_img, (cx, cy), (cx + offset_x, cy + offset_y), (255, 255, 255), 4)
                                px = int(cx + offset_x - icon_size // 2)
                                py = int(cy + offset_y - icon_size // 2)
                                overlay_transparent(draw_img, icon_img, px, py)
                except Exception as e:
                    print("Sequence marker draw error:", e)
        self._set_label_pixmap_with_crop(label, draw_img)
        # # 3. 轉為 QPixmap 並顯示
        # img_rgb = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
        # h, w, ch = img_rgb.shape
        # q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        # pixmap = QPixmap.fromImage(q_img)

        # # --- 填滿 Label 並對齊「底部中間」裁切 ---
        # label_size = label.size()
        
        # if not label_size.isEmpty():
        #     # 1. 以「填滿」為原則縮放 (會有一邊超出 Label)
        #     pixmap = pixmap.scaled(
        #         label_size, 
        #         QtCore.Qt.KeepAspectRatioByExpanding, 
        #         QtCore.Qt.SmoothTransformation
        #     )
            
        #     # 2. 計算裁切區域 (Rect)
        #     # x 軸：置中裁切 (左右各切一半)
        #     # y 軸：對齊底部 (保留最下面，裁掉上面)
        #     crop_x = (pixmap.width() - label_size.width()) // 2
        #     crop_y = pixmap.height() - label_size.height() # 關鍵：直接減去高度，保留底部
            
        #     # 防呆：確保裁切起點不為負數
        #     crop_x = max(0, crop_x)
        #     crop_y = max(0, crop_y)

        #     rect = QtCore.QRect(
        #         crop_x,
        #         crop_y,
        #         label_size.width(),
        #         label_size.height()
        #     )
            
        #     # 3. 執行裁切
        #     pixmap = pixmap.copy(rect)

        # # 4. 顯示與快取
        # label.setAlignment(QtCore.Qt.AlignCenter)
        # self.cached_keyphase_pixmaps[label] = pixmap
        # self.update_keyphase_label_image(label)
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
    def _set_label_pixmap_with_crop(self, label, img_bgr):
        """封裝重複的 Pixmap 處理與裁切邏輯"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        pixmap = QPixmap.fromImage(QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())
        
        label_size = label.size()
        if not label_size.isEmpty():
            pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            crop_x = max(0, (pixmap.width() - label_size.width()) // 2)
            crop_y = max(0, pixmap.height() - label_size.height()) # 保留底部
            pixmap = pixmap.copy(QtCore.QRect(crop_x, crop_y, label_size.width(), label_size.height()))

        label.setAlignment(Qt.AlignCenter)
        self.cached_keyphase_pixmaps[label] = pixmap
        label.setPixmap(pixmap)

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
        # 強制要求 Label 對齊中央 (選配)
        label.setAlignment(Qt.AlignCenter)

    def resizeEvent(self, event):
        super().resizeEvent(event)  

    def eventFilter1(self, source, event):
        # 監聽 Slider 的 Resize 事件
        if source == self.ui.frameSlider and event.type() == QEvent.Resize:
            slider_w = source.width()
            slider_h = source.height()
            
            # 建立樣式選項來獲取 Slider 的內部佈局細節 (Groove)
            opt = QtWidgets.QStyleOptionSlider()
            source.initStyleOption(opt)
            groove_rect = source.style().subControlRect(
                QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, source
            )
            
            # 調整 AnalysisBar 的位置與大小
            # 讓其高度足夠大 (80) 並居中對齊滑軌
            bar_h = self.analysis_bar.height() # 已經在 __init__ 設為 80
            
            # 定位在 Slider 的中心，並與滑軌 (Groove) 的寬度對齊
            self.analysis_bar.setGeometry(
                groove_rect.left(), 
                groove_rect.center().y() - bar_h // 2, 
                groove_rect.width(), 
                bar_h
            )
        
            # 重要：強制提到最上層，防止被 UI 樣式覆蓋
            self.analysis_bar.raise_()
        # if source == self.ui.frameSlider and event.type() == QEvent.Resize:
        #     # 獲取 Slider 的寬度
        #     slider_w = source.width()
        #     slider_h = source.height()
            
        #     # 通常 QSlider 的軌道位在中間，我們將 AnalysisBar 定位在稍微偏下的地方
        #     # 這裡的座標 (0, 18) 需要根據你的 Slider 樣式微調
        #     self.analysis_bar.setGeometry(0, int(slider_h/2) + 2, slider_w, 8)
            
        if event.type() == QEvent.Resize:
            if hasattr(self.ui, 'keyphaseLabel1') and source == self.ui.keyphaseLabel1:
                self.update_keyphase_label_image(self.ui.keyphaseLabel1)
            elif hasattr(self.ui, 'keyphaseLabel2') and source == self.ui.keyphaseLabel2:
                self.update_keyphase_label_image(self.ui.keyphaseLabel2)
            elif hasattr(self.ui, 'keyphaseLabel3') and source == self.ui.keyphaseLabel3:
                self.update_keyphase_label_image(self.ui.keyphaseLabel3)
        return super().eventFilter(source, event)


    def eventFilter(self, source, event):
        if source == self.ui.frameSlider and event.type() == QEvent.Resize:
            opt = QtWidgets.QStyleOptionSlider()
            source.initStyleOption(opt)
            
            # 獲取滑軌的精確矩形
            groove_rect = source.style().subControlRect(
                QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, source
            )
            
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
        # ... 其他邏輯

        if event.type() == QEvent.Resize:
            if hasattr(self.ui, 'keyphaseLabel1') and source == self.ui.keyphaseLabel1:
                self.update_keyphase_label_image(self.ui.keyphaseLabel1)
            elif hasattr(self.ui, 'keyphaseLabel2') and source == self.ui.keyphaseLabel2:
                self.update_keyphase_label_image(self.ui.keyphaseLabel2)
            elif hasattr(self.ui, 'keyphaseLabel3') and source == self.ui.keyphaseLabel3:
                self.update_keyphase_label_image(self.ui.keyphaseLabel3)
        return super().eventFilter(source, event)


    def calculateBatAnglesAtImpact(self, impact_frame: int, landing_frame: int = None):
        # 1. 確保 frame 合法
        if impact_frame <= 0: return

        # 尋找有效的當前擊球點與前一點 (過濾掉可能由 fallback 複製的相同座標)
        def find_valid_bat_3d(f_target, is_prev=False, exclude_head=None):
            search_range = [1, 2, 3, 4] if is_prev else [0, 1, 2, -1, -2]
            for offset in search_range:
                cand = f_target - offset if is_prev else f_target + offset
                if cand < 0: continue
                tail, head = self.triangulate_bat(cand)
                if tail is not None and head is not None:
                    if exclude_head is not None:
                        if np.array_equal(head[:2], exclude_head[:2]):
                            continue
                    return tail, head, cand
            return None, None, f_target - 1 if is_prev else f_target

        tail_3d, head_3d, f_curr = find_valid_bat_3d(impact_frame, is_prev=False)
        _, head_prev, f_prev = find_valid_bat_3d(f_curr, is_prev=True, exclude_head=head_3d)
        gap = max(1, f_curr - f_prev)

        if tail_3d is not None and head_3d is not None:
            # --- A. 垂直棒角 (Vertical Bat Angle) ---
            bat_vector = head_3d - tail_3d
            
            # 在 y_down 系統中，若 head_y > tail_y (棒頭在下)，則 dy 為負
            dy = tail_3d[1] - head_3d[1] 
            # 水平面的投影長度
            dxz = np.linalg.norm([bat_vector[0], bat_vector[2]])
            
            # 計算角度：dy 為負時，arctan2 會回傳負角度 (符合向下之意)
            angle_vba = np.degrees(np.arctan2(dy, dxz)) * -1
            self.ui.verticalAngleEditLabel.setText(f"{angle_vba:.1f}")

            # --- B. 揮擊軌跡指標 ---
            if head_prev is not None:
                # 取得影片 FPS (預設 30)
                fps = self.video_loader.video_fps if hasattr(self.video_loader, 'video_fps') and self.video_loader.video_fps else 30
                
                # 1. 揮棒速度計算 (Bat Speed) - 使用 mph
                # 位移向量與 3D 距離 (公尺)
                velocity_vec = head_3d - head_prev
                dist = np.linalg.norm(velocity_vec)
                
                # =========================================================================
                # (原本的計算方式，已註解以供比較)
                # =========================================================================
                # speed_ms = (dist * fps) / (2.0 * gap)
                # speed_kmh = speed_ms * 3.6 * 2.5
                # old_min, old_max = 80, 160
                # new_min, new_max = 93, 130
                # normalized_val = ((speed_kmh - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                # normalized_val = max(new_min, min(new_max, normalized_val))
                # speed_kmh_old = normalized_val
                # =========================================================================

                # =========================================================================
                # (新的計算方式：基於 landing 到 impact 之間的實際路徑與時間)
                # =========================================================================
                landing_frame_to_use = landing_frame if (landing_frame is not None and landing_frame > 0) else max(0, impact_frame - 15)
                
                path_heads_3d = {}
                for f in range(landing_frame_to_use, impact_frame + 1):
                    if f < 0: continue
                    _, head_f = self.triangulate_bat(f)
                    if head_f is not None:
                        path_heads_3d[f] = head_f
                        
                path_f = sorted(path_heads_3d.keys())
                
                # 計算總 3D 距離 (公尺)
                total_path_m = 0.0
                for i in range(len(path_f) - 1):
                    p_prev = path_heads_3d[path_f[i]]
                    p_curr = path_heads_3d[path_f[i+1]]
                    dist_val = np.linalg.norm(p_curr - p_prev)
                    total_path_m += dist_val
                    
                # 計算時間差 (秒)
                frames_diff = max(1, impact_frame - landing_frame_to_use)
                duration_sec = frames_diff / fps
                
                # 速度 (m/s) 與速度 (km/h, 原始)
                new_speed_ms = total_path_m / duration_sec
                new_speed_kmh_raw = new_speed_ms * 3.6
                
                # 新方法改用直接加 80 的偏移量計算最終速度
                speed_kmh = new_speed_kmh_raw + 80
                self.ui.batSpeedLabel.setText(f"{speed_kmh:.1f}")
                
                # 計算舊方法的速度以供比對列印
                old_min, old_max = 80, 160
                new_min, new_max = 93, 130
                velocity_vec_old = head_3d - head_prev
                dist_old = np.linalg.norm(velocity_vec_old)
                old_speed_ms = (dist_old * fps) / (2.0 * gap)
                old_speed_kmh_raw = old_speed_ms * 3.6
                old_speed_kmh_scaled = old_speed_kmh_raw * 2.5
                old_normalized_val = ((old_speed_kmh_scaled - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
                old_normalized_val = max(new_min, min(new_max, old_normalized_val))
                
                print(f"[BatMetrics 3D] 速度對比:")
                print(f"  - 舊方法 (前後鄰近幀位移): 原始={old_speed_kmh_raw:.2f} kmh, 放大={old_speed_kmh_scaled:.2f} kmh, 歸一化={old_normalized_val:.2f} kmh")
                print(f"  - 新方法 (landing到impact路徑): 原始={new_speed_kmh_raw:.2f} kmh, 偏移後 (+80)={speed_kmh:.2f} kmh, 總路徑公尺={total_path_m:.4f}, 幀數={len(path_f)}/{frames_diff}, 時間={duration_sec:.4f}秒")
                
                # 1. 取得揮擊位移向量 (Velocity vector)
                # velocity[0]: X (左右), velocity[1]: Y (上下), velocity[2]: Z (前後)
                velocity = head_3d - head_prev
                
                # 2. 計算水平面上的合成位移 (XZ 平面的長度)
                # 這是為了計算相對於地面的「爬升」角度
                vxz = np.sqrt(velocity[0]**2 + velocity[2]**2)
                
                # 3. 處理 Y 軸 (Vertical)
                # 在 y_down 系統中，head_3d[1] 變小代表向上移動。
                # 所以 vy = -(current_y - prev_y) = prev_y - current_y
                # 這樣向上揮時 vy_up 會是正值
                vy_up = -velocity[1] 
                
                # 4. 計算攻擊角度 (Attack Angle)
                # 使用 arctan2(y, x) 確保數值範圍與正負號正確
                attack_angle = np.degrees(np.arctan2(vy_up, vxz))
                attack_angle = abs(attack_angle)
                
                # 更新 UI 顯示 (對應圖片中的向上角度)
                self.ui.horizontalAngleEditLabel.setText(f"{attack_angle:.1f}")

    def loadVideo(self):
        self.video_loader.loadVideo(self.video_side_path, self.video_front_path)
        self.video_loader_raw.loadVideo(self.video_side_raw, self.video_front_raw)
        self.checkVideoLoad()

    def checkVideoLoad(self):
        """檢查影片是否讀取完成，並更新 UI 元素。"""
        # 1. 檢查是否有影片路徑
        if not self.video_loader.video_path and not self.video_loader.video_path2:
            return
            
        # 2. 若「任一」影片正在讀取中 (processed 或 raw)，定時檢查
        if self.video_loader.is_loading or self.video_loader_raw.is_loading:
            # self.ui.videoNameLabel.setText("讀取影片中")
            QTimer.singleShot(100, self.checkVideoLoad)
            return

        # 3. 影片讀取完成後更新 UI 元素
        self.updateVideoInfo()

    # ======================================================
    # 📁 取用影片資訊 person_df, bat_df, batht_df, ball_df
    # ======================================================
    def getPersonDf(self, DataFrame: pd.DataFrame, frame_num=None):
        """
        離線版，對應側面相機的 person 資料。
        介面盡量模仿 PoseEstimater.getPersonDf：
        - frame_num: 指定幀，如果為 None 則回傳整個 DataFrame
        """
        if DataFrame is None or DataFrame.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(DataFrame))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (DataFrame['frame_number'] == frame_num)
        data = DataFrame.loc[condition].copy()

        if data.empty:
            return None
        
        return data

    def getBatDf(self, DataFrame: pd.DataFrame, frame_num=None):
        """
        離線版，對應側面相機的 bat 資料。
        介面盡量模仿 PoseEstimater.getPersonDf：
        - frame_num: 指定幀，如果為 None 則回傳整個 DataFrame
        """
        if DataFrame is None or DataFrame.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(DataFrame))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (DataFrame['frame_number'] == frame_num)
        data = DataFrame.loc[condition].copy()

        if data.empty:
            return None
        
        return data

    def getBathtDf(self, DataFrame: pd.DataFrame, frame_num=None):
        """
        離線版，對應側面相機的 batht 資料。
        介面盡量模仿 PoseEstimater.getPersonDf：
        - frame_num: 指定幀，如果為 None 則回傳整個 DataFrame
        """
        if DataFrame is None or DataFrame.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(DataFrame))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (DataFrame['frame_num'] == frame_num)
        data = DataFrame.loc[condition].copy()

        if data.empty:
            return None
        
        return data

    def getBallDf(self, DataFrame: pd.DataFrame, frame_num=None):
        """
        離線版，對應側面相機的 ball 資料。
        介面盡量模仿 PoseEstimater.getPersonDf：
        - frame_num: 指定幀，如果為 None 則回傳整個 DataFrame
        """
        if DataFrame is None or DataFrame.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(DataFrame))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (DataFrame['frame_number'] == frame_num)
        data = DataFrame.loc[condition].copy()

        if data.empty:
            return None
        
        return data

    # ======================================================
    # 📁 畫到畫面上
    # ======================================================

    def updateVideoInfo(self):
        # 更新與影片相關的資訊顯示在 UI 上
        self.reset()
        self.initFrameSlider()
        self.initGraph()
        pts3d = self.triangularPersonPoints(0)
        ball_3d = self.triangulateBallPoint(0)
        tail_3d, head_3d = self.triangulate_bat(0)
        self.draw_ground_once(pts3d)
        self.draw_3d_pose(pts3d)
        self.draw_hip_shoulder_angle(pts3d)
        # self.calBackKneeAngleEditLabel(pts3d)   # 計算並更新後膝角度
        # self.calFrontKneeAngleEditLabel(pts3d)  # 計算並更新前膝角度 (建議新增)
        # self.calKPHipShoulderEditLabel(pts3d)   # 更新關鍵幀區域的肩髖分離角
        # self.draw_ball_3d(ball_3d)
        self.draw_bat_3d(tail_3d, head_3d)
        # 2. 畫出隨時間生長的軌跡 (淺色)
        pts3d1 = self.triangularPersonPoints(self.phase_result.F_liftoff)
        pts3d2 = self.triangularPersonPoints(self.phase_result.F_landing)
        pts3d3 = self.triangularPersonPoints(self.phase_result.F_impact)
        self.sethipShoulderEditLabel(1, pts3d1)
        self.sethipShoulderEditLabel(2, pts3d2)
        self.sethipShoulderEditLabel(3, pts3d3)
        self.setKneeAngleEditLabel(pts3d3)
        self.draw_swing_trajectory(0)
        self.updateFrame(frame_num = 0)
        video_size = self.video_loader.video_size
        self.ui.ResolutionLabel.setText(f"(0,0) - {video_size[0]} x {video_size[1]}")

    def setKneeAngleEditLabel(self, points_3d_centered):
        if points_3d_centered is None:
            return
        self.calKneeAngleEditLabel(points_3d_centered)

    def calKneeAngleEditLabel(self, points_3d_centered):
        if points_3d_centered is None:
            return

        # 1. 根據打擊手別判斷後腳索引
        is_righty = (self.ui.batterInput.currentText() == "右打")
        if is_righty:
            # 右打者：後腳為右腳
            f_hip_idx, f_knee_idx, f_ankle_idx = 11, 13, 24
            b_hip_idx, b_knee_idx, b_ankle_idx = 12, 14, 25
        else:
            # 左打者：後腳為左腳
            f_hip_idx, f_knee_idx, f_ankle_idx = 12, 14, 25
            b_hip_idx, b_knee_idx, b_ankle_idx = 11, 13, 24

        try:
            # 2. 取得三維座標點
            hip = points_3d_centered[b_hip_idx]
            knee = points_3d_centered[b_knee_idx]
            ankle = points_3d_centered[b_ankle_idx]

            # 3. 建立向量：以膝蓋為頂點
            # 向量 BA (膝蓋到髖部)
            v_knee_to_hip = hip - knee
            # 向量 BC (膝蓋到腳踝)
            v_knee_to_ankle = ankle - knee

            # 4. 使用內積公式計算夾角 (餘弦定理變體)
            # cos(theta) = (v1 . v2) / (|v1| * |v2|)
            dot_product = np.dot(v_knee_to_hip, v_knee_to_ankle)
            norm_hip = np.linalg.norm(v_knee_to_hip)
            norm_ankle = np.linalg.norm(v_knee_to_ankle)

            # 防止數值誤差導致超出 [-1, 1] 範圍
            cos_theta = np.clip(dot_product / (norm_hip * norm_ankle), -1.0, 1.0)
            
            # 轉為角度
            angle_deg = np.degrees(np.arccos(cos_theta))

            # 5. 更新 UI 標籤 (根據您的 UI 定義為 backKneeAngleLabel)
            self.ui.backfootLabel.setText(f"{angle_deg:.1f}")

            hip = points_3d_centered[f_hip_idx]
            knee = points_3d_centered[f_knee_idx]
            ankle = points_3d_centered[f_ankle_idx]

            # 3. 建立向量：以膝蓋為頂點
            # 向量 BA (膝蓋到髖部)
            v_knee_to_hip = hip - knee
            # 向量 BC (膝蓋到腳踝)
            v_knee_to_ankle = ankle - knee

            # 4. 使用內積公式計算夾角 (餘弦定理變體)
            # cos(theta) = (v1 . v2) / (|v1| * |v2|)
            dot_product = np.dot(v_knee_to_hip, v_knee_to_ankle)
            norm_hip = np.linalg.norm(v_knee_to_hip)
            norm_ankle = np.linalg.norm(v_knee_to_ankle)

            # 防止數值誤差導致超出 [-1, 1] 範圍
            cos_theta = np.clip(dot_product / (norm_hip * norm_ankle), -1.0, 1.0)
            
            # 轉為角度
            angle_deg = np.degrees(np.arccos(cos_theta))

            # 5. 更新 UI 標籤 (根據您的 UI 定義為 backKneeAngleLabel)
            self.ui.forefootLabel.setText(f"{angle_deg:.1f}")
        except Exception as e:
            print(f"Error in calBackKneeAngleEditLabel: {e}")

    def calBackKneeAngleEditLabel(self, points_3d_centered):
        """計算並更新 UI 上的後膝角度"""
        is_righty = (self.ui.batterInput.currentText() == "右打")
        # 索引分配：右打後腳為右(12, 14, 25)；左打後腳為左(11, 13, 24)
        b_hip, b_knee, b_ankle = (12, 14, 25) if is_righty else (11, 13, 24)
        
        angle = self._calculate_3d_joint_angle(points_3d_centered, b_hip, b_knee, b_ankle)
        if angle is not None:
            self.ui.kpbackKneeAngleLabel.setText(f"{angle:.1f}")

    def calFrontKneeAngleEditLabel(self, points_3d_centered):
        """計算並更新 UI 上的前膝角度"""
        is_righty = (self.ui.batterInput.currentText() == "右打")
        # 索引分配：右打前腳為左(11, 13, 24)；左打前腳為右(12, 14, 25)
        f_hip, f_knee, f_ankle = (11, 13, 24) if is_righty else (12, 14, 25)
        
        angle = self._calculate_3d_joint_angle(points_3d_centered, f_hip, f_knee, f_ankle)
        if angle is not None:
            self.ui.kpfrontKneeAngleLabel.setText(f"{angle:.1f}")

    def _calculate_3d_joint_angle(self, pts, idx1, idx2, idx3):
        """通用 3D 夾角計算工具"""
        try:
            v1 = pts[idx1] - pts[idx2]
            v2 = pts[idx3] - pts[idx2]
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        except:
            return None

    def sethipShoulderEditLabel(self, phase, points_3d_centered):
        if points_3d_centered is None:
            return
        self.calhipShoulderEditLabel(phase, points_3d_centered)

    def calhipShoulderEditLabel(self, phase, points_3d_centered):
        l_hip = points_3d_centered[11]
        r_hip = points_3d_centered[12]
        l_shoulder = points_3d_centered[5]
        r_shoulder = points_3d_centered[6]

        hip_pts = np.array([l_hip, r_hip], dtype=np.float32)
        shoulder_pts = np.array([l_shoulder, r_shoulder], dtype=np.float32)
        v1 = r_hip - l_hip
        v2 = r_shoulder - l_shoulder

        # 投影到 XZ 平面（MLB 定義的旋轉平面）
        v1_proj = np.array([v1[0], v1[2]])
        v2_proj = np.array([v2[0], v2[2]])

        # atan2 得到旋轉角度（有方向性）
        theta_hip = np.arctan2(v1_proj[1], v1_proj[0])
        theta_shd = np.arctan2(v2_proj[1], v2_proj[0])

        # HSS = 肩 - 髖（MLB 定義）
        angle_rad = theta_shd - theta_hip

        # 將角度包到 -180 ～ +180
        angle_deg = np.degrees(angle_rad)
        angle_deg = (angle_deg + 180) % 360 - 180
        if angle_deg < 0:
            angle_deg += 180
            
        if angle_deg > 90:
            angle_deg = abs(180 - angle_deg)
        
        label_name = f"hipShoulderEditLabel{phase}"
        # 從 self.ui 中抓取對應名稱的物件
        label_object = getattr(self.ui, label_name)
        # 設定文字
        label_object.setText(f"{angle_deg:.1f}")

    def initFrameSlider(self, total_frames=None):
        """初始化影片滑桿和相關的標籤。"""
        if total_frames is None:
            total_frames = self.video_loader.total_frames
            
        if total_frames is None:
            return
        self.ui.frameSlider.setMinimum(0)
        self.ui.frameSlider.setMaximum(total_frames - 1)
        self.ui.frameSlider.setValue(0)
        # self.ui.frameNumLabel.setText(f'0/{total_frames - 1}')
        total_seconds = 1.5
        # 初始化標籤顯示： 0.00s / 1.50s
        self.ui.frameNumLabel.setText(f'0.00/{total_seconds:.1f}s')

    def updateFrame(self, frame_num:int = None):
        """Update the displayed frame with additional analysis."""
        # 若外部沒傳，從 slider 讀（保險用）
        if frame_num is None:
            frame_num = self.ui.frameSlider.value()
        total_duration = 1.5
        current_seconds = (frame_num / (self.video_loader.total_frames - 1)) * total_duration
        # 顯示格式： 0.45s / 1.50s
        self.ui.frameNumLabel.setText(f'{current_seconds:.2f}/{total_duration:.1f}s')
        # self.ui.frameNumLabel.setText(f'{frame_num}/{self.video_loader.total_frames - 1}')
        # 取畫面
        frame = self.video_loader.getVideoImage(frame_num)
        frame_2 = self.video_loader.getVideoImage2(frame_num)
        if frame is None or frame_2 is None:
            print("Frame is None")
            return
        # 取當下 frame 的person、bat、batht、ball
        person_side_data = self.getPersonDf(self.person_side_df, frame_num)
        person_front_data = self.getPersonDf(self.person_front_df, frame_num)
        batht_side_data = self.getBathtDf(self.batht_side_df, frame_num)
        batht_front_data = self.getBathtDf(self.batht_front_df, frame_num)
        ball_side_data = self.getBallDf(self.ball_side_df, frame_num)
        ball_front_data = self.getBallDf(self.ball_front_df, frame_num)
        drawed_img = self.image_drawer.drawInfo(frame, frame_num, person_front_data, batht_front_data, ball_front_data)
        drawed_img2 = self.image_drawer_2.drawInfo(frame_2, frame_num, person_side_data, batht_side_data, ball_side_data)
        pts3d = self.triangularPersonPoints(frame_num)
        ball_3d = self.triangulateBallPoint(frame_num)
        tail_3d, head_3d = self.triangulate_bat(frame_num)
        self.draw_3d_pose(pts3d)
        self.draw_hip_shoulder_angle(pts3d)

        self.draw_bat_3d(tail_3d, head_3d)
        # 2. 畫出隨時間生長的軌跡 (淺色)
        self.draw_swing_trajectory(frame_num)
        self.showImage(drawed_img, self.view_scene, self.ui.FrameView)
        self.showImage(drawed_img2, self.view_scene2, self.ui.FrameView_2)
        # self.updateGraph(frame_num)

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        """Display an image in the QGraphicsView."""
        scene.clear()
        if image is None:
            print("Error: Image is None")
            return
        h, w = image.shape[:2]
        qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)   

    def analyzeFrame(self):
        """Analyze and process each video frame."""
        frame_num = self.ui.frameSlider.value()
        total_duration = 1.5
        current_seconds = (frame_num / (self.video_loader.total_frames - 1)) * total_duration
        # 顯示格式： 0.45s / 1.50s
        self.ui.frameNumLabel.setText(f'{current_seconds:.2f}/{total_duration:.1f}s')
        # self.ui.frameNumLabel.setText(f'{frame_num}/{self.video_loader.total_frames - 1}')
        frame = self.video_loader.getVideoImage(frame_num)
        frame_2 = self.video_loader.getVideoImage2(frame_num)
        fps = 0
        # if self.ui.showSkeletonCheckBox.isChecked():
        curr_side_person_data = self.getPersonDf(self.person_df, frame_num)
        curr_front_person_data = self.getPersonDf(self.front_person_df, frame_num)

        self.updateFrame(frame_num=frame_num)
        # pts3d = self.triangularPoints(frame_num= frame_num)
        # self.draw_ground_once(pts3d)
        # self.draw_3d_pose(pts3d)
        # self.draw_hip_shoulder_angle(pts3d)

    # ======================================================
    # Frame 控制
    # ======================================================
    def playBtnClicked(self):
        """Handle play/pause button click."""
        if self.video_loader.video_path == "" or self.video_loader.video_path2 == "":
            QMessageBox.warning(self, "無法播放影片", "請讀取影片!")
            return
        if self.video_loader.is_loading:
            QMessageBox.warning(self, "影片讀取中", "請稍等!")
            return
        self.is_play = not self.is_play
        self.ui.playBtn.setText("||" if self.is_play else "▶︎")
        if self.is_dual== False:
            self.playFrame(self.ui.frameSlider.value())
        else:
            self.playFrame4Process(self.ui.frameSlider.value())
            
    def playFrame(self, start_num: int = 0):
        """Play video frames starting from start_num."""
        for i in range(start_num, self.video_loader.total_frames):
            if not self.is_play:
                break
            self.ui.frameSlider.setValue(i)
            if i == self.video_loader.total_frames - 1 and self.is_play:
                self.playBtnClicked()
            
            # 還原原本系統 1 倍速為最低延遲 1ms 的體驗
            delay = 1
            try:
                if hasattr(self.ui, 'slowMotionInput'):
                    val_str = self.ui.slowMotionInput.currentText().replace('x', '').replace('X', '').strip()
                    current_rate = float(val_str)
                    if current_rate >= 1.0:
                        delay = 1
                    elif current_rate > 0:
                        # 當速度小於 1，成比例加上額外的延遲 (0.5倍加約33ms，0.25倍加約99ms)
                        delay = int((1.0 / current_rate - 1.0) * 33) + 1
            except:
                pass
                
            cv2.waitKey(max(1, delay))

    def nextFrame(self):
        self.frame += 1
        if self.frame >= self.total_frames:
            self.frame = 0
        self.updateFrame(self.frame)

    def setFrame(self, f):
        self.frame = f
        self.updateFrame(f)

    # ======================================================
    # reset
    # ======================================================

    def resetFrameSlider(self):
        self.ui.frameSlider.setValue(0)
        self.ui.frameSlider.setRange(0, 0)

    def reset(self):
        self.resetFrameSlider()
        self.view_scene.clear()
        self.view_scene2.clear()