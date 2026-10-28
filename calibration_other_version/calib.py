"""
相機內參校正工具（單台相機，棋盤格法）

與外參校正分開：內參只跟「相機機身 + 鏡頭」的組合有關，
與安裝位置無關，因此一台一台單獨校正即可，不需要雙機同步或 GPIO 觸發線。

使用方式：
    cd Src\\UI_Control
    python calib.py

流程：下拉選單選相機 → 連接 → 確認棋盤格規格 → 擷取 10~15 組不同角度
      → 執行校正 → 自動存到 Db/Calibration/intrinsic/
      → 第 4 區指定正面/側面，寫入 Db/Calibration/camera_roles.json

已有照片：填入相機序號及棋盤格規格 → 讀取照片資料夾 → 執行校正。
每次選擇一台相機的資料夾，不需連接相機。
"""

# Support both python -m and direct file execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QMessageBox, QFileDialog, QLineEdit, QProgressDialog,
                             QCheckBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

import PySpin
from Src.cameras.single_camera.flir_camera_system import FlirCameraSystem
from Src.calibration import calib_store

from Src.paths import CAMERA_CONFIG_DIR
CONFIG_PATH = str(CAMERA_CONFIG_DIR / "GH3_camera_config.yaml")


def list_cameras():
    """列舉目前連接的所有 FLIR 相機，回傳 [(序號, 型號), ...]"""
    result = []
    system = PySpin.System.GetInstance()
    try:
        cam_list = system.GetCameras()
        for i in range(cam_list.GetSize()):
            cam = cam_list.GetByIndex(i)
            try:
                tl = cam.TLDevice
                sn = tl.DeviceSerialNumber.GetValue()
                model = tl.DeviceModelName.GetValue()
            except Exception:
                sn, model = f"<index {i}>", "unknown"
            result.append((sn, model))
            del cam
        cam_list.Clear()
    finally:
        system.ReleaseInstance()
    return result


class CalibWindow(QMainWindow):
    # 預覽效能參數：全解析度做角點偵測太慢，預覽改用縮圖且降低偵測頻率。
    # 實際擷取樣本時仍使用全解析度，不影響校正精度。
    PREVIEW_SCALE = 0.4          # 偵測用縮圖比例
    PREVIEW_DETECT_EVERY = 3     # 每 N 次畫面更新才偵測一次

    def __init__(self):
        super().__init__()
        self.setWindowTitle("相機內參校正工具（單台）")
        self.resize(900, 900)

        self.camera = None
        self.last_frame = None
        self.objpoints = []
        self.imgpoints = []
        self.captured_count = 0
        self.image_size = None
        self._sample_spec = None
        self._photo_paths = []

        self._preview_tick = 0
        self._preview_found = False
        self._preview_corners = None

        self.initUI()
        self.refresh_cameras()
        self.refresh_role_combos()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(60)

    # ---------- UI ----------

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 相機選擇
        cam_group = QGroupBox("1. 選擇相機")
        cam_layout = QHBoxLayout(cam_group)
        self.cmb_camera = QComboBox()
        self.cmb_camera.setMinimumWidth(380)
        self.btn_refresh = QPushButton("重新掃描")
        self.btn_connect = QPushButton("連接")
        self.btn_disconnect = QPushButton("中斷連接")
        self.btn_refresh.clicked.connect(self.refresh_cameras)
        self.btn_connect.clicked.connect(self.connect_camera)
        self.btn_disconnect.clicked.connect(self.disconnect_camera)
        self.btn_disconnect.setEnabled(False)
        cam_layout.addWidget(self.cmb_camera)
        cam_layout.addWidget(self.btn_refresh)
        cam_layout.addWidget(self.btn_connect)
        cam_layout.addWidget(self.btn_disconnect)
        layout.addWidget(cam_group)

        # 棋盤格參數
        board_group = QGroupBox("2. 棋盤格規格（內角點數量 = 格子數 - 1）")
        board_layout = QHBoxLayout(board_group)
        self.spn_cols = QSpinBox()
        self.spn_cols.setRange(3, 30)
        self.spn_cols.setValue(8)
        self.spn_rows = QSpinBox()
        self.spn_rows.setRange(3, 30)
        self.spn_rows.setValue(11)
        self.spn_square = QDoubleSpinBox()
        self.spn_square.setRange(0.1, 1000.0)
        self.spn_square.setDecimals(2)
        self.spn_square.setValue(10.0)
        self.spn_square.setSuffix(" mm")
        board_layout.addWidget(QLabel("內角點 寬:"))
        board_layout.addWidget(self.spn_cols)
        board_layout.addWidget(QLabel("高:"))
        board_layout.addWidget(self.spn_rows)
        board_layout.addWidget(QLabel("每格邊長:"))
        board_layout.addWidget(self.spn_square)
        board_layout.addStretch()
        layout.addWidget(board_group)

        # 鏡頭：用標稱焦距檢查 fx 是否合理；長焦鏡頭鎖住畸變，避免估出不合理的畸變係數
        lens_group = QGroupBox("鏡頭（選填）")
        lens_layout = QHBoxLayout(lens_group)
        self.spn_focal = QDoubleSpinBox()
        self.spn_focal.setRange(0.0, 500.0)
        self.spn_focal.setDecimals(1)
        self.spn_focal.setSpecialValueText("未填")
        self.spn_focal.setSuffix(" mm")
        self.chk_lock_dist = QCheckBox("長焦：鎖住畸變（畸變係數固定為 0）")
        self.spn_focal.valueChanged.connect(
            lambda v: self.chk_lock_dist.setChecked(v >= self.TELEPHOTO_MM))
        lens_layout.addWidget(QLabel("鏡頭焦距:"))
        lens_layout.addWidget(self.spn_focal)
        lens_layout.addWidget(self.chk_lock_dist)
        lens_layout.addStretch()
        layout.addWidget(lens_group)

        self.photo_group = QGroupBox("從照片校正（不需連接相機，每次選一台相機的資料夾）")
        photo_layout = QHBoxLayout(self.photo_group)
        self.txt_photo_sn = QLineEdit()
        self.txt_photo_sn.setPlaceholderText("照片所屬的相機序號")
        self.btn_import_photos = QPushButton("讀取照片資料夾")
        self.btn_import_photos.clicked.connect(self.choose_photo_folder)
        self.lbl_photo_source = QLabel("尚未載入照片")
        photo_layout.addWidget(QLabel("相機序號："))
        photo_layout.addWidget(self.txt_photo_sn)
        photo_layout.addWidget(self.btn_import_photos)
        photo_layout.addWidget(self.lbl_photo_source, 1)
        layout.addWidget(self.photo_group)

        # 畫面
        self.lbl_view = QLabel("尚未連接相機")
        self.lbl_view.setAlignment(Qt.AlignCenter)
        self.lbl_view.setStyleSheet("border: 2px solid gray; background: #222; color: #ccc;")
        self.lbl_view.setMinimumHeight(420)
        layout.addWidget(self.lbl_view)

        # 操作按鈕
        act_group = QGroupBox("3. 擷取與校正")
        act_layout = QHBoxLayout(act_group)
        self.btn_capture = QPushButton("擷取樣本 (空白鍵)")
        self.btn_calibrate = QPushButton("執行校正")
        self.btn_reset = QPushButton("清空樣本")
        self.btn_capture.clicked.connect(self.capture_sample)
        self.btn_calibrate.clicked.connect(self.run_calibration)
        self.btn_reset.clicked.connect(self.reset_samples)
        act_layout.addWidget(self.btn_capture)
        act_layout.addWidget(self.btn_calibrate)
        act_layout.addWidget(self.btn_reset)
        layout.addWidget(act_group)

        # 角色指定：拍照時決定哪台當 primary，3D 計算時決定用哪台的內參
        role_group = QGroupBox("4. 指定正面 / 側面相機（拍外參照片與 3D 計算都依此設定）")
        role_layout = QHBoxLayout(role_group)
        self.cmb_front = QComboBox()
        self.cmb_front.setMinimumWidth(140)
        self.cmb_side = QComboBox()
        self.cmb_side.setMinimumWidth(140)
        self.btn_save_roles = QPushButton("儲存角色")
        self.btn_save_roles.clicked.connect(self.save_roles)
        role_layout.addWidget(QLabel("正面 (primary):"))
        role_layout.addWidget(self.cmb_front)
        role_layout.addWidget(QLabel("側面 (secondary):"))
        role_layout.addWidget(self.cmb_side)
        role_layout.addWidget(self.btn_save_roles)
        role_layout.addStretch()
        layout.addWidget(role_group)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMinimumHeight(160)
        layout.addWidget(self.log_area)

    def log(self, msg):
        # 最新訊息插在最上方，擷取成功與否不必捲動就看得到；多行訊息維持原本順序
        cursor = self.log_area.textCursor()
        cursor.movePosition(cursor.Start)
        cursor.insertText(f"[{datetime.now():%H:%M:%S}] {msg}\n")
        self.log_area.moveCursor(cursor.Start)

    # ---------- 相機 ----------

    def refresh_cameras(self):
        self.cmb_camera.clear()
        try:
            cams = list_cameras()
        except Exception as e:
            self.log(f"掃描相機失敗: {e}")
            return

        if not cams:
            self.log("未偵測到任何相機。請確認相機已接上、且 SpinView 等程式已關閉。")
            return

        for sn, model in cams:
            self.cmb_camera.addItem(f"SN={sn}   {model}", userData=sn)
        self.log(f"偵測到 {len(cams)} 台相機。")
        if hasattr(self, "cmb_front"):
            self.refresh_role_combos()

    def connect_camera(self):
        if self.camera is not None:
            self.log("已有相機連接中，請先中斷連接。")
            return

        sn = self.cmb_camera.currentData()
        if not sn:
            self.log("請先選擇相機。")
            return

        try:
            self.camera = FlirCameraSystem(CONFIG_PATH, sn)
            self.connected_sn = sn
            self.log(f"已連接相機 SN={sn}")
            self.btn_connect.setEnabled(False)
            self.btn_disconnect.setEnabled(True)
            self.cmb_camera.setEnabled(False)
            self.reset_samples()
        except Exception as e:
            self.camera = None
            self.log(f"連接失敗: {e}")

    def disconnect_camera(self):
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception as e:
                self.log(f"釋放相機時發生問題: {e}")
            self.camera = None
        self.last_frame = None
        self.lbl_view.setText("已中斷連接")
        self.btn_connect.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
        self.cmb_camera.setEnabled(True)
        self.log("已中斷連接。")

    # ---------- 畫面 ----------

    def update_display(self):
        if self.camera is None:
            return

        try:
            ok, raw = self.camera.get_grayscale_image()
        except Exception as e:
            self.log(f"取得畫面失敗: {e}")
            return

        if not ok or raw is None:
            return

        # 相機輸出為 BayerRG8，轉成 BGR 才能顯示與偵測
        frame = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
        self.last_frame = frame
        if self.image_size is None:
            h, w = frame.shape[:2]
            self.image_size = (w, h)

        display = frame
        self._preview_tick += 1

        # findChessboardCorners 在 1920x1084 全解析度下很慢，會拖垮畫面更新。
        # 預覽只需確認「板子有沒有被看到」，因此縮圖偵測 + 每 N 幀才做一次。
        if self._preview_tick % self.PREVIEW_DETECT_EVERY == 0:
            pattern = (self.spn_cols.value(), self.spn_rows.value())
            small = cv2.resize(frame, None, fx=self.PREVIEW_SCALE, fy=self.PREVIEW_SCALE,
                               interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            found, corners_small = cv2.findChessboardCorners(
                gray_small, pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
            )
            self._preview_found = found
            # 角點座標換算回原始解析度，才能疊在全尺寸畫面上
            self._preview_corners = corners_small / self.PREVIEW_SCALE if found else None

        if self._preview_found and self._preview_corners is not None:
            display = frame.copy()
            pattern = (self.spn_cols.value(), self.spn_rows.value())
            cv2.drawChessboardCorners(display, pattern, self._preview_corners, True)

        self.lbl_view.setPixmap(self.to_pixmap(display))

    def to_pixmap(self, bgr):
        # 先用 OpenCV 縮到顯示尺寸再交給 Qt，避免每幀搬運整張 1920x1084 的資料
        target_w = max(self.lbl_view.width(), 1)
        target_h = max(self.lbl_view.height(), 1)
        h, w = bgr.shape[:2]
        scale = min(target_w / w, target_h / h)
        if scale < 1.0:
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_img.copy())

    # ---------- 校正 ----------

    def _board_spec(self):
        return (self.spn_cols.value(), self.spn_rows.value(), self.spn_square.value())

    @staticmethod
    def _find_sample_corners(frame, pattern):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not found:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)

    def choose_photo_folder(self):
        if self.camera is not None:
            self.log("請先中斷相機連接，再讀取照片資料夾。")
            return
        if not self.txt_photo_sn.text().strip():
            self.log("請先填入照片所屬的相機序號。")
            self.txt_photo_sn.setFocus()
            return
        folder = QFileDialog.getExistingDirectory(self, "選擇單台相機的棋盤格照片資料夾",
                                                   calib_store.CALIB_DIR)
        if folder:
            self.load_photo_folder(folder, self.txt_photo_sn.text().strip())

    def load_photo_folder(self, folder, serial_number):
        """Stage a photo batch; cancellation/failure leaves existing samples intact."""
        if self.camera is not None:
            self.log("請先中斷相機連接，再讀取照片資料夾。")
            return False
        sn = str(serial_number).strip()
        if not sn or not sn.isascii() or not sn.isalnum():
            self.log("相機序號請使用英文字母或數字。")
            return False
        try:
            files = sorted(p for p in Path(folder).iterdir()
                           if p.is_file() and p.suffix.lower() in
                           {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'})
        except OSError as exc:
            self.log(f"無法讀取照片資料夾：{exc}")
            return False
        if not files:
            self.log("資料夾內沒有照片；請選擇含 JPG／PNG／BMP／TIFF 的單台相機資料夾。")
            return False

        spec = self._board_spec()
        pattern, square = spec[:2], spec[2]
        points, paths, size, preview = [], [], None, None
        progress = QProgressDialog("正在讀取棋盤格照片…", "取消", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        try:
            for index, path in enumerate(files):
                progress.setValue(index)
                progress.setLabelText(f"偵測棋盤格 {index + 1}/{len(files)}：{path.name}")
                QApplication.processEvents()
                if progress.wasCanceled():
                    self.log("已取消讀取，保留原有樣本。")
                    return False
                try:
                    # imdecode/fromfile supports Chinese Windows paths.
                    frame = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("無法解碼影像")
                    current_size = (frame.shape[1], frame.shape[0])
                    if size is not None and current_size != size:
                        raise ValueError(f"影像尺寸 {current_size} 與本批 {size} 不同")
                    corners = self._find_sample_corners(frame, pattern)
                    if corners is None:
                        raise ValueError("未完整偵測到棋盤格")
                    if points:
                        shift = np.mean(np.linalg.norm(
                            corners.reshape(-1, 2) - points[-1].reshape(-1, 2), axis=1))
                        if shift < self.DUPLICATE_SHIFT_PX:
                            raise ValueError(f"與上一張有效照片幾乎相同（位移 {shift:.1f} px）")
                    points.append(corners)
                    paths.append(str(path.resolve()))
                    size, preview = current_size, frame
                except (OSError, ValueError, cv2.error) as exc:
                    self.log(f"略過 {path.name}：{exc}")
            QApplication.processEvents()
            if progress.wasCanceled():
                self.log("已取消讀取，保留原有樣本。")
                return False
        finally:
            progress.close()
        if not points:
            self.log("沒有可用的棋盤格照片，保留原有樣本；請確認內角點數量與照片。")
            return False

        objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
        self.objpoints = [objp.copy() for _ in points]
        self.imgpoints = points
        self.captured_count = len(points)
        self.image_size = size
        self.connected_sn = sn
        self._sample_spec = spec
        self._photo_paths = paths
        self.last_frame = preview
        self.txt_photo_sn.setText(sn)
        self.lbl_photo_source.setText(f"SN{sn}｜{len(points)}/{len(files)} 張有效照片")
        self.lbl_photo_source.setToolTip(str(Path(folder).resolve()))
        display = preview.copy()
        cv2.drawChessboardCorners(display, pattern, points[-1], True)
        self.lbl_view.setPixmap(self.to_pixmap(display))
        covered, _ = self._coverage()
        self.log(f"已載入 SN{sn}：{len(points)}/{len(files)} 張有效照片，尺寸 {size}，"
                 f"覆蓋 {covered}/9 區。已取代先前樣本；按「執行校正」計算內參。")
        return True

    def capture_sample(self):
        if self.camera is None or self.last_frame is None:
            self.log("尚未連接相機或畫面未就緒。")
            return

        pattern = (self.spn_cols.value(), self.spn_rows.value())
        square = self.spn_square.value()
        if self._sample_spec is not None and self._sample_spec != self._board_spec():
            self.log("棋盤格規格已變更，請恢復原規格或清空樣本後重新擷取。")
            return
        size = (self.last_frame.shape[1], self.last_frame.shape[0])
        if self.imgpoints and self.image_size != size:
            self.log("影像尺寸與既有樣本不同，請清空樣本後重新擷取。")
            return
        corners = self._find_sample_corners(self.last_frame, pattern)
        if corners is None:
            self.log("擷取失敗：未完整偵測到棋盤格。請調整角度/距離，確認整塊板子都在畫面內。")
            return

        # 板子沒移動就重複擷取，等於同一組資料算好幾次，會讓誤差假性偏低
        if self.imgpoints:
            previous = np.asarray(self.imgpoints[-1]).reshape(-1, 2)
            current = corners.reshape(-1, 2)
            if current.shape != previous.shape:
                self.log("未擷取：棋盤格角點數與既有樣本不同，請恢復原規格或清空樣本後重拍。")
                return
            shift = np.mean(np.linalg.norm(current - previous, axis=1))
            if shift < self.DUPLICATE_SHIFT_PX:
                self.log(f"未擷取：與上一組幾乎相同（平均位移 {shift:.1f} px），請移動或傾斜板子後再擷取。")
                return

        objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
        objp *= square

        self.objpoints.append(objp)
        self.imgpoints.append(corners)
        self.captured_count += 1
        self.image_size = size
        self._sample_spec = self._board_spec()

        covered, missing = self._coverage()
        msg = f"已擷取第 {self.captured_count} 組樣本｜畫面覆蓋 {covered}/9 區"
        if missing:
            msg += f"，尚缺：{'、'.join(missing)}"
        self.log(msg)

    # 品質判斷門檻
    RMS_GOOD = 0.5               # RMS 低於此值為良好
    RMS_MAX = 1.0                # RMS 高於此值判定不合格
    MIN_SAMPLES = 15             # 建議樣本數
    MIN_TILT_DEG = 20.0          # 至少要有一組傾斜超過此角度
    DUPLICATE_SHIFT_PX = 5.0     # 與上一組平均位移小於此值視為重複
    PIXEL_SIZE_MM = 0.00586      # GS3-U3-23S6C 像素大小 5.86 µm，用來把鏡頭焦距換算成 fx
    TELEPHOTO_MM = 25.0          # 焦距達此值自動勾選「鎖住畸變」
    FX_TOLERANCE = 0.15          # fx 與焦距推算值相差超過 15% 就警告
    AXIS_MIN_TILT_DEG = 10.0     # 傾斜超過此角度的樣本才拿來看「往哪個方向轉」
    AXIS_CONCENTRATION = 0.8     # 轉軸方向集中度超過此值（1＝全部同一軸）就警告
    REGION_NAMES = ["左上", "上", "右上", "左", "中", "右", "左下", "下", "右下"]

    def _coverage(self):
        """把畫面切成 3x3 九宮格，回傳 (已覆蓋區數, 未覆蓋的區名)"""
        if self.image_size is None or not self.imgpoints:
            return 0, list(self.REGION_NAMES)
        w, h = self.image_size
        pts = np.vstack([c.reshape(-1, 2) for c in self.imgpoints])
        col = np.clip((pts[:, 0] / w * 3).astype(int), 0, 2)
        row = np.clip((pts[:, 1] / h * 3).astype(int), 0, 2)
        hit = set((row * 3 + col).tolist())
        missing = [name for i, name in enumerate(self.REGION_NAMES) if i not in hit]
        return len(hit), missing

    def _calibrate(self):
        """執行校正並計算每組樣本的誤差與傾斜角"""
        flags = 0
        if self.chk_lock_dist.isChecked():
            # 長焦鏡頭視角窄、幾乎沒有畸變，照片裡沒有足夠資訊估畸變，
            # 放著不鎖會估出不合理的係數（2026-10-08 正面 50 mm 的 k2 = 33.5），去畸變時反而把影像扭曲
            flags = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                     | cv2.CALIB_ZERO_TANGENT_DIST)
        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.image_size, None, None, flags=flags
        )
        per_image_rms = []
        tilts = []
        self._tilt_axes = []     # 每組板子的傾斜轉軸方向（度，影像平面內）
        for i in range(len(self.objpoints)):
            proj, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            d = np.linalg.norm(self.imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
            per_image_rms.append(float(np.sqrt(np.mean(d ** 2))))
            # 板子法向量與相機光軸的夾角
            R, _ = cv2.Rodrigues(rvecs[i])
            tilts.append(float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))))
            # 板子是繞哪個軸轉離正對相機：轉軸 = 光軸 × 板子法向量（落在影像平面內）
            normal = R[:, 2] * np.sign(R[2, 2] or 1.0)
            self._tilt_axes.append(float(np.degrees(np.arctan2(normal[0], -normal[1]))))
        return rms, mtx, dist, np.array(per_image_rms), np.array(tilts)

    def _assess(self, rms, per_image_rms, tilts, mtx=None):
        """依門檻判斷品質，回傳 (結論, 問題清單, 異常樣本索引)"""
        problems = []
        # RMS 小不代表 fx 對：長焦時 fx 與距離可以互相抵消，RMS 照樣很小，所以另外用鏡頭焦距檢查
        focal = self.spn_focal.value()
        if mtx is not None and focal > 0:
            expected = focal / self.PIXEL_SIZE_MM
            ratio = mtx[0, 0] / expected
            if abs(ratio - 1) > self.FX_TOLERANCE:
                problems.append(f"fx={mtx[0, 0]:.0f}，與 {focal:g} mm 鏡頭推算的 {expected:.0f} 相差 "
                                f"{(ratio - 1) * 100:+.0f}%：請確認鏡頭焦距，並讓板子往上下、左右不同方向傾斜 30° 以上（只朝同一方向轉會讓 fx 算不準）")
        bad = [i for i, e in enumerate(per_image_rms)
               if e > self.RMS_MAX and e > 2 * rms]

        if rms > self.RMS_MAX:
            verdict = "不合格，建議重拍"
            problems.append(f"整體誤差 {rms:.3f} px 超過 {self.RMS_MAX} px")
        elif rms > self.RMS_GOOD:
            verdict = "可用，但建議補拍改善"
            problems.append(f"整體誤差 {rms:.3f} px 介於 {self.RMS_GOOD}～{self.RMS_MAX} px")
        else:
            verdict = "良好"

        if bad:
            problems.append(f"{len(bad)} 組樣本誤差異常（第 {', '.join(str(i + 1) for i in bad)} 組），"
                            "可能是模糊或晃動")
        covered, missing = self._coverage()
        if missing:
            problems.append(f"畫面覆蓋 {covered}/9 區，缺：{'、'.join(missing)}（邊角沒拍到，畸變會校不準）")
        axis_concentration = self._axis_concentration(tilts)
        if axis_concentration is not None and axis_concentration > self.AXIS_CONCENTRATION:
            problems.append(f"板子幾乎都朝同一個方向轉（集中度 {axis_concentration:.2f}），fx 可能算不準："
                            "請讓板子上下、左右、斜向都傾斜（不要為了讓另一台相機看到而只朝一個方向轉）")
        if tilts.max() < self.MIN_TILT_DEG:
            problems.append(f"最大傾斜僅 {tilts.max():.0f}°，請加入傾斜 {self.MIN_TILT_DEG:.0f}° 以上的樣本")
        if self.captured_count < self.MIN_SAMPLES:
            problems.append(f"樣本僅 {self.captured_count} 組，建議至少 {self.MIN_SAMPLES} 組")

        if verdict == "良好" and problems:
            verdict = "良好，但有改善空間"
        return verdict, problems, bad

    def _axis_concentration(self, tilts):
        """傾斜轉軸方向的集中度（0～1）。轉軸沒有正負之分，所以角度先乘 2 再取平均向量長度；
        只朝同一方向轉（轉軸互相平行）時接近 1，這時焦距與距離分不開，fx 會算不準。"""
        axes = np.radians([a for a, t in zip(getattr(self, "_tilt_axes", []), tilts) if t >= self.AXIS_MIN_TILT_DEG])
        if len(axes) < 3:
            return None
        return float(np.hypot(np.mean(np.cos(2 * axes)), np.mean(np.sin(2 * axes))))

    def run_calibration(self):
        if self._sample_spec is not None and self._sample_spec != self._board_spec():
            self.log("棋盤格規格與樣本不符，請恢復原規格或重新讀取／擷取照片。")
            return
        if self.captured_count < 10:
            self.log(f"樣本僅 {self.captured_count} 組，建議至少 10 組不同角度再校正。")
            return
        if self.image_size is None:
            self.log("尚未取得影像尺寸。")
            return

        self.log("開始計算校正參數...")
        QApplication.processEvents()

        rms, mtx, dist, per_image_rms, tilts = self._calibrate()
        verdict, problems, bad = self._assess(rms, per_image_rms, tilts, mtx)

        # 有異常樣本時，詢問是否剔除後重算
        if bad and self.captured_count - len(bad) >= 10:
            reply = QMessageBox.question(
                self, "發現異常樣本",
                f"第 {', '.join(str(i + 1) for i in bad)} 組樣本誤差明顯偏高"
                f"（{', '.join(f'{per_image_rms[i]:.2f}' for i in bad)} px），"
                "可能是拍攝時模糊或晃動。\n\n要剔除這些樣本並重新計算嗎？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                for i in sorted(bad, reverse=True):
                    del self.objpoints[i]
                    del self.imgpoints[i]
                    if self._photo_paths:
                        del self._photo_paths[i]
                self.captured_count = len(self.imgpoints)
                self.log(f"已剔除 {len(bad)} 組異常樣本，剩 {self.captured_count} 組，重新計算...")
                QApplication.processEvents()
                rms, mtx, dist, per_image_rms, tilts = self._calibrate()
                verdict, problems, bad = self._assess(rms, per_image_rms, tilts, mtx)

        # 結果合成一則訊息，避免「最新在上」時多行順序顛倒
        report = (
            f"校正完成 SN={self.connected_sn}｜判定：{verdict}\n"
            f"  重投影誤差 RMS: {rms:.4f} px  (< {self.RMS_GOOD} 良好，> {self.RMS_MAX} 須重拍)\n"
            f"  單組誤差範圍: {per_image_rms.min():.3f} ~ {per_image_rms.max():.3f} px\n"
            f"  板子傾斜範圍: {tilts.min():.0f}° ~ {tilts.max():.0f}°\n"
            f"  fx={mtx[0,0]:.2f}  fy={mtx[1,1]:.2f}\n"
            f"  cx={mtx[0,2]:.2f}  cy={mtx[1,2]:.2f}\n"
            f"  畸變係數: {np.round(dist.ravel(), 6)}"
        )
        if problems:
            report += "\n  待改善：\n" + "\n".join(f"   - {p}" for p in problems)
        self.log(report)

        # 不合格時先確認，避免蓋掉之前較好的結果
        if rms > self.RMS_MAX:
            reply = QMessageBox.warning(
                self, "校正不合格",
                f"重投影誤差 {rms:.3f} px，超過 {self.RMS_MAX} px。\n\n"
                + "\n".join(f"・{p}" for p in problems)
                + "\n\n可能原因：棋盤格規格填錯、板子不平、樣本模糊。\n"
                "建議「清空樣本」後重拍。\n\n仍要儲存這次結果嗎？（會覆蓋這台相機之前的校正）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                self.log("未儲存本次結果。")
                return
        elif problems:
            QMessageBox.information(
                self, f"校正結果：{verdict}",
                f"重投影誤差 {rms:.3f} px。結果已儲存，但以下項目可以改善：\n\n"
                + "\n".join(f"・{p}" for p in problems))

        mean_err = float(np.mean(per_image_rms))
        os.makedirs(calib_store.INTRINSIC_DIR, exist_ok=True)
        out_name = os.path.join(calib_store.INTRINSIC_DIR, f"calib_SN{self.connected_sn}.npz")
        np.savez(
            out_name,
            mtx=mtx, dist=dist,
            image_size=np.array(self.image_size),
            rms=rms, mean_reproj_err=mean_err,
            per_image_rms=per_image_rms, tilt_deg=tilts,
            pattern=np.array([self.spn_cols.value(), self.spn_rows.value()]),
            square_size=self.spn_square.value(),
            serial_number=self.connected_sn,
            sample_count=self.captured_count,
            source_images=np.asarray(self._photo_paths, dtype=str),
        )
        self.log(f"已儲存至 {out_name}")

        self._append_to_json(mtx, dist, rms, mean_err, verdict)

    def _append_to_json(self, mtx, dist, rms, mean_err, verdict):
        """把這台相機的內參累積寫入 intrinsics_all.json（以序號為鍵）。
        3D 計算時再依 camera_roles.json 取出正面/側面各自的內參。
        """
        entry = {
            "intrinsic_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "fx": float(mtx[0, 0]),
            "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]),
            "cy": float(mtx[1, 2]),
            "image_size": list(self.image_size),
            "rms": float(rms),
            "mean_reproj_err": float(mean_err),
            "verdict": verdict,
            "pattern": [self.spn_cols.value(), self.spn_rows.value()],
            "square_size_mm": self.spn_square.value(),
            "sample_count": self.captured_count,
            "distortion_locked": self.chk_lock_dist.isChecked(),
            "lens_focal_mm": self.spn_focal.value() or None,
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
        }
        if self._photo_paths:
            entry["source_images"] = list(self._photo_paths)

        try:
            count = calib_store.save_intrinsic(self.connected_sn, entry)
        except Exception as e:
            self.log(f"寫入 {calib_store.INTRINSICS_JSON} 失敗：{e}")
            return

        self.log(f"已累積寫入 {calib_store.INTRINSICS_JSON}（目前共 {count} 台）")
        self.refresh_role_combos()

    def refresh_role_combos(self):
        """更新正面/側面下拉選單：可選「已校正」或「目前連接」的相機，並預選現有角色"""
        try:
            sns = list(calib_store.load_intrinsics().keys())
        except Exception:
            sns = []
        for i in range(self.cmb_camera.count()):
            sn = self.cmb_camera.itemData(i)
            if sn and sn not in sns:
                sns.append(sn)

        roles = calib_store.load_roles()
        for cmb, role_idx in ((self.cmb_front, 0), (self.cmb_side, 1)):
            current = cmb.currentText() or (roles[role_idx] if roles else "")
            cmb.clear()
            cmb.addItems(sns)
            idx = cmb.findText(current)
            if idx >= 0:
                cmb.setCurrentIndex(idx)

    def save_roles(self):
        """寫入 camera_roles.json：拍外參照片時決定 primary，3D 計算時決定內參"""
        front_sn = self.cmb_front.currentText()
        side_sn = self.cmb_side.currentText()

        if not front_sn or not side_sn:
            self.log("請先選擇正面與側面相機。")
            return
        if front_sn == side_sn:
            self.log("正面與側面不能是同一台相機。")
            return

        calib_store.save_roles(front_sn, side_sn)
        intr = calib_store.load_intrinsics()
        missing = [sn for sn in (front_sn, side_sn) if sn not in intr]
        msg = (f"已儲存角色至 {calib_store.ROLES_JSON}\n"
               f"  正面 = SN{front_sn}\n"
               f"  側面 = SN{side_sn}")
        if missing:
            msg += f"\n  注意：{', '.join('SN' + s for s in missing)} 尚未做內參校正"
        self.log(msg)

    def reset_samples(self):
        self.objpoints = []
        self.imgpoints = []
        self.captured_count = 0
        self.image_size = None
        self._sample_spec = None
        self._photo_paths = []
        self.lbl_photo_source.setText("尚未載入照片")
        self.lbl_photo_source.setToolTip("")
        self.log("已清空樣本。")

    # ---------- 其他 ----------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.capture_sample()
        elif event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.timer.stop()
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = CalibWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
