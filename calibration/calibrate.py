"""單一入口：兩台棋盤格內參 → 棋盤格雙機同步拍照 → 自動偵測角點 → 外參（R、t，mm）。

啟動：python calibration/calibrate.py（可從任何工作目錄執行）
相機固定後，依畫面指示移動棋盤格；外參照片須兩台同時完整看到棋盤格才會存檔。
本工具不載入 YOLO / ViTPose。calib.py 仍可單獨用來校正一台相機的內參。
"""

# Support both python -m and direct file execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime

from paths import CAMERA_CONFIG_DIR, DB_DIR
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
qt_plugins = Path(os.environ.get("CONDA_PREFIX", sys.prefix)) / "Library/plugins/platforms"
if qt_plugins.is_dir():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins))

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QShortcut, QSpinBox,
    QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

import calib_store
from calibration_workflow import (
    MAX_STEREO_RMS, MIN_PAIRS, RECOMMENDED_PAIRS, estimate_chessboard_stereo, match_image_pairs,
    publish_intrinsics, publish_stereo, save_json, validate_intrinsic, validate_roles,
)


def read_image(file):
    image = cv2.imdecode(np.fromfile(str(file), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("無法讀取影像：{}".format(file))
    return image


def write_image(file, image):
    ok, encoded = cv2.imencode(Path(file).suffix, image)
    if not ok:
        raise OSError("無法編碼影像：{}".format(file))
    encoded.tofile(str(file))


class Job(QThread):
    """Run OpenCV calculations without blocking the Qt event loop."""
    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.function()
        except Exception as exc:
            self.error = str(exc)


class CaptureWorker(QThread):
    """Own both cameras in one worker; release them before the next step."""
    failed = pyqtSignal(str)

    def __init__(self, roles, parent):
        super().__init__(parent)
        self.roles = roles
        self.latest = None

    def run(self):
        cameras = []
        try:
            from camera_objects import DualFlirSystem, FlirCameraSystem
            for sn, config in zip(self.roles, ("GH3_camera_config.yaml", "GH3_camera_config_2.yaml")):
                camera = FlirCameraSystem(str(CAMERA_CONFIG_DIR / config), sn)
                cameras.append(camera)
            pair = DualFlirSystem(*cameras)
            failures = 0
            while not self.isInterruptionRequested():
                ok, front, side = pair.get_grayscale_images()
                if not ok:
                    failures += 1
                    if failures >= 3:
                        raise RuntimeError("雙機取像失敗，請確認 GPIO 同步線、USB 連接及其他程式是否占用相機。")
                    continue
                failures = 0
                # Assignment is atomic; the UI reads both images from one pair.
                self.latest = (time.monotonic(), cv2.cvtColor(front, cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(side, cv2.COLOR_GRAY2BGR))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.latest = None
            for camera in reversed(cameras):
                try:
                    camera.release()
                except Exception as exc:
                    self.failed.emit("釋放相機失敗：{}".format(exc))


class ImageView(QGraphicsView):
    """顯示相機畫面；滾輪縮放、拖曳平移，換畫面時保留目前的縮放位置。"""

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(300, 230)
        self.setBackgroundBrush(QColor("#20252b"))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.image_shape = None

    def set_image(self, image):
        self.scene().clear()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self.scene().addPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(0, 0, width, height)
        # 預覽每 100 ms 換一張，只在第一張或尺寸改變時縮放至完整畫面，才能放大檢查角點
        if self.image_shape != (height, width):
            self.image_shape = (height, width)
            self.fit_image()

    def fit_image(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def mouseDoubleClickEvent(self, event):
        self.fit_image()

    def wheelEvent(self, event):
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.05 < self.transform().m11() * factor < 20:
            self.scale(factor, factor)


class CalibrationWindow(QMainWindow):
    # 外參拍照預覽：縮圖偵測 + 每 N 次更新偵測一次，避免全解析度偵測拖慢畫面
    PREVIEW_SCALE = 0.4
    PREVIEW_DETECT_EVERY = 3

    def __init__(self):
        super().__init__()
        self.setWindowTitle("相機校正精靈 — 棋盤格內參 / 外參")
        self.resize(1200, 900)
        self.state = "setup"
        self.session = None
        self.roles = None
        self.entries = {}
        self.names = []
        self.preview_tick = 0
        self.preview_found = {"cf": False, "cs": False}
        self.preview_corners = {"cf": None, "cs": None}
        self.panel = None
        self.capture = None
        self.job = None
        self.after_capture = None
        self.last_capture_time = None
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.step_label = QLabel("1 設定 → 2 正面內參 → 3 側面內參 → 4 雙機拍棋盤格 → 5 計算外參")
        self.step_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px;")
        layout.addWidget(self.step_label)
        self.instructions = QLabel()
        self.instructions.setWordWrap(True)
        layout.addWidget(self.instructions)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.make_setup_page()
        self.make_pair_page()
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(140)
        layout.addWidget(self.log_area)
        actions = QHBoxLayout()
        self.cancel_button = QPushButton("取消本次校正")
        self.cancel_button.clicked.connect(self.cancel)
        self.next_button = QPushButton("開始校正")
        self.next_button.setMinimumHeight(42)
        self.next_button.clicked.connect(self.advance)
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        actions.addWidget(self.next_button)
        layout.addLayout(actions)
        self.space = QShortcut(QKeySequence("Space"), self)
        self.space.activated.connect(self.capture_sample)
        self.preview = QTimer(self)
        self.preview.timeout.connect(self.update_preview)
        self.preview.start(100)
        self.reset_ui()
        self.scan_cameras()

    def make_setup_page(self):
        self.setup_page = QWidget()
        layout = QVBoxLayout(self.setup_page)
        box = QGroupBox("相機與棋盤格設定")
        form = QFormLayout(box)
        self.front_combo, self.side_combo = QComboBox(), QComboBox()
        for combo in (self.front_combo, self.side_combo):
            combo.setEditable(True)
        form.addRow("正面相機序號（primary）", self.front_combo)
        form.addRow("側面相機序號（secondary）", self.side_combo)
        self.scan_button = QPushButton("重新掃描相機")
        self.scan_button.clicked.connect(self.scan_cameras)
        form.addRow(self.scan_button)
        self.scan_status = QLabel()
        self.scan_status.setWordWrap(True)
        form.addRow(self.scan_status)
        self.reuse = QCheckBox("沿用這兩台已有內參，只重做外參（鏡頭、焦距、解析度須相同）")
        form.addRow(self.reuse)
        self.intrinsic_only = QCheckBox("只做內參（相機還沒架到球場位置時使用，外參之後再做）")
        form.addRow(self.intrinsic_only)
        self.cols, self.rows = QSpinBox(), QSpinBox()
        for spin, value in ((self.cols, 8), (self.rows, 7)):
            spin.setRange(3, 30)
            spin.setValue(value)
        self.square = QDoubleSpinBox()
        self.square.setRange(0.1, 1000)
        self.square.setValue(100)
        self.square.setSuffix(" mm")
        form.addRow("棋盤格內角點：寬（格子數 − 1）", self.cols)
        form.addRow("棋盤格內角點：高（格子數 − 1）", self.rows)
        form.addRow("每格實際邊長", self.square)
        # 鏡頭焦距決定內參初值與長焦鎖定（>= calib.CalibWindow.TELEPHOTO_MM 自動鎖光心與高階畸變）
        self.front_focal, self.side_focal = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, value in ((self.front_focal, 35), (self.side_focal, 8)):
            spin.setRange(0, 500)
            spin.setDecimals(1)
            spin.setSpecialValueText("未填")
            spin.setSuffix(" mm")
            spin.setValue(value)
        form.addRow("正面鏡頭焦距", self.front_focal)
        form.addRow("側面鏡頭焦距", self.side_focal)
        layout.addWidget(box)
        note = QLabel("準備：固定兩台相機與焦距，關閉 main.py / SpinView 等占用相機的程式。\n"
                      "內參：每台拍至少 15 組不同位置、距離及傾斜角度的棋盤格，涵蓋畫面四角。\n"
                      f"外參：接 GPIO 同步線，拍 {RECOMMENDED_PAIRS} 組以上兩台都完整看到的棋盤格"
                      f"（至少 {MIN_PAIRS} 組），涵蓋打擊區不同位置與深度。\n"
                      "照片可在同一視窗拍攝，也可匯入含 cf / cs 的既有照片資料夾。\n"
                      "完成後才更新目前校正值；每次資料獨立保存，供日後方法比較。")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch()
        self.pages.addWidget(self.setup_page)

    def make_pair_page(self):
        self.pair_page = QWidget()
        layout = QVBoxLayout(self.pair_page)
        images = QHBoxLayout()
        self.views = {}
        self.view_titles = {}
        for role, title in (("cf", "正面"), ("cs", "側面")):
            column = QVBoxLayout()
            label = QLabel(title)
            column.addWidget(label)
            self.view_titles[role] = label
            view = ImageView()
            column.addWidget(view)
            self.views[role] = view
            images.addLayout(column)
        layout.addLayout(images, 1)
        self.pair_status = QLabel()
        layout.addWidget(self.pair_status)
        self.capture_actions = QWidget()
        buttons = QHBoxLayout(self.capture_actions)
        self.connect_button = QPushButton("連接雙相機")
        self.connect_button.clicked.connect(self.connect_pair)
        self.shot_button = QPushButton("拍一組（空白鍵）")
        self.shot_button.clicked.connect(self.capture_pair)
        self.import_button = QPushButton("匯入 cf / cs 照片資料夾")
        self.import_button.clicked.connect(self.import_pairs)
        for button in (self.connect_button, self.shot_button, self.import_button):
            buttons.addWidget(button)
        layout.addWidget(self.capture_actions)
        self.pages.addWidget(self.pair_page)

    def log(self, message):
        self.log_area.append("[{}] {}".format(datetime.now().strftime("%H:%M:%S"), message))

    def fail(self, message):
        self.log(message)
        QMessageBox.warning(self, "尚未完成校正", message)

    def scan_cameras(self):
        saved, serials = None, set()
        try:
            saved = calib_store.load_roles()
            if saved:
                serials.update(saved)
            serials.update(calib_store.load_intrinsics())
        except Exception as exc:
            self.log("既有校正檔無法讀取：{}".format(exc))
        connected = []
        try:
            from calib import list_cameras
            connected = sorted({sn for sn, _ in list_cameras()})
        except Exception as exc:
            message = "無法掃描相機；可輸入序號並使用既有內參與照片：{}".format(exc)
        else:
            if connected:
                message = "掃描完成：偵測到 {} 台相機（{}）。".format(len(connected), "、".join(connected))
            else:
                message = "未偵測到相機，請確認 USB 連線並關閉其他相機程式後重新掃描。"
        self.scan_status.setText(message)
        self.log(message)

        # Connected devices come first; stored serials remain available for imports.
        choices = connected + sorted(serials.difference(connected))
        selected = [combo.currentText().strip() or (saved[i] if saved else "")
                    for i, combo in enumerate((self.front_combo, self.side_combo))]
        if connected and not self.reuse.isChecked():
            # A fresh capture must not default to a disconnected, historical rig.
            selected = [sn if sn in connected else "" for sn in selected]
            if selected[0] == selected[1]:
                selected[1] = ""
            defaults = connected
        else:
            defaults = choices
        for i in range(len(selected)):
            if not selected[i]:
                selected[i] = next((sn for sn in defaults if sn not in selected), "")
        for i, combo in enumerate((self.front_combo, self.side_combo)):
            combo.clear()
            combo.addItems(choices)
            combo.setCurrentIndex(-1)
            combo.setCurrentText(selected[i])

    def reset_ui(self):
        self.state = "setup"
        self.pages.setCurrentWidget(self.setup_page)
        self.setup_page.setEnabled(True)
        self.next_button.setText("開始校正")
        self.next_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.instructions.setText("選擇正面、側面相機及棋盤格規格後，按「開始校正」。")

    def advance(self):
        try:
            if self.state == "setup":
                self.start_session()
            elif self.state.startswith("intrinsic"):
                self.calculate_intrinsic()
            elif self.state == "capture":
                if len(self.names) < MIN_PAIRS:
                    raise ValueError("至少拍攝或匯入 {} 組雙機照片。".format(MIN_PAIRS))
                self.stop_capture(self.calculate_extrinsic)
            elif self.state == "done":
                self.reset_ui()
        except Exception as exc:
            self.fail(str(exc))

    def start_session(self):
        roles = (self.front_combo.currentText().strip(), self.side_combo.currentText().strip())
        validate_roles(*roles)
        if self.reuse.isChecked() and self.intrinsic_only.isChecked():
            raise ValueError("「沿用已有內參」與「只做內參」只能擇一。")
        entries = {}
        if self.reuse.isChecked():
            stored = calib_store.load_intrinsics()
            for sn in roles:
                if sn not in stored:
                    raise ValueError("SN{} 尚無內參，請取消「沿用已有內參」。".format(sn))
                validate_intrinsic(stored[sn])
                entries[sn] = stored[sn]
        self.roles, self.entries = roles, entries
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session = Path(calib_store.CALIB_DIR) / "sessions" / (stamp + "_chessboard")
        for folder in ("cf", "cs", "intrinsic"):
            (self.session / folder).mkdir(parents=True, exist_ok=True)
        self.names = []
        self.last_capture_time = None
        self.persist_session("started")
        self.setup_page.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.log("本次資料：{}".format(self.session))
        if entries:
            self.begin_capture()
        else:
            self.begin_intrinsic(0)

    def persist_session(self, status):
        save_json(self.session / "session.json", {
            "method": "intrinsic_only" if self.intrinsic_only.isChecked() else "chessboard_stereo_pnp",
            "front": self.roles[0], "side": self.roles[1],
            "status": status, "intrinsics": self.entries,
            "pattern": [self.cols.value(), self.rows.value()], "square_size_mm": self.square.value(),
            "lens_focal_mm": {"front": self.front_focal.value() or None,
                              "side": self.side_focal.value() or None},
            "image_names": self.names,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

    def begin_intrinsic(self, role_index):
        import calib
        sample_dir = self.session / "intrinsic" / ("images_SN" + self.roles[role_index])
        sample_dir.mkdir(parents=True, exist_ok=True)

        class Collector(calib.CalibWindow):
            def __init__(collector):
                collector.sample_files = []
                super().__init__()

            def reset_samples(collector):
                super().reset_samples()
                collector.sample_files = []

            def capture_sample(collector):
                count = collector.captured_count
                super().capture_sample()
                if collector.captured_count > count:
                    try:
                        # Keep originals for later method comparison/recalibration.
                        stamp = datetime.now().strftime("%H%M%S_%f")
                        filename = stamp + ".png"
                        write_image(sample_dir / filename, collector.last_frame)
                        collector.sample_files.append(filename)
                    except Exception as exc:
                        collector.objpoints.pop()
                        collector.imgpoints.pop()
                        collector.captured_count -= 1
                        collector.log("未接受本組：原始照片儲存失敗 {}".format(exc))

            def keyPressEvent(collector, event):
                # Window-wide Space shortcut is handled by the wizard.
                if event.key() != Qt.Key_Escape:
                    super().keyPressEvent(event)

        # The legacy collector is reused; only this new process's config is changed.
        calib.CONFIG_PATH = str(CAMERA_CONFIG_DIR /
                                ("GH3_camera_config.yaml" if role_index == 0 else "GH3_camera_config_2.yaml"))
        self.state = "intrinsic_{}".format(role_index)
        self.panel = Collector()
        self.panel.timer.stop()
        self.panel.lbl_view.setMinimumHeight(280)
        self.panel.log_area.setMinimumHeight(80)
        self.panel.log_area.setMaximumHeight(120)
        self.panel.cmb_camera.parentWidget().hide()
        self.panel.cmb_front.parentWidget().hide()
        self.panel.photo_group.hide()
        self.panel.btn_calibrate.hide()
        self.panel.spn_cols.setValue(self.cols.value())
        self.panel.spn_rows.setValue(self.rows.value())
        self.panel.spn_square.setValue(self.square.value())
        self.panel.spn_cols.parentWidget().setEnabled(False)
        # 設定焦距會連動勾選長焦鎖定；面板上仍可手動改勾
        focal = (self.front_focal if role_index == 0 else self.side_focal).value()
        self.panel.spn_focal.setValue(focal)
        sn = self.roles[role_index]
        index = self.panel.cmb_camera.findData(sn)
        if index < 0:
            self.dispose_panel()
            self.reset_ui()
            raise ValueError("未偵測到 SN{}，請接上相機後重新開始。".format(sn))
        self.panel.cmb_camera.setCurrentIndex(index)
        self.panel.connect_camera()
        if self.panel.camera is None:
            self.dispose_panel()
            self.reset_ui()
            raise ValueError("無法連接 SN{}，請關閉其他相機程式後重試。".format(sn))
        self.pages.addWidget(self.panel)
        self.pages.setCurrentWidget(self.panel)
        self.panel.timer.start(60)
        title = "正面" if role_index == 0 else "側面"
        self.instructions.setText("{} SN{}：移動、傾斜棋盤格，按空白鍵擷取至少 15 組；完成後按右下角計算。"
                                  .format(title, sn))
        self.next_button.setText("計算{}內參並繼續".format(title))

    def dispose_panel(self):
        if self.panel is not None:
            self.panel.timer.stop()
            self.panel.disconnect_camera()
            self.pages.removeWidget(self.panel)
            self.panel.deleteLater()
            self.panel = None

    def start_job(self, function, success, failure=None):
        self.next_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.pages.setEnabled(False)
        job = Job(function, self)
        self.job = job

        def finished():
            self.job = None
            self.next_button.setEnabled(True)
            self.cancel_button.setEnabled(True)
            self.pages.setEnabled(True)
            if self.panel is not None:
                self.panel.timer.start(60)
            try:
                if job.error:
                    raise ValueError(job.error)
                success(job.result)
            except Exception as exc:
                (failure or self.fail)(str(exc))
            job.deleteLater()

        job.finished.connect(finished)
        job.start()

    def calculate_intrinsic(self):
        if self.panel.captured_count < 15:
            raise ValueError("目前 {} 組，請擷取至少 15 組不同角度的棋盤格。".format(self.panel.captured_count))
        self.panel.timer.stop()
        self.log("正在計算內參…")
        self.start_job(self.panel._calibrate, self.intrinsic_finished)

    def intrinsic_finished(self, result):
        rms, K, dist, errors, tilts = result
        panel = self.panel
        verdict, problems, _ = panel._assess(rms, errors, tilts, K)
        if not np.isfinite(rms) or rms > panel.RMS_MAX:
            raise ValueError("內參 RMS {:.3f} px，須低於或等於 {} px。請清空樣本後改善光線、清晰度再拍。"
                             .format(rms, panel.RMS_MAX))
        role_index = int(self.state.rsplit("_", 1)[1])
        sn = self.roles[role_index]
        entry = {
            "intrinsic_matrix": K.tolist(), "distortion_coefficients": dist.tolist(),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "image_size": list(panel.image_size), "rms": float(rms),
            "mean_reproj_err": float(np.mean(errors)), "verdict": verdict,
            "pattern": [self.cols.value(), self.rows.value()], "square_size_mm": self.square.value(),
            "sample_count": panel.captured_count, "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "sample_images": ["intrinsic/images_SN{}/{}".format(sn, name) for name in panel.sample_files],
            "distortion_locked": panel.chk_lock_dist.isChecked(), "lens_focal_mm": panel.spn_focal.value() or None,
            "calibration_flags": int(panel._last_flags),
        }
        validate_intrinsic(entry)
        np.savez(str(self.session / "intrinsic" / ("calib_SN{}.npz".format(sn))),
                 mtx=K, dist=dist, image_size=panel.image_size, rms=rms, per_image_rms=errors,
                 mean_reproj_err=np.mean(errors), tilt_deg=tilts, pattern=entry["pattern"],
                 square_size=self.square.value(), serial_number=sn, sample_count=panel.captured_count,
                 objpoints=np.asarray(panel.objpoints), imgpoints=np.asarray(panel.imgpoints),
                 sample_images=np.asarray(entry["sample_images"]))
        self.entries[sn] = entry
        self.persist_session("intrinsic_{}_complete".format(role_index))
        self.log("SN{} 內參完成，RMS {:.3f} px。{}".format(sn, rms, "；".join(problems)))
        self.dispose_panel()
        if role_index == 0:
            self.begin_intrinsic(1)
        elif self.intrinsic_only.isChecked():
            self.finish_intrinsic_only()
        else:
            self.begin_capture()

    def finish_intrinsic_only(self):
        """兩台內參完成就發布（內參＋正面／側面角色），不拍外參；相機架好後再勾「沿用已有內參」做外參。"""
        publish_intrinsics(calib_store.CALIB_DIR, self.session, self.roles, self.entries,
                           datetime.now().isoformat(timespec="seconds"))
        self.state = "done"
        try:
            self.persist_session("complete")
        except OSError as exc:
            self.log("內參已套用，但工作階段狀態無法更新：{}".format(exc))
        self.instructions.setText("內參完成，已更新 Db/Calibration。相機架好後，勾選「沿用已有內參」再做一次外參。")
        self.log("兩台內參與正面／側面角色已更新；外參（stereo.json）沒有改動，若相機有移動請重新做外參。")
        self.log("照片與內參樣本：{}".format(self.session))
        self.next_button.setText("開始另一次校正")
        self.cancel_button.setEnabled(False)

    def begin_capture(self):
        self.state = "capture"
        self.pages.setCurrentWidget(self.pair_page)
        self.capture_actions.show()
        self.connect_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.shot_button.setEnabled(False)
        self.next_button.setText("計算外參並儲存")
        self.instructions.setText(f"接好 GPIO 同步線後連接雙相機。棋盤格每個位置停穩、兩台畫面都出現彩色角點後按空白鍵；"
                                  f"建議 {RECOMMENDED_PAIRS} 組以上（至少 {MIN_PAIRS} 組），"
                                  "涵蓋打擊區的左右、前後、高低，並讓板子朝不同方向傾斜。")
        self.pair_status.setText("已拍 0 組；也可匯入既有 cf / cs 照片。")
        self.preview_found = {"cf": False, "cs": False}
        self.preview_corners = {"cf": None, "cs": None}
        for role, view in self.views.items():
            view.scene().clear()
            view.image_shape = None
        self.update_view_titles()

    def update_view_titles(self):
        for role, label in self.view_titles.items():
            title = ("正面 SN" if role == "cf" else "側面 SN") + self.roles[0 if role == "cf" else 1]
            found = self.preview_found[role]
            label.setText("{}　{}".format(title, "✔ 偵測到棋盤格" if found else "✘ 未偵測到棋盤格"))
            label.setStyleSheet("color: {}; font-weight: bold;".format("#15803d" if found else "#b91c1c"))

    def detect_preview(self, image):
        pattern = (self.cols.value(), self.rows.value())
        small = cv2.resize(image, None, fx=self.PREVIEW_SCALE, fy=self.PREVIEW_SCALE,
                           interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        return found, (corners / self.PREVIEW_SCALE if found else None)

    def connect_pair(self):
        if self.capture is not None:
            return
        self.connect_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.capture = CaptureWorker(self.roles, self)
        self.capture.failed.connect(self.log)
        self.capture.finished.connect(self.capture_stopped)
        self.capture.start()
        self.log("正在連接雙相機…")

    def update_preview(self):
        if self.state != "capture" or self.capture is None or self.after_capture is not None:
            return
        pair = self.capture.latest
        fresh = pair is not None and time.monotonic() - pair[0] < 2
        if not fresh:
            self.shot_button.setEnabled(False)
            return
        self.preview_tick += 1
        if self.preview_tick % self.PREVIEW_DETECT_EVERY == 0:
            for role, image in zip(("cf", "cs"), pair[1:]):
                self.preview_found[role], self.preview_corners[role] = self.detect_preview(image)
            self.update_view_titles()
        pattern = (self.cols.value(), self.rows.value())
        for role, image in zip(("cf", "cs"), pair[1:]):
            if self.preview_found[role]:
                image = image.copy()
                cv2.drawChessboardCorners(image, pattern, self.preview_corners[role], True)
            self.views[role].set_image(image)
        # 外參照片必須兩台同時看到完整棋盤格，否則該組無法配對
        self.shot_button.setEnabled(all(self.preview_found.values()))

    def stop_capture(self, callback):
        if self.capture is None:
            callback()
            return
        self.after_capture = callback
        self.next_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.shot_button.setEnabled(False)
        self.capture.requestInterruption()
        self.log("正在釋放雙相機…")

    def capture_stopped(self):
        if self.capture is not None:
            self.capture.deleteLater()
            self.capture = None
        self.connect_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.shot_button.setEnabled(False)
        self.next_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        callback, self.after_capture = self.after_capture, None
        if callback:
            try:
                callback()
            except Exception as exc:
                self.fail(str(exc))
        elif self.state == "capture":
            self.log("雙相機已停止，可重新連接或匯入照片。")

    def capture_sample(self):
        if self.job is not None or self.after_capture is not None:
            return
        if self.state.startswith("intrinsic") and self.panel is not None:
            self.panel.capture_sample()
        elif self.state == "capture":
            self.capture_pair()

    def check_image_size(self, image, role):
        sn = self.roles[0 if role == "cf" else 1]
        size = (image.shape[1], image.shape[0])
        if tuple(self.entries[sn]["image_size"]) != size:
            raise ValueError("{} 影像尺寸 {} 與內參 {} 不符，請使用同解析度／裁切範圍重新校正。"
                             .format(role, size, self.entries[sn]["image_size"]))

    def capture_pair(self):
        try:
            pair = self.capture.latest if self.capture is not None else None
            if pair is None or time.monotonic() - pair[0] > 2:
                raise ValueError("尚未取得新鮮的雙機畫面，請先連接相機。")
            if pair[0] == self.last_capture_time:
                return
            if not all(self.preview_found.values()):
                missing = "、".join(("正面" if r == "cf" else "側面") for r, ok in self.preview_found.items() if not ok)
                raise ValueError("{}未偵測到完整棋盤格，這組無法用於外參；請調整板子位置後再拍。".format(missing))
            # PNG 無損，避免 JPG 壓縮影響亞像素角點
            name = "{:02d}.png".format(len(self.names) + 1)
            for role, image in zip(("cf", "cs"), pair[1:]):
                self.check_image_size(image, role)
            paths = [self.session / role / name for role in ("cf", "cs")]
            try:
                for file, image in zip(paths, pair[1:]):
                    write_image(file, image)
            except Exception:
                for file in paths:
                    if file.exists():
                        file.unlink()
                raise
            self.last_capture_time = pair[0]
            self.names.append(name)
            self.persist_session("capture")
            remain = max(0, RECOMMENDED_PAIRS - len(self.names))
            self.pair_status.setText("已拍 {} 組；請移動、傾斜棋盤格再拍下一組{}。".format(
                len(self.names), "（建議再拍 {} 組）".format(remain) if remain else ""))
        except Exception as exc:
            self.fail(str(exc))

    def import_pairs(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇內含 cf 與 cs 的校正照片資料夾",
                                                   str(DB_DIR / "Record/Calibrate_Picture"))
        if not folder:
            return
        try:
            if self.names:
                raise ValueError("本次已有照片，請取消並開新一次校正後再匯入，避免混用不同相機擺位。")
            names = match_image_pairs(folder)
            # Validate every image before accepting any pair into this session.
            for name in names:
                for role in ("cf", "cs"):
                    self.check_image_size(read_image(Path(folder) / role / name), role)
            for name in names:
                for role in ("cf", "cs"):
                    shutil.copy2(str(Path(folder) / role / name), str(self.session / role / name))
            self.names = names
            self.persist_session("capture_imported")
            self.pair_status.setText("已匯入 {} 組。請確認照片來自所選序號及目前相機擺位。".format(len(names)))
            # Imported filenames need not be numeric; keep this run import-only.
            self.connect_button.setEnabled(False)
            self.import_button.setEnabled(False)
        except Exception as exc:
            self.fail(str(exc))

    def calculate_extrinsic(self):
        self.state = "calculate"
        self.capture_actions.hide()
        self.log("正在偵測 {} 組照片的棋盤格角點並計算外參…".format(len(self.names)))
        roles, entries = self.roles, dict(self.entries)
        pattern, square = (self.cols.value(), self.rows.value()), self.square.value()
        self.start_job(lambda: estimate_chessboard_stereo(self.session, roles, entries, pattern, square),
                       self.extrinsic_finished, self.extrinsic_failed)

    def extrinsic_failed(self, message):
        # 計算失敗時回到拍照頁，可補拍後再算；已拍的照片保留在本次資料夾
        self.begin_capture()
        self.pair_status.setText("已拍 {} 組；外參計算失敗，可補拍後再算。".format(len(self.names)))
        self.fail(message)

    def extrinsic_finished(self, report):
        save_json(self.session / "report.json", report)
        fit = report["fit"]
        checks = report["held_out_board_checks"]
        held_rmse = float(np.median([c["square_rmse_mm"] for c in checks]))
        summary = ("外參結果：Stereo RMS {:.3f} px｜基線 {:.0f} mm｜訓練 {} 組、驗證 {} 組、剔除 {} 組｜"
                   "驗證組格子邊長誤差中位數 {:.2f} mm"
                   .format(fit["stereo_rms_px"], report["baseline_mm"], len(report["training_pairs"]),
                           len(checks), len(report["rejected_pairs"]), held_rmse))
        self.log(summary)
        for role, names in report["detection_failed"].items():
            if names:
                self.log("{}未偵測到棋盤格：{}".format("正面" if role == "cf" else "側面", "、".join(names)))
        if report["rejected_pairs"]:
            self.log("姿態不一致而剔除：{}".format("、".join(report["rejected_pairs"])))
        for warning in report["warnings"]:
            self.log("REVIEW：{}".format(warning))
        if fit["stereo_rms_px"] > MAX_STEREO_RMS:
            raise ValueError("Stereo RMS {:.2f} px 超過 {} px，未更新校正檔（報告已存於 {}）。\n"
                             "請確認棋盤格規格、兩台內參，並補拍不同位置與深度的照片。"
                             .format(fit["stereo_rms_px"], MAX_STEREO_RMS, self.session / "report.json"))
        publish_stereo(calib_store.CALIB_DIR, self.session, self.roles, self.entries, report)
        self.state = "done"
        try:
            self.persist_session("complete")
        except OSError as exc:
            self.log("校正檔已套用，但工作階段狀態無法更新：{}".format(exc))
        self.instructions.setText("校正完成，已更新 Db/Calibration/extrinsic/stereo.json。")
        self.log("照片、角點預覽（corners/）、內參樣本與報告：{}".format(self.session))
        self.log("驗證組與訓練組來自同一批照片，仍建議用已知長度的物體現場驗證實際尺寸。")
        self.next_button.setText("開始另一次校正")
        self.cancel_button.setEnabled(False)

    def cancel(self):
        if self.job is not None:
            return
        def finish():
            self.dispose_panel()
            if self.session is not None:
                try:
                    self.persist_session("cancelled")
                except OSError as exc:
                    self.log("無法更新草稿狀態：{}".format(exc))
            self.log("已取消；目前使用的校正值未變更，本次草稿保留在工作階段資料夾。")
            self.reset_ui()
        self.stop_capture(finish)

    def closeEvent(self, event):
        if self.job is not None:
            self.log("正在計算，完成後即可關閉。")
            event.ignore()
            return
        if self.capture is not None:
            event.ignore()
            self.stop_capture(self.close)
            return
        self.dispose_panel()
        if self.session is not None and self.state not in ("setup", "done"):
            try:
                self.persist_session("interrupted")
            except OSError:
                pass
        self.preview.stop()
        event.accept()


def main():
    # Load native camera dependencies before starting Qt and calculation workers.
    # Missing SDK is still handled by the camera scan UI.
    try:
        import calib
    except ImportError:
        pass
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
