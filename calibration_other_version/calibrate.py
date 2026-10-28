"""單一入口：兩台棋盤格內參 → 校正桿雙機拍照 → 標註 → 外參。

啟動：python Src/calibration/calibrate.py（可從任何工作目錄執行）
相機固定後，依畫面指示移動棋盤格／校正桿；左右影像均依相同順序點 6 個球心。
本工具不載入 YOLO / ViTPose。既有 calib.py、01、02 工具仍可獨立使用。
"""

# Support both python -m and direct file execution from any working directory.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime

# The independent tools do not initialize the legacy camera/intrinsic wizard.
if __name__ == "__main__" and "--two-rods" in sys.argv:
    from Src.calibration.calibrate_two_rods import main as rods_main
    sys.exit(rods_main([argument for argument in sys.argv[1:] if argument != "--two-rods"]))
if __name__ == "__main__" and "--wood-frame" in sys.argv:
    from Src.calibration.calibrate_wood_frame import main as wood_main
    sys.exit(wood_main([argument for argument in sys.argv[1:] if argument != "--wood-frame"]))
if __name__ == "__main__" and "--manual-coordinates" in sys.argv:
    from Src.calibration.calibrate_manual_coordinates import main as manual_main
    sys.exit(manual_main([argument for argument in sys.argv[1:] if argument != "--manual-coordinates"]))
if __name__ == "__main__" and "--cross" in sys.argv:
    raise SystemExit("十字架工具已移除，請改用 --wood-frame。")

UI_DIR = Path(__file__).resolve().parent
from Src.paths import CAMERA_CONFIG_DIR
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
qt_plugins = Path(os.environ.get("CONDA_PREFIX", sys.prefix)) / "Library/plugins/platforms"
if qt_plugins.is_dir():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins))

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QShortcut, QSpinBox,
    QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

from Src.calibration import calib_store
from Src.calibration.calibration_workflow import (
    MIN_PAIRS, estimate_wand_fundamental, match_image_pairs,
    publish_calibration, publish_intrinsics, save_json, validate_intrinsic, validate_roles,
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
            from Src.cameras.single_camera.flir_camera_system import FlirCameraSystem
            from Src.cameras.two_cameras.dual_flir_system import DualFlirSystem
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
                self.latest = (time.monotonic(), cv2.cvtColor(front, cv2.COLOR_BayerBG2BGR),
                               cv2.cvtColor(side, cv2.COLOR_BayerBG2BGR))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.latest = None
            for camera in reversed(cameras):
                try:
                    camera.release()
                except Exception as exc:
                    self.failed.emit("釋放相機失敗：{}".format(exc))


class PointView(QGraphicsView):
    """Image coordinates remain correct after fitting, panning and zooming."""
    edited = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(300, 230)
        self.setBackgroundBrush(QColor("#20252b"))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.points = []
        self.image_shape = None
        self.markers = []
        self.editable = False

    def set_image(self, image, points=None, editable=False):
        self.scene().clear()
        self.markers = []
        self.points = points if points is not None else []
        self.editable = editable
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        self.image_shape = (height, width)
        qimage = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self.scene().addPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(0, 0, width, height)
        self.fit_image()
        self.draw_points()

    def fit_image(self):
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def draw_points(self):
        for marker in self.markers:
            self.scene().removeItem(marker)
        self.markers = []
        for i, (x, y) in enumerate(self.points):
            self.markers.append(self.scene().addEllipse(x-4, y-4, 8, 8, QPen(QColor("#ff4545"), 2)))
            text = self.scene().addText(str(i + 1))
            text.setDefaultTextColor(QColor("#ffff00"))
            text.setPos(x + 5, y - 15)
            self.markers.append(text)

    def mousePressEvent(self, event):
        if self.editable and not (event.modifiers() & Qt.ControlModifier):
            position = self.mapToScene(event.pos())
            if event.button() == Qt.RightButton and self.points:
                self.points.pop()
            elif event.button() == Qt.LeftButton and len(self.points) < 6:
                height, width = self.image_shape
                if not (0 <= position.x() < width and 0 <= position.y() < height):
                    return
                if any(np.hypot(x-position.x(), y-position.y()) < 1 for x, y in self.points):
                    return
                self.points.append([position.x(), position.y()])
            else:
                return
            self.draw_points()
            self.edited.emit()
        else:
            super().mousePressEvent(event)

    def wheelEvent(self, event):
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.05 < self.transform().m11() * factor < 20:
            self.scale(factor, factor)


class CalibrationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("相機校正精靈 — 棋盤格內參 / 校正桿外參")
        self.resize(1200, 900)
        self.state = "setup"
        self.session = None
        self.roles = None
        self.entries = {}
        self.points = {"cf": {}, "cs": {}}
        self.names = []
        self.index = 0
        self.panel = None
        self.capture = None
        self.job = None
        self.after_capture = None
        self.last_capture_time = None
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.step_label = QLabel("1 設定 → 2 正面內參 → 3 側面內參 → 4 拍照 → 5 標註與計算")
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
        self.intrinsic_only = QCheckBox("只做內參（外參改用 RoMa：在校正中心選 RoMa，不拍校正桿）")
        form.addRow(self.intrinsic_only)
        self.cols, self.rows = QSpinBox(), QSpinBox()
        for spin, value in ((self.cols, 8), (self.rows, 11)):
            spin.setRange(3, 30)
            spin.setValue(value)
        self.square = QDoubleSpinBox()
        self.square.setRange(0.1, 1000)
        self.square.setValue(10)
        self.square.setSuffix(" mm")
        form.addRow("棋盤格內角點：寬（格子數 − 1）", self.cols)
        form.addRow("棋盤格內角點：高（格子數 − 1）", self.rows)
        form.addRow("每格實際邊長", self.square)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.1, 10)
        self.threshold.setValue(3)
        self.threshold.setSuffix(" px")
        form.addRow("外參內點誤差門檻", self.threshold)
        layout.addWidget(box)
        note = QLabel("準備：固定兩台相機與焦距，關閉 main.py / SpinView 等占用相機的程式。\n"
                      "內參：每台拍至少 15 組不同位置、距離及傾斜角度的棋盤格。\n"
                      "外參：接 GPIO 同步線，拍至少 4 組不同位置、角度與深度的校正桿。\n"
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
            view = PointView()
            view.edited.connect(self.points_changed)
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
        self.annotation_actions = QWidget()
        buttons = QHBoxLayout(self.annotation_actions)
        self.previous_button = QPushButton("上一組")
        self.previous_button.clicked.connect(lambda: self.move_pair(-1))
        self.following_button = QPushButton("下一組")
        self.following_button.clicked.connect(lambda: self.move_pair(1))
        fit = QPushButton("縮放至完整影像")
        fit.clicked.connect(lambda: [view.fit_image() for view in self.views.values()])
        for button in (self.previous_button, self.following_button, fit):
            buttons.addWidget(button)
        layout.addWidget(self.annotation_actions)
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
            from Src.calibration.calib import list_cameras
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
                self.stop_capture(self.begin_annotation)
            elif self.state == "annotate":
                self.calculate_extrinsic()
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
        self.session = Path(calib_store.CALIB_DIR) / "sessions" / (stamp + "_wand")
        for folder in ("cf", "cs", "intrinsic"):
            (self.session / folder).mkdir(parents=True, exist_ok=True)
        self.names, self.points = [], {"cf": {}, "cs": {}}
        self.index = 0
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
            "method": "intrinsic_only" if self.intrinsic_only.isChecked() else "wand_manual_ransac",
            "front": self.roles[0], "side": self.roles[1],
            "status": status, "intrinsics": self.entries,
            "pattern": [self.cols.value(), self.rows.value()], "square_size_mm": self.square.value(),
            "threshold_px": self.threshold.value(), "image_names": self.names,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

    def begin_intrinsic(self, role_index):
        import Src.calibration.calib as calib
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

    def start_job(self, function, success):
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
                self.fail(str(exc))
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
        """兩台內參完成就發布（內參＋正面／側面角色），不拍校正桿；外參之後用 RoMa 計算。"""
        publish_intrinsics(calib_store.CALIB_DIR, self.session, self.roles, self.entries,
                           datetime.now().isoformat(timespec="seconds"))
        self.state = "done"
        try:
            self.persist_session("complete")
        except OSError as exc:
            self.log("內參已套用，但工作階段狀態無法更新：{}".format(exc))
        self.instructions.setText("內參完成，已更新 Db/Calibration。外參請到校正中心（start_calibrate.bat）選 RoMa，用當天的錄影計算。")
        self.log("兩台內參與正面／側面角色已更新；外參（fundamental.json）沒有改動，若相機有移動請用 RoMa 重新計算。")
        self.log("照片與內參樣本：{}".format(self.session))
        self.next_button.setText("開始另一次校正")
        self.cancel_button.setEnabled(False)

    def begin_capture(self):
        self.state = "capture"
        self.pages.setCurrentWidget(self.pair_page)
        self.capture_actions.show()
        self.annotation_actions.hide()
        self.connect_button.setEnabled(True)
        self.import_button.setEnabled(True)
        self.shot_button.setEnabled(False)
        self.next_button.setText("開始標註")
        self.instructions.setText("接好 GPIO 同步線後連接雙相機。校正桿每個位置停穩後按空白鍵，至少 4 組；"
                                  "涵蓋不同角度及深度，兩台都須看見同樣 6 個球心。")
        self.pair_status.setText("已拍 0 組；也可匯入既有 cf / cs 照片。")
        for role, view in self.views.items():
            view.scene().clear()
            view.editable = False
            self.view_titles[role].setText(("正面 SN" if role == "cf" else "側面 SN") +
                                          self.roles[0 if role == "cf" else 1])

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
        self.shot_button.setEnabled(fresh)
        if fresh:
            for role, image in zip(("cf", "cs"), pair[1:]):
                self.views[role].set_image(image)

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
            name = "{:02d}.jpg".format(len(self.names) + 1)
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
            self.pair_status.setText("已拍 {} 組；請移動校正桿再拍下一組。".format(len(self.names)))
        except Exception as exc:
            self.fail(str(exc))

    def import_pairs(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇內含 cf 與 cs 的校正照片資料夾",
                                                   str(UI_DIR.parents[1] / "Db/Record/Calibrate_Picture"))
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

    def begin_annotation(self):
        self.state = "annotate"
        self.capture_actions.hide()
        self.annotation_actions.show()
        self.next_button.setText("計算並儲存校正")
        self.instructions.setText("兩張照片依相同實體球心順序點 1–6。左鍵新增、右鍵撤銷；"
                                  "滾輪縮放，Ctrl＋拖曳平移。點位自動存成草稿；完成所有照片後計算。")
        self.index = 0
        self.show_pair()

    def show_pair(self):
        name = self.names[self.index]
        for role in ("cf", "cs"):
            points = self.points[role].setdefault(name, [])
            self.views[role].set_image(read_image(self.session / role / name), points, True)
        self.previous_button.setEnabled(self.index > 0)
        self.following_button.setEnabled(self.index < len(self.names) - 1)
        self.points_changed()

    def points_changed(self):
        if self.state != "annotate":
            return
        name = self.names[self.index]
        completed = sum(len(self.points["cf"].get(n, [])) == 6 and
                        len(self.points["cs"].get(n, [])) == 6 for n in self.names)
        self.pair_status.setText("第 {}/{} 組 {}｜正面 {}/6、側面 {}/6｜完成 {}/{} 組"
                                 .format(self.index+1, len(self.names), name,
                                         len(self.points["cf"][name]), len(self.points["cs"][name]),
                                         completed, len(self.names)))
        try:
            for role in ("cf", "cs"):
                save_json(self.session / ("selected_points_{}.json".format(role)), self.points[role])
        except Exception as exc:
            self.fail("草稿儲存失敗：{}".format(exc))

    def move_pair(self, offset):
        self.index = max(0, min(len(self.names)-1, self.index + offset))
        self.show_pair()

    def calculate_extrinsic(self):
        for name in self.names:
            if any(len(self.points[role].get(name, [])) != 6 for role in ("cf", "cs")):
                raise ValueError("{} 尚未完成左右各 6 點，請補齊後再計算。".format(name))
        self.log("正在計算外參並檢查內點品質…")
        threshold = self.threshold.value()
        self.start_job(lambda: estimate_wand_fundamental(self.points["cf"], self.points["cs"],
                                                        threshold), self.extrinsic_finished)

    def extrinsic_finished(self, report):
        report.update(front=self.roles[0], side=self.roles[1])
        save_json(self.session / "report.json", report)
        publish_calibration(calib_store.CALIB_DIR, self.session, self.roles, self.entries, report, self.points)
        self.state = "done"
        try:
            self.persist_session("complete")
        except OSError as exc:
            self.log("校正檔已套用，但工作階段狀態無法更新：{}".format(exc))
        for view in self.views.values():
            view.editable = False
        self.annotation_actions.hide()
        self.instructions.setText("校正完成，已更新 Db/Calibration。主程式下次錄影會採用這組校正值。")
        self.log("校正完成：內點 {}/{}，內點 RMS {:.3f} px；全體中位誤差 {:.3f} px。"
                 .format(report["inliers"], report["total_points"], report["inlier_rms_px"],
                         report["median_error_px"]))
        self.log("照片、內參樣本、點位與報告：{}".format(self.session))
        self.log("外參採原始像素座標，與現有 3D 流程一致；誤差為擬合結果，仍需現場驗證實際尺寸。")
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
        import Src.calibration.calib as calib
    except ImportError:
        pass
    app = QApplication(sys.argv)
    window = CalibrationWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
