"""校正精靈：兩台相機同步拍棋盤格，同一批照片算兩台內參與外參（R、t，mm）。

啟動：python calibration/calibration.py（可從任何工作目錄執行）
兩台都看到棋盤格的照片存成配對（cf/、cs/ 同名），內參與外參都用；
只有一台看到的照片存進 intrinsic/images_SN<序號>/，只用於那台內參（補畫面四角）。
本工具不載入 YOLO / ViTPose。單台相機內參另見 intrinsic_window.py（calibration.py --single）。
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

from paths import CAMERA_CONFIG_DIR
# 正面、側面相機的設定檔（曝光、增益的預設值也在這裡）
CAMERA_CONFIGS = (CAMERA_CONFIG_DIR / "GH3_camera_config.yaml", CAMERA_CONFIG_DIR / "GH3_camera_config_2.yaml")
# 尚無 camera_roles.json 時的預設：正面 35 mm、側面 8 mm
DEFAULT_ROLES = ("25083472", "21091470")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
qt_plugins = Path(os.environ.get("CONDA_PREFIX", sys.prefix)) / "Library/plugins/platforms"
if qt_plugins.is_dir():
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins))

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QShortcut, QSpinBox,
    QStackedWidget, QTextEdit, QVBoxLayout, QWidget,
)

from camera_controls import CameraControls
import intrinsic
import storage
from stereo import (
    MAX_STEREO_RMS, MIN_PAIRS, RECOMMENDED_PAIRS, estimate_chessboard_stereo, match_image_pairs,
    pair_problems, save_rejected,
    publish_stereo, save_json, validate_intrinsic, validate_roles,
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
    """Own both cameras in one worker; release them before calculating."""
    failed = pyqtSignal(str)
    opened = pyqtSignal(int, object)            # 相機索引、get_exposure_settings()
    applied = pyqtSignal(int, float, float)     # 相機索引、實際曝光 μs、增益 dB

    def __init__(self, roles, parent):
        super().__init__(parent)
        self.roles = roles
        self.latest = None
        # UI 執行緒放入 {相機索引: (曝光, 增益)}，由本執行緒在兩張畫面之間套用，不與取像同時操作相機
        self.requests = {}

    def run(self):
        cameras = []
        try:
            from camera_objects import DualFlirSystem, FlirCameraSystem
            for sn, config in zip(self.roles, CAMERA_CONFIGS):
                camera = FlirCameraSystem(str(config), sn)
                cameras.append(camera)
            pair = DualFlirSystem(*cameras)
            self.check_sync(*cameras)
            settings = [camera.get_exposure_settings() for camera in cameras]
            # 側面是 secondary（觸發模式），回報的曝光上限不受幀率限制，但實際跟著正面的幀率觸發，
            # 曝光超過正面的上限就會漏掉觸發，因此兩台都以正面（free-run）的上限為準
            limit = settings[0]["exposure_range"][1]
            for camera, setting in zip(cameras, settings):
                low, high = setting["exposure_range"]
                setting["exposure_range"] = (low, min(high, limit))
                if setting["exposure_time"] > limit:
                    setting["exposure_time"] = camera.set_exposure_time(limit)
            for index, setting in enumerate(settings):
                self.opened.emit(index, setting)
            failures = 0
            while not self.isInterruptionRequested():
                for index in list(self.requests):
                    exposure_time, gain = self.requests.pop(index)
                    try:
                        camera = cameras[index]
                        self.applied.emit(index, camera.set_exposure_time(min(exposure_time, limit)),
                                          camera.set_gain(gain))
                    except Exception as exc:
                        self.failed.emit("調整曝光／增益失敗：{}".format(exc))
                try:
                    ok, front, side = pair.get_grayscale_images()
                except Exception:
                    # 取像逾時（例如拍攝中同步線鬆脫）會丟例外，和取到不完整畫面一樣計入失敗
                    ok = False
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

    @staticmethod
    def grab(camera, attempts=3):
        """取一張畫面；逾時或不完整重試幾次，都失敗回傳 False"""
        for _ in range(attempts):
            try:
                if camera.get_grayscale_image()[0]:
                    return True
            except Exception:
                pass
        return False

    def check_sync(self, front, side):
        """連線後先各取一張，取不到時判斷是相機本身、同步線接反還是沒有觸發訊號，回報可直接處理的原因。

        正面是 primary（自己取像，從 Line2 送出觸發），側面是 secondary（等 Line3 收到觸發才拍）。
        正面取得到、側面取不到就是同步問題；再把方向對調試一次：
        對調後正面收得到側面的觸發，代表線接反了，否則是線沒接好或損壞。
        """
        if not self.grab(front):
            raise RuntimeError("正面 SN{} 取不到畫面（與同步線無關）：請確認 USB 連線，"
                               "並關閉 main.py、SpinView 等占用相機的程式。".format(front.serial_number))
        if self.grab(side):
            return
        # 取像中不能改觸發設定（設定函式遇到不可寫入的節點會直接略過），先停止取像再對調
        for camera in (front, side):
            camera.stop_acquisition()
        side.configure_gpio_primary()
        side.disable_trigger_mode()
        front.configure_gpio_secondary()
        self.grab(side, attempts=1)     # 側面開始取像才會送出觸發
        if self.grab(front):
            raise RuntimeError(
                "同步線接反了：目前是側面 SN{} 送出觸發、正面 SN{} 接收，程式需要正面送出觸發。\n"
                "請把同步線兩端對調（正面接觸發輸出端），或確認設定頁的正面／側面序號沒有選反，再按「連接雙相機」。"
                .format(side.serial_number, front.serial_number))
        raise RuntimeError(
            "側面 SN{} 收不到正面 SN{} 的同步觸發，對調方向也收不到：同步線沒接好、接錯腳位或損壞。\n"
            "請檢查兩台相機的 GPIO 接頭是否插緊，再按「連接雙相機」。"
            .format(side.serial_number, front.serial_number))


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
    # 拍照預覽：縮圖偵測 + 每 N 次更新偵測一次，避免全解析度偵測拖慢畫面。
    # 側面 8 mm 廣角裡的棋盤格很小，0.4 倍時 20261019 的側面照片只偵測到 10/33 張，0.7 倍全部偵測得到
    PREVIEW_SCALE = 0.7
    PREVIEW_DETECT_EVERY = 3
    VIEWS = ("cf", "cs")             # 畫面：正面、側面
    LENSES = ("front", "side")       # 對應 session.json／內參計算的名稱
    TITLES = ("正面", "側面")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("相機校正精靈 — 雙視角棋盤格內參 / 外參")
        self.resize(1200, 900)
        self.state = "setup"
        self.session = None
        self.roles = None
        self.entries = {}
        self.reused = False
        self.imported = False
        self.names = []
        self.reset_samples()
        self.preview_tick = 0
        self.preview_found = {"cf": False, "cs": False}
        self.preview_corners = {"cf": None, "cs": None}
        self.capture = None
        self.job = None
        self.after_capture = None
        self.last_capture_time = None
        # 每台相機最後的曝光／增益 {序號: (μs, dB)}，重新連接時沿用並記入 session.json
        self.camera_settings = {}
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.step_label = QLabel("1 設定 → 2 雙機同時拍棋盤格 → 3 計算兩台內參與外參")
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
        # 鏡頭焦距決定內參初值與長焦鎖定（>= intrinsic.TELEPHOTO_MM 自動鎖光心與高階畸變），也用來即時估傾斜角
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
        note = QLabel("準備：固定兩台相機與焦距，接好 GPIO 同步線，關閉 main.py / SpinView 等占用相機的程式。\n"
                      "拍攝：兩台同時拍，兩台都看到棋盤格存成一組配對（兩台內參＋外參都用），"
                      "只有一台看到時只當那台的內參照片。\n"
                      f"每台內參至少 {intrinsic.MIN_SAMPLES} 張、涵蓋畫面四角並往不同方向傾斜；"
                      f"配對至少 {MIN_PAIRS} 組（建議 {RECOMMENDED_PAIRS} 組），涵蓋打擊區不同位置與深度。\n"
                      "也可匯入含 cf / cs 的既有照片資料夾。\n"
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
        self.camera_status = {}
        self.pair_controls = {}
        for index, (role, title) in enumerate(zip(self.VIEWS, self.TITLES)):
            column = QVBoxLayout()
            label = QLabel(title)
            column.addWidget(label)
            self.view_titles[role] = label
            view = ImageView()
            column.addWidget(view, 1)
            self.views[role] = view
            # 這台目前的內參樣本數、覆蓋與傾斜，拍攝中即時更新
            status = QLabel()
            status.setWordWrap(True)
            column.addWidget(status)
            self.camera_status[role] = status
            controls = CameraControls(title + "曝光")
            controls.changed.connect(lambda exposure, gain, index=index: self.request_camera_settings(
                index, exposure, gain))
            column.addWidget(controls)
            self.pair_controls[role] = controls
            images.addLayout(column)
        layout.addLayout(images, 1)
        self.pair_status = QLabel()
        self.pair_status.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.pair_status)
        self.capture_actions = QWidget()
        buttons = QHBoxLayout(self.capture_actions)
        self.connect_button = QPushButton("連接雙相機")
        self.connect_button.clicked.connect(self.connect_pair)
        self.shot_button = QPushButton("拍照（空白鍵）")
        self.shot_button.clicked.connect(self.capture_shot)
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
            saved = storage.load_roles()
            if saved:
                serials.update(saved)
            serials.update(storage.load_intrinsics())
        except Exception as exc:
            self.log("既有校正檔無法讀取：{}".format(exc))
        connected = []
        try:
            from intrinsic_window import list_cameras
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
        selected = [combo.currentText().strip() or (saved or DEFAULT_ROLES)[i]
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

    def reset_samples(self):
        """本次拍攝的即時統計：每台的內參角點、傾斜估計與影像尺寸，以及外參配對在正面畫面的角點與姿態"""
        self.shot_count = 0
        self.samples = {"cf": [], "cs": []}
        self.tilts = {"cf": [], "cs": []}
        self.image_sizes = {"cf": None, "cs": None}
        self.pair_corners = []
        self.pair_poses = []

    def advance(self):
        try:
            if self.state == "setup":
                self.start_session()
            elif self.state == "capture":
                if self.ready_to_calculate():
                    self.stop_capture(self.calculate)
            elif self.state == "done":
                self.reset_ui()
        except Exception as exc:
            self.fail(str(exc))

    def start_session(self):
        roles = (self.front_combo.currentText().strip(), self.side_combo.currentText().strip())
        validate_roles(*roles)
        entries = {}
        if self.reuse.isChecked():
            stored = storage.load_intrinsics()
            for sn in roles:
                if sn not in stored:
                    raise ValueError("SN{} 尚無內參，請取消「沿用已有內參」。".format(sn))
                validate_intrinsic(stored[sn])
                entries[sn] = stored[sn]
        self.roles, self.entries, self.reused, self.imported = roles, entries, bool(entries), False
        self.session = Path(storage.new_session_dir())
        for folder in ("cf", "cs", "intrinsic"):
            (self.session / folder).mkdir(parents=True, exist_ok=True)
        self.names = []
        self.reset_samples()
        self.last_capture_time = None
        self.persist_session("started")
        self.setup_page.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.log("本次資料：{}".format(self.session))
        if entries:
            # 沿用的內參也存一份到本次資料夾，外參報告才知道用的是哪一版
            save_json(self.session / "intrinsics.json", entries)
        self.begin_capture()

    def persist_session(self, status):
        save_json(self.session / "session.json", {
            "method": "chessboard_dual_view",
            "front": self.roles[0], "side": self.roles[1],
            "status": status, "intrinsics": self.entries, "reused_intrinsics": self.reused,
            "pattern": [self.cols.value(), self.rows.value()], "square_size_mm": self.square.value(),
            "lens_focal_mm": {"front": self.front_focal.value() or None,
                              "side": self.side_focal.value() or None},
            "image_names": self.names,
            "shot_count": self.shot_count,
            "camera_settings": {sn: {"exposure_time_us": exposure, "gain_db": gain}
                                for sn, (exposure, gain) in self.camera_settings.items()},
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

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
            try:
                if job.error:
                    raise ValueError(job.error)
                success(job.result)
            except Exception as exc:
                (failure or self.fail)(str(exc))
            job.deleteLater()

        job.finished.connect(finished)
        job.start()

    def begin_capture(self):
        self.state = "capture"
        self.pages.setCurrentWidget(self.pair_page)
        self.capture_actions.show()
        self.connect_button.setEnabled(not self.imported)
        self.import_button.setEnabled(not self.imported and not self.names)
        self.shot_button.setEnabled(False)
        self.next_button.setText("計算內參與外參並儲存")
        if self.reused:
            self.instructions.setText(
                f"沿用已有內參，只拍外參：接好 GPIO 同步線後連接雙相機，兩台畫面都出現彩色角點時按空白鍵。"
                f"建議 {RECOMMENDED_PAIRS} 組以上（至少 {MIN_PAIRS} 組），涵蓋打擊區的左右、前後、高低並朝不同方向傾斜。")
        else:
            self.instructions.setText(
                "接好 GPIO 同步線後連接雙相機，按空白鍵兩台同時拍：兩台都出現彩色角點存成一組配對；"
                "只有一台看到時只存那台，當作內參照片（用來補畫面四角）。"
                f"每台內參至少 {intrinsic.MIN_SAMPLES} 張，配對至少 {MIN_PAIRS} 組（建議 {RECOMMENDED_PAIRS} 組）；"
                "依下方提示補拍缺少的角落與傾斜方向。")
        self.preview_found = {"cf": False, "cs": False}
        self.preview_corners = {"cf": None, "cs": None}
        for role, view in self.views.items():
            view.scene().clear()
            view.image_shape = None
        self.update_view_titles()
        self.update_capture_status()

    def update_view_titles(self):
        for index, role in enumerate(self.VIEWS):
            title = "{} SN{}".format(self.TITLES[index], self.roles[index])
            found = self.preview_found[role]
            self.view_titles[role].setText("{}　{}".format(title, "✔ 偵測到棋盤格" if found else "✘ 未偵測到棋盤格"))
            self.view_titles[role].setStyleSheet(
                "color: {}; font-weight: bold;".format("#15803d" if found else "#b91c1c"))

    def camera_problems(self, role):
        """這台目前內參樣本的問題（覆蓋、傾斜、張數），與校正後的品質判斷同一套門檻"""
        tilts = [t for t, _ in self.tilts[role]]
        axes = [a for _, a in self.tilts[role]]
        return intrinsic.sample_problems(self.samples[role], self.image_sizes[role], tilts, axes)

    def update_capture_status(self):
        for index, role in enumerate(self.VIEWS):
            label = self.camera_status[role]
            if self.reused:
                label.setText("沿用已有內參")
                label.setStyleSheet("color: #475569;")
                continue
            if self.imported:
                label.setText("已匯入照片，計算時才會檢查內參樣本")
                label.setStyleSheet("color: #475569;")
                continue
            count = len(self.samples[role])
            covered, _ = intrinsic.coverage(self.samples[role], self.image_sizes[role])
            tilted = sum(t >= intrinsic.MIN_TILT_DEG for t, _ in self.tilts[role])
            problems = self.camera_problems(role)
            text = "內參 {}/{} 張｜畫面覆蓋 {}/9｜傾斜 ≥{:.0f}° {} 張".format(
                count, intrinsic.MIN_SAMPLES, covered, intrinsic.MIN_TILT_DEG, tilted)
            if problems:
                text += "\n" + "\n".join("• " + p for p in problems)
            else:
                text += "\n✔ 張數、覆蓋與傾斜都足夠"
            label.setText(text)
            label.setStyleSheet("color: {};".format("#b45309" if problems else "#15803d"))
        problems = self.pair_problems()
        text = "外參配對 {} 組（至少 {}、建議 {}）｜已拍 {} 次".format(
            len(self.names), MIN_PAIRS, RECOMMENDED_PAIRS, self.shot_count)
        if self.imported:
            text += "；已匯入照片，計算時才會檢查配對"
        elif problems:
            text += "\n" + "\n".join("• " + p for p in problems)
        elif self.names:
            text += "\n✔ 組數、分布、前後距離與傾斜都足夠"
        self.pair_status.setText(text)
        self.pair_status.setStyleSheet("font-weight: bold; color: {};".format(
            "#b91c1c" if len(self.names) < MIN_PAIRS else "#b45309" if problems else "#15803d"))

    def pair_problems(self):
        """外參配對目前的問題（組數、分布、前後距離、傾斜）；匯入的照片沒有即時統計"""
        if self.imported:
            return [] if len(self.names) >= RECOMMENDED_PAIRS else [
                "配對 {} 組，建議 {} 組以上".format(len(self.names), RECOMMENDED_PAIRS)]
        return pair_problems(self.pair_corners, self.image_sizes["cf"], self.pair_poses)

    def ready_to_calculate(self):
        """數量不足就擋下；數量夠但覆蓋、傾斜有問題時讓使用者決定是否補拍"""
        blocking = []
        if len(self.names) < MIN_PAIRS:
            blocking.append("配對 {} 組，至少需要 {} 組兩台都看到棋盤格的照片。".format(len(self.names), MIN_PAIRS))
        check_samples = not self.reused and not self.imported
        if check_samples:
            for index, role in enumerate(self.VIEWS):
                if len(self.samples[role]) < intrinsic.MIN_SAMPLES:
                    blocking.append("{}內參 {} 張，至少需要 {} 張。".format(
                        self.TITLES[index], len(self.samples[role]), intrinsic.MIN_SAMPLES))
        if blocking:
            raise ValueError("\n".join(blocking))
        warnings = []
        if check_samples:
            for index, role in enumerate(self.VIEWS):
                warnings += ["{}：{}".format(self.TITLES[index], p) for p in self.camera_problems(role)]
        warnings += ["外參：{}".format(p) for p in self.pair_problems()]
        if not warnings:
            return True
        answer = QMessageBox.question(
            self, "照片品質提醒", "目前還有以下可改善的地方：\n\n" + "\n".join("• " + w for w in warnings)
            + "\n\n仍要直接計算嗎？（選「否」可繼續補拍）",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def detect_preview(self, image, scale=None):
        scale = scale or self.PREVIEW_SCALE
        pattern = (self.cols.value(), self.rows.value())
        small = image if scale == 1 else cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        return found, (corners / scale if found else None)

    def connect_pair(self):
        if self.capture is not None:
            return
        self.connect_button.setEnabled(False)
        self.import_button.setEnabled(False)
        self.capture = CaptureWorker(self.roles, self)
        self.capture.failed.connect(self.log)
        self.capture.opened.connect(self.pair_camera_opened)
        self.capture.applied.connect(self.pair_settings_applied)
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
            for role, image in zip(self.VIEWS, pair[1:]):
                self.preview_found[role], self.preview_corners[role] = self.detect_preview(image)
            self.update_view_titles()
        pattern = (self.cols.value(), self.rows.value())
        for role, image in zip(self.VIEWS, pair[1:]):
            if self.preview_found[role]:
                image = image.copy()
                cv2.drawChessboardCorners(image, pattern, self.preview_corners[role], True)
            self.views[role].set_image(image)
        # 沿用內參時只需要外參配對，必須兩台都看到；否則任一台看到就能當那台的內參照片
        found = self.preview_found.values()
        self.shot_button.setEnabled(all(found) if self.reused else any(found))

    def pair_camera_opened(self, index, settings):
        controls = self.pair_controls[self.VIEWS[index]]
        controls.bind(settings, str(CAMERA_CONFIGS[index]))
        sn = self.roles[index]
        if sn in self.camera_settings:
            controls.set_values(*self.camera_settings[sn])
        else:
            self.camera_settings[sn] = controls.values()

    def request_camera_settings(self, index, exposure_time, gain):
        if self.capture is not None:
            self.capture.requests[index] = (exposure_time, gain)

    def pair_settings_applied(self, index, exposure_time, gain):
        self.pair_controls[self.VIEWS[index]].show_actual(exposure_time, gain)
        self.camera_settings[self.roles[index]] = (exposure_time, gain)

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
        for controls in self.pair_controls.values():
            controls.unbind()
        self.connect_button.setEnabled(not self.imported)
        self.import_button.setEnabled(not self.imported and not self.names)
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
        if self.job is None and self.after_capture is None and self.state == "capture":
            self.capture_shot()

    def check_image_size(self, image, role):
        sn = self.roles[self.VIEWS.index(role)]
        if sn not in self.entries:
            return
        size = (image.shape[1], image.shape[0])
        if tuple(self.entries[sn]["image_size"]) != size:
            raise ValueError("{} 影像尺寸 {} 與內參 {} 不符，請使用同解析度／裁切範圍重新校正。"
                             .format(role, size, self.entries[sn]["image_size"]))

    def capture_shot(self):
        """兩台同時拍：兩台都偵測到存成配對（cf/、cs/ 同名），只有一台偵測到存進該台的 intrinsic/images_SN<序號>/"""
        try:
            pair = self.capture.latest if self.capture is not None else None
            if pair is None or time.monotonic() - pair[0] > 2:
                raise ValueError("尚未取得新鮮的雙機畫面，請先連接相機。")
            if pair[0] == self.last_capture_time:
                return
            pattern = (self.cols.value(), self.rows.value())
            # 預覽是縮圖偵測且每幾張才偵測一次，存檔前用全解析度再確認並取得亞像素角點；
            # 預覽沒看到的那台也快速檢查一次全解析度，避免漏掉剛移進畫面的板子
            found = {}
            for role, image in zip(self.VIEWS, pair[1:]):
                if self.preview_found[role] or self.detect_preview(image, scale=1)[0]:
                    corners = intrinsic.find_corners(image, pattern)
                    if corners is not None:
                        found[role] = corners
            if self.reused and len(found) < 2:
                raise ValueError("沿用內參時只拍外參，兩台都要完整看到棋盤格；請調整板子位置後再拍。")
            if not found:
                raise ValueError("兩台都未完整偵測到棋盤格，請調整板子位置後再拍。")
            # 板子沒移動就重複收，等於同一組資料算好幾次，會讓誤差假性偏低
            moved = [role for role, corners in found.items()
                     if not self.samples[role]
                     or intrinsic.mean_shift(corners, self.samples[role][-1]) >= intrinsic.DUPLICATE_SHIFT_PX]
            if not moved:
                raise ValueError("與上一張幾乎相同，請移動或傾斜棋盤格後再拍。")
            for role, image in zip(self.VIEWS, pair[1:]):
                if role in found:
                    self.check_image_size(image, role)

            # PNG 無損，避免 JPG 壓縮影響亞像素角點；配對與單台照片共用編號，檔名不重複
            name = "{:02d}.png".format(self.shot_count + 1)
            images = dict(zip(self.VIEWS, pair[1:]))
            if len(found) == 2:
                files = {role: self.session / role / name for role in self.VIEWS}
            else:
                role = next(iter(found))
                folder = self.session / "intrinsic" / "images_SN{}".format(self.roles[self.VIEWS.index(role)])
                folder.mkdir(parents=True, exist_ok=True)
                files = {role: folder / name}
            try:
                for role, file in files.items():
                    write_image(file, images[role])
            except Exception:
                for file in files.values():
                    if file.exists():
                        file.unlink()
                raise

            self.last_capture_time = pair[0]
            self.shot_count += 1
            square = self.square.value()
            if len(found) == 2:
                self.names.append(name)
                # 外參的分布、距離與傾斜以正面為準（正面視野窄，大致就是兩台共同看到的區域）
                front = images["cf"]
                self.image_sizes["cf"] = (front.shape[1], front.shape[0])
                self.pair_corners.append(found["cf"])
                pose = intrinsic.estimate_pose(found["cf"], pattern, square, self.image_sizes["cf"],
                                               self.front_focal.value())
                if pose is not None:
                    self.pair_poses.append(pose)
            for role in moved:
                index = self.VIEWS.index(role)
                size = (images[role].shape[1], images[role].shape[0])
                self.samples[role].append(found[role])
                self.image_sizes[role] = size
                focal = (self.front_focal, self.side_focal)[index].value()
                tilt = intrinsic.estimate_tilt(found[role], pattern, square, size, focal)
                if tilt is not None:
                    self.tilts[role].append(tilt)
            self.persist_session("capture")
            kind = "配對" if len(found) == 2 else "只有{}（內參）".format(self.TITLES[self.VIEWS.index(next(iter(found)))])
            self.log("第 {} 張：{}{}".format(self.shot_count, kind, "" if len(moved) == len(found) else
                                            "；{}與上一張幾乎相同，不計入內參".format(
                                                "、".join(self.TITLES[self.VIEWS.index(r)]
                                                         for r in found if r not in moved))))
            self.update_capture_status()
        except Exception as exc:
            self.fail(str(exc))

    def import_pairs(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇內含 cf 與 cs 的校正照片資料夾",
                                                   storage.SESSIONS_DIR)
        if not folder:
            return
        try:
            if self.names or self.shot_count:
                raise ValueError("本次已有照片，請取消並開新一次校正後再匯入，避免混用不同相機擺位。")
            folder = Path(folder)
            names = match_image_pairs(folder)
            # Validate every image before accepting any pair into this session.
            for name in names:
                for role in self.VIEWS:
                    self.check_image_size(read_image(folder / role / name), role)
            for name in names:
                for role in self.VIEWS:
                    shutil.copy2(str(folder / role / name), str(self.session / role / name))
            # 來源若是之前的 session，只有單台看到的內參照片也一起帶進來
            extra = 0
            for sn in self.roles:
                source = folder / "intrinsic" / "images_SN{}".format(sn)
                if not self.reused and source.is_dir():
                    target = self.session / "intrinsic" / source.name
                    target.mkdir(parents=True, exist_ok=True)
                    for file in intrinsic.list_images(source):
                        shutil.copy2(str(file), str(target / file.name))
                        extra += 1
            self.names = names
            self.imported = True
            self.persist_session("capture_imported")
            self.log("已匯入 {} 組配對{}。請確認照片來自所選序號及目前相機擺位。".format(
                len(names), "、{} 張單台內參照片".format(extra) if extra else ""))
            # Imported filenames need not be numeric; keep this run import-only.
            self.connect_button.setEnabled(False)
            self.import_button.setEnabled(False)
            self.update_capture_status()
        except Exception as exc:
            self.fail(str(exc))

    def calculate(self):
        self.state = "calculate"
        self.capture_actions.hide()
        session, roles, reused = self.session, self.roles, self.reused
        pattern, square = (self.cols.value(), self.rows.value()), self.square.value()
        focals = {"front": self.front_focal.value(), "side": self.side_focal.value()}
        entries = dict(self.entries)
        # 計算在背景執行緒，訊息先收集起來，結束後再寫到 log
        self.job_messages = messages = []

        def work():
            if not reused:
                for lens, sn in zip(self.LENSES, roles):
                    entry, _ = intrinsic.calibrate_session_camera(session, lens, sn, pattern, square,
                                                                  focals[lens], messages.append)
                    if entry is None:
                        raise ValueError("SN{} 有效內參照片不足，請補拍。".format(sn))
                    validate_intrinsic(entry)
                    if entry["rms"] > intrinsic.RMS_MAX:
                        raise ValueError("SN{} 內參 RMS {:.3f} px 超過 {} px，請依 log 的提示補拍（模糊、反光、覆蓋不足）。"
                                         .format(sn, entry["rms"], intrinsic.RMS_MAX))
                    entries[sn] = entry
                save_json(session / "intrinsics.json", entries)
            messages.append("== 外參（cf/、cs/ 同名照片）")
            return entries, estimate_chessboard_stereo(session, roles, entries, pattern, square)

        self.log("正在計算{}外參（{} 組配對）…".format("" if reused else "兩台內參與", len(self.names)))
        self.start_job(work, self.calculation_finished, self.calculation_failed)

    def flush_job_messages(self):
        for message in self.job_messages:
            self.log(message)
        self.job_messages = []

    def calculation_failed(self, message):
        # 計算失敗時回到拍照頁，已拍的照片與統計保留，可補拍後再算
        self.flush_job_messages()
        self.begin_capture()
        self.fail(message)

    def calculation_finished(self, result):
        self.flush_job_messages()
        self.entries, report = result
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
            save_rejected(self.session, report)
            raise ValueError("Stereo RMS {:.2f} px 超過 {} px，未更新校正檔（報告已存於 {}）。\n"
                             "請確認棋盤格規格，並補拍不同位置與深度的配對照片。"
                             .format(fit["stereo_rms_px"], MAX_STEREO_RMS, self.session / "stereo_failed.json"))
        _, is_current = publish_stereo(storage.CALIB_DIR, self.session, self.roles, self.entries, report)
        self.state = "done"
        try:
            self.persist_session("complete")
        except OSError as exc:
            self.log("校正檔已套用，但工作階段狀態無法更新：{}".format(exc))
        self.instructions.setText("校正完成，已更新 Db/Calibration/stereo.json；之後的錄影都會用這次校正。"
                                  if is_current else
                                  "校正完成並已標記可用；目前使用中的是日期更新的另一次校正。")
        self.log("照片、角點預覽（corners/）、內參樣本與報告：{}".format(self.session))
        self.log("驗證組與訓練組來自同一批照片，仍建議用已知長度的物體現場驗證實際尺寸。")
        self.next_button.setText("開始另一次校正")
        self.cancel_button.setEnabled(False)

    def cancel(self):
        if self.job is not None:
            return
        def finish():
            if self.discard_empty_session():
                self.log("已取消；本次沒有拍到照片，已刪除空的工作階段資料夾。")
            else:
                try:
                    self.persist_session("cancelled")
                except OSError as exc:
                    self.log("無法更新草稿狀態：{}".format(exc))
                self.log("已取消；目前使用的校正值未變更，本次草稿保留在工作階段資料夾。")
            self.reset_ui()
        self.stop_capture(finish)

    def discard_empty_session(self):
        """本次一張照片都沒有就刪除 session 資料夾（例如開始後才發現相機沒畫面而取消），回傳是否已刪除"""
        if self.session is None or not self.session.is_dir():
            return False
        if any(p.suffix.lower() in intrinsic.IMAGE_SUFFIXES for p in self.session.rglob("*") if p.is_file()):
            return False
        try:
            shutil.rmtree(str(self.session))
        except OSError as exc:
            self.log("無法刪除空的工作階段資料夾 {}：{}".format(self.session, exc))
            return False
        self.session = None
        return True

    def closeEvent(self, event):
        if self.job is not None:
            self.log("正在計算，完成後即可關閉。")
            event.ignore()
            return
        if self.capture is not None:
            event.ignore()
            self.stop_capture(self.close)
            return
        if self.session is not None and self.state not in ("setup", "done") and not self.discard_empty_session():
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
        import intrinsic_window
    except ImportError:
        pass
    from qt_app import create_app
    app = create_app(sys.argv)
    window = CalibrationWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
