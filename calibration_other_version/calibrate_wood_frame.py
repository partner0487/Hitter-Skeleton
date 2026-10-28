"""Guided wooden-frame stereo PnP calibration (Python 3.8 / PyQt5)."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import copy
import json
import time
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QAbstractItemView, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QProgressBar, QProgressDialog, QPushButton, QShortcut, QSpinBox, QSplitter,
    QStackedWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

from Src.calibration import calib_store
from Src.calibration.calibration_workflow import (
    save_json, validate_intrinsic, validate_roles,
)
from Src.calibration.wood_frame import (
    wood_frame_target, fit_wood_frame_rig, save_wood_frame_rig,
)
from Src.calibration.calibrate import CaptureWorker, Job, write_image
from Src.calibration.calibrate_manual_coordinates import AnnotationCanvas
from Src.calibration.calibration_import import PairImportWorker, remove_staged_images


ROLES = ("cf", "cs")
POINT_IDS = {"cf": ["F1", "F2", "F3", "F4"], "cs": ["S1", "S2", "S3", "S4"]}
GUIDANCE = (
    "中央位置：木框完整入鏡，兩面標記都要清楚。",
    "保持相機固定，將木框向左移動，停穩再拍。",
    "將木框向右移動，確認兩台仍看得見各自四點。",
    "將木框提高，避免標記貼近畫面邊界。",
    "將木框降低，四點不要被支架遮住。",
    "將木框移近一些，保持兩面都清楚。",
    "將木框移遠一些，標記仍須足夠大。",
    "木框輕微向左轉，兩面不可接近側刃。",
    "木框輕微向右轉，兩面不可接近側刃。",
    "木框輕微前傾，四點不要重疊。",
    "木框輕微後傾，保持影像清晰。",
    "左上區域：改變位置並略微旋轉木框。",
    "右上區域：改變位置並略微旋轉木框。",
    "較近的中央位置：確認清晰度與兩面可見性。",
    "回到測量區域中央，最後確認兩台相機都沒有移動。",
)


def qimage_from_bgr(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                  QImage.Format_RGB888).copy()


def read_image(path):
    # AnnotationCanvas uses unrotated source pixels. Ignore JPEG EXIF orientation
    # here too, including 180-degree rotations which keep width/height unchanged.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                         cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("無法讀取影像：" + str(path))
    return image


def load_intrinsic_choices(path):
    """Accept both serial-indexed intrinsics and the existing chessboard file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("內參 JSON 必須為物件格式。")
    if data.get("schema") == "chessboard_intrinsics_v1":
        entries = data["cameras"]
    else:
        entries = data
    if not isinstance(entries, dict):
        raise ValueError("內參 cameras 必須為物件格式。")
    result = {}
    for key, entry in entries.items():
        if isinstance(entry, dict) and "intrinsic_matrix" in entry:
            try:
                validate_intrinsic(entry)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("內參 {} 格式無效：{}".format(key, exc)) from exc
            result[str(key)] = entry
    if not result:
        raise ValueError("檔案中沒有可用的相機內參。")
    return result


class WoodFrameWindow(QMainWindow):
    point_ids = POINT_IDS
    session_schema = "wood_frame_session_v1"
    session_suffix = "_wood_frame"
    minimum_pairs = 5
    maximum_pairs = 30
    target_definition = staticmethod(wood_frame_target)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("木框 PnP 外參校正")
        self.resize(1280, 860)
        self.session = None
        self.pairs = []
        self.cameras = {}
        self.roles = ()
        self.capture = None
        self.after_capture = None
        self.job = None
        self.result = None
        self.last_capture_time = None
        self.last_preview_time = None
        self.import_dialog = None
        self.close_after_import = False
        self.capture_error = None
        self.index = 0
        self.phase = "setup"
        self.entries = {}
        self.intrinsic_source = str(Path(calib_store.INTRINSICS_JSON))
        self._build_ui()
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.update_preview)
        self.preview_timer.start(100)
        self.shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut.activated.connect(self.capture_pair)
        self._load_defaults()

    @staticmethod
    def button(text, callback, layout):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("木框外參校正")
        title.setStyleSheet("font-size: 23px; font-weight: 600;")
        self.title_label = title
        layout.addWidget(title)
        self.steps = QLabel("1 設定內參  →  2 拍攝約 15 組  →  3 標記四點  →  4 儲存最佳組")
        layout.addWidget(self.steps)
        self.instruction = QLabel()
        self.instruction.setWordWrap(True)
        self.instruction.setStyleSheet("background:#eaf2fb; color:#15334e; padding:12px;")
        layout.addWidget(self.instruction)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        setup = QWidget()
        self.setup_page = setup
        form = QVBoxLayout(setup)
        geometry = QLabel(
            "固定木框座標（mm），紙板厚度 0.3 cm = 3 mm，已包含在座標中。\n"
            "正面：F1 (80,320,-3) → F2 (320,320,-3) → F3 (320,80,-3) → F4 (80,80,-3)\n"
            "側面：S1 (403,320,30) → S2 (403,320,270) → S3 (403,80,270) → S4 (403,80,30)\n"
            "標記實體角點的編號必須固定；不可依照片上的左右位置重新排序。")
        geometry.setWordWrap(True)
        self.geometry_label = geometry
        form.addWidget(geometry)
        choices = QGroupBox("使用已校正的內參")
        fields = QFormLayout(choices)
        self.intrinsic_label = QLabel()
        self.intrinsic_label.setWordWrap(True)
        fields.addRow(self.intrinsic_label)
        self.combos = {}
        self.serials = {}
        from PyQt5.QtWidgets import QLineEdit
        for role, title in zip(ROLES, ("正面 CF", "側面 CS")):
            combo = QComboBox()
            self.combos[role] = combo
            fields.addRow(title + " 內參", combo)
            serial = QLineEdit()
            serial.setPlaceholderText("實體相機序號（即時拍攝必填）")
            self.serials[role] = serial
            fields.addRow(title + " 序號", serial)
            combo.currentTextChanged.connect(
                lambda key, r=role: self.serials[r].setText(key if key not in ROLES else ""))
        self.target_count = QSpinBox()
        self.target_count.setRange(5, 30)
        self.target_count.setValue(15)
        fields.addRow("目標雙機組數", self.target_count)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.1, 10)
        self.threshold.setValue(3)
        self.threshold.setSuffix(" px")
        fields.addRow("單張重投影 RMS 上限", self.threshold)
        form.addWidget(choices)
        setup_actions = QHBoxLayout()
        self.button("選擇內參 JSON…", self.choose_intrinsics, setup_actions)
        self.button("載入未完成的工作階段…", self.resume_session, setup_actions)
        self.button("開始木框校正", self.start_session, setup_actions)
        form.addLayout(setup_actions)
        note = QLabel("先用 calib.py 完成兩台內參。拍外參時保持相機位置、焦距、解析度及裁切不變。\n"
                      "每一組是同一木框姿態的正面＋側面照片；即時拍攝需接 GPIO 同步線。")
        note.setWordWrap(True)
        form.addWidget(note)
        form.addStretch()
        self.pages.addWidget(setup)

        pair_page = QWidget()
        self.pair_page = pair_page
        pair_layout = QVBoxLayout(pair_page)
        self.progress = QProgressBar()
        pair_layout.addWidget(self.progress)
        self.pair_label = QLabel()
        pair_layout.addWidget(self.pair_label)
        splitter = QSplitter(Qt.Horizontal)
        self.views, self.point_combos = {}, {}
        for role, title in zip(ROLES, ("正面 CF · F1 → F4", "側面 CS · S1 → S4")):
            pane = QWidget()
            col = QVBoxLayout(pane)
            col.addWidget(QLabel(title))
            view = AnnotationCanvas()
            self.views[role] = view
            view.clicked.connect(lambda x, y, r=role: self.mark_point(r, x, y))
            col.addWidget(view, 1)
            row = QHBoxLayout()
            combo = QComboBox()
            combo.addItems(self.point_ids[role])
            self.point_combos[role] = combo
            row.addWidget(QLabel("目前標記"))
            row.addWidget(combo)
            self.button("撤銷一點", lambda checked=False, r=role: self.undo_point(r), row)
            self.button("符合視窗", view.fit_image, row)
            col.addLayout(row)
            splitter.addWidget(pane)
        pair_layout.addWidget(splitter, 1)
        self.capture_bar = QWidget()
        row = QHBoxLayout(self.capture_bar)
        self.connect_button = self.button("連接雙相機", self.connect_pair, row)
        self.shot_button = self.button("拍攝一組（空白鍵）", self.capture_pair, row)
        self.shot_button.setEnabled(False)
        self.stop_button = self.button("停止相機", lambda: self.stop_capture(), row)
        self.import_button = self.button("匯入 cf／cs 照片資料夾…", self.import_pairs, row)
        self.annotate_button = self.button("完成拍攝，開始標記", self.begin_annotation, row)
        pair_layout.addWidget(self.capture_bar)
        self.annotation_bar = QWidget()
        row = QHBoxLayout(self.annotation_bar)
        self.include_pair = QCheckBox("採用這組（取消可略過誤拍）")
        self.include_pair.setChecked(True)
        self.include_pair.toggled.connect(self.set_pair_included)
        row.addWidget(self.include_pair)
        self.previous_button = self.button("上一組", lambda: self.move_pair(-1), row)
        self.next_button = self.button("下一組", lambda: self.move_pair(1), row)
        self.button("返回拍攝／補拍", self.return_to_capture, row)
        self.calculate_button = self.button("計算並比較所有組", self.calculate, row)
        pair_layout.addWidget(self.annotation_bar)
        self.annotation_bar.hide()
        self.pages.addWidget(pair_page)

        result_page = QWidget()
        result_layout = QVBoxLayout(result_page)
        self.result_summary = QLabel()
        self.result_summary.setWordWrap(True)
        self.result_summary.setStyleSheet("font-size:16px; padding:10px;")
        result_layout.addWidget(self.result_summary)
        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(
            ["組別", "結果", "正面 RMS (px)", "側面 RMS (px)", "跨組誤差 (px)", "說明"])
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        result_layout.addWidget(self.result_table, 1)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(160)
        result_layout.addWidget(self.result_text)
        row = QHBoxLayout()
        self.button("檢視最佳組照片", self.show_best_pair, row)
        self.button("返回標記／修正", self.begin_annotation, row)
        self.button("儲存最佳一組外參…", self.export_result, row)
        self.button("開啟手動原點／XYZ／尺度 UI", self.open_manual, row)
        result_layout.addLayout(row)
        self.pages.addWidget(result_page)
        self.result_page = result_page
        bottom = QHBoxLayout()
        self.status = QLabel("準備設定。")
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.reset_button = self.button("新工作階段", self.reset, bottom)
        layout.addLayout(bottom)
        self.set_setup()

    def set_setup(self):
        self.phase = "setup"
        self.pages.setCurrentWidget(self.setup_page)
        self.instruction.setText("選擇正面與側面內參，再開始。木框尺寸與點位已固定，不需要輸入尺寸。")

    def fail(self, message):
        self.status.setText(message)
        QMessageBox.warning(self, "木框校正", message)

    def _load_defaults(self):
        try:
            if Path(self.intrinsic_source).exists():
                self.set_intrinsics(load_intrinsic_choices(self.intrinsic_source), self.intrinsic_source)
            saved = calib_store.load_roles()
            if saved:
                for role, sn in zip(ROLES, saved):
                    index = self.combos[role].findText(sn)
                    if index >= 0:
                        self.combos[role].setCurrentIndex(index)
                    self.serials[role].setText(sn)
        except (ValueError, KeyError, OSError) as exc:
            self.status.setText("尚無完整內參，請選擇內參 JSON：" + str(exc))

    def set_intrinsics(self, entries, source):
        self.entries = entries
        self.intrinsic_source = str(source)
        self.intrinsic_label.setText(str(source))
        keys = list(entries)
        for i, role in enumerate(ROLES):
            self.combos[role].clear()
            self.combos[role].addItems(keys)
            if role in keys:
                self.combos[role].setCurrentText(role)
            else:
                self.combos[role].setCurrentIndex(min(i, len(keys) - 1))

    def choose_intrinsics(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇內參 JSON", self.intrinsic_source, "JSON (*.json)")
        if path:
            try:
                self.set_intrinsics(load_intrinsic_choices(path), path)
            except (ValueError, KeyError, OSError) as exc:
                self.fail(str(exc))

    def start_session(self):
        try:
            keys = [self.combos[r].currentText() for r in ROLES]
            if not all(k in self.entries for k in keys) or keys[0] == keys[1]:
                raise ValueError("請分別選擇正面、側面兩組不同內參。")
            cameras = {r: copy.deepcopy(self.entries[k]) for r, k in zip(ROLES, keys)}
            for camera in cameras.values():
                validate_intrinsic(camera)
            session = Path(calib_store.CALIB_DIR) / "sessions" / (
                datetime.now().strftime("%Y%m%d_%H%M%S_%f") + self.session_suffix)
            for role in ROLES:
                (session / role).mkdir(parents=True, exist_ok=True)
            self.cameras, self.session = cameras, session
            self.roles = tuple(self.serials[r].text().strip() for r in ROLES)
            self.pairs, self.result = [], None
            self.last_capture_time = None
            self.active_target_count = self.target_count.value()
            self.active_threshold = self.threshold.value()
            self.persist()
            self.return_to_capture()
        except (ValueError, OSError) as exc:
            self.fail(str(exc))

    def persist(self):
        if self.session is None:
            return
        save_json(self.session / "session.json", {
            "schema": self.session_schema, "target": self.target_definition(),
            "cameras": self.cameras, "roles": list(self.roles),
            "target_count": self.active_target_count, "max_rms_px": self.active_threshold,
            "intrinsic_source": self.intrinsic_source, "pairs": self.pairs,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

    def resume_session(self, path=None):
        if self.job is not None or self.capture is not None:
            return False
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "選擇工作階段 session.json",
                                                  str(Path(calib_store.CALIB_DIR) / "sessions"), "JSON (*.json)")
        if not path:
            return False
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("工作階段 JSON 必須為物件格式。")
            if data.get("schema") != self.session_schema or data["target"] != self.target_definition():
                raise ValueError("工作階段格式或木框點位不符。")
            cameras = data["cameras"]
            for role in ROLES:
                validate_intrinsic(cameras[role])
            pairs = data["pairs"]
            if not isinstance(pairs, list):
                raise ValueError("工作階段 pairs 必須為陣列。")
            names = set()
            for pair in pairs:
                if not isinstance(pair, dict) or not isinstance(pair.get("excluded", False), bool):
                    raise ValueError("工作階段組別格式無效。")
                if not isinstance(pair["id"], str) or pair["id"] in names:
                    raise ValueError("工作階段含重複組別。")
                names.add(pair["id"])
                for role in ROLES:
                    if not isinstance(pair[role], dict) or not isinstance(pair[role].get("points"), dict):
                        raise ValueError("工作階段的標記 points 必須為物件。")
                    image = read_image(pair[role]["path"])
                    size = [image.shape[1], image.shape[0]]
                    if size != list(cameras[role]["image_size"]) or size != pair[role]["image_size"]:
                        raise ValueError("工作階段照片與內參尺寸不符。")
                    for key, point in pair[role]["points"].items():
                        xy = np.asarray(point, float)
                        if (key not in self.point_ids[role] or xy.shape != (2,)
                                or not np.isfinite(xy).all() or np.any(xy < 0)
                                or xy[0] >= size[0] or xy[1] >= size[1]):
                            raise ValueError("工作階段包含無效標記。")
            target_count = int(data["target_count"])
            threshold = float(data["max_rms_px"])
            if not self.minimum_pairs <= target_count <= self.maximum_pairs or not np.isfinite(threshold) or not 0.1 <= threshold <= 10:
                raise ValueError("工作階段的取樣數或誤差門檻無效。")
            roles = data["roles"]
            if not isinstance(roles, list) or len(roles) != 2 or not all(isinstance(r, str) for r in roles):
                raise ValueError("工作階段缺少相機序號欄位。")
            self.session, self.pairs, self.cameras = Path(path).parent, pairs, cameras
            self.roles = tuple(roles)
            self.active_target_count, self.active_threshold = target_count, threshold
            self.intrinsic_source = data.get("intrinsic_source", "")
            self.result, self.last_capture_time = None, None
            if self.pairs:
                self.begin_annotation()
            else:
                self.return_to_capture()
            return True
        except (ValueError, KeyError, TypeError, AttributeError, OSError, cv2.error) as exc:
            self.fail("無法載入工作階段：" + str(exc))
            return False

    def return_to_capture(self):
        if self.job is not None:
            return
        self.phase = "capture"
        self.result = None
        self.last_preview_time = None
        for view in self.views.values():
            view.clear_image()
        self.pages.setCurrentWidget(self.pair_page)
        self.capture_bar.show()
        self.annotation_bar.hide()
        self.update_progress()

    def update_progress(self):
        total = len(self.pairs)
        self.progress.setMaximum(max(total, self.active_target_count))
        self.progress.setValue(total)
        self.pair_label.setText("已拍攝／匯入 {} 組（目標 {} 組，每組正面＋側面各一張）".format(
            total, self.active_target_count))
        self.instruction.setText("第 {} 組提示：{} 相機全程保持固定，木框停穩後再拍。".format(
            total + 1, GUIDANCE[total % len(GUIDANCE)]) if total < self.active_target_count else
            "已達目標組數。可繼續補拍，或按「完成拍攝，開始標記」。")
        self.status.setText("工作階段：" + str(self.session))
        self.connect_button.setEnabled(self.capture is None)
        self.import_button.setEnabled(self.capture is None)
        self.stop_button.setEnabled(self.capture is not None)
        self.annotate_button.setEnabled(bool(self.pairs))

    def connect_pair(self):
        if self.capture is not None or self.phase != "capture" or self.job is not None:
            return
        try:
            validate_roles(*self.roles)
            self.capture_error = None
            self.last_preview_time = None
            self.capture = CaptureWorker(self.roles, self)
            self.capture.failed.connect(self.capture_failed)
            self.capture.finished.connect(self.capture_stopped)
            self.capture.start()
            self.update_progress()
            self.status.setText("正在連接雙相機；請確認 GPIO 同步線與 USB 連線。")
        except (ValueError, OSError) as exc:
            self.fail(str(exc))

    def capture_failed(self, message):
        self.capture_error = message
        self.status.setText(message)

    def update_preview(self):
        if self.phase != "capture" or self.capture is None or self.after_capture is not None:
            return
        pair = self.capture.latest
        fresh = pair is not None and time.monotonic() - pair[0] < 2
        self.shot_button.setEnabled(fresh)
        if fresh and pair[0] != self.last_preview_time:
            for role, image in zip(ROLES, pair[1:]):
                self.views[role].update_preview_image(qimage_from_bgr(image))
            self.last_preview_time = pair[0]

    def stop_capture(self, callback=None):
        if self.capture is None:
            if callback:
                callback()
            return
        self.after_capture = callback
        self.capture.requestInterruption()
        self.capture_bar.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.status.setText("正在停止並釋放雙相機…")

    def capture_stopped(self):
        if self.capture is not None:
            self.capture.deleteLater()
            self.capture = None
        self.shot_button.setEnabled(False)
        self.capture_bar.setEnabled(True)
        self.reset_button.setEnabled(True)
        callback, self.after_capture = self.after_capture, None
        if self.phase == "capture":
            self.update_progress()
        if callback:
            callback()
        if self.capture_error:
            self.status.setText(self.capture_error)

    def check_size(self, role, image):
        size = [image.shape[1], image.shape[0]]
        if size != list(self.cameras[role]["image_size"]):
            raise ValueError("{} 影像尺寸 {} 與內參 {} 不符；請確認解析度及裁切。".format(
                role.upper(), size, self.cameras[role]["image_size"]))
        return size

    def capture_pair(self):
        if self.phase != "capture" or self.job is not None or not self.capture_bar.isEnabled():
            return
        try:
            snapshot = self.capture.latest if self.capture is not None else None
            if snapshot is None or time.monotonic() - snapshot[0] > 2:
                raise ValueError("尚未收到新的雙機影像，請先連接相機。")
            if snapshot[0] == self.last_capture_time:
                return
            pair_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            pair = {"id": pair_id}
            for role, image in zip(ROLES, snapshot[1:]):
                size = self.check_size(role, image)
                pair[role] = {"points": {}, "image_size": size,
                              "path": str((self.session / role / (pair_id + ".png")).resolve())}
            written = []
            try:
                for role, image in zip(ROLES, snapshot[1:]):
                    written.append(Path(pair[role]["path"]))
                    write_image(pair[role]["path"], image)
                self.pairs.append(pair)
                try:
                    self.persist()
                except Exception:
                    self.pairs.pop()
                    raise
            except Exception:
                remove_staged_images(written)
                raise
            self.last_capture_time = snapshot[0]
            self.update_progress()
        except (ValueError, OSError, cv2.error) as exc:
            self.fail(str(exc))

    def import_pairs(self, folder=None):
        if self.capture is not None or self.job is not None or self.phase != "capture":
            return
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "選擇包含 cf 與 cs 的同名成對照片資料夾")
        if not folder:
            return
        self.pages.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.status.setText("正在背景匯入照片，可取消；完成後才會保存工作階段。")
        self.job = PairImportWorker(folder, self.session, self.cameras, read_image, self)
        self.import_dialog = QProgressDialog("準備匯入照片…", "取消匯入", 0, 0, self)
        self.import_dialog.setWindowTitle("匯入雙機照片")
        # The main controls are disabled. Non-modal progress avoids setValue()
        # pumping nested events while the finished handler commits the import.
        self.import_dialog.setWindowModality(Qt.NonModal)
        self.import_dialog.setAutoClose(False)
        self.import_dialog.setAutoReset(False)
        self.import_dialog.canceled.connect(self.job.cancel)
        self.job.progress.connect(self.import_progress)
        self.job.finished.connect(self.import_finished)
        self.import_dialog.show()
        self.job.start()

    def import_progress(self, completed, total, name):
        if self.import_dialog is not None:
            self.import_dialog.setRange(0, total)
            self.import_dialog.setLabelText("匯入照片 {}/{}：{}".format(completed, total, name))
            self.import_dialog.setValue(completed)

    def import_finished(self):
        job, self.job = self.job, None
        closing, self.close_after_import = self.close_after_import, False
        if self.import_dialog is not None:
            self.import_dialog.hide()
            self.import_dialog.deleteLater()
            self.import_dialog = None
        self.pages.setEnabled(True)
        self.reset_button.setEnabled(True)
        pending = job.result or []
        indices = []
        try:
            if job.error:
                raise ValueError(job.error)
            if job.cancelled or job.isInterruptionRequested() or closing:
                remove_staged_images(p[r]["path"] for p in pending for r in ROLES)
                self.status.setText("已取消匯入，既有照片與標記已保留。")
                return
            old_pairs = self.pairs
            self.pairs = self.pairs + pending
            try:
                self.persist()
            except Exception:
                self.pairs = old_pairs
                remove_staged_images(p[r]["path"] for p in pending for r in ROLES)
                raise
            self.result = None
            self.update_progress()
            indices = list(range(len(old_pairs), len(self.pairs)))
        except (ValueError, OSError, cv2.error) as exc:
            self.fail(str(exc))
        finally:
            job.deleteLater()
            if closing:
                self.close()
        if indices:
            self.on_pairs_imported(indices)

    def on_pairs_imported(self, indices):
        """Subclass hook, called after the import is durably committed."""

    def begin_annotation(self):
        if self.job is not None:
            return
        if not self.pairs:
            self.fail("請先拍攝或匯入照片。")
            return
        if self.capture is not None:
            self.stop_capture(self.begin_annotation)
            return
        self.phase = "annotate"
        self.pages.setCurrentWidget(self.pair_page)
        self.capture_bar.hide()
        self.annotation_bar.show()
        self.index = min(self.index, len(self.pairs) - 1)
        self.show_pair()

    def show_pair(self):
        pair = self.pairs[self.index]
        for view in self.views.values():
            view.clear_image()
        try:
            for role in ROLES:
                self.views[role].set_image(pair[role]["path"])
                self.views[role].set_points(pair[role]["points"])
                missing = [key for key in self.point_ids[role] if key not in pair[role]["points"]]
                self.point_combos[role].setCurrentText(missing[0] if missing else self.point_ids[role][0])
        except (ValueError, OSError) as exc:
            for view in self.views.values():
                view.clear_image()
            self.fail(str(exc))
        self.update_annotation_status()

    def update_annotation_status(self):
        selected = [pair for pair in self.pairs if not pair.get("excluded", False)]
        completed = sum(all(set(pair[r]["points"]) == set(self.point_ids[r]) for r in ROLES)
                        for pair in selected)
        self.progress.setMaximum(max(1, len(selected)))
        self.progress.setValue(completed)
        pair = self.pairs[self.index]
        self.include_pair.blockSignals(True)
        self.include_pair.setChecked(not pair.get("excluded", False))
        self.include_pair.blockSignals(False)
        self.pair_label.setText("第 {}/{} 組 · {}｜正面 {}/4、側面 {}/4｜完成 {}/{} 組，略過 {} 組".format(
            self.index + 1, len(self.pairs), pair.get("original_name", pair["id"]),
            len(pair["cf"]["points"]), len(pair["cs"]["points"]), completed, len(selected),
            len(self.pairs) - len(selected)))
        self.instruction.setText(
            "依實體編號點選：F1 (80,320,-3) → F2 (320,320,-3) → F3 (320,80,-3) → F4 (80,80,-3)；"
            "S1 (403,320,30) → S2 (403,320,270) → S3 (403,80,270) → S4 (403,80,30)。\n"
            "左鍵標記；選擇點名可重標；滾輪縮放、中鍵拖曳平移。點位自動保存。")
        self.previous_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.pairs) - 1)
        self.calculate_button.setEnabled(completed == len(selected) and completed >= self.minimum_pairs)

    def set_pair_included(self, included):
        if self.phase != "annotate" or self.job is not None:
            return
        pair = self.pairs[self.index]
        previous = pair.get("excluded", False)
        pair["excluded"] = not included
        try:
            self.persist()
            self.result = None
        except OSError as exc:
            pair["excluded"] = previous
            self.fail("略過設定未保存：" + str(exc))
        self.update_annotation_status()

    def mark_point(self, role, x, y):
        if self.phase != "annotate" or self.job is not None:
            return False
        size = self.views[role].image_size
        if (size is None or not np.isfinite([x, y]).all()
                or not 0 <= x < size[0] or not 0 <= y < size[1]):
            return False
        points = self.pairs[self.index][role]["points"]
        key = self.point_combos[role].currentText()
        old = copy.deepcopy(points)
        points[key] = [float(x), float(y)]
        try:
            self.persist()
        except OSError as exc:
            points.clear()
            points.update(old)
            self.fail("點位未保存：" + str(exc))
            return False
        self.result = None
        self.views[role].set_points(points)
        missing = [point for point in self.point_ids[role] if point not in points]
        if missing:
            self.point_combos[role].setCurrentText(missing[0])
        self.update_annotation_status()
        return True

    def undo_point(self, role):
        if self.phase != "annotate" or self.job is not None:
            return False
        points = self.pairs[self.index][role]["points"]
        if points:
            key = next(reversed(points))
            previous = points.pop(key)
            try:
                self.persist()
            except OSError as exc:
                points[key] = previous
                self.fail("撤銷未保存：" + str(exc))
                return False
            self.point_combos[role].setCurrentText(key)
            self.views[role].set_points(points)
            self.result = None
            self.update_annotation_status()
            return True
        return False

    def move_pair(self, offset):
        self.index = max(0, min(len(self.pairs) - 1, self.index + offset))
        self.show_pair()

    def calculate(self):
        if self.job is not None:
            return
        selected = [p for p in self.pairs if not p.get("excluded", False)]
        if len(selected) < 5 or any(set(p[r]["points"]) != set(self.point_ids[r])
                                   for p in selected for r in ROLES):
            self.fail("至少需要 5 組，且每組必須完成正面及側面各 4 點；建議拍攝 15 組。")
            return
        cameras, pairs = copy.deepcopy(self.cameras), copy.deepcopy(selected)
        threshold = self.active_threshold
        self.pages.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.status.setText("正在解算各組 PnP，檢查平面解與雙機一致性…")
        self.job = Job(lambda: fit_wood_frame_rig(cameras, self.target_definition(), pairs,
                                                 min_pairs=5, max_rms_px=threshold), self)
        self.job.finished.connect(self.calculation_finished)
        self.job.start()

    def calculation_finished(self):
        job, self.job = self.job, None
        self.pages.setEnabled(True)
        self.reset_button.setEnabled(True)
        try:
            if job.error:
                raise ValueError(job.error)
            self.result = job.result
            self.result["camera_serials"] = dict(zip(ROLES, self.roles))
            self.result["source_session"] = str(self.session.resolve())
            self.result["source_intrinsics"] = self.intrinsic_source
            self.result["manually_excluded_pairs"] = [
                p["id"] for p in self.pairs if p.get("excluded", False)]
            save_wood_frame_rig(self.session / "best_stereo.json", self.result)
            self.phase = "result"
            self.pages.setCurrentWidget(self.result_page)
            self.instruction.setText("已比較完成。下方保留評分與各組診斷；按「儲存最佳一組外參」匯出供後續讀取。")
            self.show_results()
            self.status.setText("工作階段結果已保存：" + str(self.session / "best_stereo.json"))
        except (ValueError, OSError) as exc:
            self.result = None
            self.fail(str(exc))
        finally:
            job.deleteLater()

    def show_results(self):
        fit = self.result["fit"]
        best = self.result["best_pair_id"]
        number = next(i + 1 for i, p in enumerate(self.pairs) if p["id"] == best)
        self.result_summary.setText(
            "最佳：第 {} 組　｜　雙機基線 {:.1f} mm\n"
            "本組 RMS {:.3f} px　｜　其他組中位跨相機誤差 {:.3f} px　｜　一致組數 {}/{}".format(
                number, self.result["baseline_mm"], fit["pair_rms_px"],
                fit["other_pairs_median_rms_px"], fit["support_pairs"],
                len(self.result["candidate_diagnostics"])))
        diagnostics = self.result["candidate_diagnostics"]
        self.result_table.setRowCount(len(diagnostics))
        statuses = {"best": "★ 最佳", "accepted": "通過", "rejected": "未採用"}
        for row, diagnostic in enumerate(diagnostics):
            def value(number):
                return "—" if number is None else "{:.3f}".format(number)
            pair_number = next(i + 1 for i, p in enumerate(self.pairs) if p["id"] == diagnostic["id"])
            items = [str(pair_number), statuses.get(diagnostic["status"], diagnostic["status"]),
                     value(diagnostic["cameras"].get("cf", {}).get("rms_px")),
                     value(diagnostic["cameras"].get("cs", {}).get("rms_px")),
                     value(diagnostic.get("cross_projection_rms_px")), diagnostic.get("reason", "")]
            for col, item in enumerate(items):
                self.result_table.setItem(row, col, QTableWidgetItem(item))
        notes = ["依其他照片的雙機幾何一致性與本組重投影誤差選出一組；所有評分隨外參一起保存。",
                 "世界座標依最佳照片中木框的實際位置與方向。更換木框姿態不影響雙機相對外參。",
                 "誤差為照片檢查結果，請再以現場已知距離確認實際量測精度。"]
        notes.extend(self.result.get("warnings", []))
        self.result_text.setPlainText("\n".join(notes))

    def show_best_pair(self):
        if self.result is not None:
            self.index = next(i for i, p in enumerate(self.pairs)
                              if p["id"] == self.result["best_pair_id"])
            self.begin_annotation()

    def export_result(self):
        if self.result is None:
            return
        default = Path(calib_store.CALIB_DIR) / "wood_frame" / "stereo.json"
        path, _ = QFileDialog.getSaveFileName(self, "儲存最佳一組 PnP 外參", str(default), "JSON (*.json)")
        if not path:
            return
        try:
            save_wood_frame_rig(path, self.result)
            self.status.setText("已儲存：" + str(path))
            QMessageBox.information(self, "校正完成", "最佳一組外參已儲存。\n" + str(path)
                                    + "\n可由公制 3D／量測工具讀取此 stereo.json。")
        except (ValueError, OSError) as exc:
            self.fail("儲存失敗，可重試：" + str(exc))

    def open_manual(self):
        from Src.calibration.calibrate_manual_coordinates import ManualCoordinatesWindow
        if not hasattr(self, "manual_window"):
            self.manual_window = ManualCoordinatesWindow()
        self.manual_window.show()
        self.manual_window.raise_()

    def reset(self):
        if self.job is not None:
            return
        if self.capture is not None:
            self.stop_capture(self.reset)
            return
        self.session, self.result = None, None
        self.pairs = []
        self.index = 0
        self.set_setup()
        self.status.setText("既有工作階段已保留，可重新載入。")

    def closeEvent(self, event):
        if isinstance(self.job, PairImportWorker):
            self.close_after_import = True
            self.job.cancel()
            self.status.setText("正在取消匯入並清理暫存照片，完成後關閉。")
            event.ignore()
        elif self.job is not None:
            self.status.setText("計算中，完成後即可關閉。")
            event.ignore()
        elif self.capture is not None:
            event.ignore()
            self.stop_capture(self.close)
        else:
            self.preview_timer.stop()
            event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="木框 PnP：引導拍照、標記、選取最佳外參")
    parser.parse_args(argv)
    # Resolve optional native SDK dependencies on the main thread before Qt
    # starts worker threads, as in the existing calibration entry point.
    try:
        from Src.cameras.single_camera.flir_camera_system import FlirCameraSystem
        from Src.cameras.two_cameras.dual_flir_system import DualFlirSystem
    except ImportError:
        pass  # Offline photo import remains available; capture reports the error.
    app = QApplication.instance() or QApplication(sys.argv)
    window = WoodFrameWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
