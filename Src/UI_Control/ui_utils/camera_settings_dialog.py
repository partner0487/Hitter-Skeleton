"""相機設定視窗：上方選正面、側面各用哪台相機（做法同校正工具：掃描連接的相機再選序號），
下方正面、側面兩台並排，調曝光時間與增益。

相機選擇：序號存到 Db/Calibration/camera_roles.json（與校正共用）；正面可選「不使用」，
只接側面一台就錄成純側面 2D（utils/camera_roles.py）。相機開著時不能換，關掉相機再選，下次開啟生效。

改動即時套用到相機，並自動寫回各自的 yaml（只改 exposure_time、gain_value 兩行，見
FlirCameraSystem.update_exposure），下次開相機沿用。相機沒開時欄位不能調。
"""
import math
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout)

from utils import camera_roles

NO_FRONT = "不使用（只錄側面，純側面 2D）"


class CameraPanel(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.camera = None
        form = QFormLayout(self)
        self.exposure = QDoubleSpinBox()
        self.exposure.setDecimals(0)
        self.exposure.setSingleStep(100)
        self.exposure.setSuffix(" μs")
        self.gain = QDoubleSpinBox()
        self.gain.setDecimals(1)
        self.gain.setSingleStep(1)
        self.gain.setSuffix(" dB")
        for spin in (self.exposure, self.gain):
            # 打字時不要每按一鍵就送到相機，按 Enter 或離開欄位才套用；方向鍵／滾輪仍即時套用
            spin.setKeyboardTracking(False)
            spin.setAlignment(Qt.AlignRight)
            spin.setMinimumWidth(140)
            spin.valueChanged.connect(self.apply)
        self.exposure_range = QLabel()
        self.gain_range = QLabel()
        self.info = QLabel()
        self.status = QLabel()
        for label in (self.exposure_range, self.gain_range, self.info, self.status):
            label.setStyleSheet("color: gray;")
        form.addRow("曝光時間", self.exposure)
        form.addRow("", self.exposure_range)
        form.addRow("增益", self.gain)
        form.addRow("", self.gain_range)
        form.addRow("", self.info)
        form.addRow("", self.status)
        self.bind(None)

    def bind(self, camera, idle_text="請先在錄影頁勾選「開啟相機」"):
        """camera 為 FlirCameraSystem；None 表示相機沒開（或這台沒用），狀態列顯示 idle_text"""
        self.camera = camera
        settings = None
        if camera is not None:
            try:
                settings = camera.get_exposure_settings()
            except Exception as e:
                self.status.setText(f"讀取失敗：{e}")
        for spin in (self.exposure, self.gain):
            spin.setEnabled(settings is not None)
        if settings is None:
            for spin in (self.exposure, self.gain):
                spin.blockSignals(True)
                spin.setSpecialValueText("—")
                spin.setRange(0, 0)
                spin.blockSignals(False)
            for label in (self.exposure_range, self.gain_range, self.info):
                label.setText("")
            if camera is None:
                self.status.setText(idle_text)
            return
        for spin, value, (low, high) in ((self.exposure, settings["exposure_time"], settings["exposure_range"]),
                                         (self.gain, settings["gain"], settings["gain_range"])):
            spin.blockSignals(True)
            spin.setSpecialValueText("")
            step = 10 ** -spin.decimals()  # 直接 setRange 會把上限四捨五入而超過相機上限
            spin.setRange(math.ceil(low / step) * step, math.floor(high / step) * step)
            spin.setValue(value)
            spin.blockSignals(False)
        self.exposure_range.setText("可設 {:g} ~ {:g} μs（上限受幀率限制）".format(self.exposure.minimum(),
                                                                                self.exposure.maximum()))
        self.gain_range.setText("可設 {:.1f} ~ {:.1f} dB".format(self.gain.minimum(), self.gain.maximum()))
        self.info.setText("序號 {}｜{} × {}｜{} fps".format(camera.serial_number, camera.get_width(),
                                                           camera.get_height(), camera.get_fps()))
        self.status.setText("")

    def apply(self, *_):
        if self.camera is None:
            return
        try:
            exposure, gain = self.camera.update_exposure(self.exposure.value(), self.gain.value())
        except Exception as e:
            self.status.setText(f"設定失敗：{e}")
            return
        for spin, value in ((self.exposure, exposure), (self.gain, gain)):
            spin.blockSignals(True)
            spin.setValue(value)   # 顯示實際套用的值（超出範圍會被夾住）
            spin.blockSignals(False)
        self.status.setText(f"已自動儲存到 {os.path.basename(self.camera.config_yaml_path)}")


class CameraRolesBox(QGroupBox):
    """正面、側面各用哪台相機；改了立即存檔，下次開相機生效"""

    def __init__(self):
        super().__init__("相機選擇")
        form = QFormLayout(self)
        self.front_combo, self.side_combo = QComboBox(), QComboBox()
        for combo in (self.front_combo, self.side_combo):
            combo.setMinimumWidth(260)
            combo.activated.connect(self.save)
        form.addRow("正面相機", self.front_combo)
        form.addRow("側面相機", self.side_combo)
        self.scan_button = QPushButton("重新掃描相機")
        self.scan_button.clicked.connect(self.scan)
        form.addRow("", self.scan_button)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray;")
        form.addRow("", self.status)
        self.connected = []

    def scan(self):
        try:
            self.connected = camera_roles.list_cameras()
        except Exception as e:
            self.connected = []
            self.status.setText(f"無法掃描相機：{e}")
        else:
            found = "、".join(self.connected) if self.connected else "沒有偵測到相機，請確認 USB 連線並關閉其他相機程式"
            self.status.setText(f"目前連接：{found}。擊球觸發與 2D 分析只用側面，正面只用於 3D。")
        self.fill()

    def fill(self):
        front, side = camera_roles.load_roles()
        serials = list(self.connected) + sorted({front, side}.difference(self.connected))
        label = lambda sn: sn if sn in self.connected else f"{sn}（未連接）"
        for combo, current, extra in ((self.front_combo, front if camera_roles.use_front() else None, [NO_FRONT]),
                                      (self.side_combo, side, [])):
            combo.blockSignals(True)
            combo.clear()
            for text in extra:
                combo.addItem(text, None)
            for sn in serials:
                combo.addItem(label(sn), sn)
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.blockSignals(False)

    def save(self, *_):
        front, side = self.front_combo.currentData(), self.side_combo.currentData()
        if not side:
            return
        if front == side:
            self.status.setText("正面和側面不能是同一台相機，未儲存。")
            self.fill()
            return
        saved_front, _ = camera_roles.load_roles()
        camera_roles.save_roles(front or saved_front, side)   # 不使用正面時保留原本的正面序號給校正
        camera_roles.set_use_front(front is not None)
        mode = "正面＋側面（2D + 3D）" if front else "只用側面（純側面 2D）"
        self.status.setText(f"已儲存：{mode}，下次開啟相機生效。")

    def setLocked(self, locked):
        """相機開著時不能換"""
        for widget in (self.front_combo, self.side_combo, self.scan_button):
            widget.setEnabled(not locked)
        self.setTitle("相機選擇（關閉相機後才能更換）" if locked else "相機選擇")


class CameraSettingsDialog(QDialog):
    """cameras：回傳 (正面, 側面) FlirCameraSystem 的函式，相機沒開時回傳 None"""

    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.cameras = cameras
        self.setWindowTitle("相機設定")
        layout = QVBoxLayout(self)
        self.roles = CameraRolesBox()
        layout.addWidget(self.roles)
        row = QHBoxLayout()
        self.front = CameraPanel("正面相機")
        self.side = CameraPanel("側面相機")
        row.addWidget(self.front)
        row.addWidget(self.side)
        layout.addLayout(row)
        note = QLabel("改動會即時套用到相機，並自動存回 camera_config 的 yaml，下次開相機沿用。")
        note.setStyleSheet("color: gray;")
        layout.addWidget(note)
        close = QPushButton("關閉")
        close.clicked.connect(self.close)
        layout.addWidget(close, 0, Qt.AlignRight)

    def refresh(self):
        pair = self.cameras()
        front, side = pair if pair is not None else (None, None)
        self.front.bind(front, "未使用正面相機（純側面 2D）" if pair is not None else "請先在錄影頁勾選「開啟相機」")
        self.side.bind(side)
        self.roles.setLocked(pair is not None)

    def showEvent(self, event):
        if self.cameras() is None:
            self.roles.scan()   # 相機開著時已占用，不掃描，只顯示目前的選擇
        else:
            self.roles.fill()
        self.refresh()
        super().showEvent(event)
