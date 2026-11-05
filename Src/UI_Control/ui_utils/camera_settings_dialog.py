"""相機設定視窗：正面、側面兩台並排，調曝光時間與增益。

改動即時套用到相機，並自動寫回各自的 yaml（只改 exposure_time、gain_value 兩行，見
FlirCameraSystem.update_exposure），下次開相機沿用。相機沒開時欄位不能調。
"""
import math
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QVBoxLayout)


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

    def bind(self, camera):
        """camera 為 FlirCameraSystem；None 表示相機沒開"""
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
                self.status.setText("請先在錄影頁勾選「開啟相機」")
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


class CameraSettingsDialog(QDialog):
    """cameras：回傳 (正面, 側面) FlirCameraSystem 的函式，相機沒開時回傳 None"""

    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.cameras = cameras
        self.setWindowTitle("相機設定")
        layout = QVBoxLayout(self)
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
        self.front.bind(front)
        self.side.bind(side)

    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)
