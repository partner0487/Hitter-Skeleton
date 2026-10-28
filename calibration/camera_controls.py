"""曝光時間與增益調整：精靈與單相機視窗共用。

數值改變時發出 changed(曝光 μs, 增益 dB)，由持有相機的一方套用後呼叫 show_actual 顯示實際值
（超出相機範圍時會被夾住）。「存成預設」寫回 calibration/camera_config/*.yaml，下次開相機就用這組。
"""
import re

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton


def save_yaml_defaults(config_path, exposure_time, gain):
    """只改 yaml 的 exposure_time、gain_value 兩行，保留其他設定與註解"""
    with open(config_path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    for key, value in (("exposure_time", exposure_time), ("gain_value", gain)):
        text, count = re.subn(r"(?m)^(\s*{}:\s*)[-\d.]+".format(key),
                              lambda m: m.group(1) + "{:g}".format(round(value, 1)), text, count=1)
        if count != 1:
            raise ValueError("{} 找不到 {}".format(config_path, key))
    with open(config_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


class CameraControls(QGroupBox):
    changed = pyqtSignal(float, float)
    reload_requested = pyqtSignal()

    def __init__(self, title="相機曝光"):
        super().__init__(title)
        self.config_path = None
        layout = QHBoxLayout(self)
        self.exposure = QDoubleSpinBox()
        self.exposure.setDecimals(0)
        self.exposure.setRange(1, 1000000)
        self.exposure.setSingleStep(100)
        self.exposure.setSuffix(" μs")
        self.gain = QDoubleSpinBox()
        self.gain.setDecimals(1)
        self.gain.setRange(0, 48)
        self.gain.setSingleStep(1)
        self.gain.setSuffix(" dB")
        for spin in (self.exposure, self.gain):
            # 打字時不要每按一鍵就送到相機，按 Enter 或離開欄位才套用；方向鍵／滾輪仍即時套用
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._emit)
        self.save_button = QPushButton("存成預設")
        self.save_button.setToolTip("寫回 camera_config 的 yaml，下次開相機就使用這組數值")
        self.save_button.clicked.connect(self.save_default)
        self.reload_button = QPushButton("重新載入相機")
        self.reload_button.setToolTip("關閉後重新開啟相機並套用目前數值；畫面卡住或設定沒反應時使用，已拍的照片保留")
        self.reload_button.clicked.connect(self.reload_requested)
        self.status = QLabel()
        layout.addWidget(QLabel("曝光時間"))
        layout.addWidget(self.exposure)
        layout.addWidget(QLabel("增益"))
        layout.addWidget(self.gain)
        layout.addWidget(self.save_button)
        layout.addWidget(self.reload_button)
        layout.addWidget(self.status, 1)
        self.setEnabled(False)

    def bind(self, settings, config_path):
        """相機開好後呼叫；settings 為 FlirCameraSystem.get_exposure_settings() 的結果"""
        self.config_path = config_path
        for spin, value, (low, high) in ((self.exposure, settings["exposure_time"], settings["exposure_range"]),
                                         (self.gain, settings["gain"], settings["gain_range"])):
            spin.blockSignals(True)
            spin.setRange(low, high)
            spin.setValue(value)
            spin.blockSignals(False)
        self.exposure.setToolTip("相機允許 {:.0f} ~ {:.0f} μs（上限受幀率限制）".format(*settings["exposure_range"]))
        self.status.setText("")
        self.setEnabled(True)

    def unbind(self):
        self.setEnabled(False)

    def values(self):
        return self.exposure.value(), self.gain.value()

    def set_values(self, exposure_time, gain):
        """改數值並發出 changed（用於沿用上一階段的設定）"""
        self.show_actual(exposure_time, gain)
        self._emit()

    def show_actual(self, exposure_time, gain):
        for spin, value in ((self.exposure, exposure_time), (self.gain, gain)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.status.setText("")

    def _emit(self, *_):
        self.changed.emit(*self.values())

    def save_default(self):
        try:
            save_yaml_defaults(self.config_path, *self.values())
            self.status.setText("已存成預設")
        except (OSError, ValueError, TypeError) as exc:
            self.status.setText("儲存失敗：{}".format(exc))
