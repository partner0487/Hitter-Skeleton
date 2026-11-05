"""球棒設定視窗：目前選的打者的棒長（吋）與棒重（盎司），改動自動存到 Db/<打者>/bat.json。"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from utils.bat_profile import load_bat, reset_bat, save_bat


class BatSettingsDialog(QDialog):
    """player：回傳目前打者名稱的函式"""

    def __init__(self, player, parent=None):
        super().__init__(parent)
        self.player = player
        self.setWindowTitle("球棒設定")
        layout = QVBoxLayout(self)
        self.title = QLabel()
        layout.addWidget(self.title)
        form = QFormLayout()
        self.length = QDoubleSpinBox()
        self.length.setRange(24, 36)
        self.length.setDecimals(1)
        self.length.setSingleStep(0.5)
        self.length.setSuffix(" 吋")
        self.weight = QDoubleSpinBox()
        self.weight.setRange(0, 40)
        self.weight.setDecimals(1)
        self.weight.setSingleStep(0.5)
        self.weight.setSuffix(" oz")
        self.weight.setSpecialValueText("未填")
        for spin in (self.length, self.weight):
            spin.setKeyboardTracking(False)
            spin.setAlignment(Qt.AlignRight)
            spin.setMinimumWidth(140)
            spin.valueChanged.connect(self.save)
        self.length_m = QLabel()
        form.addRow("棒長", self.length)
        form.addRow("", self.length_m)
        form.addRow("棒重", self.weight)
        layout.addLayout(form)
        self.status = QLabel()
        note = QLabel("棒長用於 2D 揮棒速度與比例換算，也用來檢查 3D 量到的棒長；\n棒重目前只記錄。沒設定時使用預設棒長。")
        for label in (self.length_m, self.status, note):
            label.setStyleSheet("color: gray;")
        layout.addWidget(self.status)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        self.reset_button = QPushButton("恢復預設")
        self.reset_button.clicked.connect(self.reset)
        close = QPushButton("關閉")
        close.clicked.connect(self.close)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def refresh(self):
        name = self.player()
        profile = load_bat(name)
        self.title.setText(f"打者：{name or '（未選擇）'}")
        for spin, value in ((self.length, profile.length_in), (self.weight, profile.weight_oz or 0)):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.setEnabled(bool(name))
            spin.blockSignals(False)
        self.reset_button.setEnabled(bool(name) and not profile.is_default)
        self.showStatus(profile)

    def showStatus(self, profile):
        self.length_m.setText(f"= {profile.length_m:.3f} m")
        self.status.setText("使用預設棒長（尚未設定）" if profile.is_default else "已自動儲存")

    def save(self, *_):
        name = self.player()
        if not name:
            return
        save_bat(name, self.length.value(), self.weight.value() or None)
        self.reset_button.setEnabled(True)
        self.showStatus(load_bat(name))

    def reset(self):
        name = self.player()
        if name:
            reset_bat(name)
        self.refresh()

    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)
