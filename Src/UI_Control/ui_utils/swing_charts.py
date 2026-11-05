"""3D 頁「分析」分頁的曲線圖視窗：肩髖分離角、甜蜜點速度，標出離地／著地／擊球與目前幀。

資料與指標同源：分離角來自 body_metrics.separation_series（正 = 肩落後髖），
甜蜜點速度來自 bat_speed3d 過濾後的 3D 球棒軌跡（相鄰兩個有效幀的位移 ÷ 時間）。
"""
from typing import Dict, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout

PHASE_LINES = (("F_liftoff", "a 離地", "#FF3232"), ("F_landing", "b 著地", "#FF9600"), ("F_impact", "c 擊球", "#00C8FF"))
CURVE_COLOR = "#3060C0"


class SwingChartsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("曲線圖")
        self.resize(820, 620)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        layout = QVBoxLayout(self)
        self.sep_plot = self._make_plot("肩髖分離角（正 = 肩落後髖）", "角度 (°)")
        self.speed_plot = self._make_plot("甜蜜點速度", "速度 (km/h)")
        self.speed_plot.setXLink(self.sep_plot)
        layout.addWidget(self.sep_plot)
        layout.addWidget(self.speed_plot)
        self.cursors = [self._cursor(p) for p in (self.sep_plot, self.speed_plot)]

    @staticmethod
    def _make_plot(title, ylabel):
        plot = pg.PlotWidget(title=title)
        plot.setLabel("bottom", "幀")
        plot.setLabel("left", ylabel)
        plot.showGrid(x=True, y=True, alpha=0.25)
        return plot

    @staticmethod
    def _cursor(plot):
        line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#888888", width=1, style=Qt.DashLine))
        plot.addItem(line, ignoreBounds=True)
        return line

    def set_data(self, separation: Dict[int, float], sweet_speed: Dict[int, float], phase_result,
                 total_frames: int):
        for plot, series in ((self.sep_plot, separation), (self.speed_plot, sweet_speed)):
            for item in list(plot.listDataItems()):
                plot.removeItem(item)
            for item in [i for i in plot.items() if getattr(i, "_phase_line", False)]:
                plot.removeItem(item)
            if series:
                frames = np.array(sorted(series))
                values = np.array([series[f] for f in frames])
                plot.plot(frames, values, pen=pg.mkPen(CURVE_COLOR, width=2), symbol="o", symbolSize=4,
                          symbolBrush=CURVE_COLOR, symbolPen=None)
            if plot is self.sep_plot:
                zero = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("#BBBBBB", width=1))
                zero._phase_line = True
                plot.addItem(zero, ignoreBounds=True)
            for attr, text, color in PHASE_LINES:
                f = getattr(phase_result, attr, None) if phase_result is not None else None
                if f is None:
                    continue
                line = pg.InfiniteLine(pos=f, angle=90, pen=pg.mkPen(color, width=2),
                                       label=text, labelOpts={"position": 0.95, "color": color})
                line._phase_line = True
                plot.addItem(line, ignoreBounds=True)
        if total_frames:
            self.sep_plot.setXRange(0, max(total_frames - 1, 1), padding=0.01)

    def set_frame(self, frame: Optional[int]):
        if frame is None:
            return
        for line in self.cursors:
            line.setPos(frame)
