"""疊在影片滑桿上的階段進度條（2D、3D、回放頁共用）：各階段色塊、a／b／c 關鍵幀標記與各段秒數。"""
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from utils.fps_utils import NOMINAL_DURATION_S

BAR_HEIGHT = 12
MARKER_RADIUS = 12
MARKER_COLORS = {"a": QColor(255, 50, 50), "b": QColor(255, 150, 0), "c": QColor(0, 200, 255)}  # 離地、著地、擊球


class AnalysisBar(QWidget):
    def __init__(self, parent=None, show_seconds: bool = True):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.segments = []   # [(起點比例, 終點比例, 顏色), ...]
        self.markers = []    # [(比例, 'a'/'b'/'c'), ...]
        self.show_seconds = show_seconds
        self.time_of_ratio = None
        self.handle_px = 0   # 滑桿把手寬度，由滑桿的 resize 事件設定

    def set_segments(self, segments):
        self.segments = segments
        self.update()

    def set_markers(self, markers):
        self.markers = markers
        self.update()

    def set_time_of_ratio(self, time_of_ratio):
        """進度條比例 → 秒數，各段秒數用它相減（與時間標籤同一套換算）"""
        self.time_of_ratio = time_of_ratio
        self.update()

    def _x(self, ratio, w):
        """比例 → x 像素，與 QSlider 把手中心相同：把手中心從 handle_px/2 走到 w - handle_px/2，
        直接用 ratio * w 時越靠兩端越偏，把手停在擊球幀也壓不到 c 標記"""
        return int(round(self.handle_px / 2 + ratio * (w - self.handle_px)))

    def paintEvent(self, event):
        if not self.segments:
            return
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            bar_y = int(h * 0.52) - BAR_HEIGHT // 2  # 色塊置中偏下，上方留給秒數

            for start, end, color in self.segments:
                x = self._x(start, w)
                if self._x(end, w) - x > 0:
                    painter.fillRect(QRect(x, bar_y, self._x(end, w) - x, BAR_HEIGHT), QColor(color))

            # 關鍵幀標記：圓圈貼齊色塊下緣
            r = MARKER_RADIUS
            for ratio, text in self.markers:
                mx = self._x(max(0.0, min(ratio, 1.0)), w)
                rect = QRect(mx - r, bar_y + BAR_HEIGHT, r * 2, r * 2)
                painter.setBrush(QBrush(MARKER_COLORS.get(str(text), QColor(Qt.gray))))
                painter.setPen(QPen(Qt.black, 1))
                painter.drawEllipse(rect)
                painter.setFont(QFont("Arial", 12, QFont.Bold))
                painter.setPen(QPen(Qt.white))
                painter.drawText(rect, Qt.AlignCenter, str(text))

            if self.show_seconds:
                time_of = self.time_of_ratio or (lambda ratio: ratio * NOMINAL_DURATION_S)
                painter.setFont(QFont("Arial", 14, QFont.Bold))
                for start, end, color in self.segments:
                    x = self._x(start, w)
                    seg_w = self._x(end, w) - x
                    if seg_w < 15:
                        continue
                    painter.setPen(QColor(color))
                    text_h = 24
                    painter.drawText(QRect(x, bar_y - text_h - 4, seg_w, text_h), Qt.AlignCenter,
                                     f"{time_of(end) - time_of(start):.2f}s")
        finally:
            painter.end()
