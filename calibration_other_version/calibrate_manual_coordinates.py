"""手動設定兩個視角的原點、XYZ 投影方向及參考尺度。

python -m Src.calibration.calibrate_manual_coordinates
"""
import argparse
from datetime import datetime, timezone
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PyQt5.QtCore import QEvent, QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QImageReader, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFormLayout,
    QGraphicsItem, QGraphicsScene, QGraphicsView, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QTabWidget,
    QSizePolicy, QVBoxLayout, QWidget,
)

from Src.calibration.manual_coordinates import (
    LIMITATION, VIEW_NAMES, load_document, make_document, save_document,
)
from Src.paths import DB_DIR


IMAGE_FILTER = "影像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;所有檔案 (*)"
POINT_COLORS = {"O": "#ffffff", "X": "#ff6767", "Y": "#54df9b",
                "Z": "#69a9ff", "A": "#ffd467", "B": "#ffd467"}


def read_qimage(path):
    """Use unrotated source pixels, shared with the saved coordinate convention."""
    reader = QImageReader(str(path))
    reader.setAutoTransform(False)
    image = reader.read()
    if image.isNull():
        raise ValueError("無法讀取影像：{}\n{}".format(path, reader.errorString()))
    return image


class AnnotationCanvas(QGraphicsView):
    """Clickable source-pixel canvas with wheel zoom and middle-button panning.

    ``set_image(path_or_qimage)`` returns ``(width, height)``.
    ``set_points({label: [x, y]}, origin_key=None)`` draws labels and optional
    axis rays. ``clicked`` emits source pixel x/y and never image-border clicks.
    """
    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setMinimumSize(240, 220)
        self.setBackgroundBrush(QColor("#20252b"))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_size = None
        self.image_path = None
        self._points = {}
        self._markers = []
        self._pixmap_item = None
        self._origin_key = None
        self._pan_start = None
        self._auto_fit = True

    def set_image(self, source):
        if isinstance(source, QImage):
            image, path = source, None
        else:
            image, path = read_qimage(source), str(Path(source).resolve())
        if image.isNull():
            raise ValueError("影像是空白的。")
        self.scene().clear()
        self._pixmap_item = None
        self._markers = []
        self._points = {}
        self.image_path = path
        self.image_size = (image.width(), image.height())
        self._pixmap_item = self.scene().addPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(0, 0, *self.image_size)
        self.fit_image()
        return self.image_size

    def update_preview_image(self, image):
        """Update a live QImage without resetting zoom, pan or existing overlays.

        Resolution changes still go through ``set_image`` to reset the canvas.
        Callers clear annotations once when entering live preview.
        """
        if not isinstance(image, QImage) or image.isNull():
            raise ValueError("影像是空白的。")
        if self._pixmap_item is None or self.image_size != (image.width(), image.height()):
            return self.set_image(image)
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.image_path = None
        return self.image_size

    def clear_image(self):
        self.scene().clear()
        self._pixmap_item = None
        self._markers = []
        self._points = {}
        self.image_size = None
        self.image_path = None

    def set_points(self, points, origin_key=None):
        self._points = {key: list(point) for key, point in points.items()
                        if point is not None}
        self._origin_key = origin_key
        self._draw_points()

    def _draw_points(self):
        for item in self._markers:
            self.scene().removeItem(item)
        self._markers = []
        if self.image_size is None:
            return
        origin = self._points.get(self._origin_key)
        for label, point in self._points.items():
            x, y = point
            color = QColor(POINT_COLORS.get(label, "#ffd467"))
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            if origin is not None and label != self._origin_key:
                self._markers.append(self.scene().addLine(*origin, x, y, pen))
            dot = self.scene().addEllipse(-4, -4, 8, 8, pen)
            dot.setPos(x, y)
            dot.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            self._markers.append(dot)
            text = self.scene().addSimpleText("+" + label if label in ("X", "Y", "Z") else label)
            text.setBrush(color)
            text.setPos(x, y)
            text.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            self._markers.append(text)
        if origin is None and "A" in self._points and "B" in self._points:
            pen = QPen(QColor(POINT_COLORS["A"]), 2)
            pen.setCosmetic(True)
            self._markers.append(self.scene().addLine(*self._points["A"], *self._points["B"], pen))

    def fit_image(self):
        if self.image_size is not None:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        self._auto_fit = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_image()

    def wheelEvent(self, event):
        if self.image_size is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        scale = self.transform().m11() * factor
        if 0.01 <= scale <= 100:
            self.scale(factor, factor)
            self._auto_fit = False
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_start = QPoint(event.pos())
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.LeftButton and self.image_size is not None:
            point = self.mapToScene(event.pos())
            width, height = self.image_size
            if 0 <= point.x() < width and 0 <= point.y() < height:
                self.clicked.emit(point.x(), point.y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = QPoint(event.pos())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._auto_fit = False
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._pan_start = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class LengthInput(QLineEdit):
    """Retain imported decimal precision instead of silently rounding a spin box."""
    valueChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("請輸入實際量測長度")
        self.textChanged.connect(lambda: self.valueChanged.emit(self.value()))

    def value(self):
        try:
            return float(self.text())
        except ValueError:
            return float("nan")

    def setValue(self, value):
        self.setText(repr(float(value)))


class AnnotationPanel(QGroupBox):
    changed = pyqtSignal()

    def __init__(self, title, mode, parent=None):
        super().__init__(title, parent)
        self.mode = mode
        self.labels = ("O", "X", "Y", "Z") if mode == "axes" else ("A", "B")
        self.points = {}
        self.image = None
        self.image_available = False
        layout = QVBoxLayout(self)
        self.header = QWidget()
        header_layout = QVBoxLayout(self.header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.header)
        controls = QHBoxLayout()
        self.choose_button = QPushButton("選擇圖片")
        self.choose_button.clicked.connect(self.choose_image)
        controls.addWidget(self.choose_button)
        self.target = QComboBox()
        for label in self.labels:
            name = "原點 O" if label == "O" else (
                "+" + label + " 軸方向" if mode == "axes" else "端點 " + label)
            self.target.addItem(name, label)
        controls.addWidget(self.target)
        fit_button = QPushButton("符合畫面")
        fit_button.clicked.connect(lambda: self.canvas.fit_image())
        controls.addWidget(fit_button)
        clear_button = QPushButton("清除標記")
        clear_button.clicked.connect(self.clear_points)
        controls.addWidget(clear_button)
        header_layout.addLayout(controls)
        self.file_label = QLabel("尚未選擇影像")
        self.file_label.setWordWrap(True)
        self.file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        header_layout.addWidget(self.file_label)
        self.canvas = AnnotationCanvas()
        self.canvas.clicked.connect(self.mark_point)
        layout.addWidget(self.canvas, 1)
        self.details = QWidget()
        layout.addWidget(self.details)
        layout = QVBoxLayout(self.details)
        layout.setContentsMargins(0, 0, 0, 0)
        self.point_label = QLabel("依序在圖片點選；可切換目標修正任一點。")
        self.point_label.setWordWrap(True)
        self.point_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.point_label)
        if mode == "scale":
            row = QHBoxLayout()
            row.addWidget(QLabel("A–B 真實長度"))
            self.length = LengthInput()
            self.length.valueChanged.connect(self.update_points)
            self.unit = QComboBox()
            self.unit.addItems(["mm", "cm", "m"])
            self.unit.setCurrentText("cm")
            self.unit.currentTextChanged.connect(self.update_points)
            row.addWidget(self.length, 1)
            row.addWidget(self.unit)
            layout.addLayout(row)
            self.scale_label = QLabel("請選圖、標出 A／B 並輸入實際量測長度。")
            self.scale_label.setWordWrap(True)
            layout.addWidget(self.scale_label)
        layout.addWidget(QLabel("左鍵標記 · 滾輪縮放 · 按住滑鼠中鍵平移"))

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "選擇原始影像", "", IMAGE_FILTER)
        if not path:
            return
        try:
            self.set_image(path, preserve_missing=True)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "影像讀取失敗", str(exc))

    def set_image(self, path, preserve_missing=False):
        image = read_qimage(path)
        size = (image.width(), image.height())
        preserve = preserve_missing and self.image is not None and not self.image_available
        if preserve and size != (self.image["width"], self.image["height"]):
            raise ValueError("替代圖片尺寸與原始標記不同，無法沿用像素座標。請選擇相同尺寸的原圖。")
        self.canvas.set_image(image)
        self.canvas.image_path = str(Path(path).resolve())
        self.image = {"path": self.canvas.image_path, "width": size[0], "height": size[1]}
        self.image_available = True
        if not preserve:
            self.points = {}
        self.file_label.setText("{}  ·  {} × {} px".format(Path(path).name, *size))
        self.file_label.setToolTip(str(path))
        self.target.setCurrentIndex(0)
        self.update_points()

    def clear_points(self):
        self.points = {}
        self.target.setCurrentIndex(0)
        self.update_points()

    def mark_point(self, x, y):
        if not self.image_available:
            return
        self.points[self.target.currentData()] = [float(x), float(y)]
        index = self.target.currentIndex()
        if index + 1 < self.target.count():
            self.target.setCurrentIndex(index + 1)
        self.update_points()

    def update_points(self, *args):
        self.canvas.set_points(self.points, "O" if self.mode == "axes" else None)
        self.point_label.setText("   ".join(
            "{}: {}".format(label, "({:.1f}, {:.1f})".format(*self.points[label])
                            if label in self.points else "未標記")
            for label in self.labels))
        if self.mode == "scale" and hasattr(self, "scale_label"):
            import math
            from Src.calibration.manual_coordinates import UNIT_TO_MM
            if all(label in self.points for label in ("A", "B")):
                a, b = self.points["A"], self.points["B"]
                distance = math.hypot(b[0] - a[0], b[1] - a[1])
                if distance > 1e-6 and math.isfinite(self.length.value()) and self.length.value() > 0:
                    scale = self.length.value() * UNIT_TO_MM[self.unit.currentText()] / distance
                    self.scale_label.setText(
                        "{:.2f} px → {:.6g} mm/px（參考線段所在深度）".format(distance, scale))
                else:
                    self.scale_label.setText("兩端不可重合，真實長度必須大於 0。")
            else:
                self.scale_label.setText("請選圖、標出 A／B 並輸入實際量測長度。")
        self.changed.emit()

    def restore(self, image, points, length=None, unit=None):
        """Restore data even when source images moved; a replacement can relink it."""
        problem = None
        self.image = dict(image)
        self.points = dict(points)
        try:
            loaded = read_qimage(image["path"])
            if (loaded.width(), loaded.height()) != (image["width"], image["height"]):
                raise ValueError("影像尺寸與標記檔不一致")
            self.canvas.set_image(loaded)
            self.canvas.image_path = image["path"]
            self.image_available = True
            self.file_label.setText("{}  ·  {} × {} px".format(
                Path(image["path"]).name, image["width"], image["height"]))
        except (OSError, ValueError) as exc:
            self.canvas.clear_image()
            self.image_available = False
            self.file_label.setText("圖片無法讀取；請選擇相同原圖重新連結：" + Path(image["path"]).name)
            problem = "{}：{}".format(image["path"], exc)
        self.file_label.setToolTip(image["path"])
        if self.mode == "scale":
            self.length.setValue(length)
            self.unit.setCurrentText(unit)
        self.update_points()
        return problem


class ManualCoordinatesWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("木框場景｜手動原點、XYZ 軸與尺度")
        self.resize(1240, 850)
        self._dirty = False
        self._loading = False
        self._created_at = datetime.now(timezone.utc).isoformat()
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        heading = QLabel("1 選正面／側面原圖 → 2 標記共同原點與 +XYZ → 3 標記已知長度 → 4 匯出 JSON")
        heading.setWordWrap(True)
        heading.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px 0;")
        layout.addWidget(heading)
        instructions = QLabel(
            "兩張圖請標記同一個實體原點，+XYZ 使用相同的右手座標方向。"
            "軸端點只代表正方向，不代表軸長；若某軸正對鏡頭，請改用能辨認該軸的拍攝角度。\n"
            "尺度圖可另外選擇：相機、解析度及拍攝設定應相同，參考線段盡量與量測物位於同一深度。")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.tabs = QTabWidget()
        self.panels = {}
        for role, name in VIEW_NAMES.items():
            page = QWidget()
            row = QHBoxLayout(page)
            axes = AnnotationPanel(name + "｜原點與 XYZ", "axes")
            scale = AnnotationPanel(name + "｜真實尺度", "scale")
            row.addWidget(axes, 1)
            row.addWidget(scale, 1)
            self.panels[role] = {"axes": axes, "scale": scale}
            axes.installEventFilter(self)
            scale.installEventFilter(self)
            axes.changed.connect(self._changed)
            scale.changed.connect(self._changed)
            self.tabs.addTab(page, name)
        layout.addWidget(self.tabs, 1)
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("選填：場地、原點位置與 XYZ 定義，例如 +X 往本壘、+Y 往一壘、+Z 向上")
        self.notes.textChanged.connect(self._changed)
        notes_layout = QFormLayout()
        notes_layout.addRow("座標定義備註", self.notes)
        layout.addLayout(notes_layout)
        limitation = QLabel(LIMITATION)
        limitation.setWordWrap(True)
        limitation.setStyleSheet("color: #8a5a12; padding: 5px;")
        layout.addWidget(limitation)
        row = QHBoxLayout()
        self.status_label = QLabel("尚未完成兩個視角的標記")
        self.status_label.setWordWrap(True)
        row.addWidget(self.status_label, 1)
        self.import_button = QPushButton("讀取標記 JSON")
        self.import_button.clicked.connect(self.import_dialog)
        self.export_button = QPushButton("匯出結果 JSON")
        self.export_button.clicked.connect(self.export_dialog)
        row.addWidget(self.import_button)
        row.addWidget(self.export_button)
        layout.addLayout(row)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.LayoutRequest, QEvent.Resize):
            for panels in self.panels.values():
                if watched not in panels.values():
                    continue
                # Reserve matching space above and below both images, including
                # wrapped filenames/coordinates and the scale-only controls.
                for name in ("header", "details"):
                    widgets = [getattr(panel, name) for panel in panels.values()]
                    height = max(
                        max(widget.layout().sizeHint().height(),
                            widget.layout().totalHeightForWidth(widget.width()))
                        for widget in widgets)
                    for widget in widgets:
                        if widget.minimumHeight() != height or widget.maximumHeight() != height:
                            widget.setFixedHeight(height)
                break
        return super().eventFilter(watched, event)

    def _changed(self, *args):
        if self._loading:
            return
        self._dirty = True
        try:
            self.document()
        except ValueError as exc:
            self.status_label.setText("尚未完成：" + str(exc))
        else:
            self.status_label.setText("正面與側面標記完成，可匯出（有尚未儲存的修改）")

    def document(self):
        views = {}
        for role, panels in self.panels.items():
            axes, scale = panels["axes"], panels["scale"]
            views[role] = {
                "axes_image": axes.image,
                "origin_px": axes.points.get("O"),
                "axes_px": {axis: axes.points.get(axis) for axis in ("X", "Y", "Z")},
                "scale": {
                    "image": scale.image,
                    "endpoints_px": [scale.points.get("A"), scale.points.get("B")],
                    "length": scale.length.value(), "unit": scale.unit.currentText(),
                },
            }
        document = make_document(views, self.notes.text())
        document["created_at"] = self._created_at
        return document

    def save_to(self, path):
        result = save_document(path, self.document())
        self._dirty = False
        self.status_label.setText("已匯出：" + str(result))
        return result

    def load_from(self, path):
        document = load_document(path)  # Validate fully before changing any fields.
        problems = []
        self._loading = True
        try:
            for role, view in document["views"].items():
                panels = self.panels[role]
                points = dict(view["axes_px"], O=view["origin_px"])
                problem = panels["axes"].restore(view["axes_image"], points)
                if problem:
                    problems.append(problem)
                scale = view["scale"]
                problem = panels["scale"].restore(
                    scale["image"], dict(zip(("A", "B"), scale["endpoints_px"])),
                    scale["length"], scale["unit"])
                if problem:
                    problems.append(problem)
            self.notes.setText(document["notes"])
            self._created_at = document["created_at"]
        finally:
            self._loading = False
        self._dirty = False
        self.status_label.setText(
            "已讀取；{} 張來源圖需重新連結。標記資料已保留。".format(len(problems))
            if problems else "已讀取：" + str(path))
        return problems

    def export_dialog(self):
        try:
            self.document()
        except ValueError as exc:
            QMessageBox.warning(self, "請完成標記", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "匯出原點、座標軸與尺度",
            str(DB_DIR / "Calibration" / "wood_frame" / "manual_coordinates.json"), "JSON (*.json)")
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self.save_to(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "儲存失敗", str(exc))

    def import_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "讀取手動標記", "", "JSON (*.json)")
        if not path:
            return
        if self._dirty and QMessageBox.question(
                self, "尚未匯出", "目前有未匯出的修改。要讀取檔案並取代目前標記嗎？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            problems = self.load_from(path)
            if problems:
                QMessageBox.information(self, "請重新連結原圖", "\n\n".join(problems))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "讀取失敗", str(exc))

    def closeEvent(self, event):
        if self._dirty and QMessageBox.question(
                self, "尚未匯出", "目前標記尚未匯出。確定離開？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            event.ignore()
            return
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="正面／側面影像的手動原點、XYZ 投影方向與真實尺度標記 UI。")
    parser.add_argument("--load", metavar="JSON", help="啟動時讀取已匯出的手動標記 JSON")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    window = ManualCoordinatesWindow()
    if args.load:
        try:
            problems = window.load_from(args.load)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
        if problems:
            QMessageBox.information(window, "請重新連結原圖", "\n\n".join(problems))
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
