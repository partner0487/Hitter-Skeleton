"""關鍵幀指標區的排版：每列最多兩組「標題｜數值」，2D、3D 各區塊用同一套格線、靠上對齊；
各欄（離地、著地、擊球、Rapsodo）的標題列、指標區一樣高，圖片底部才會對齊。

.ui 產生的版面是一層層的 QHBoxLayout，標題、數值、單位各自分開、寬度比例不一，
這裡把原有元件取出重排（物件名稱不變，其他程式照常 setText），沒用到的元件（空白、單位欄）隱藏。
"""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSizePolicy

MISSING = "—"
PAIR_GAP_PX = 24
VALUE_SAMPLE = "000.0 km/h"   # 數值欄寬度以此為準


def _detach(layout):
    """取出 layout（含巢狀 layout）裡的所有元件"""
    widgets = []
    while layout.count():
        item = layout.takeAt(0)
        if item.widget() is not None:
            widgets.append(item.widget())
        elif item.layout() is not None:
            widgets += _detach(item.layout())
    return widgets


def arrange_metrics(blocks):
    """blocks：[(grid, rows), ...]，rows 每列為 [(標題 QLabel, 數值 QLabel) 或 None（空一格）, ...]；grid 原有但沒列出的元件隱藏。
    同一頁的標題欄、數值欄寬度一致（取最寬的標題與 VALUE_SAMPLE），各區塊的數字上下、左右都對齊"""
    pairs = [pair for _, rows in blocks for row in rows for pair in row if pair is not None]
    if not pairs:
        return
    for caption, _ in pairs:  # 標題統一以半形冒號結尾（.ui 裡有的有、有的沒有）
        caption.setText(caption.text().strip().rstrip(":：") + ":")
    caption_w = max(c.fontMetrics().horizontalAdvance(c.text()) for c, _ in pairs) + 8
    value_w = max(v.fontMetrics().horizontalAdvance(VALUE_SAMPLE) for _, v in pairs)
    max_rows = max(len(rows) for _, rows in blocks)
    for grid, rows in blocks:
        used = {w for row in rows for pair in row if pair is not None for w in pair}
        for w in _detach(grid):
            if w not in used:
                w.hide()
        for r, row in enumerate(rows):
            for c, pair in enumerate(row):
                if pair is None:
                    continue
                caption, value = pair
                caption.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                caption.setMinimumWidth(caption_w)
                value.setMinimumWidth(value_w)
                grid.addWidget(caption, r, c * 3)
                grid.addWidget(value, r, c * 3 + 1)
                caption.show()
                value.show()
        # 列數較少的區塊補空白列，同一頁每個指標區一樣高
        template = next(pair[0] for row in rows for pair in row if pair is not None)
        for r in range(len(rows), max_rows):
            grid.addWidget(new_label_like(template, " "), r, 0)
        # 欄：標題｜數值｜間隔｜標題｜數值｜剩餘；多出的寬度給最右欄，兩組指標的位置固定
        for col, stretch in enumerate((0, 0, 0, 0, 0, 1)):
            grid.setColumnStretch(col, stretch)
        grid.setColumnMinimumWidth(2, PAIR_GAP_PX)
        grid.setRowStretch(max_rows, 1)


def align_columns(columns):
    """columns：各欄的 QVBoxLayout，內容依序為 標題列、圖片 QLabel、指標區（layout 或 groupBox）。
    標題列統一高度、指標區只取需要的高度（已由 arrange_metrics 補成相同列數），剩下的高度全給圖片，
    各欄的圖片格子因此一樣高、底部對齊"""
    headers = [col.itemAt(0).layout() for col in columns]
    header_widgets = [h.itemAt(i).widget() for h in headers for i in range(h.count()) if h.itemAt(i).widget()]
    header_h = max(w.sizeHint().height() for w in header_widgets)
    for w in header_widgets:
        w.setFixedHeight(header_h)
    for col in columns:
        image = col.itemAt(1).widget()
        image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        image.setMinimumSize(50, 50)
        metrics = col.itemAt(2)
        if metrics.widget() is not None:
            metrics.widget().setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        col.setStretch(0, 0)
        col.setStretch(1, 1)
        col.setStretch(2, 0)


def new_label_like(template: QLabel, text: str = "") -> QLabel:
    """與 template 同字型、同父元件的新 QLabel（例如 3D 頁新增的最大分離角）"""
    label = QLabel(text, template.parentWidget())
    label.setFont(template.font())
    return label


def fmt(value, unit: str = "", digits: int = 1, signed: bool = False) -> str:
    """數值 + 單位；沒有數值時顯示 —"""
    if value is None:
        return MISSING
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MISSING
    if number != number:  # NaN
        return MISSING
    text = f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"
    return f"{text}{unit}" if unit in ("°",) else f"{text} {unit}".rstrip()


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rapsodo_texts(data: dict):
    """Rapsodo data.json → (球初速, 出球角度, 打擊距離) 的顯示文字；mph 換 km/h、ft 換 m，空值顯示 —"""
    exit_mph, launch, dist_ft = (_number(data.get(k)) for k in ("ExitVelocity", "LaunchAngle", "Distance"))
    return (fmt(exit_mph * 1.60934 if exit_mph is not None else None, "km/h"),
            fmt(launch, "°"),
            fmt(dist_ft * 0.3048 if dist_ft is not None else None, "m"))
