"""把下拉選單換成同一排的圓點選項。

下拉選單本身保留（隱藏）當作資料來源：其他程式照樣讀 currentIndex()／currentText()、
接 currentIndexChanged，不必改；圓點與下拉選單雙向同步。
"""
from PyQt5.QtWidgets import QButtonGroup, QHBoxLayout, QRadioButton, QWidget


def batter_radio_buttons(combo) -> QButtonGroup:
    """在 combo 原本的位置放一排圓點（每個選項一個），回傳按鈕群組"""
    container = QWidget(combo.parentWidget())
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    group = QButtonGroup(container)
    for i in range(combo.count()):
        radio = QRadioButton(combo.itemText(i))
        radio.setFont(combo.font())
        group.addButton(radio, i)
        row.addWidget(radio)
    group.button(max(combo.currentIndex(), 0)).setChecked(True)
    group.idClicked.connect(combo.setCurrentIndex)

    def follow(index):
        button = group.button(index)
        if button is not None and not button.isChecked():
            button.setChecked(True)
    combo.currentIndexChanged.connect(follow)

    layout = combo.parentWidget().layout()
    parent_layout = _layout_containing(layout, combo) if layout is not None else None
    if parent_layout is not None:
        parent_layout.replaceWidget(combo, container)
    combo.hide()
    combo.radio_container = container   # 停用選項時連圓點一起停用（分析頁處理影片時鎖住打擊方向）
    return group


def _layout_containing(layout, widget):
    """遞迴找出直接包含 widget 的 layout"""
    if layout.indexOf(widget) >= 0:
        return layout
    for i in range(layout.count()):
        child = layout.itemAt(i).layout()
        if child is not None:
            found = _layout_containing(child, widget)
            if found is not None:
                return found
    return None
