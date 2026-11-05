"""3D 版主程式：錄影（現場錄影 → 先顯示 2D → 背景算完換成 3D）、分析（過往錄影的 3D 分析）、比較（指標與分析頁相同）。

main.py 維持原本的 2D／3D／回放三頁（廠商的基本需求：即時錄影與 2D 分析）。
"""
import os
import sys

# 環境變數必須在 import PyQt5 / ultralytics 之前設定才會生效
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    os.environ.setdefault(
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        os.path.join(conda_prefix, "Library", "plugins", "platforms"),
    )

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui.main_window import Ui_MainWindow
from utils.model import Model
from record_widget import RecordWidget
from analysis3d_widget import Analysis3DWidget
from compare3d_widget import Compare3DWidget
from ui_utils.radio_choice import batter_radio_buttons

APP_FONT_FAMILY = "Microsoft JhengHei"


class Main3D(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.setWindowTitle("棒球分析系統（3D）")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.recordWidget = RecordWidget(self.model)
        self.ui.Two_d_Tab.addTab(self.recordWidget, "錄影")
        self.analysisWidget = Analysis3DWidget(self.model)
        self.ui.Two_d_Tab.addTab(self.analysisWidget, "分析")
        batter_radio_buttons(self.analysisWidget.ui.batterInput)   # 右打／左打改成圓點（錄影頁在 RecordWidget 裡換）
        self.compareWidget = Compare3DWidget(self.model, self.recordWidget.worker3d)   # 與錄影頁共用背景 3D 程序
        self.ui.Two_d_Tab.addTab(self.compareWidget, "比較")

    def closeEvent(self, event):
        # 關閉視窗前停止相機、計時器、背景執行緒與背景 3D 程序
        for widget in (self.recordWidget, self.analysisWidget, self.compareWidget):
            try:
                widget.shutdown()
            except Exception as e:
                print(f"{type(widget).__name__} shutdown 失敗: {e}")
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 統一用微軟正黑體：.ui 沒指定字型，Windows 繁中預設是新細明體，與 Arial 等混用不一致
    app.setFont(QFont(APP_FONT_FAMILY, app.font().pointSize()))
    main_window = Main3D()
    main_window.show()
    sys.exit(app.exec_())
