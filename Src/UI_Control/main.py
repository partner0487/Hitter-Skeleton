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

from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
# from dual_video_onlyProcess_widget import PoseHitVideoTabControlDualInit
from ui.main_window import Ui_MainWindow
# Import the Mode Select UI
from utils.model import Model
from analysis3d_widget import Analysis3DWidget 
from analysis2d_widget import Analysis2dWidget
from compare_widget import CompareWidget

class Main(QMainWindow):
    def __init__(self, initial_tab_index=0):
        super(Main, self).__init__()
        self.model = Model()  
        self.setWindowTitle("棒球分析系統")
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_tabs()

    def init_tabs(self):
        # self.hitterRecordWidget = HitterRecordWidget(self.model)
        # self.ui.Two_d_Tab.addTab(self.hitterRecordWidget, "Record")
        self.analysis2dWidget = Analysis2dWidget(self.model)
        self.ui.Two_d_Tab.addTab(self.analysis2dWidget, "2D")
        self.analysis3dWidget = Analysis3DWidget(self.model)
        self.ui.Two_d_Tab.addTab(self.analysis3dWidget, "3D")
        self.compareWidget = CompareWidget(self.model)
        self.ui.Two_d_Tab.addTab(self.compareWidget, "回放")

    def closeEvent(self, event):
        # 關閉視窗前停止所有相機、計時器與背景執行緒，避免 QThread 未結束就被銷毀
        for widget in (self.analysis2dWidget, self.analysis3dWidget, self.compareWidget):
            try:
                widget.shutdown()
            except Exception as e:
                print(f"{type(widget).__name__} shutdown 失敗: {e}")
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())
