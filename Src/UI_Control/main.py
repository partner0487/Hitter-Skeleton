import sys
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(os.environ["CONDA_PREFIX"], "Library", "plugins", "platforms")
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
# from dual_video_onlyProcess_widget import PoseHitVideoTabControlDualInit
from ui.main_window import Ui_MainWindow
# Import the Mode Select UI
from utils.model import Model
from analysis3d_widget import Analysis3DWidget 
from analysis2d_widget import Analysis2dWidget
from compare_widget import CompareWidget
import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main()
    main_window.show()
    sys.exit(app.exec_())
