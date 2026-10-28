"""建立 QApplication：開啟高 DPI 縮放並放大字型，4K 螢幕上按鈕與文字才不會太小。"""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QGuiApplication
from PyQt5.QtWidgets import QApplication

FONT_POINT_SIZE = 11


def create_app(argv):
    # 必須在建立 QApplication 之前設定；PassThrough 讓 150%、250% 等縮放比例不被四捨五入
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    font = QFont("Microsoft JhengHei UI")
    font.setPointSize(FONT_POINT_SIZE)
    app.setFont(font)
    return app
