"""建立 QApplication：開啟高 DPI 縮放並放大字型，4K 螢幕上按鈕與文字才不會太小。

另外記錄錯誤：PyQt5 在 slot 裡遇到未處理的例外會直接結束程式，改成只印出錯誤；
相機驅動等原生程式崩潰時，堆疊寫入 calibration/crash.log。
"""
import faulthandler
import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QGuiApplication
from PyQt5.QtWidgets import QApplication

FONT_POINT_SIZE = 11


def _print_exception(kind, value, tb):
    traceback.print_exception(kind, value, tb)


def create_app(argv):
    from paths import CALIBRATION_DIR
    sys.excepthook = _print_exception
    # 檔案需在程式結束前保持開啟，faulthandler 才寫得進去
    create_app.crash_log = open(CALIBRATION_DIR / "crash.log", "a", encoding="utf-8")
    faulthandler.enable(create_app.crash_log)
    # 必須在建立 QApplication 之前設定；PassThrough 讓 150%、250% 等縮放比例不被四捨五入
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    font = QFont("Microsoft JhengHei UI")
    font.setPointSize(FONT_POINT_SIZE)
    app.setFont(font)
    return app
