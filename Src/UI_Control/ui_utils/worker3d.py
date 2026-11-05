"""背景 3D 程序（tools/process3d_worker.py）的介面端：錄影頁與比較頁共用同一個程序，模型只載入一次。

request(錄影資料夾, "right"|"left") 排入工作；算完發出 done(資料夾, metrics_3d.json 路徑) 或 failed(資料夾, 原因)。
資料夾字串原樣送出、原樣回傳，各頁用它對應自己在等的錄影。
"""
import os
import sys

from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

WORKER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "process3d_worker.py")
METRICS_FILE = "metrics_3d.json"


class Worker3D(QObject):
    done = pyqtSignal(str, str)
    failed = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.buffer = ""

    def start(self):
        self.process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("QT_QPA_PLATFORM", "offscreen")
        self.process.setProcessEnvironment(env)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read)
        self.process.finished.connect(lambda *_: print("[背景 3D] 程序已結束"))
        self.process.start(sys.executable, ["-u", WORKER])

    def request(self, folder, side):
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self.start()
        self.process.write(f"{folder}\t{side}\n".encode("utf-8"))

    def read(self):
        self.buffer += bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        *lines, self.buffer = self.buffer.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            if line.startswith("WORKER "):
                print(f"[背景 3D] {line}")
                kind, _, rest = line[len("WORKER "):].partition(" ")
                folder, _, detail = rest.partition("\t")
                if kind == "DONE":
                    self.done.emit(folder, detail)
                elif kind == "ERROR":
                    self.failed.emit(folder, detail)
            elif "Traceback" in line or "Error" in line:
                print(f"[背景 3D] {line}")

    def shutdown(self):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.process.closeWriteChannel()  # 背景程序讀到 EOF 後自行結束
            if not self.process.waitForFinished(3000):
                self.process.kill()
