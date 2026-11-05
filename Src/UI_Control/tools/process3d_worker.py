"""背景 3D 處理程序：給 main3d.py 的錄影頁用，錄完先顯示 2D，這裡在另一個程序把 3D 算完。

在不顯示視窗的情況下執行分析頁（Analysis3DWidget）同一套流程，數值與分析頁完全相同：
1. 處理影片（雙相機偵測），結果存到錄影資料夾的 CF_ 子資料夾，側面結果覆蓋給 2D；
   已處理過（例如批量處理過的舊錄影）就跳過這步
2. 載入資料夾分析，把各指標的顯示文字存成 CF_ 子資料夾裡的 metrics_3d.json

模型只在啟動時載入一次，之後常駐等工作。另開程序是因為處理迴圈會佔住介面執行緒約 15 秒，
放在錄影頁同一個程序裡會卡住相機畫面與揮棒觸發。

通訊（一行一筆）：
  stdin  ：<錄影資料夾>\t<right|left>
  stdout ：WORKER READY ／ WORKER START <資料夾> ／ WORKER DONE <資料夾>\t<metrics_3d.json>
           ／ WORKER ERROR <資料夾>\t<原因>；其他輸出為一般記錄
"""
import json
import os
import queue
import sys
import threading

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
UI_CONTROL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, UI_CONTROL)
os.chdir(UI_CONTROL)

from PyQt5.QtCore import QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

METRICS_FILE = "metrics_3d.json"
PROCESSED_FILES = ("person_side.json", "person_front.json", "bat_side.json", "bat_front.json")
JOB_TIMEOUT_S = 180


def say(*parts):
    print("WORKER", *parts, flush=True)


def main():
    app = QApplication(sys.argv)
    # 不顯示視窗時，對話框會一直等人按，改成只印出
    for name in ("warning", "information", "critical"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: print("[對話框]", *[str(x) for x in a[1:3]], flush=True)))
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

    from utils.model import Model
    from analysis3d_widget import Analysis3DWidget

    widget = Analysis3DWidget(Model())
    widget.headless = True
    widget.resize(1920, 1040)

    jobs = queue.Queue()
    state = {"folder": None, "analyzing": None}
    watchdog = QTimer()
    watchdog.setSingleShot(True)

    def read_stdin():
        for line in sys.stdin:
            if line.strip():
                jobs.put(line.rstrip("\n"))
        jobs.put(None)  # 主程式關閉

    def finish(ok, detail):
        watchdog.stop()
        folder = state["folder"]
        state["folder"] = state["analyzing"] = None
        say("DONE" if ok else "ERROR", f"{folder}\t{detail}")

    def next_job():
        if state["folder"] is not None:
            return
        try:
            line = jobs.get_nowait()
        except queue.Empty:
            return
        if line is None:
            app.quit()
            return
        folder, _, side = line.partition("\t")
        pairs = widget.findClipPairs(folder)
        if not pairs:
            say("ERROR", f"{folder}\t找不到成對的 CF*.mp4 與 CS*.mp4")
            return
        index = widget.ui.batterInput.findText("左打" if side.strip() == "left" else "右打")
        if index >= 0:
            # 不觸發「換打擊方向就重算上一支」，校正會在載入時依打擊方向重新挑選
            widget.ui.batterInput.blockSignals(True)
            widget.ui.batterInput.setCurrentIndex(index)
            widget.ui.batterInput.blockSignals(False)
        state["folder"] = folder
        say("START", folder)
        watchdog.start(JOB_TIMEOUT_S * 1000)
        # 已處理過（CF_ 子資料夾有雙相機偵測結果）就只做分析，不重跑偵測
        cf = pairs[0][0]
        output = os.path.join(os.path.dirname(cf), os.path.splitext(os.path.basename(cf))[0])
        if all(os.path.isfile(os.path.join(output, n)) for n in PROCESSED_FILES):
            state["analyzing"] = output
            widget.load_from_folder(output)
        else:
            widget.loadVideo4Process(*pairs[0])

    def on_processed(output_folder):
        # 處理完 → 載入資料夾分析（延後執行，等處理迴圈結束）
        if state["folder"] is None:
            return
        state["analyzing"] = output_folder
        QTimer.singleShot(300, lambda: widget.load_from_folder(output_folder))

    original_update = widget.updateVideoInfo

    def update_and_report(*args, **kwargs):
        result = original_update(*args, **kwargs)
        out = state["analyzing"]
        if out:
            path = os.path.join(out, METRICS_FILE)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(widget.metricTexts(), f, ensure_ascii=False, indent=1)
                finish(True, path)
            except OSError as e:
                finish(False, f"寫入 {path} 失敗：{e}")
        return result

    widget.updateVideoInfo = update_and_report
    widget.processFinished.connect(on_processed)
    watchdog.timeout.connect(lambda: finish(False, f"超過 {JOB_TIMEOUT_S} 秒沒有完成"))

    threading.Thread(target=read_stdin, daemon=True).start()
    poll = QTimer()
    poll.timeout.connect(next_job)
    poll.start(200)
    say("READY")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
