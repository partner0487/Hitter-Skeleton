"""main3d.py 的錄影頁：現場錄影與錄完的 2D 即時分析沿用 Analysis2dWidget（與 main.py 的 2D 頁相同），
2D 結果顯示後交給背景程序（tools/process3d_worker.py）做 3D，算完把指標換成 3D 的數值並補上 3D 才有的指標。

- 右側控制區把原本的「相機」「設定」分頁合成一欄，不提供載入影片與最近錄影清單（過往錄影在分析頁看）
- 指標區排成與分析頁相同的格線；肩髖分離角、垂直棒角、最大分離角、骨盆／軀幹轉速 2D 量不到，3D 算完前顯示「計算中…」
- 左上角標題後面標示目前數值的來源：2D 結果（3D 分析中…）→ 3D
- 已做過 3D 的錄影（CF_ 子資料夾裡有 metrics_3d.json）直接顯示，不重算
- 3D 做不了（沒有校正、缺正面影片等）時保留 2D 數值，3D 才有的指標顯示 —
"""
import json
import os
import sys

import math

from PyQt5.QtCore import QProcess, QProcessEnvironment, Qt
from PyQt5.QtWidgets import QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from analysis2d_widget import Analysis2dWidget
from ui_utils.radio_choice import batter_radio_buttons
from ui_utils.metric_grid import MISSING, align_columns, arrange_metrics, new_label_like
from utils.paths import clip_title

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "process3d_worker.py")
METRICS_FILE = "metrics_3d.json"
PENDING = "計算中…"
BLOCK_GAP_PX = 16


class RecordWidget(Analysis2dWidget):
    def __init__(self, model, parent=None):
        super().__init__(model, parent)
        self.worker = None
        self.worker_buffer = ""
        self.result_source = ""   # 標題後的來源標示
        self.mergeControlPanel()
        self.startWorker()

    # ---------- 右側控制區：相機與設定合成一欄，不提供選檔（過往錄影在分析頁看） ----------
    def mergeControlPanel(self):
        ui = self.ui
        for widget in (ui.fileName, ui.fileNameLabel, ui.loadVideoBtn, ui.listView):
            widget.hide()
        # 相機分頁底部的彈性空白會把下面的區塊擠開，拿掉
        for i in reversed(range(ui.gridLayout_3.count())):
            if ui.gridLayout_3.itemAt(i).spacerItem() is not None:
                ui.gridLayout_3.removeItem(ui.gridLayout_3.itemAt(i))
        camera_box = QGroupBox("相機")
        camera_layout = QVBoxLayout(camera_box)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        ui.tabWidget.removeTab(ui.tabWidget.indexOf(ui.tab))
        camera_layout.addWidget(ui.tab)
        ui.tab.show()

        panel = QWidget()
        panel.setFont(ui.tabWidget.font())   # 原本字型繼承自分頁元件（12pt）
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(ui.groupBox)        # 打擊者、打擊方向、模式
        layout.addWidget(camera_box)         # 相機選擇、曝光、開啟相機、開始打擊、手動錄影、打區
        ui.fileGroupBox_2.setFont(panel.font())  # .ui 裡標題是 10pt，與其他區塊一致
        layout.addWidget(ui.fileGroupBox_2)  # 關鍵幀跳轉
        layout.addStretch(1)

        index = ui.horizontalLayout_2.indexOf(ui.tabWidget)
        ui.horizontalLayout_2.removeWidget(ui.tabWidget)
        ui.tabWidget.hide()
        ui.horizontalLayout_2.insertWidget(index, panel, 2)
        self.addExposureControls()
        self.batterRadios = batter_radio_buttons(ui.batterInput)

    # ---------- 相機：只調曝光時間與增益（紅藍白平衡拿掉；原本標「曝光」的滑桿其實是增益） ----------
    def addExposureControls(self):
        ui = self.ui
        for widget in (ui.gain, ui.red_ratio, ui.blue_ratio, ui.gainSlider, ui.redRatioSlider, ui.blueRatioSlider,
                       ui.gainValue, ui.redRatioValue, ui.blueRatioValue):
            widget.hide()
        self.exposureSpin = QDoubleSpinBox()
        self.exposureSpin.setDecimals(0)
        self.exposureSpin.setRange(1, 1000000)
        self.exposureSpin.setSingleStep(100)
        self.exposureSpin.setSuffix(" μs")
        self.gainSpin = QDoubleSpinBox()
        self.gainSpin.setDecimals(1)
        self.gainSpin.setRange(0, 48)
        self.gainSpin.setSingleStep(1)
        self.gainSpin.setSuffix(" dB")
        row = QHBoxLayout()
        for caption, spin in (("曝光時間", self.exposureSpin), ("增益", self.gainSpin)):
            # 打字時不要每按一鍵就送到相機，按 Enter 或離開欄位才套用；方向鍵／滾輪仍即時套用
            spin.setKeyboardTracking(False)
            spin.setAlignment(Qt.AlignRight)
            spin.valueChanged.connect(self.onExposureChanged)
            row.addWidget(QLabel(caption))
            row.addWidget(spin, 1)
        ui.verticalLayout_12.addLayout(row)
        self.disableExposureControls()

    def disableExposureControls(self):
        """相機沒開：欄位顯示「—」且不能調"""
        for spin in (self.exposureSpin, self.gainSpin):
            spin.blockSignals(True)
            spin.setSpecialValueText("—")
            spin.setValue(spin.minimum())
            spin.setEnabled(False)
            spin.blockSignals(False)

    def updateCameraSliders(self):
        """開相機、切換正面／側面時，顯示該相機目前的曝光時間與增益"""
        try:
            cam = self.get_selected_camera()
            settings = cam.get_exposure_settings()
        except Exception as e:  # 相機還沒開
            print(f"讀取相機曝光設定失敗: {e}")
            return
        for spin, value, (low, high) in ((self.exposureSpin, settings["exposure_time"], settings["exposure_range"]),
                                         (self.gainSpin, settings["gain"], settings["gain_range"])):
            spin.blockSignals(True)
            step = 10 ** -spin.decimals()  # 直接 setRange 會把上限四捨五入而超過相機上限
            spin.setSpecialValueText("")
            spin.setRange(math.ceil(low / step) * step, math.floor(high / step) * step)
            spin.setValue(value)
            spin.setEnabled(True)
            spin.blockSignals(False)
        self.exposureSpin.setToolTip("相機允許 {:.0f} ~ {:.0f} μs（上限受幀率限制）".format(*settings["exposure_range"]))
        self.gainSpin.setToolTip("相機允許 {:.1f} ~ {:.1f} dB".format(*settings["gain_range"]))

    def onExposureChanged(self, *_):
        try:
            cam = self.get_selected_camera()
            exposure, gain = cam.update_exposure(self.exposureSpin.value(), self.gainSpin.value())
        except Exception as e:
            print(f"設定相機曝光失敗: {e}")
            return
        for spin, value in ((self.exposureSpin, exposure), (self.gainSpin, gain)):
            spin.blockSignals(True)
            spin.setValue(value)   # 顯示實際套用的值（超出範圍會被夾住）
            spin.blockSignals(False)

    def toggleCamera(self, state: int):
        super().toggleCamera(state)
        if state != 2:
            self.disableExposureControls()

    def populateRecentVideosList(self):
        pass  # 錄影頁不列過往錄影

    # ---------- 版面：與分析頁相同 ----------
    def arrangeMetricLabels(self):
        ui = self.ui
        ui.horizontalAngleLabel.setText("攻擊角度:")
        ui.verticalAngleLabel3.setText("垂直棒角:")
        self.strideTimeCaption = new_label_like(ui.centerGravity, "跨步時間:")
        self.strideTimeLabel = new_label_like(ui.centerGravityLabel)
        self.swingTimeCaption = new_label_like(ui.centerGravity, "擊球時間:")
        self.swingTimeLabel = new_label_like(ui.centerGravityLabel)
        self.maxSepCaption = new_label_like(ui.hipShoulderLabel2, "最大分離角:")
        self.maxSepLabel = new_label_like(ui.hipShoulderEditLabel2)
        self.pelvisRotCaption = new_label_like(ui.hipShoulderLabel2, "骨盆轉速:")
        self.pelvisRotLabel = new_label_like(ui.hipShoulderEditLabel2)
        self.torsoRotCaption = new_label_like(ui.hipShoulderLabel2, "軀幹轉速:")
        self.torsoRotLabel = new_label_like(ui.hipShoulderEditLabel2)
        arrange_metrics([
            (ui.gridLayout_6, [[(ui.centerGravity, ui.centerGravityLabel), (self.strideTimeCaption, self.strideTimeLabel)],
                               [(ui.hipShoulderLabel1, ui.hipShoulderEditLabel1), (self.swingTimeCaption, self.swingTimeLabel)]]),
            (ui.gridLayout_4, [[(ui.centerGravity2, ui.centerGravityLabel2), (ui.strideDis, ui.strideDisLabel)],
                               [(ui.hipShoulderLabel2, ui.hipShoulderEditLabel2), (self.maxSepCaption, self.maxSepLabel)],
                               [(self.pelvisRotCaption, self.pelvisRotLabel), (self.torsoRotCaption, self.torsoRotLabel)]]),
            (ui.gridLayout_5, [[(ui.batSpeed, ui.batSpeedLabel), (ui.horizontalAngleLabel, ui.horizontalAngleEditLabel)],
                               [(ui.forefoot3, ui.forefootLabel3), (ui.backfoot3, ui.backfootLabel3)],
                               [(ui.hipShoulderLabel3, ui.hipShoulderEditLabel3), (ui.verticalAngleLabel3, ui.verticalAngleEditLabel3)]]),
            (ui.gridLayout_8, [[(ui.exitV, ui.exitVLabel), (ui.lanuchAngle, ui.lanuchAngleLabel)],
                               [(ui.attackDis, ui.attackDisLabel)]]),
        ])
        align_columns([ui.verticalLayout_5, ui.verticalLayout_7, ui.verticalLayout_9, ui.verticalLayout_10])
        # 錄影頁的影片區較寬、指標區較窄，每塊右邊留間隔，數值才不會黏到下一塊的標題
        for grid in (ui.gridLayout_6, ui.gridLayout_4, ui.gridLayout_5):
            grid.setContentsMargins(0, 0, BLOCK_GAP_PX, 0)

    def metricLabels(self) -> dict:
        """metrics_3d.json 的鍵 → 本頁的 QLabel（鍵名見 Analysis3DWidget.metricTexts）"""
        ui = self.ui
        return {
            "centerGravityLabel": ui.centerGravityLabel, "centerGravityLabel2": ui.centerGravityLabel2,
            "strideDisLabel": ui.strideDisLabel, "batSpeedLabel": ui.batSpeedLabel,
            "horizontalAngleEditLabel": ui.horizontalAngleEditLabel, "verticalAngle": ui.verticalAngleEditLabel3,
            "frontKnee": ui.forefootLabel3, "backKnee": ui.backfootLabel3,
            "hipShoulderEditLabel1": ui.hipShoulderEditLabel1, "hipShoulderEditLabel2": ui.hipShoulderEditLabel2,
            "hipShoulderEditLabel3": ui.hipShoulderEditLabel3, "maxSep": self.maxSepLabel,
            "pelvisRot": self.pelvisRotLabel, "torsoRot": self.torsoRotLabel,
            "strideTime": self.strideTimeLabel, "swingTime": self.swingTimeLabel,
        }

    def only3dLabels(self):
        ui = self.ui
        return (ui.hipShoulderEditLabel1, ui.hipShoulderEditLabel2, ui.hipShoulderEditLabel3, self.maxSepLabel,
                self.pelvisRotLabel, self.torsoRotLabel, ui.verticalAngleEditLabel3)

    # ---------- 2D 結果出來後要 3D ----------
    def currentFolder(self):
        path = getattr(self.video_loader, "video_path", None)
        return os.path.dirname(path) if path else None

    def outputFolder(self, folder):
        """3D 結果的 CF_ 子資料夾（與 3D 處理的命名相同：正面影片檔名去掉副檔名）"""
        path = getattr(self.video_loader, "video_path", None)
        return os.path.join(folder, os.path.splitext(os.path.basename(path))[0]) if path else None

    def setPhaseResult(self, phase_result):
        super().setPhaseResult(phase_result)
        if getattr(self, "_applying_3d", False):
            return
        folder = self.currentFolder()
        if not folder:
            return
        for label in self.only3dLabels():
            label.setText(PENDING)
        saved = os.path.join(self.outputFolder(folder) or "", METRICS_FILE)
        if os.path.isfile(saved):
            self.apply3D(folder, saved)
            return
        self.setSource("2D 結果（3D 分析中…）")
        self.request3D(folder)

    def setSource(self, text):
        self.result_source = text
        title = clip_title(self.video_loader.video_path) if getattr(self.video_loader, "video_path", None) else ""
        self.ui.ResolutionLabel.setText(f"{title}　［{text}］" if title else f"［{text}］")

    def apply3D(self, folder, metrics_path):
        if folder != self.currentFolder():
            return  # 已經換到別支錄影
        try:
            with open(metrics_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            self.fail3D(folder, f"讀不到 {metrics_path}：{e}")
            return
        # 3D 判斷的關鍵幀與 2D 不同時（偶爾差一幀），畫面上的關鍵幀跟著 3D
        phases = data.get("phases", {})
        pr = getattr(self, "phase_result", None)
        if pr is not None and any(phases.get(k) is not None and getattr(pr, k) != phases[k] for k in phases):
            for k, v in phases.items():
                if v is not None:
                    setattr(pr, k, int(v))
            self._applying_3d = True
            try:
                super().setPhaseResult(pr)
            finally:
                self._applying_3d = False
        labels = self.metricLabels()
        tooltips = data.get("tooltips", {})
        for key, text in data.get("texts", {}).items():
            if key in labels:
                labels[key].setText(text)
                labels[key].setToolTip(tooltips.get(key, ""))
        # 著地圖上的跨步距離標註也換成 3D
        if data.get("stride_cm") is not None and pr is not None:
            self.stride_cm = data["stride_cm"]
            self.display_phase_image(self.ui.keyphaseLabel2, pr.F_landing, 'landing')
        self.setSource("3D")

    def fail3D(self, folder, reason):
        if folder != self.currentFolder():
            return
        print(f"[錄影頁] 3D 無法分析：{reason}")
        for label in self.only3dLabels():
            label.setText(MISSING)
        self.setSource("2D 結果（3D 無法分析）")

    # ---------- 背景程序 ----------
    def startWorker(self):
        self.worker = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("QT_QPA_PLATFORM", "offscreen")
        self.worker.setProcessEnvironment(env)
        self.worker.setProcessChannelMode(QProcess.MergedChannels)
        self.worker.readyReadStandardOutput.connect(self.readWorker)
        self.worker.finished.connect(lambda *_: print("[錄影頁] 背景 3D 程序已結束"))
        self.worker.start(sys.executable, ["-u", WORKER])

    def request3D(self, folder):
        if self.worker is None or self.worker.state() == QProcess.NotRunning:
            self.startWorker()
        side = "right" if self.ui.batterInput.currentText() == "右打" else "left"
        self.worker.write(f"{folder}\t{side}\n".encode("utf-8"))

    def readWorker(self):
        self.worker_buffer += bytes(self.worker.readAllStandardOutput()).decode("utf-8", errors="replace")
        *lines, self.worker_buffer = self.worker_buffer.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            if line.startswith("WORKER "):
                print(f"[錄影頁] {line}")
                kind, _, rest = line[len("WORKER "):].partition(" ")
                folder, _, detail = rest.partition("\t")
                if kind == "DONE":
                    self.apply3D(folder, detail)
                elif kind == "ERROR":
                    self.fail3D(folder, detail)
            elif "Traceback" in line or "Error" in line:
                print(f"[背景 3D] {line}")

    def shutdown(self):
        super().shutdown()
        if self.worker is not None and self.worker.state() != QProcess.NotRunning:
            self.worker.closeWriteChannel()  # 背景程序讀到 EOF 後自行結束
            if not self.worker.waitForFinished(3000):
                self.worker.kill()
