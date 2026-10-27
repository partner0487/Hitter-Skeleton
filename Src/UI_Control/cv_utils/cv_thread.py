import cv2
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from utils.timer import Timer
import queue

class VideoToImagesThread(QThread):
    emit_signal = pyqtSignal([list,int,int])   
    _run_flag=True
    def __init__(self,video_path):
        super(VideoToImagesThread, self).__init__()
        self.video_path=video_path

    def video_to_frame(self, input_video_path):
        video_images = []
        vidcap = cv2.VideoCapture(input_video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        frame_counter = 0
        count = 0
        while success:
            # # load in even number frame only
            # if not frame_counter & 1:
            video_images.append(image)
            count += 1
            success, image = vidcap.read()
            # frame_counter += 1
        vidcap.release()
        fps = int(fps) >> 1
        # set image count labels
        return video_images, fps, count

    def run(self):
        # capture from web cam
        video_images, fps, count= self.video_to_frame(self.video_path)
        _run_flag=False
        self.emit_signal.emit(video_images, fps, count)
        
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # if self.cap!=None:
        #     self.cap.release()
        print("stop video to image thread")
    
    def isFinished(self):
        print(" finish thread")

import EasyPySpin

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        self.cap = EasyPySpin.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -1)
        self.cap.set(cv2.CAP_PROP_GAIN, -1)
        self.cap.set_pyspin_value("PixelFormat", "BayerRG8")
        self.cap.set_pyspin_value("AcquisitionMode", 0)
        self.cap.set_pyspin_value("ExposureAuto", "Off")
        # self.cap.set_pyspin_value("ExposureTime", 1000.0)
        self.cap.set_pyspin_value("TriggerMode", "Off")
        self.cap.set_pyspin_value("AcquisitionFrameRate", 160.0)
        self.cap.set_pyspin_value("AcquisitionFrameRateControlEnabled", "False")
        print(self.cap.get_pyspin_value("AcquisitionFrameRate"))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera with index {camera_index}")
        self.running = False

    def start_capture(self):
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_capture(self):
        self.running = False
        self.wait()

    def run(self):
        count = 0
        while self.running:     
            ret, frame = self.cap.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
            if ret:
                if count % 3 ==0:
                    self.frame_ready.emit(rgb_frame)  # 發送影像
            else:  # 例外處理
                print("Warning: Failed to capture frame")
                break
            count +=1
        self.cap.release()

class VideoWriterThread(QThread):
    # 使用 signal 傳遞狀態更新
    writeFinished = pyqtSignal()

    def __init__(self, filepath, frame_width, frame_height, fps=60, codec='mp4v', max_queue=300):
        super().__init__()
        self.filepath = filepath
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.codec = codec
        self.is_writing = False
        # 佇列上限避免磁碟寫太慢時記憶體無限成長
        self.frame_queue = queue.Queue(maxsize=max_queue)
        self.dropped_frames = 0

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.frame_width, self.frame_height))
        if not writer.isOpened():
            print(f"無法開啟影片寫入器: {self.filepath}")
            self.is_writing = False
            self.writeFinished.emit()
            return

        # 逐幀從佇列取出寫入，收到 None 代表結束（佇列中剩餘的幀會先寫完）
        while True:
            frame = self.frame_queue.get()
            if frame is None:
                break
            writer.write(frame)

        # 錄製結束後釋放資源
        writer.release()
        if self.dropped_frames:
            print(f"警告: {self.filepath} 因佇列已滿丟棄 {self.dropped_frames} 幀")
        self.writeFinished.emit()

    def start_writing(self):
        self.is_writing = True
        self.start()

    def stop_writing(self):
        # 可重複呼叫：只有第一次會送出結束訊號
        if not self.is_writing:
            return
        self.is_writing = False
        self.frame_queue.put(None)

    def write_frame(self, frame):
        if not self.is_writing:
            return
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1

    def release(self):
        self.stop_writing()
        self.wait()  # 等待執行緒安全結束
