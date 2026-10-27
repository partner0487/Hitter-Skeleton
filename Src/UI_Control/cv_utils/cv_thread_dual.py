import cv2
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from utils.timer import Timer
from utils.paths import CAMERA_CONFIG_DIR


class VideoToImagesThread(QThread):
    emit_signal = pyqtSignal([list, list, int,int])   
    def __init__(self,video_path, video_path2):
        super(VideoToImagesThread, self).__init__()
        self.video_path= video_path
        self.video_path2= video_path2
        self._run_flag = True

    def video_to_frame(self, input_video_path, input_video_path2):
        video_images = []
        video_images2 = []
        
        vidcap = cv2.VideoCapture(input_video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        
        vidcap2 = cv2.VideoCapture(input_video_path2)

        # 讀取第一部影片
        while self._run_flag:
            success, image = vidcap.read()
            if not success: break
            video_images.append(image)
        vidcap.release()

        # 讀取第二部影片
        while self._run_flag:
            success2, image2 = vidcap2.read()
            if not success2: break
            video_images2.append(image2)
        vidcap2.release()

        # 回傳最大長度與個別列表
        count = max(len(video_images), len(video_images2))
        return video_images, video_images2, int(fps), count

    def run(self):
        video_images, video_images2, fps, count= self.video_to_frame(self.video_path,self.video_path2)
        # 被中止時不送出讀到一半的資料
        if not self._run_flag:
            return
        self.emit_signal.emit(video_images, video_images2, fps, count)

    def stop(self):
        """Sets run flag to False; caller should wait() for the thread to finish"""
        self._run_flag = False
        print("stop video to image thread")

from camera_objects.single_camera import FlirCameraSystem 
from camera_objects.two_cameras import DualFlirSystem

class VideoCaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray,np.ndarray)

    def __init__(self, parent=None, camera_index=0):
        super().__init__(parent)
        CONFIG = str(CAMERA_CONFIG_DIR / "GH3_camera_config.yaml")
        CONFIG_2 = str(CAMERA_CONFIG_DIR / "GH3_camera_config_2.yaml")
        SN1 = "25083472" # 側面
        # SN2 = "25083472" # 正面
        SN2 = "21091470"
        self.camera1 = FlirCameraSystem(CONFIG,SN1)
        self.camera2 = FlirCameraSystem(CONFIG_2,SN2)
        self.cameras = DualFlirSystem(self.camera1,self.camera2)
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
        fail_count = 0
        max_fail = 30  # 連續失敗超過此次數才視為相機斷線
        try:
            while self.running:
                ret, frame_1, frame_2 = self.cameras.get_grayscale_images()
                if not ret or frame_1 is None or frame_2 is None:
                    fail_count += 1
                    if fail_count >= max_fail:
                        print(f"Error: 連續 {max_fail} 次擷取失敗，停止相機")
                        break
                    continue
                fail_count = 0

                # 只轉換要送出的幀，避免浪費運算
                if count % 3 == 0:
                    rgb_frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BayerBG2BGR)
                    rgb_frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BayerBG2BGR)
                    self.frame_ready.emit(rgb_frame_1, rgb_frame_2)  # 發送影像
                count += 1
        finally:
            # 無論正常結束或發生例外都要釋放相機
            self.cameras.release()


class VideoWriterThread(QThread):
    # 使用 signal 傳遞狀態更新
    writeFinished = pyqtSignal()

    def __init__(self, filepath, frame_width, frame_height, fps=60, codec='mp4v'):
        super().__init__()
        self.filepath = filepath
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.codec = codec
        self.is_writing = False

    def run(self):
        # 在 run 方法中進行視頻寫入
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        # 持續寫入直到 is_writing 被設置為 False
        while self.is_writing:
            if hasattr(self, 'frame') and self.frame is not None:
                self.writer.write(self.frame)

        # 錄製結束後釋放資源
        self.writer.release()
        self.writeFinished.emit()

    def start_writing(self):
        self.is_writing = True
        self.start()

    def stop_writing(self):
        self.is_writing = False

    def write_frame(self, frame):
        self.frame = frame

    def release(self):
        self.stop_writing()
        self.wait()  # 等待執行緒安全結束