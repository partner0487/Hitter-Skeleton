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
        # 只有側面影片的錄影（純側面 2D）沒有正面影片：該路徑為 None，回傳空的影格列表
        caps = [cv2.VideoCapture(p) if p else None for p in (input_video_path, input_video_path2)]
        fps = next((c.get(cv2.CAP_PROP_FPS) for c in caps if c is not None), 0)

        count = 0
        # 兩支影片同步讀取，以較短者為準
        while self._run_flag:
            reads = [c.read() if c is not None else (True, None) for c in caps]
            if not all(ok for ok, _ in reads):
                break
            for images, (_, image) in zip((video_images, video_images2), reads):
                if image is not None:
                    images.append(image)
            count += 1
            if all(c is None for c in caps):
                break

        for c in caps:
            if c is not None:
                c.release()
        # 回傳影片標記的 fps（原本右移一位減半，3D 頁所有速度都少一半）
        fps = int(round(fps))
        return video_images, video_images2, fps, count

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