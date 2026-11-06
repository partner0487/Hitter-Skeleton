import cv2
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import sys
import cv2
import queue
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from utils.timer import Timer
from utils.paths import CAMERA_CONFIG_DIR
import time

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
        fps = 0
        for path, images in ((input_video_path, video_images), (input_video_path2, video_images2)):
            if not path:
                continue
            vidcap = cv2.VideoCapture(path)
            fps = fps or vidcap.get(cv2.CAP_PROP_FPS)
            while self._run_flag:
                success, image = vidcap.read()
                if not success: break
                images.append(image)
            vidcap.release()

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

class VideoCaptureThread(QThread):
    # 正面相機沒用時第一個畫面為 None（純側面錄影），所以用 object 而非 np.ndarray
    frame_ready = pyqtSignal(object, object, float)
    # 相機 180 fps，每 EMIT_EVERY 幀送出一次（180 → 60 fps，與論文資料集一致）
    EMIT_EVERY = 3


    def __init__(self, parent=None, front_sn=None, side_sn=None):
        """front_sn 為 None 時只開側面相機；序號由 utils/camera_roles.resolve_cameras 決定"""
        super().__init__(parent)
        CONFIG = str(CAMERA_CONFIG_DIR / "GH3_camera_config.yaml")    # 正面
        CONFIG_2 = str(CAMERA_CONFIG_DIR / "GH3_camera_config_2.yaml")  # 側面
        self.camera2 = FlirCameraSystem(CONFIG_2, side_sn)
        try:
            self.camera1 = FlirCameraSystem(CONFIG, front_sn) if front_sn else None
        except Exception:
            try:
                self.camera2.release()
            except Exception:
                pass
            raise
        self.running = False

    @property
    def side_only(self):
        return self.camera1 is None

    def start_capture(self):
        self.running = True
        if not self.isRunning():
            self.start()

    def stop_capture(self):
        self.running = False
        self.wait()

    def grab(self):
        """(成功, 正面, 側面)；只開側面時正面為 None"""
        ok2, frame_2 = self.camera2.get_grayscale_image()
        if self.camera1 is None:
            return ok2 and frame_2 is not None, None, frame_2
        ok1, frame_1 = self.camera1.get_grayscale_image()
        return ok1 and ok2 and frame_1 is not None and frame_2 is not None, frame_1, frame_2

    def run(self):
        count = 0
        fail_count = 0
        max_fail = 30  # 連續失敗超過此次數才視為相機斷線
        try:
            while self.running:
                ret, frame_1, frame_2 = self.grab()
                grab_time = time.perf_counter()
                if not ret:
                    fail_count += 1
                    if fail_count >= max_fail:
                        print(f"Error: 連續 {max_fail} 次擷取失敗，停止相機")
                        break
                    continue
                fail_count = 0

                # 只轉換要送出的幀，避免浪費運算
                if count % self.EMIT_EVERY == 0:
                    rgb_frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BayerBG2BGR) if frame_1 is not None else None
                    rgb_frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BayerBG2BGR)
                    self.frame_ready.emit(rgb_frame_1, rgb_frame_2, grab_time)
                count += 1
        finally:
            # 無論正常結束或發生例外都要釋放相機
            for camera in (self.camera1, self.camera2):
                if camera is not None:
                    camera.release()


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
