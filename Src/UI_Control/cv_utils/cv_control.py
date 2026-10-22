import cv2
import numpy as np
import queue
import os
import pandas as pd
from enum import Enum
from PyQt5.QtWidgets import QGraphicsScene
from utils.vis_image_dual import ImageDrawer
from cv_utils.cv_thread import VideoCaptureThread, VideoWriterThread, VideoToImagesThread
from PyQt5.QtWidgets import *
import EasyPySpin

class Camera:
    def __init__(self, camera_idx:int = 0):
        self.camera_idx = camera_idx
        self.is_opened = False
        self.frame_count = 0
        self.fps_control = 1
        self.frame_buffer = queue.Queue()
        self.video_path = None
        self.video_writer = None
        self.video_thread = None
        self.frame_size=None

    def open_camera(self):
        # 開啟相機，並設置回調來處理每一幀
        self.frame_count = 0
        self.video_thread = VideoCaptureThread(camera_index=self.camera_idx)
        self.video_thread.frame_ready.connect(self.buffer_frame)
        self.video_thread.start_capture()
        self.is_opened = True

    def close_camera(self):
        # 關閉相機，清理資源
        if self.video_thread is not None:
            self.video_thread.stop_capture()
            self.video_thread = None
        self.is_opened = False
        self.frame_buffer = queue.Queue()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

    def toggle_camera(self, is_checked:bool):
        # 根據checkbox狀態切換相機
        if is_checked:
            self.open_camera()
            frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
            fps = int(self.video_thread.cap.get_pyspin_value("AcquisitionFrameRate"))
            self.frame_size=(frame_width,frame_height)
            return (frame_width, frame_height, fps)
        else:
            self.close_camera()
            self.frame_size=None
            return (0, 0, 0)

    def buffer_frame(self, frame:np.ndarray):
        # 接收每一幀並進行處理
        self.frame_count += 1
        if self.is_opened and self.frame_count % self.fps_control ==0:
            self.frame_buffer.put(frame)
            # print(self.frame_buffer.qsize())

        if self.video_writer is not None and self.video_writer.is_writing:
            self.video_writer.write_frame(frame)

    def start_recording(self, filename: str):
        # 開始錄製影片
        if self.video_thread is None:
            return
        frame_width = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_thread.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_thread.cap.get_pyspin_value("AcquisitionFrameRate"))
        # fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
        self.video_path = filename
        self.video_writer = VideoWriterThread(filename, frame_width, frame_height, fps=fps)
        self.video_writer.start_writing()

    def stop_recording(self):
        # 停止錄製影片
        if self.video_writer is not None:
            self.video_writer.stop_writing()  # 停止寫入
            self.video_writer.release()  # 釋放執行緒

    def set_camera_idx(self, new_idx: int):
        self.camera_idx = new_idx

    def set_fps_control(self, fps:int):
        self.fps_control = fps

class DataType(Enum):
    DEFAULT = {"name": "default", "tips": "", "filter": ""}
    IMAGE = {"name": "image", "tips": "",
             "filter": "Image files (*.jpeg *.png *.tiff *.psd *.pdf *.eps *.gif)"}
    VIDEO = {"name": "video", "tips": "",
             "filter": "Video files ( *.WEBM *.MPG *.MP2 *.MPEG *.MPE *.MPV *.OGG *.MP4 *.M4P *.M4V *.AVI *.WMV *.MOV *.QT *.FLV *.SWF *.MKV)"}
    CSV = {"name": "csv", "tips": "",
           "filter": "Video files (*.csv)"}
    FOLDER = {"name": "folder", "tips": "", "filter": ""}

class VideoLoader:
    def __init__(self, image_drawer: ImageDrawer =None):
        self.image_drawer = image_drawer
        self.video_path = None
        self.folder_path = None
        self.video_size = None
        self.video_fps = None
        self.video_frames = None
        self.total_frames = None
        self.video_name = None
        self.bat_json_name = None
        self.swingData_json_name = None
        self.is_loading = False
    
    def loadVideo(self, video_path:str = None):
        options = QFileDialog.Options()
        if video_path is None:
            video_path, _ = QFileDialog.getOpenFileName(None, "Select Video File", "", "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
            if not video_path:
                return
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.folder_path = os.path.dirname(video_path)
        self.is_loading = True
        self.v_t = VideoToImagesThread(self.video_path)
        self.v_t.emit_signal.connect(self.video_to_frame)
        self.v_t.start()


    def video_to_frame(self, video_frames, fps, count):
        self.total_frames = count
        self.video_frames = video_frames
        self.video_fps = fps
        self.video_size = (self.video_frames[0].shape[1], self.video_frames[0].shape[0])
        
        self.close_thread(self.v_t)

    def close_thread(self, thread):
        thread.stop()
        self.is_loading = False
        thread = None
    
    def getVideoImage(self, frame_num:int) -> np.ndarray:
        return self.video_frames[frame_num].copy()
    
    def saveBatAnalyzeData(self):
        batAnalyzeData = {
            "SwingSpeed": int(self.image_drawer.batSwingSpeed) if self.image_drawer.batSwingSpeed is not None else "",
            "SwingTime": int(self.image_drawer.batSwingTime) if self.image_drawer.batSwingTime is not None else "",
            "SwingHorizontalAngle": int(self.image_drawer.batHorizonAngle) if self.image_drawer.batHorizonAngle is not None else ""
        }
        batAnalyzeData_df = {key: [value] for key, value in batAnalyzeData.items()}

        return batAnalyzeData_df

    def saveVideo(self, process:bool):
        output_folder = os.path.join("../../Db/Record", self.video_name)
        os.makedirs(output_folder, exist_ok=True)

        save_batAnalyzeData_df = pd.DataFrame(self.saveBatAnalyzeData())

        json_path = os.path.join(output_folder, f"{self.video_name}.json")
        bat_json_path = os.path.join(output_folder, f"{self.video_name}_bat.json")
        bat_analyze_json_path = os.path.join(output_folder, f"{self.video_name}_SwingData.json")

        self.bat_json_name = f"{self.video_name}_bat"
        self.swingData_json_name = f"{self.video_name}_SwingData"

        save_person_df = self.image_drawer.pose_estimater.person_df
        save_bat_df = self.image_drawer.pose_estimater.bat_df

        save_person_df.to_json(json_path, orient='records')
        if save_bat_df is not None:
            if not save_bat_df.empty : 
                save_bat_df.to_json(bat_json_path, orient='records')
                save_batAnalyzeData_df.to_json(bat_analyze_json_path, orient='records')

        if process:
            save_location = os.path.join(output_folder, f"{self.video_name}_Sk26.mp4")
        else: save_location = os.path.join(output_folder, f"{self.video_name}.mp4")
        video_writer = cv2.VideoWriter(save_location, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)

        if not video_writer.isOpened():
            print("Error while opening video writer!")
            return
        if process:
            for frame_num, frame in enumerate(self.video_frames):
                image = self.image_drawer.drawInfo(img = frame, frame_num = frame_num)
                video_writer.write(image)
        else:
            for frame_num, frame in enumerate(self.video_frames):
                # image = self.image_drawer.drawInfo(img = frame, frame_num = frame_num)
                video_writer.write(frame)
        video_writer.release()
        print("Store video success")

    def reset(self):
        self.video_path = None
        self.folder_path = None
        self.video_size = None
        self.video_fps = None
        self.video_frames = None
        self.total_frames = None
        self.video_name = None
        self.is_loading = False

class JsonLoader:
    def __init__(self, folder_path:str = None, file_name:str = None):
        self.folder_path = folder_path
        self.file_name = file_name
        self.person_df = pd.DataFrame()
        self.bat_df = pd.DataFrame()
        self.SwingData_df = pd.DataFrame()

    def load(self):
        if self.folder_path == None or self.file_name == None:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        json_path = os.path.join(self.folder_path, f"{self.file_name}.json")
        bat_json_path = os.path.join(self.folder_path, f"{self.file_name}_bat.json")
        swingData_json_path = os.path.join(self.folder_path, f"{self.file_name}_SwingData.json")
        if not os.path.exists(json_path):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        self.person_df = pd.read_json(json_path)
        self.bat_df = pd.read_json(bat_json_path)
        self.SwingData_df = pd.read_json(swingData_json_path)
        return self.person_df, self.bat_df, self.SwingData_df
