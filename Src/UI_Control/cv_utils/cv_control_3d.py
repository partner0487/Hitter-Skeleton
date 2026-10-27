import cv2
import numpy as np
import queue
import os
import pandas as pd
from enum import Enum
from PyQt5.QtWidgets import QGraphicsScene
from cv_utils.cv_thread_3d import VideoToImagesThread
from PyQt5.QtWidgets import *
import EasyPySpin

class DataType(Enum):
    DEFAULT = {"name": "default", "tips": "", "filter": ""}
    IMAGE = {"name": "image", "tips": "",
             "filter": "Image files (*.jpeg *.png *.tiff *.psd *.pdf *.eps *.gif)"}
    VIDEO = {"name": "video", "tips": "",
             "filter": "Video files ( *.WEBM *.MPG *.MP2 *.MPEG *.MPE *.MPV *.OGG *.MP4 *.M4P *.M4V *.AVI *.WMV *.MOV *.QT *.FLV *.SWF *.MKV)"}
    CSV = {"name": "csv", "tips": "",
           "filter": "Video files (*.csv)"}
    FOLDER = {"name": "folder", "tips": "", "filter": ""}

class DualJsonLoader:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path

    def _read_json(self, name: str):
        path = os.path.join(self.folder_path, name)
        if os.path.exists(path):
            return pd.read_json(path, orient='records')
        return pd.DataFrame()

    def load(self):
        data = {
            "person_side": self._read_json("person_side.json"),
            "person_front": self._read_json("person_front.json"),
            "bat_side": self._read_json("bat_side.json"),
            "bat_front": self._read_json("bat_front.json"),
            "batht_side": self._read_json("batht_side.json"),
            "batht_front": self._read_json("batht_front.json"),
            "ball_side": self._read_json("ball_side.json"),
            "ball_front": self._read_json("ball_front.json"),
        }
        return data



class VideoLoader:
    def __init__(self):
        self.video_path = None
        self.folder_path = None

        self.video_path2 = None
        self.folder_path2 = None

        self.video_size = None
        self.video_fps = None
        self.video_frames1 = None
        self.video_frames2 = None
        self.total_frames = None
        self.is_loading = False
    
    def loadVideo(self, video_path:str = None, video_path2:str = None):
        self.video_path = video_path
        self.video_path2 = video_path2
        self.is_loading = True
        self.v_t = VideoToImagesThread(self.video_path, self.video_path2)
        self.v_t.emit_signal.connect(self.video_to_frame)
        self.v_t.start()
        print("Load Video Sucess!")


    def video_to_frame(self, video_frames1, video_frames2, fps, count):
        self.total_frames = count
        self.video_frames1 = video_frames1
        self.video_fps = fps
        self.video_size = (self.video_frames1[0].shape[1], self.video_frames1[0].shape[0])
        self.total_frames = count
        self.video_frames2 = video_frames2
        self.video_size2 = (self.video_frames2[0].shape[1], self.video_frames2[0].shape[0])
        self.close_thread(self.v_t)

    def close_thread(self, thread):
        thread.stop()
        self.is_loading = False
        thread = None
    
    def getVideoImage(self, frame_num:int) -> np.ndarray:
        if self.video_frames1 is None or len(self.video_frames1) == 0:
            return None  # Offline 不會 crash
        idx = min(frame_num, len(self.video_frames1) - 1)
        return self.video_frames1[idx].copy()
    
    def getVideoImage2(self, frame_num:int) -> np.ndarray:
        if self.video_frames2 is None or len(self.video_frames2) == 0:
            return None  # Offline 不會 crash
        idx = min(frame_num, len(self.video_frames2) - 1)
        return self.video_frames2[idx].copy()


    def reset(self):
        self.video_path = None
        self.folder_path = None
        self.video_path2 = None
        self.folder_path2 = None
        self.video_size = None
        self.video_fps = None
        self.video_frames1 = None
        self.video_frames2 = None
        self.total_frames = None
        self.video_name = None
        self.video_name2 = None
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
        print(swingData_json_path)
        if not os.path.exists(json_path):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        self.person_df = pd.read_json(json_path)
        self.bat_df = pd.read_json(bat_json_path)
        self.SwingData_df = pd.read_json(swingData_json_path)
        return self.person_df, self.bat_df, self.SwingData_df
