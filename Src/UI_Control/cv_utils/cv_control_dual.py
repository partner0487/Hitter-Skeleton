import cv2
import numpy as np
import queue
import os
import pandas as pd
from enum import Enum
from PyQt5.QtWidgets import QGraphicsScene
from utils.vis_image_dual import ImageDrawer
from cv_utils.cv_thread_dual import VideoCaptureThread, VideoWriterThread, VideoToImagesThread
from PyQt5.QtWidgets import *
import EasyPySpin
import cv2
import collections
import threading
from datetime import datetime
import os
import glob

class ContinuousRecorder:
    def __init__(self, fps=180, pre_roll_sec=2, post_roll_sec=1):
        self.fps = fps
        # self.pre_roll_frames = int(fps * pre_roll_sec)
        # self.post_roll_frames = int(fps * post_roll_sec)
        
        # 使用 deque 作為環形緩衝區，儲存 (frame1, frame2)
        # 50 表示錄製觸發「前」的 50 幀畫面。若要錄更前面可加大數字。
        self.buffer = collections.deque(maxlen=50)
        
        self.is_recording_post = False
        self.post_count = 0
        self.trigger_data = [] # 用於暫存觸發後的影像 tuples
        self.is_saving = False
        self.full_video = []

    def add_frame(self, frame1, frame2):
        if self.is_saving: return
        
        # 永遠在更新緩衝區
        if not self.is_recording_post:
            self.buffer.append((frame1.copy(), frame2.copy()))
        else:
            # 如果處於觸發後的「後一秒」錄製狀態
            self.trigger_data.append((frame1.copy(), frame2.copy()))
            self.post_count += 1
            
            if self.post_count >= 40: # 假設 30 是你要的後段長度
                # 將緩衝區的所有內容轉存到 full_video
                self.full_video = list(self.buffer) + self.trigger_data
                self.is_recording_post = False 
                print("片段擷取完成，等待寫入磁碟...")

    def trigger(self):
        """當偵測到球碰到球棒時呼叫此函數"""
        if not self.is_recording_post and not self.is_saving:
            print("Event Triggered! Recording post-roll...")
            self.is_recording_post = True
            self.post_count = 0
            self.trigger_data = []

    def reset_trigger(self):
        self.is_recording_post = False
        self.post_count = 0
        self.trigger_data = []
        self.full_video = []

class Camera:
    # 即時預覽只需最新畫面，限制緩衝避免延遲累積與記憶體成長
    FRAME_BUFFER_SIZE = 2

    def __init__(self, camera_idx:int = 0):
        self.camera_idx = camera_idx
        self.is_opened = False
        self.frame_count = 0
        self.fps_control = 15
        self.frame_buffer = queue.Queue(maxsize=self.FRAME_BUFFER_SIZE)
        self.frame_buffer_2 = queue.Queue(maxsize=self.FRAME_BUFFER_SIZE)
        self.video_path = None
        self.video_path2 = None
        self.video_writer1 = None
        self.video_writer2 = None
        self.video_thread = None
        self.frame_size=None
        self.continuous_recorder = ContinuousRecorder(fps=180, pre_roll_sec=2, post_roll_sec=1)

    def _put_latest(self, buffer: queue.Queue, frame: np.ndarray):
        # 佇列滿時丟掉最舊的幀，只保留最新畫面
        if buffer.full():
            try:
                buffer.get_nowait()
            except queue.Empty:
                pass
        buffer.put_nowait(frame)

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
        self.frame_buffer_2 = queue.Queue()
        if self.video_writer1 is not None:
            self.video_writer1.release()
            self.video_writer1 = None
            self.video_writer2.release()
            self.video_writer2 = None

    def toggle_camera(self, is_checked:bool):
        # 根據checkbox狀態切換相機
        if is_checked:
            self.open_camera()
            frame_width = int(self.video_thread.camera1.get_width())
            frame_height = int(self.video_thread.camera1.get_height())
            fps = int(self.video_thread.camera1.get_fps())
            self.frame_size=(frame_width,frame_height)
            return (frame_width, frame_height, fps)
        else:
            self.close_camera()
            self.frame_size=None
            return (0, 0, 0)

    def buffer_frame(self, frame:np.ndarray, frame2:np.ndarray):
        # 接收每一幀並進行處理
        self.frame_count += 1
        
        # 更新 ContinuousRecorder
        if self.continuous_recorder:
            self.continuous_recorder.add_frame(frame, frame2)
            
        if self.is_opened and self.frame_count % self.fps_control ==0:
            self._put_latest(self.frame_buffer, frame)
            self._put_latest(self.frame_buffer_2, frame2)

        if self.video_writer1 is not None and self.video_writer1.is_writing:
            self.video_writer1.write_frame(frame)
            self.video_writer2.write_frame(frame2)

    def start_recording(self, filename: str, filename2:str):
        # 開始錄製影片
        if self.video_thread is None:
            return
        frame_width = int(self.video_thread.camera1.get_width())
        frame_height = int(self.video_thread.camera1.get_height())
        fps = int(self.video_thread.camera1.get_fps())
        # fps = int(self.video_thread.cap.get(cv2.CAP_PROP_FPS))
        self.video_path = filename
        self.video_path2 = filename2
        self.video_writer1 = VideoWriterThread(filename, frame_width, frame_height, fps=fps)
        self.video_writer2 = VideoWriterThread(filename2, frame_width, frame_height, fps=fps)
        self.video_writer1.start_writing()
        self.video_writer2.start_writing()

    def stop_recording(self):
        # 停止錄製影片
        if self.video_writer1 is not None or self.video_writer2 is not None:
            self.video_writer1.stop_writing()  # 停止寫入
            self.video_writer2.stop_writing()  # 停止寫入
            self.video_writer2.release()  # 釋放執行緒
            self.video_writer1.release()  # 釋放執行緒

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
            "person_3d": self._read_json("person_3d.json"),
        }
        return data



class VideoLoader:
    def __init__(self, image_drawer: ImageDrawer =None, image_drawer2: ImageDrawer =None):
        self.image_drawer = image_drawer
        self.image_drawer2 = image_drawer2
        
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
        self.bat_json_name = None
        self.swingData_json_name = None
        self.is_loading = False
        self.output_folder = None
    
    def loadVideo(self, video_path:str = None, video_path2:str = None):
        self.reset()
        options = QFileDialog.Options()
        initial_dir = "../../Db/Record"
        # if video_path is None or video_path2 is None:
        #     video_path, _ = QFileDialog.getOpenFileName(None, "選取正面攝影機", initial_dir, "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        #     video_path2, _ = QFileDialog.getOpenFileName(None, "選取側面攝影機", video_path, "Video Files (*.mp4 *.avi);;All Files (*)", options=options)
        #     if not video_path or not video_path2:
        #         return
        # 如果沒有傳入路徑，則讓使用者選取資料夾
        if video_path is None or video_path2 is None:
            folder_selected = QFileDialog.getExistingDirectory(None, "選取包含影片的資料夾", initial_dir, options=options)
            
            if not folder_selected:
                return

            # 搜尋資料夾內符合 CF 與 CS 開頭的影片檔
            # 使用 glob 可以同時處理不同副檔名
            cf_files = glob.glob(os.path.join(folder_selected, "CF*.mp4")) + glob.glob(os.path.join(folder_selected, "CF*.avi"))
            cs_files = glob.glob(os.path.join(folder_selected, "CS*.mp4")) + glob.glob(os.path.join(folder_selected, "CS*.avi"))

            if not cf_files or not cs_files:
                print("找不到對應的 CF 或 CS 影片檔案！")
                return
            
            # 預設抓取找到的第一個檔案
            video_path = cf_files[0]
            video_path2 = cs_files[0]

        print("Load Video Sucess!")
        self.video_path = video_path
        self.video_path2 = video_path2
        self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        self.video_name2 = os.path.splitext(os.path.basename(self.video_path))[0]

        self.folder_path = os.path.dirname(video_path)
        self.folder_path2 = os.path.dirname(video_path2)

        self.is_loading = True
        self.v_t = VideoToImagesThread(self.video_path, self.video_path2)
        self.v_t.emit_signal.connect(self.video_to_frame)
        self.v_t.start()

    def abort_loading(self):
        if self.is_loading and hasattr(self, 'v_t') and self.v_t is not None:
            self.v_t.stop()
            self.v_t.wait()
            # Disconnect the signal so it doesn't interfere with future loads
            try:
                self.v_t.emit_signal.disconnect(self.video_to_frame)
            except TypeError:
                pass
            self.v_t = None
            self.is_loading = False
            self.reset()
            print("Video loading aborted.")


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
            return None
        safe_num = max(0, min(frame_num, len(self.video_frames1) - 1))
        return self.video_frames1[safe_num].copy()
    
    def getVideoImage2(self, frame_num:int) -> np.ndarray:
        if self.video_frames2 is None or len(self.video_frames2) == 0:
            return None
        safe_num = max(0, min(frame_num, len(self.video_frames2) - 1))
        return self.video_frames2[safe_num].copy()

    def saveBatAnalyzeData1(self):
        batAnalyzeData1 = {
            "SwingSpeed": int(self.image_drawer.batSwingSpeed) if self.image_drawer.batSwingSpeed is not None else "",
            "SwingTime": int(self.image_drawer.batSwingTime) if self.image_drawer.batSwingTime is not None else "",
            "SwingHorizontalAngle": int(self.image_drawer.batHorizonAngle) if self.image_drawer.batHorizonAngle is not None else ""
        }
        batAnalyzeData_df1 = {key: [value] for key, value in batAnalyzeData1.items()}

        return batAnalyzeData_df1

    def saveBatAnalyzeData2(self):
        batAnalyzeData2 = {
            "SwingSpeed": int(self.image_drawer2.batSwingSpeed) if self.image_drawer2.batSwingSpeed is not None else "",
            "SwingTime": int(self.image_drawer2.batSwingTime) if self.image_drawer2.batSwingTime is not None else "",
            "SwingHorizontalAngle": int(self.image_drawer2.batHorizonAngle) if self.image_drawer2.batHorizonAngle is not None else ""
        }
        batAnalyzeData_df2 = {key: [value] for key, value in batAnalyzeData2.items()}

        return batAnalyzeData_df2

    def saveVideo(self, process:bool):
        
        # self.output_folder = os.path.join("../../Db/Record", self.video_name)
        output_folder = self.folder_path
        os.makedirs(output_folder, exist_ok=True)
        json_path_side = os.path.join(output_folder, f"{self.video_name}_side.json")
        json_path_front = os.path.join(output_folder, f"{self.video_name}_front.json")
        bat_json_path_side = os.path.join(output_folder, f"{self.video_name}_batSide.json")
        bat_json_path_front = os.path.join(output_folder, f"{self.video_name}_batFront.json")
        batht_json_path_side = os.path.join(output_folder, f"{self.video_name}_bathtSide.json")
        batht_json_path_front = os.path.join(output_folder, f"{self.video_name}_bathtFront.json")

        ball_json_path_side = os.path.join(output_folder, f"{self.video_name}_ballSide.json")
        ball_json_path_front = os.path.join(output_folder, f"{self.video_name}_ballFront.json")

        save_person_df_side = self.image_drawer.pose_estimater.person_df
        save_person_df_front = self.image_drawer2.pose_estimater.person_df
        save_bat_df_side = self.image_drawer.pose_estimater.bat_df
        save_bat_df_front = self.image_drawer2.pose_estimater.bat_df
        save_ball_df_side = self.image_drawer.pose_estimater.ball_df
        save_ball_df_front = self.image_drawer2.pose_estimater.ball_df

        save_batht_df_side = self.image_drawer.bat_analyzer.batHeadTail_df
        save_batht_df_front = self.image_drawer2.bat_analyzer.batHeadTail_df

        save_person_df_side.to_json(json_path_side, orient='records')
        save_person_df_front.to_json(json_path_front, orient='records')
        save_bat_df_side.to_json(bat_json_path_side, orient='records')
        save_bat_df_front.to_json(bat_json_path_front, orient='records')
        save_ball_df_side.to_json(ball_json_path_side, orient='records')
        save_ball_df_front.to_json(ball_json_path_front, orient='records')
        save_batht_df_side.to_json(batht_json_path_side, orient='records')
        save_batht_df_front.to_json(batht_json_path_front, orient='records')

        if process:
            save_location1 = os.path.join(output_folder, f"{self.video_name}_side_Sk26.mp4")
            save_location2 = os.path.join(output_folder, f"{self.video_name}_front_Sk26.mp4")
            video_writer1 = cv2.VideoWriter(save_location1, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)
            video_writer2 = cv2.VideoWriter(save_location2, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)
            if not video_writer1.isOpened():
                print("Error while opening video writer!")
                return
            if not video_writer2.isOpened():
                print("Error while opening video writer!")
                return
            for frame_num, frame in enumerate(self.video_frames1):
                image = self.image_drawer2.drawInfo(img = frame, frame_num = frame_num)
                video_writer1.write(image)
            for frame_num, frame in enumerate(self.video_frames2):
                image = self.image_drawer.drawInfo(img = frame, frame_num = frame_num)
                video_writer2.write(image)
        else: 
            save_location1 = os.path.join(output_folder, f"{self.video_name}_side.mp4")
            save_location2 = os.path.join(output_folder, f"{self.video_name}_front.mp4")
            video_writer1 = cv2.VideoWriter(save_location1, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)
            video_writer2 = cv2.VideoWriter(save_location2, cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, self.video_size)
            if not video_writer1.isOpened():
                print("Error while opening video writer!")
                return
            if not video_writer2.isOpened():
                print("Error while opening video writer!")
                return
            for frame_num, frame in enumerate(self.video_frames1):
                video_writer1.write(frame)
            for frame_num, frame in enumerate(self.video_frames2):
                video_writer2.write(frame)
        video_writer1.release()
        video_writer2.release()
        print("Store video success")

    def saveDualVideo(self, process: bool):
        # 1. 建立輸出資料夾：Db/Record/<video_name>/
        # self.output_folder = os.path.join("../../Db/Record", self.video_name)
        
        # Create a subfolder using the video filename (without extension)
        video_filename = os.path.splitext(os.path.basename(self.video_path))[0]
        output_folder = os.path.join(self.folder_path, video_filename)
        
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # 2. JSON 檔命名規則：<kind>_<view>.json
        #   person_side.json, person_front.json
        #   bat_side.json,    bat_front.json
        #   batht_side.json,  batht_front.json
        #   ball_side.json,   ball_front.json
        json_path_side_person = os.path.join(output_folder, "person_side.json")
        json_path_front_person = os.path.join(output_folder, "person_front.json")

        json_path_side_bat = os.path.join(output_folder, "bat_side.json")
        json_path_front_bat = os.path.join(output_folder, "bat_front.json")

        json_path_side_batht = os.path.join(output_folder, "batht_side.json")
        json_path_front_batht = os.path.join(output_folder, "batht_front.json")

        json_path_side_ball = os.path.join(output_folder, "ball_side.json")
        json_path_front_ball = os.path.join(output_folder, "ball_front.json")

        # 3. 取出要存的 DataFrame
        save_person_df_side = self.image_drawer.pose_estimater.person_df
        save_person_df_front = self.image_drawer2.pose_estimater.person_df

        save_bat_df_side = self.image_drawer.pose_estimater.bat_df
        save_bat_df_front = self.image_drawer2.pose_estimater.bat_df

        save_ball_df_side = self.image_drawer.pose_estimater.ball_df
        save_ball_df_front = self.image_drawer2.pose_estimater.ball_df

        save_batht_df_side = self.image_drawer.bat_analyzer.batHeadTail_df
        save_batht_df_front = self.image_drawer2.bat_analyzer.batHeadTail_df

        # 4. 寫 JSON（orient='records' 一樣保留）
        save_person_df_side.to_json(json_path_side_person, orient='records')
        save_person_df_front.to_json(json_path_front_person, orient='records')

        save_bat_df_side.to_json(json_path_side_bat, orient='records')
        save_bat_df_front.to_json(json_path_front_bat, orient='records')

        save_ball_df_side.to_json(json_path_side_ball, orient='records')
        save_ball_df_front.to_json(json_path_front_ball, orient='records')

        save_batht_df_side.to_json(json_path_side_batht, orient='records')
        save_batht_df_front.to_json(json_path_front_batht, orient='records')

        if process:
            # video_writer1 吃到的是 CF 的影像 -> 對應 front
            save_location1 = os.path.join(output_folder, "skel_front.mp4")
            # video_writer2 吃到的是 CS 的影像 -> 對應 side
            save_location2 = os.path.join(output_folder, "skel_side.mp4")
        else:
            save_location1 = os.path.join(output_folder, "raw_front.mp4")
            save_location2 = os.path.join(output_folder, "raw_side.mp4")

        video_writer1 = cv2.VideoWriter(
            save_location1,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.video_fps,
            self.video_size
        )
        video_writer2 = cv2.VideoWriter(
            save_location2,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.video_fps,
            self.video_size
        )

        if not video_writer1.isOpened():
            print("Error while opening video writer1!")
            return
        if not video_writer2.isOpened():
            print("Error while opening video writer2!")
            return

        if process:
            # 畫上骨架再寫出
            for frame_num, frame in enumerate(self.video_frames1):
                image = self.image_drawer2.drawInfo(img=frame, frame_num=frame_num)
                video_writer1.write(image)
            for frame_num, frame in enumerate(self.video_frames2):
                image = self.image_drawer.drawInfo(img=frame, frame_num=frame_num)
                video_writer2.write(image)
        else:
            # 直接存原始畫面
            for frame in self.video_frames1:
                video_writer1.write(frame)
            for frame in self.video_frames2:
                video_writer2.write(frame)

        video_writer1.release()
        video_writer2.release()
        print("Store video success, folder =", output_folder)

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
