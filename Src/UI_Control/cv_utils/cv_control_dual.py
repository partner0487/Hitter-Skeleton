import cv2
import numpy as np
import queue
import os
import pandas as pd
from enum import Enum
from PyQt5.QtWidgets import QGraphicsScene
from utils.vis_image_dual import ImageDrawer
from cv_utils.cv_thread_dual import (VideoCaptureThread, VideoWriterThread, VideoToImagesThread, analysis_step,
                                     PRE_ROLL_S, POST_ROLL_S, TRIGGER_COOLDOWN_S)
from utils.paths import RECORD_DIR
from utils.camera_roles import resolve_cameras
from PyQt5.QtWidgets import *
import EasyPySpin
import cv2
import collections
import threading
from datetime import datetime
import os
import glob

class ContinuousRecorder:
    """自動錄影的觸發前後緩衝：在擷取執行緒以相機完整 fps（180，廠商 160）保存原始畫面。

    存相機原始 Bayer（每台每幀約 2 MB，存檔時才轉彩色），180 fps 的 0.83 秒緩衝兩台約 0.6 GB，
    與原本存 60 fps 彩色畫面相同。觸發前後以秒計，換 fps 時分析用影片仍是 1.5 秒。
    add() 在擷取執行緒呼叫，其餘在介面執行緒，以 lock 保護。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.cooldown_until = 0.0
        self.configure(180)

    def configure(self, camera_fps: float):
        """開相機時依相機 fps 設定緩衝長度"""
        with self._lock:
            self.camera_fps = float(camera_fps)
            self.step = analysis_step(camera_fps)
            self.post_frames = max(1, int(round(POST_ROLL_S * camera_fps)))
            self.buffer = collections.deque(maxlen=max(1, int(round(PRE_ROLL_S * camera_fps))))
            self._reset_locked()

    def _reset_locked(self):
        self.is_recording_post = False
        self.is_saving = False      # 片段收完到存檔完成之間，不再觸發
        self.pre_frames = []
        self.post_data = []
        self.clip = None

    def add(self, raw1, raw2, grab_time):
        """擷取執行緒每一幀呼叫（raw1 為 None 表示純側面）；陣列需為呼叫端已複製的新物件"""
        item = (raw1, raw2, grab_time)
        with self._lock:
            self.buffer.append(item)   # 一直滾動，下一次觸發時才有完整的觸發前畫面
            if not self.is_recording_post:
                return
            self.post_data.append(item)
            if len(self.post_data) >= self.post_frames:
                frames = self.pre_frames + self.post_data
                self.clip = {
                    "frames1": [f for f, _, _ in frames],
                    "frames2": [f for _, f, _ in frames],
                    "times": [t for _, _, t in frames],
                    "camera_fps": self.camera_fps,
                    "step": self.step,
                }
                self.pre_frames, self.post_data = [], []
                self.is_recording_post = False
                self.is_saving = True
                print(f"片段擷取完成（{len(frames)} 幀，{self.camera_fps:.0f} fps），等待寫入磁碟...")

    def trigger(self, now: float) -> bool:
        """開始收觸發後的畫面；錄製中、存檔中或冷卻中回傳 False"""
        with self._lock:
            if self.is_recording_post or self.is_saving or now < self.cooldown_until:
                return False
            self.pre_frames = list(self.buffer)
            self.post_data = []
            self.is_recording_post = True
            self.cooldown_until = now + TRIGGER_COOLDOWN_S
            print("Event Triggered! Recording post-roll...")
            return True

    def cooling_down(self, now: float) -> bool:
        return now < self.cooldown_until

    def take_clip(self):
        """取出收完的片段（dict）；還沒收完回傳 None"""
        with self._lock:
            return self.clip

    def reset_trigger(self):
        with self._lock:
            self._reset_locked()

class Camera:
    # 即時預覽只需最新畫面，限制緩衝避免延遲累積與記憶體成長
    FRAME_BUFFER_SIZE = 2

    def __init__(self, camera_idx:int = 0):
        self.camera_idx = camera_idx
        self.is_opened = False
        self.frame_count = 0
        self.fps_control = 2  # 60 fps 再取 1/2 → 30 fps 給 realtime ViTPose
        self.frame_buffer = queue.Queue(maxsize=self.FRAME_BUFFER_SIZE)
        self.frame_buffer_2 = queue.Queue(maxsize=self.FRAME_BUFFER_SIZE)
        self.video_path = None
        self.video_path2 = None
        self.video_writer1 = None
        self.video_writer2 = None
        self.video_thread = None
        self.notice = ""   # 開相機時的提示（例如找不到正面相機、改成只錄側面）
        self.frame_size=None
        self.continuous_recorder = ContinuousRecorder()
        # 擷取執行緒的畫面變化觸發成功時（在介面執行緒）呼叫，安排存檔
        self.on_triggered = None

    def _put_latest(self, buffer: queue.Queue, frame: np.ndarray):
        # 佇列滿時丟掉最舊的幀，只保留最新畫面
        if buffer.full():
            try:
                buffer.get_nowait()
            except queue.Empty:
                pass
        buffer.put_nowait(frame)

    def get_stream_fps(self) -> float:
        """buffer_frame 實際收到的名目 fps（相機 fps ÷ 抽幀間隔）"""
        return self.video_thread.camera_fps / self.video_thread.emit_every

    def set_motion_roi(self, roi):
        """畫面變化觸發的 ROI（側面畫面座標 [x, y, w, h]）；None 表示不偵測"""
        if self.video_thread is not None:
            self.video_thread.motion_roi = tuple(int(v) for v in roi) if roi else None

    def _handle_triggered(self):
        if self.on_triggered is not None:
            self.on_triggered()

    @property
    def side_only(self) -> bool:
        """只開了側面相機（沒接正面或相機設定選了不使用），錄影只寫 CS_ 影片"""
        return self.video_thread is not None and self.video_thread.side_only

    def open_camera(self):
        """開啟相機，並設置回調來處理每一幀。用哪幾台由相機設定決定（utils/camera_roles.py）；
        側面相機沒接上時丟 RuntimeError；正面沒接上改成只開側面，原因放在 self.notice"""
        self.frame_count = 0
        front_sn, side_sn, self.notice = resolve_cameras()
        self.video_thread = VideoCaptureThread(front_sn=front_sn, side_sn=side_sn, recorder=self.continuous_recorder)
        self.video_thread.frame_ready.connect(self.buffer_frame)
        self.video_thread.triggered.connect(self._handle_triggered)
        self.video_thread.start_capture()
        self.is_opened = True

    def close_camera(self):
        # 關閉相機，清理資源
        if self.video_thread is not None:
            self.video_thread.stop_capture()
            self.video_thread = None

        self.is_opened = False
        self.frame_buffer = queue.Queue(maxsize=self.FRAME_BUFFER_SIZE)
        self.frame_buffer_2 = queue.Queue(maxsize=self.FRAME_BUFFER_SIZE)
        self.stop_recording()

    def toggle_camera(self, is_checked:bool):
        # 根據checkbox狀態切換相機；畫面大小與 fps 以側面相機為準（純側面錄影沒有正面）
        if is_checked:
            self.open_camera()
            camera = self.video_thread.camera2
            frame_width = int(camera.get_width())
            frame_height = int(camera.get_height())
            fps = int(camera.get_fps())
            self.frame_size=(frame_width,frame_height)
            return (frame_width, frame_height, fps)
        else:
            self.close_camera()
            self.frame_size=None
            return (0, 0, 0)

    def buffer_frame(self, frame, frame2:np.ndarray, grab_time:float):
        # 接收每一幀並進行處理；只開側面相機時 frame（正面）為 None
        # 自動錄影的緩衝與畫面變化觸發在擷取執行緒以完整 fps 處理（ContinuousRecorder），這裡只負責顯示與手動錄影
        self.frame_count += 1

        if self.is_opened and self.frame_count % self.fps_control ==0:
            if frame is not None:
                self._put_latest(self.frame_buffer, frame)
            self._put_latest(self.frame_buffer_2, frame2)

        for writer, image in ((self.video_writer1, frame), (self.video_writer2, frame2)):
            if writer is not None and writer.is_writing and image is not None:
                writer.write_frame(image)

    def start_recording(self, filename, filename2:str):
        # 開始錄製影片；filename（正面）為 None 時只錄側面
        if self.video_thread is None:
            return
        camera = self.video_thread.camera2
        frame_width = int(camera.get_width())
        frame_height = int(camera.get_height())
        fps = self.get_stream_fps()
        self.video_path = filename
        self.video_path2 = filename2
        self.video_writer1 = VideoWriterThread(filename, frame_width, frame_height, fps=fps) if filename else None
        self.video_writer2 = VideoWriterThread(filename2, frame_width, frame_height, fps=fps)
        for writer in (self.video_writer1, self.video_writer2):
            if writer is not None:
                writer.start_writing()

    def stop_recording(self):
        # 停止錄製影片：先送出結束訊號，再等兩個寫入執行緒寫完
        writers = [w for w in (self.video_writer1, self.video_writer2) if w is not None]
        for writer in writers:
            writer.stop_writing()
        for writer in writers:
            writer.release()
        self.video_writer1 = self.video_writer2 = None

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
        initial_dir = str(RECORD_DIR)
        # 如果沒有傳入路徑，則讓使用者選取資料夾；只有側面影片（CS）時 video_path 為 None，做純側面 2D
        if video_path is None and video_path2 is None:
            folder_selected = QFileDialog.getExistingDirectory(None, "選取包含影片的資料夾", initial_dir, options=options)
            
            if not folder_selected:
                return

            # 搜尋資料夾內符合 CF 與 CS 開頭的影片檔
            # 使用 glob 可以同時處理不同副檔名
            cf_files = glob.glob(os.path.join(folder_selected, "CF*.mp4")) + glob.glob(os.path.join(folder_selected, "CF*.avi"))
            cs_files = glob.glob(os.path.join(folder_selected, "CS*.mp4")) + glob.glob(os.path.join(folder_selected, "CS*.avi"))

            if not cs_files:
                print("找不到側面 CS 影片檔案！")
                return
            
            # 預設抓取找到的第一個檔案
            video_path = cf_files[0] if cf_files else None
            video_path2 = cs_files[0]

        self.video_path = video_path
        self.video_path2 = video_path2
        name = lambda path: os.path.splitext(os.path.basename(path))[0] if path else None
        self.video_name = name(self.video_path)
        self.video_name2 = name(self.video_path2)

        self.folder_path = os.path.dirname(video_path or video_path2)
        self.folder_path2 = os.path.dirname(video_path2 or video_path)

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
        self.total_frames = count
        self.video_frames2 = video_frames2
        # 純側面錄影沒有正面影格，大小都用側面的
        size = lambda frames: (frames[0].shape[1], frames[0].shape[0]) if frames else None
        self.video_size2 = size(self.video_frames2)
        self.video_size = size(self.video_frames1) or self.video_size2
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


    def saveVideo(self, process:bool):
        
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

    def saveResults(self):
        """把偵測結果存成 JSON。不另存 raw_／skel_ 影片：介面直接讀原始 CF／CS 影片、依 JSON 即時畫骨架
        （utils/paths.source_videos），每支片段省下約 3 秒的畫圖與影片壓縮"""
        # 1. 建立輸出資料夾：<錄影資料夾>/<video_name>/
        
        # Create a subfolder using the video filename (without extension)
        # 結果子資料夾以正面影片檔名命名（CF_…）；純側面錄影以側面影片檔名命名（CS_…）
        video_filename = os.path.splitext(os.path.basename(self.video_path or self.video_path2))[0]
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

        print("Store results success, folder =", output_folder)

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
        if not os.path.exists(json_path):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        self.person_df = pd.read_json(json_path)
        self.bat_df = pd.read_json(bat_json_path)
        self.SwingData_df = pd.read_json(swingData_json_path)
        return self.person_df, self.bat_df, self.SwingData_df
