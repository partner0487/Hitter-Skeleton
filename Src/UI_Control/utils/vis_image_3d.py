import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from .analyze import PoseAnalyzer, BatAnalyzer
from skeleton.detect_skeleton_dual import PoseEstimater
import os
from scipy.signal import savgol_filter
import sys
import pyqtgraph as pg
from vispy.scene import visuals


try:
    colors = np.round(
        np.array(plt.get_cmap('gist_rainbow').colors) * 255
    ).astype(np.uint8)[:, ::-1].tolist()
except AttributeError:  # if palette has not pre-defined colors
    colors = np.round(
        np.array(plt.get_cmap('gist_rainbow')(np.linspace(0, 1, 10))) * 255
    ).astype(np.uint8)[:, -2::-1].tolist()

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(__file__)

class ImageDrawer():
    def __init__(self, pose_estimater = None, pose_analyzer = None, bat_analyzer = None):
        self.pose_estimater = pose_estimater
        self.pose_analyzer = pose_analyzer
        self.bat_analyzer = bat_analyzer
        self.font_path = os.path.join(application_path, 'R-PMingLiU-TW-2.ttf')
        self.fontStyle = ImageFont.truetype(self.font_path, 20)
        
        self.bat_smoothed_traj = []
        
        self.show_grid = False
        self.show_bbox = False
        self.show_skeleton = False
        self.show_traj = False
        self.show_region = False
        self.show_angle_info = False
        self.show_countdown = False
        self.show_overlay = False
        self.batHorizonAngle = None
        self.batSwingSpeed = None
        self.batSwingTime = None
        self.angle_info_pos = (0,0)
        self.region = [(100, 250), (450, 600)]
        self.joints = {
            "coco": {
                "keypoints": {
                    0: "鼻子",
                    1: "左眼",
                    2: "右眼",
                    3: "左耳",
                    4: "右耳",
                    5: "左肩",
                    6: "右肩",
                    7: "左手肘",
                    8: "右手肘",
                    9: "左手腕",
                    10: "右手腕",
                    11: "左髋",
                    12: "右髋",
                    13: "左膝",
                    14: "右膝",
                    15: "左腳踝",
                    16: "右腳踝"
                },
                "skeleton_links": [
                    [0, 1], [0, 2], [1, 3], [2, 4], # 頭
                    #軀幹
                    [5, 7], [7, 9],                 #左手
                    [6, 8], [8, 10],                #右手
                    [11, 13], [13, 15],   #左腿
                    [12, 14], [14, 16],   #右腿
                ],
                "left_points_indices": [[5, 7], [7, 9], [11, 13], [13, 15]],  # Indices of left hand, leg, and foot keypoints
                "right_points_indices": [[6, 8], [8, 10], [12, 14], [14, 16]]  # Indices of right hand, leg, and foot keypoints
            },
            "haple":{
                "keypoints": {
                    0: "鼻子",
                    1: "左眼",
                    2: "右眼",
                    3: "左耳",
                    4: "右耳",
                    5: "左肩",
                    6: "右肩",
                    7: "左肘",
                    8: "右肘",
                    9: "左腕",
                    10: "右腕",
                    11: "左髖",
                    12: "右髖",
                    13: "左膝",
                    14: "右膝",
                    15: "左踝",
                    16: "右踝",
                    17: "頭部",
                    18: "頸部",
                    19: "臀部",
                    20: "左大腳趾",
                    21: "右大腳趾",
                    22: "左小腳趾",
                    23: "右小腳趾",
                    24: "左腳跟",
                    25: "右腳跟"
                },
                "skeleton_links":[
                    [0, 1], [0, 2], [1, 3], [2, 4], # 頭
                    [5, 18], [6, 18], [17, 18],[18, 19],#軀幹
                    [5, 7], [7, 9],                 #左手
                    [6, 8], [8, 10],                #右手
                    [19, 11], [11, 13], [13, 15],   #左腿
                    [19, 12], [12, 14], [14, 16],   #右腿
                    [20, 24], [22, 24], [15, 24],   #左腳
                    [21, 25], [23, 25], [16, 25]    #右腳
                ],
                
                "left_points_indices": [[5, 18], [5, 7], [7, 9],[19, 11], [11, 13], [13, 15], [20, 24], [22, 24], [15, 24]],  # Indices of left hand, leg, and foot keypoints
                "right_points_indices": [[6, 18], [6, 8], [8, 10], [19, 12], [12, 14], [14, 16], [21, 25], [23, 25], [16, 25]],  # Indices of right hand, leg, and foot keypoints
                "angle_dict":{
                    # 'l_elbow_angle': [5, 7, 9],
                    '右手肘': [6, 8, 10]
                    # 'l_shoulder_angle': [18, 5, 7],
                    # 'r_shoulder_angle': [18, 6, 8],
                    # 'l_knee_angle': [11, 13, 15],
                    # 'r_knee_angle': [12, 14, 16]
                }
            },
        }


    def drawInfo(self, img:np.ndarray, frame_num:int=None, person_data:list = None, batht_data:list = None, ball_data:list = None, countdown_time:int = None):
        if img is None:
            return
        image = img.copy()
        if self.show_region:
            image = self.drawRegion(image)
        
        if countdown_time is not None:
            self.show_countdown = True

        if self.show_countdown:
            image = self.drawCountdown(image, countdown_time)

        if self.show_grid :
            image = self.drawGrid(image)
        
        if self.show_bbox:
            image = self.drawBbox(image, person_data)
        
        if self.show_skeleton:
            image = self.drawPointsandSkeleton(image, person_data, self.joints['haple']['skeleton_links'], points_palette_samples=10)
            image = self.draw_bat(image, batht_data, frame_num)
            image = self.draw_ball(image, ball_data, frame_num)
        # if self.show_angle_info:
        #     image = self.drawAngleInfo(image, frame_num)

        # image = self.draw_bat(image, frame_num)
        # image = self.drawBatTraj(image, frame_num)

        return image
    
    def drawCountdown(self, img:np.ndarray,countdown_time:int):
        height, width, _ = img.shape
        text = str(countdown_time) 
        
        # 設置文字的大小、顏色與位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 20
        color = (0, 0, 255)  # 紅色
        thickness = 15

        # 計算文字的邊界框來居中顯示
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        # 在透明圖層上繪製文字
        cv2.putText(img, text, (text_x,text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        if countdown_time == 0:
            self.show_countdown = False

        return img

    def draw_bat(self, image, batht_data, frame_num: int = None):
        """
        在 image 上畫出該 frame_num 對應的球棒頭尾連線。
        batht_data: pandas DataFrame，至少包含 'head', 'tail', 'frame_num' 三欄。
        """
        if batht_data is None or len(batht_data) == 0:
            return image

        # 若有給 frame_num，就依 frame_num 過濾；若 batht_data 已經是單一幀，也可以略過這步
        if frame_num is not None and 'frame_num' in batht_data.columns:
            row_df = batht_data.loc[batht_data['frame_num'] == frame_num]
            if row_df.empty:
                return image
            row = row_df.iloc[0]
        else:
            # 當作只有一筆資料
            row = batht_data.iloc[0]

        bat_head = row['head']   # 例如 [1250.43, 296.18]
        bat_tail = row['tail']   # 例如 [1194.09, 395.16]

        if bat_head is None or bat_tail is None:
            return image

        if len(bat_head) < 2 or len(bat_tail) < 2:
            return image

        # 轉成整數 pixel 座標
        x1, y1 = map(int, bat_head[:2])
        x2, y2 = map(int, bat_tail[:2])

        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), thickness=15)
        return image

    def draw_ball(self, image, ball_data, frame_num: int = None):
        """
        在 image 上畫出該 frame_num 對應球的圓點（淺藍色，大小依 bbox 決定）。
        ball_data: pandas DataFrame，至少包含 'bbox', 'frame_number'。
                bbox 格式: [[x1, y1, x2, y2]]
        """
        if ball_data is None or len(ball_data) == 0:
            return image

        # 依 frame_num 取出對應資料；若 ball_data 已先過濾成單一幀，可以不傳 frame_num
        if frame_num is not None and 'frame_number' in ball_data.columns:
            row_df = ball_data.loc[ball_data['frame_number'] == frame_num]
            if row_df.empty:
                return image
            row = row_df.iloc[0]
        else:
            row = ball_data.iloc[0]

        bbox = row.get("bbox", None)
        if bbox is None or not isinstance(bbox, list) or len(bbox) == 0:
            return image

        # bbox: [[x1, y1, x2, y2]]
        x1, y1, x2, y2 = bbox[0]
        w = float(x2 - x1)
        h = float(y2 - y1)
        if w <= 0 or h <= 0:
            return image

        # 中心點
        cx = int((x1 + x2) / 2.0)
        cy = int((y1 + y2) / 2.0)

        # 半徑：取寬高較小值的一半，並給一個下限避免太小
        radius = int(0.5 * min(w, h))
        radius = max(radius, 2)

        # 淺藍色（BGR）
        light_blue = (255, 255, 128)  # 你可依需求微調

        cv2.circle(image, (cx, cy), radius=radius, color=light_blue, thickness=-1)

        return image

    def drawCross(self, image, x, y, length=5, color=(0, 0, 255), thickness=2):
        cv2.line(image, (x, y - length), (x, y + length), color, thickness)
        cv2.line(image, (x - length, y), (x + length, y), color, thickness)

    def drawGrid(self, image:np.ndarray):
        #return image:np.ndarray
        
        height, width = image.shape[:2]
        self.drawCross(image,int(width/2),int(height/2),length=20,color=(0,0,255),thickness = 3)
        # 計算垂直線的位置
        vertical_interval = width // 5
        vertical_lines = [vertical_interval * i for i in range(1, 5)]

        # 計算水平線的位置
        horizontal_interval = height // 5
        horizontal_lines = [horizontal_interval * i for i in range(1, 5)]

        # 畫垂直線
        for x in vertical_lines:
            cv2.line(image, (x, 0), (x, height), (0, 255, 0), 2)

        # 畫水平線
        for y in horizontal_lines:
            cv2.line(image, (0, y), (width, y), (0, 255, 0), 2)

        return image

    def drawBbox(self, image:np.ndarray, person_df:pd.DataFrame):
        if person_df.empty:
            return image
        person_ids = person_df['person_id']
        person_bbox = person_df['bbox']
        for id, bbox in zip(person_ids, person_bbox):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = tuple(colors[id % len(colors)])
            color = (0,255,0)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            image = cv2.putText(image, str(id), (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1.5, color, 2)
        return image


    def drawRegion(self, img:np.ndarray):
        cv2.rectangle(img, self.region[0], self.region[1], (0, 255, 0), -1)
        return img

    def drawPoints(self, image, points, person_idx, color_palette='gist_rainbow', palette_samples=10, confidence_threshold=0.3):
        """
        Draws `points` on `image`.

        Args:
            image: image in opencv format
            points: list of points to be drawn.
                Shape: (nof_points, 3)
                Format: each point should contain (y, x, confidence)
            color_palette: name of a matplotlib color palette
                Default: 'tab20'
            palette_samples: number of different colors sampled from the `color_palette`
                Default: 16
            confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
                Default: 0.5

        Returns:
            A new image with overlaid points

        """
        try:
            colors = np.round(
                np.array(plt.get_cmap(color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            colors = np.round(
                np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()

        circle_size = max(1, min(image.shape[:2]) // 160)  # ToDo Shape it taking into account the size of the detection
        # circle_size = max(2, int(np.sqrt(np.max(np.max(points, axis=0) - np.min(points, axis=0)) // 16)))
        for i, pt in enumerate(points):
        
            unlabel = False if pt[0] != 0 and pt[1] != 0 else True
            if pt[2] > confidence_threshold and not unlabel:
                image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[person_idx % len(colors)]), -1)

        return image

    def drawSkeleton(self, image, points, skeleton, color_palette='Set2', palette_samples='jet', person_index=0,
                    confidence_threshold=0.5):
        """
        Draws a `skeleton` on `image`.

        Args:
            image: image in opencv format
            points: list of points to be drawn.
                Shape: (nof_points, 3)
                Format: each point should contain (y, x, confidence)
            skeleton: list of joints to be drawn
                Shape: (nof_joints, 2)
                Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
            color_palette: name of a matplotlib color palette
                Default: 'Set2'
            palette_samples: number of different colors sampled from the `color_palette`
                Default: 8
            person_index: index of the person in `image`
                Default: 0
            confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
                Default: 0.5

        Returns:
            A new image with overlaid joints

        """
        try:
            colors = np.round(
                np.array(plt.get_cmap(color_palette).colors) * 255
            ).astype(np.uint8)[:, ::-1].tolist()
        except AttributeError:  # if palette has not pre-defined colors
            colors = np.round(
                np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
            ).astype(np.uint8)[:, -2::-1].tolist()
        right_skeleton = self.joints['haple']['right_points_indices']
        left_skeleton = self.joints['haple']['left_points_indices']
        
        for i, joint in enumerate(skeleton):
            pt1, pt2 = points[joint]
            pt1_unlabel = False if pt1[0] != 0 and pt1[1] != 0 else True
            pt2_unlabel = False if pt2[0] != 0 and pt2[1] != 0 else True
            skeleton_color = tuple(colors[person_index % len(colors)])
            skeleton_color = (0, 165, 255)
            if joint in right_skeleton:
                skeleton_color = (240, 176, 0)
            elif joint in left_skeleton:
                skeleton_color = (0, 0, 255)
            if pt1[2] > confidence_threshold and not pt1_unlabel and pt2[2] > confidence_threshold and not pt2_unlabel:
                image = cv2.line(
                    image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                    skeleton_color , 6
                )
        return image

    def drawPointsandSkeleton(self, image, person_df, skeleton, points_color_palette='gist_rainbow', points_palette_samples=10,
                                skeleton_color_palette='Set2', skeleton_palette_samples='jet', confidence_threshold=0.3):
        """
        Draws `points` and `skeleton` on `image`.

        Args:
            image: image in opencv format
            points: list of points to be drawn.
                Shape: (nof_points, 3)
                Format: each point should contain (y, x, confidence)
            skeleton: list of joints to be drawn
                Shape: (nof_joints, 2)
                Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
            points_color_palette: name of a matplotlib color palette
                Default: 'tab20'
            points_palette_samples: number of different colors sampled from the `color_palette`
                Default: 16
            skeleton_color_palette: name of a matplotlib color palette
                Default: 'Set2'
            skeleton_palette_samples: number of different colors sampled from the `color_palette`
                Default: 8
            person_index: index of the person in `image`
                Default: 0
            confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
                Default: 0.5

        Returns:
            A new image with overlaid joints

        """
        if person_df is None:
            return image
        if person_df.empty:
            return image
        person_data = self.df_to_points(person_df)
        for person_id, points in person_data.items(): 
            image = self.drawSkeleton(image, points, skeleton,person_index=person_id)
            image = self.drawPoints(image, points,person_idx=person_id)
        return image

    def df_to_points(self, person_df):
        person_data = {}
        person_ids = person_df['person_id']
        person_kpts = person_df['keypoints']
        for id, kpts in zip(person_ids, person_kpts):
            person_data[id] = np.array(self.swapValues(kpts))
        return person_data
    

    def swapValues(self, kpts):
        return [[item[1], item[0], item[2]] for item in kpts]

    def setShowBbox(self, status:bool):
        self.show_bbox = status
    
    def setShowSkeleton(self, status:bool):
        self.show_skeleton = status
    
    def setShowGrid(self, status:bool):
        self.show_grid = status
        
    def setShowRegion(self, status:bool):
        self.show_region = status

    def reset(self):
        self.show_grid = False
        self.show_bbox = False
        self.show_skeleton = False