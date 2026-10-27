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
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import GLLinePlotItem
from vispy.scene import visuals
# from OpenGL.GL import *

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
    def __init__(self, pose_estimater: PoseEstimater=None, pose_analyzer:PoseAnalyzer=None, bat_analyzer:BatAnalyzer=None, angle_name:str = "右手肘"):
        self.font_path = os.path.join(application_path, 'R-PMingLiU-TW-2.ttf')
        self.fontStyle = ImageFont.truetype(self.font_path, 20)
        self.pose_estimater = pose_estimater
        self.pose_analyzer = pose_analyzer

        self.bat_analyzer = bat_analyzer # 待修正
        self.bat_smoothed_traj = []
        
        self.angle_name = angle_name
        self.show_grid = False
        self.show_bbox = False
        self.show_skeleton = False
        self.show_traj = False
        self.show_region = False
        self.show_angle_info = False
        self.show_overlay = False
        self.batHorizonAngle = None
        self.batSwingSpeed = None
        self.batSwingTime = None
        self.angle_info_pos = (0,0)
        self.region = [(100, 250), (450, 600)]

    def drawInfo(self, img:np.ndarray, frame_num:int=None, kpt_buffer:list = None):
        if img is None:
            return
        image = img.copy()
        curr_person_df = self.pose_estimater.getPersonDf(frame_num = frame_num, is_select=True)
        if self.show_region:
            image = self.drawRegion(image)
        
        if self.show_grid :
            image = self.drawGrid(image)
        
        if self.show_bbox:
            image = self.drawBbox(image, curr_person_df)
        
        if self.show_skeleton:
            image = self.drawPointsandSkeleton(image, curr_person_df, self.pose_estimater.joints['haple']['skeleton_links'], points_palette_samples=10)
            
        if self.show_traj:
            image = self.drawTraj(image, kpt_buffer)

        if self.show_angle_info:
            image = self.drawAngleInfo(image, frame_num)

        image = self.draw_bat(image, frame_num)
        # image = self.drawBatTraj(image, frame_num)

        return image

    def draw_bat(self, image, frame_num:int=None):
        # 畫球棒
        if self.bat_analyzer == None: return image
        bat_head, bat_tail = self.bat_analyzer.getBatHeadTailData(frame_num=frame_num) 
        if len(bat_head) and len(bat_tail):
            bat_head = (int(np.array(bat_head[0])), int(np.array(bat_head[1])))
            bat_tail = (int(np.array(bat_tail[0])), int(np.array(bat_tail[1])))
            cv2.line(image, bat_head, bat_tail, (0, 255, 255), thickness=15)
        return image

    def drawOverlay(self, image, frame_num:int=None):
        df = pd.DataFrame()
        if self.bat_analyzer.start2DrawTrajIdx is not None:
            if self.bat_analyzer.end2DrawTrajIdx is not None:
                # 根据 frame_num 范围过滤 df
                if frame_num >= self.bat_analyzer.start2DrawTrajIdx and frame_num <= self.bat_analyzer.end2DrawTrajIdx:
                    df = self.bat_analyzer.batHeadTail_df[
                        (self.bat_analyzer.batHeadTail_df['frame_num'] <= frame_num) & 
                        (self.bat_analyzer.batHeadTail_df['frame_num'] >= self.bat_analyzer.start2DrawTrajIdx)
                    ]
                elif frame_num >= self.bat_analyzer.end2DrawTrajIdx:
                    df = self.bat_analyzer.batHeadTail_df[
                        (self.bat_analyzer.batHeadTail_df['frame_num'] <= self.bat_analyzer.end2DrawTrajIdx) & 
                        (self.bat_analyzer.batHeadTail_df['frame_num'] >= self.bat_analyzer.start2DrawTrajIdx)
                    ]
            else:
                # 仅在 start2DrawTrajIdx 已设置时，绘制轨迹到当前帧
                if frame_num >= self.bat_analyzer.start2DrawTrajIdx:
                    df = self.bat_analyzer.batHeadTail_df[
                        (self.bat_analyzer.batHeadTail_df['frame_num'] <= frame_num)
                    ]
            h, w = image.shape[:2]
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    head = tuple(map(int, row['head']))
                    tail = tuple(map(int, row['tail']))
                    if head[0] >= w / 2 or head[1] >= h / 3:
                        cv2.line(image, head, tail, (0, 255, 255), thickness=15)

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

    def drawTraj(self, img: np.ndarray, kpt_buffer: list):
        if not kpt_buffer or len(kpt_buffer) < 2:
            return img

        # 將座標轉換為整數
        int_kpt_buffer = [tuple(map(int, kpt)) for kpt in kpt_buffer]
        
        # 迭代相鄰的兩個點，並畫出軌跡線
        for (f_kptx, f_kpty), (s_kptx, s_kpty) in zip(int_kpt_buffer[:-1], int_kpt_buffer[1:]):
            cv2.line(img, (f_kptx, f_kpty), (s_kptx, s_kpty), (0, 255, 0), 5)

        return img

    def drawAngleInfo(self, img: np.ndarray, frame_num: int) -> np.ndarray:
        # 从 pose_analyzer 获取角度数据
        _, angle_info = self.pose_analyzer.get_frame_angle_data(frame_num, self.angle_name)
        if (len(angle_info) == 0):
            return img
        # 提取角度值，并将其转换为整数
        angle_value = int(angle_info[0])
        
        # 提取坐标并将其转换为整数元组
        p = tuple(map(int, angle_info[1][1]))
        # 使用 cv2.line 绘制线条
        cv2.line(img, p, (self.angle_info_pos[0] + 20, self.angle_info_pos[1] + 320), (0, 0, 0), 2)
        
        # 使用 cv2.putText 绘制角度值
        img = cv2.putText(img, str(angle_value), (self.angle_info_pos[0] - 50, self.angle_info_pos[1] + 420), 
                        cv2.FONT_HERSHEY_COMPLEX, 3.5, (0, 255, 0), 3)
        
        return img

    def drawRegion(self, img:np.ndarray):
        # 取得影像的高度與寬度
        h, w = img.shape[:2]
        
        # 計算中心點
        center_x, center_y = w // 2, h // 2
    
        # 設定顏色與粗細 (顏色: 綠色, 粗細: 2)
        color = (0, 255, 0)
        thickness = 2
        
        # 畫水平線 (從左到右)
        cv2.line(img, (0, center_y), (w, center_y), color, thickness)
        
        # 畫垂直線 (從上到下)
        cv2.line(img, (center_x, 0), (center_x, h), color, thickness)
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
        right_skeleton = self.pose_estimater.joints['haple']['right_points_indices']
        left_skeleton = self.pose_estimater.joints['haple']['left_points_indices']
        
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


    def draw_3d_pose(self, gl_widget, points3d):
        """
        Draw 3D pose on the given GLViewWidget (gl_widget).

        Args:
            gl_widget: the pyqtgraph.opengl.GLViewWidget to draw on.
            points3d: list or np.array of [x, y, z] joint positions.
        """
        # 1. 清除原有的內容
        for item in gl_widget.items[:]:  # 複製避免錯誤
            if not isinstance(item, gl.GLGridItem):
                gl_widget.removeItem(item)

        # 2. 畫關節點 (blue)
        points = np.array(points3d)
        # scatter = GLScatterPlotItem(
        #     pos=points,
        #     size=2,
        #     color=(0, 0.2, 0.2, 1),  # cyan
        #     pxMode=False  # 改為 True 嘗試修正顯示問題
        # )
        # gl_widget.addItem(scatter)

        # 3. 畫骨架線段 (red)
        skeleton = self.pose_estimater.joints['haple']['skeleton_links']
        for joint in skeleton:
            pt1 = points[joint[0]]
            pt2 = points[joint[1]]
            line_pts = np.array([pt1, pt2])
            line = GLLinePlotItem(
                pos=line_pts,
                color=(1, 0, 0, 1),  # red
                width=5,
                antialias=True
            )
            gl_widget.addItem(line)


    

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

    def setShowTraj(self, status:bool):
        self.show_traj = status

    def setShowAngleInfo(self, status:bool):
        self.show_angle_info = status
        self.setAngleInfoPos()

    def setAngleInfoPos(self):
        person_df = self.pose_estimater.getPersonDf(is_select=True)
        if person_df is None:
            return
        self.angle_info_pos = person_df.iloc[0]['keypoints'][19]
        self.angle_info_pos = tuple(map(int,self.angle_info_pos))

    def setShowOverlay(self, status:bool):
        self.show_overlay = status

    def reset(self):
        self.show_grid = False
        self.show_bbox = False
        self.show_skeleton = False
        self.show_traj = False
        self.show_angle_info = False
        self.angle_info_pos = (0,0)
        
        self.bat_smoothed_traj = []
        self.batHorizonAngle = None
        self.batSwingSpeed = None
        self.batSwingTime = None
        self.show_region = False
        self.show_overlay = False