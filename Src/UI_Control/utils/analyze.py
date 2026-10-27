import numpy as np
import pandas as pd
from skeleton.detect_skeleton_dual import PoseEstimater
from utils.one_euro_filter import OneEuroFilter
from utils.fps_utils import DEFAULT_FPS

class PoseAnalyzer:
    def __init__(self, pose_estimater:PoseEstimater):
        self.pose_estimater = pose_estimater
        self.angle_dict = self.pose_estimater.joints['haple']['angle_dict']
        self.analyze_info = []
        self.analyze_df = pd.DataFrame()
        self.processed_frames = set()
        self.distancePerPixel = 0


    def addAnalyzeInfo(self, frame_num: int):
        """Analyze information for each frame up to the current frame."""
        if self.pose_estimater.person_id is None:
            return pd.DataFrame()
        
        person_kpt = self.pose_estimater.getPersonDf(frame_num= frame_num, is_select= True,is_kpt=True)
        
        if person_kpt is None:
            return
        info = {
            'frame_number': frame_num,
            'angle': self._update_analyze_information(person_kpt)
        }

        if frame_num not in self.processed_frames:
            self.processed_frames.add(frame_num)
            self.analyze_info.append(info)
            self.analyze_df = pd.DataFrame(self.analyze_info)
            self.analyze_df = self.analyze_df.sort_values(by='frame_number').reset_index(drop=True)

    def _calculate_angle(self, A, B, C):
        """Calculate the angle between three points A, B, and C."""
        BA = np.array(A) - np.array(B)
        BC = np.array(C) - np.array(B)
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)
        cos_angle = dot_product / (magnitude_BA * magnitude_BC)
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle_rad)

    def _update_analyze_information(self, person_kpt):
        """Update and return analyze information for the given keypoints."""
        info = {}
        for angle_name, kpt_list in self.angle_dict.items():
            A = person_kpt[kpt_list[0]][:2]
            B = person_kpt[kpt_list[1]][:2]
            C = person_kpt[kpt_list[2]][:2]
            info[angle_name] = [self._calculate_angle(A, B, C), [np.array(A), np.array(B), np.array(C)]]
        return info

    def get_frame_angle_data(self, frame_num: int = None, angle_name: str = None) :
        # print(self.analyze_df)
        if self.analyze_df.empty:
            return pd.DataFrame(), []
        condition = pd.Series([True] * len(self.analyze_df))

        # 根據 frame_num 過濾數據
        if frame_num is not None:
            condition &= (self.analyze_df['frame_number'] == frame_num)

        data = self.analyze_df.loc[condition]
        
        # 如果數據為空則返回
        if data.empty:
            return None, []

        if angle_name is not None:
            # 如果指定了 angle_name 且有特定幀數，返回該幀的角度數據
            if frame_num is not None:
                angle_value = data['angle'].iloc[0][angle_name]
                return data, angle_value
            else:
                # 如果未指定 frame_num，則返回該 angle_name 的所有幀數及對應角度
                frame_numbers = self.analyze_df['frame_number'].unique()
                angles = [row['angle'][angle_name][0] for _, row in self.analyze_df.iterrows() if angle_name in row['angle']]
                return frame_numbers, angles
        
        return data, []

    def reset(self):
        self.analyze_info = []
        self.analyze_df = pd.DataFrame()
        self.processed_frames = set()


class BatAnalyzer:
    def __init__(self, pose_estimater:PoseEstimater):
        self.pose_estimater = pose_estimater
        self.person_id = pose_estimater.person_id
        self.smooth_filter = OneEuroFilter(min_cutoff= 0.1, beta=0.1)
        self.batterDirection = 0

        self.bat_head = None
        self.bat_tail = None
        self.pre_bat_head = None
        self.pre_bat_tail = None

        self.speed_data = []
        self.batHeadMovingSpeed_df = pd.DataFrame()

        self.angle_data = []
        self.batHeadAngle_df = pd.DataFrame()

        # Parameters
        self.fps = DEFAULT_FPS
        self.batsize = 81
        self.batHeadTail = []
        self.batHeadTail_df = pd.DataFrame()
        self.start2DrawTrajIdx = None

        self.liftLeg = False
        self.recordLiftFeetPosition = None
        self.originFeetPosition = None

        self.end2DrawTrajIdx = None
        self.lowPointFrame = None
        self.preLowPointFrame = None
        self.lowPoint = None
        self.lowPointAngle = None


    def calculate2PointsDistance(self, point1, point2): 
        return np.abs(np.linalg.norm(np.array(point1) - np.array(point2)))
    
    def detStart2DrawTraj(self, frame_num: int= None): # 判斷是否開始畫traj
        # if self.start2DrawTrajIdx == None:
        #     # 還沒開始才要判斷
        #     person_kpt = self.pose_estimater.getPersonDf(frame_num= frame_num, is_select= True, is_kpt=True)
        #     # meanwrist = (np.array(person_kpt[9][:2]) + np.array(person_kpt[10][:2]))/2
        #     wrist1 = np.array(person_kpt[9][:2])
        #     wrist2 = np.array(person_kpt[10][:2])
        #     if wrist1[1] <= np.array(person_kpt[18][:2][1]) or wrist2[1] <= np.array(person_kpt[18][:2][1]) :
        #         self.start2DrawTrajIdx = frame_num
        if self.start2DrawTrajIdx is None:
            person_kpt = self.pose_estimater.getPersonDf(frame_num= frame_num, is_select= True, is_kpt=True)
            if self.batterDirection == 0:
                ankle = np.array(person_kpt[15][:2])
            elif self.batterDirection == 1:
                ankle = np.array(person_kpt[16][:2])
            if self.originFeetPosition is None:
                self.originFeetPosition = ankle[1]
                self.recordLiftFeetPosition = ankle[1]
            elif ankle[1] < self.recordLiftFeetPosition: 
                self.recordLiftFeetPosition = ankle[1]
            elif self.liftLeg == False and ankle[1] > (self.recordLiftFeetPosition + 15):
                self.liftLeg = True
                self.start2DrawTrajIdx = frame_num

    def detEnd2DrawTraj(self, frame_num: int= None): # 判斷是否結束畫traj
        if self.start2DrawTrajIdx != None and self.lowPointFrame != None and self.end2DrawTrajIdx == None:
            # 還沒結束才要判斷
            # 透過棒尾跟頸部去判斷
            person_kpt = self.pose_estimater.getPersonDf(frame_num= frame_num, is_select= True, is_kpt=True)
            tail = np.array(self.batHeadTail_df['tail'][self.batHeadTail_df['frame_num'] == frame_num])
            if tail[0][1] <= person_kpt[18][:2][1]:
                self.end2DrawTrajIdx = frame_num

    # 把 mask 轉成head、tail可以判斷
    def appendHeadTail(self, frame_num: int = None):
        # 找出frame_num = frame_num的棒頭及棒尾
        if self.pose_estimater.person_id is None :
            # 如果沒選人回傳空字串
            return [], []
        else:
            # 有選人
            # 判斷kpt是否存在
            person_kpt = self.pose_estimater.getPersonDf(frame_num= frame_num, is_select= True, is_kpt=True)
            if person_kpt is None: # kpt沒東西 回傳空值
                return [], []

            else:
                # 有偵測到人的kpt，看球棒
                bat_data = self.pose_estimater.getBatDF(frame_num= frame_num)
                if bat_data.empty: return [], [] # 沒有偵測到球棒 回傳空值
                # else calculate head tail 有偵測到球棒 開始計算棒頭棒尾

                # 找距離手腕最遠的地方為棒頭
                # bat_mask_xy = np.array(bat_data['mask'].values.tolist())[0][1][0]
                bat_mask_list = bat_data['mask'].values.tolist()

                # 過濾掉不符合條件的資料，確保每個元素都有兩個部分
                bat_mask_xy = [item[1][0] for item in bat_mask_list if len(item) > 1 and len(item[1]) > 0]
                if not bat_mask_xy:
                    return [], []  # 如果過濾後沒資料，回傳空值

                # 取第一個有效的 mask 資料
                bat_mask_xy = np.array(bat_mask_xy[0])


                meanwrist = (np.array(person_kpt[9][:2]) + np.array(person_kpt[10][:2]))/2
                d_head_wrist = np.linalg.norm(bat_mask_xy - meanwrist, axis=1)
                max_index = np.argmax(d_head_wrist)
                bat_head = np.array(bat_mask_xy[max_index])
                d_head_tail = np.linalg.norm(bat_mask_xy - bat_head, axis=1)
                max_index_tail = np.argmax(d_head_tail)
                bat_tail = bat_mask_xy[max_index_tail]
                if self.pre_bat_head is not None and self.pre_bat_tail is not None:
                    bat_head = np.array(bat_head)
                    bat_tail = np.array(bat_tail)
                    
                    bat_head = self.smooth_filter(bat_head, self.pre_bat_head)
                    bat_tail = self.smooth_filter(bat_tail, self.pre_bat_tail)

                # 確認當前 frame_num 是否已經存在
                # 若是透過slider去拉會有問題 因此需做該步驟
                if not any(item['frame_num'] == frame_num for item in self.batHeadTail):
                    bat_tail_head_info = {
                        'head': bat_head,
                        'tail' : bat_tail,
                        'frame_num': np.array(bat_data['frame_number'].values.tolist())[0]
                    }
                    self.pre_bat_head = bat_head
                    self.pre_bat_tail = bat_tail
                    self.batHeadTail.append(bat_tail_head_info)
                    self.batHeadTail_df = pd.DataFrame(self.batHeadTail)
    
    def getBatHeadTailData(self, frame_num: int = None):
        # print(self.batHeadTail_df)
        if self.batHeadTail_df.empty:
           return [], [] 
        batHeadTailData = self.batHeadTail_df[self.batHeadTail_df['frame_num'] == frame_num]
        if batHeadTailData.empty:
            return [], []
        bat_haed = batHeadTailData['head'].iloc[0]
        bat_tail = batHeadTailData['tail'].iloc[0]
        return bat_haed, bat_tail


    def appendSpeed(self, frame_num:int = None):
        # 如果是最後一個才要去加，否則取用就好
        if self.batHeadTail_df.empty:
            return
        if frame_num == self.batHeadTail_df['frame_num'].iloc[-1]:
            if self.start2DrawTrajIdx != None and self.end2DrawTrajIdx == None and self.batHeadTail_df.shape[0] >= 1 :
                # 當開始畫trajectory 並且資料達到2個可以計算speed
                distancePerPixel = float(self.batsize) / self.calculate2PointsDistance(self.batHeadTail_df['head'].iloc[-1], self.batHeadTail_df['tail'].iloc[-1])
                distance = distancePerPixel * self.calculate2PointsDistance(self.batHeadTail_df['head'].iloc[-1], self.batHeadTail_df['head'].iloc[-2])
                startFrame = self.batHeadTail_df['frame_num'].iloc[-2]
                endFrame = self.batHeadTail_df['frame_num'].iloc[-1]
                throughtFrame = endFrame - startFrame
                speed = (distance / (float(1/self.fps) * throughtFrame)) / 100
                speed_info = {
                    'speed': float(speed),
                    'distance':float(distance),
                    'thougthFrame':int(throughtFrame),
                    'startFrame': int(startFrame),
                    'endFrame' : int(endFrame)
                }

                if self.batHeadMovingSpeed_df.empty == True:
                    self.speed_data.append(speed_info)
                    self.batHeadMovingSpeed_df = pd.DataFrame(self.speed_data)
                elif not (self.batHeadMovingSpeed_df['endFrame'] == frame_num).any():
                    self.speed_data.append(speed_info)
                    self.batHeadMovingSpeed_df = pd.DataFrame(self.speed_data)


    def meanBatSpeed(self):
        if self.batHeadMovingSpeed_df is None or self.batHeadMovingSpeed_df.empty or 'endFrame' not in self.batHeadMovingSpeed_df.columns:
            if self.batHeadTail_df is None or self.batHeadTail_df.empty or self.start2DrawTrajIdx is None or self.lowPointFrame is None:
                return None, None
            
            traj_df = self.batHeadTail_df[
                (self.batHeadTail_df['frame_num'] >= self.start2DrawTrajIdx) &
                (self.batHeadTail_df['frame_num'] <= self.lowPointFrame)
            ].sort_values('frame_num')
            
            if len(traj_df) < 2:
                return None, None
                
            total_distance_cm = 0
            for i in range(1, len(traj_df)):
                prev = traj_df.iloc[i-1]
                curr = traj_df.iloc[i]
                
                dist_px = self.calculate2PointsDistance(curr['head'], curr['tail'])
                if dist_px == 0: continue
                dist_per_px = float(self.batsize) / dist_px
                
                head_dist_px = self.calculate2PointsDistance(curr['head'], prev['head'])
                dist_cm = head_dist_px * dist_per_px
                total_distance_cm += dist_cm
                
            total_frames = traj_df.iloc[-1]['frame_num'] - traj_df.iloc[0]['frame_num']
            total_time_ms = total_frames * (1.0 / self.fps) * 1000.0
            
            if total_time_ms != 0:
                mean_speed = (total_distance_cm / 100.0) / (total_time_ms / 1000.0)
                return mean_speed, total_time_ms
            return None, None

        filtered_df = self.batHeadMovingSpeed_df[
            (self.batHeadMovingSpeed_df['endFrame'] >= self.start2DrawTrajIdx) & 
            (self.batHeadMovingSpeed_df['endFrame'] <= self.lowPointFrame)
        ]
        weighted_distance = (filtered_df['distance']).sum() / 100 # cm > m
        total_througth_time = filtered_df['thougthFrame'].sum() * float(1/self.fps) * 1000 # ms
        if total_througth_time != 0:
            weighted_mean_speed = (weighted_distance) / (total_througth_time / 1000)
            return weighted_mean_speed, total_througth_time
        else:
            return None, None


    def getSwingLowerFrame(self, h, frame_num:int =None):
        if self.start2DrawTrajIdx != None and self.end2DrawTrajIdx == None:
            if self.lowPointFrame is None and self.batHeadTail_df is not None and self.batHeadTail_df.empty is not True :
                filter_df = np.array(self.batHeadTail_df['head'][self.batHeadTail_df['frame_num']== frame_num])
                if filter_df.size > 0  :
                    point = filter_df[0]
                    if point[1] >= h/2:
                        if self.lowPoint is None:
                            self.lowPoint = point
                            self.preLowPoint = frame_num
                        elif point[1] >= self.lowPoint[1]:
                            self.lowPoint = point
                            self.preLowPointFrame = frame_num
                        elif point[1] < self.lowPoint[1]:
                            self.lowPointFrame = self.preLowPointFrame
    
    def calAngleInSwingLowerFrame(self):
        if self.lowPointFrame != None and self.end2DrawTrajIdx != None:
            prev_idx = self.batHeadTail_df[self.batHeadTail_df['frame_num'] == self.lowPointFrame].index
            curr_idx = self.batHeadTail_df[self.batHeadTail_df['frame_num'] == self.lowPointFrame].index + 1
            red_line_prev = np.array(tuple(map(int, self.batHeadTail_df['head'].iloc[prev_idx].values[0])))
            red_line_curr = np.array(tuple(map(int, self.batHeadTail_df['head'].iloc[curr_idx].values[0])))
            vector = red_line_curr - red_line_prev

            horizontal_vector = np.array([1, 0])  # 水平向量
            dot_product = np.dot(vector, horizontal_vector)
            vector_norm = np.linalg.norm(vector)
            horizontal_norm = np.linalg.norm(horizontal_vector)

            if vector_norm == 0:
                print("Invalid vector with zero length.")
                return None

            cos_angle = dot_product / (vector_norm * horizontal_norm)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) 
            angle_in_degrees = np.degrees(angle)

            return angle_in_degrees
        return None


    def appendBatHeadAngle(self, frame_num:int = None):
        if self.batHeadTail_df.empty is False and self.batHeadTail_df.shape[0] >= 2 and self.start2DrawTrajIdx !=  None and self.end2DrawTrajIdx == None:
            prev_idx = self.batHeadTail_df[self.batHeadTail_df['frame_num'] == frame_num].index -1
            curr_idx = self.batHeadTail_df[self.batHeadTail_df['frame_num'] == frame_num].index
            prev = np.array(tuple(map(int, self.batHeadTail_df['head'].iloc[prev_idx].values[0])))
            curr = np.array(tuple(map(int, self.batHeadTail_df['head'].iloc[curr_idx].values[0])))
            vector = curr - prev

            angle_rad = np.arctan2(vector[1], vector[0])  
            angle_in_degrees = np.degrees(angle_rad)  

            if angle_in_degrees < 0:
                angle_in_degrees += 360

            angle_info = {
                'angle': int(angle_in_degrees),
                'preFrameNum': frame_num - 1,
                'curFrameNum': frame_num
            }



            if self.start2DrawTrajIdx !=  None  and self.end2DrawTrajIdx ==  None:
                self.angle_data.append(angle_info)
                self.batHeadAngle_df = pd.DataFrame(self.angle_data)
            elif self.end2DrawTrajIdx != None and frame_num >= self.start2DrawTrajIdx and frame_num <= self.end2DrawTrajIdx:
                if not (self.batHeadAngle_df['curFrameNum'] == frame_num).any():
                    self.angle_data.append(angle_info)
                    self.batHeadAngle_df = pd.DataFrame(self.angle_data)

    def getAngleData(self, frame_num:int = None):
        if self.batHeadAngle_df.empty is False:
            print(frame_num)
            print(self.batHeadAngle_df['curFrameNum'].iloc[-1])
            print(self.batHeadAngle_df)
            idx = self.batHeadAngle_df[self.batHeadAngle_df['curFrameNum'] == frame_num].index
            angle = np.array(tuple(map(int, self.batHeadAngle_df['angle'].iloc[idx])))
            return angle
        else: return None

    def reset(self):
        self.smooth_filter = OneEuroFilter()
        self.batsize = 87
        self.batterDirection = 0

        self.bat_head = None
        self.bat_tail = None
        self.pre_bat_head = None
        self.pre_bat_tail = None

        self.speed_data = []
        self.batHeadMovingSpeed_df = pd.DataFrame()

        self.angle_data = []
        self.batHeadAngle_df = pd.DataFrame()

        self.batHeadTail = []
        self.batHeadTail_df = pd.DataFrame()
        self.start2DrawTrajIdx = None

        self.liftLeg = False
        self.recordLiftFeetPosition = None
        self.originFeetPosition = None

        self.end2DrawTrajIdx = None
        self.lowPointFrame = None
        self.preLowPointFrame = None
        self.lowPoint = None
        self.lowPointAngle = None


  



class JointAreaChecker:
    def __init__(self, image_size:tuple):
        self.image_width = image_size[0]
        self.image_height = image_size[1]

        # 設置1/5區域的邊界
        self.region_width = image_size[0]
        self.region_height = image_size[1] // 5

        # 區域左上角的坐標
        self.region_top_left = (0, 0)
        self.region_bottom_right = (self.region_width, self.region_height)

    def is_joint_in_area(self, joint_position:tuple)->bool:
        """檢查關節點是否在定義的區域內"""
        if joint_position is None:
            return False
        x, y = joint_position
        
        in_area = (self.region_top_left[0] <= x <= self.region_bottom_right[0] and
                   self.region_top_left[1] <= y <= self.region_bottom_right[1])
        return in_area
    
    def is_ready_for_hitting(self, writstl, writstr, neck):

        if (writstr == None) or (writstl == None) or (neck == None):
            return False
        mean_wrist = (writstl[1] + writstr[1]) / 2
        if mean_wrist < neck[1]: return True
        else: return False