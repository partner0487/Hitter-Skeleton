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

    
