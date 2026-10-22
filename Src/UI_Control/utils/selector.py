import pandas as pd
import numpy as np

class PersonSelector:
    def __init__(self):
        self.selected_id = None

    def select(self, x: float = 0.0, y: float = 0.0, search_person_df: pd.DataFrame = pd.DataFrame()):
        """選擇最大 bbox 或指定座標範圍內的 bbox，並記錄選擇的 person_id."""
        if search_person_df.empty:
            return

        max_area = -1

        for _, row in search_person_df.iterrows():
            person_id = row['person_id']
            x1, y1, x2, y2 = map(int, row['bbox'])
            
            # 如果 x 和 y 都為 0，無需座標篩選，直接找最大 bbox
            if not (x == 0 and y == 0):
                if not (x1 <= x <= x2 and y1 <= y <= y2):
                    continue

            # 計算 bbox 面積
            area = (x2 - x1) * (y2 - y1)
            # 更新最大面積的 bbox
            if area > max_area:
                max_area = area
                self.selected_id = person_id

    
    def get_select_id(self):
        print(self.selected_id)
        return self.selected_id
    
    def reset(self):
        self.selected_id = None
        
class KptSelector:
    def __init__(self):
        self.selected_id = None

    def select(self,x: float, y: float , search_person_df:pd.DataFrame = pd.DataFrame() ):
        def calculate_distance(point1, point2):
            return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        if search_person_df.empty :
            return
        selected_id = None
        min_distance = float('inf')
        for _, row in search_person_df.iterrows():
            person_kpts = row['keypoints']
            for kpt_id, kpt in enumerate(person_kpts):
                
                kptx, kpty, _, _= map(int, kpt)
        
                distance = calculate_distance([kptx, kpty], [x, y])
                if distance < min_distance:
                    min_distance = distance
                    selected_id = kpt_id

        self.selected_id = selected_id
    
    def get_select_id(self):
        print(self.selected_id)
        return self.selected_id

    def reset(self):
        self.selected_id = None