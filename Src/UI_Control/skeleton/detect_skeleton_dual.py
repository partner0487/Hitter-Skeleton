import numpy as np
import pandas as pd
from utils.one_euro_filter import OneEuroFilter
import os
import sys
import numpy as np
from utils.model import Model
from mmengine.logging import print_log
from scipy.signal import savgol_filter
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from utils.timer import FPSTimer
from utils.bat_roi import detect_bat_near_hands, detect_bats_near_hands, grip_distance, hands_and_height
from ultralytics import YOLO
from mmcv.transforms import Compose
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
import torch
import torch.autograd.profiler as profiler
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "tracker"))

try:
    from mmdet.apis import inference_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

# 整張畫面 YOLO：人（0）、球（32）、球棒（34）
FULL_FRAME_KWARGS = dict(task='segment', classes=[0, 32, 34], conf=0.15, verbose=False)


def topdown_batch(pose_model, jobs):
    """多張畫面的骨架一起推論（雙相機同一幀），jobs 為 [(img, bboxes (n, 4) xyxy), ...]，
    回傳各張畫面的結果列表，與逐張呼叫 mmpose inference_topdown 相同（沒有框時以整張畫面為框）"""
    scope = pose_model.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)
    pipeline = getattr(pose_model, "_topdown_pipeline", None)
    if pipeline is None:
        pipeline = pose_model._topdown_pipeline = Compose(pose_model.cfg.test_dataloader.dataset.pipeline)
    data_list, counts = [], []
    for img, bboxes in jobs:
        if bboxes is None or len(bboxes) == 0:
            h, w = img.shape[:2]
            bboxes = np.array([[0, 0, w, h]], dtype=np.float32)
        for bbox in bboxes:
            data_info = dict(img=img, bbox=bbox[None], bbox_score=np.ones(1, dtype=np.float32))
            data_info.update(pose_model.dataset_meta)
            data_list.append(pipeline(data_info))
        counts.append(len(bboxes))
    with torch.no_grad():
        results = pose_model.test_step(pseudo_collate(data_list))
    out, start = [], 0
    for n in counts:
        out.append(results[start:start + n])
        start += n
    return out

class PoseEstimater:
    def __init__(self, model: Model =None):
        
        self.model = model
        self.img_shape = (0,0,0)
        self.person_df = pd.DataFrame()
        self.pre_person_df = pd.DataFrame()
        self.prev_bbox_center = None
        # 對極幾何導引（正面相機用）：(F, 側面打者 bbox)，見 filterBiggestTargets；None 時不使用
        self.epipolar_guide = None

        self.bat_df = pd.DataFrame()
        self.has_pre_bat_df = False
        self.pre_bat_df = pd.DataFrame()
        self.last_hands = None
        self.bat_data = []
        self.ball_df = pd.DataFrame()
        self.ball_data = []
        self.person_id = None
        self.kpt_id = None
        self.pitch_hand_id = 10
        self.fps = None
        self.person_data = []
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []

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

    def mergePersonData(self, pred_instances, person_ids: list, frame_num: int = None):
        person_bboxes = pred_instances['bboxes']
        if frame_num is None:
            self.person_data = []

        for person, pid, bbox in zip(pred_instances, person_ids, person_bboxes):
            keypoints_data = np.hstack((
                np.round(person['keypoints'][0], 2),
                np.round(person['keypoint_scores'][0], 2).reshape(-1, 1),
                np.full((len(person['keypoints'][0]), 1), False, dtype=bool)
            ))

            new_kpts = np.full((len(self.joints['haple']['keypoints']), keypoints_data.shape[1]), 0.9)
            new_kpts[:26] = keypoints_data
            person_info = {
                'person_id': pid,
                'bbox': bbox,
                'keypoints': new_kpts
            }
            if frame_num is not None:
                person_info['frame_number'] = frame_num

            self.person_data.append(person_info)

        return pd.DataFrame(self.person_data)

    def _stillHeld(self) -> bool:
        """沿用上一幀的球棒前，確認它仍在這一幀打者的手上（utils/bat_roi.grip_distance）；
        沒確認時不沿用：TSG 曾一路沿用 50 幀，手早就離開，甚至沿用到後方其他球員的球棒"""
        hands = getattr(self, "last_hands", None)
        if hands is None or self.pre_bat_df.empty:
            return False
        try:
            poly = np.asarray(self.pre_bat_df.iloc[0]["mask"][1][0], dtype=np.float32).reshape(-1, 2)
        except (IndexError, KeyError, TypeError, ValueError):
            return False
        return grip_distance(poly, *hands) is not None

    def mergeBatData(self, bbox, mask, frame_num: int = None):
        if mask is None:
            # 當下沒偵測到，但 pre_bat_df 裡有舊資料、而且仍在打者手上
            if self._stillHeld():
                self.has_pre_bat_df = True
                bat_info = self.pre_bat_df.iloc[0].to_dict()
                if frame_num is not None:
                    bat_info['frame_number'] = frame_num
                if frame_num is not None and frame_num >= 0:
                    self.bat_data.append(bat_info)
                return pd.DataFrame(self.bat_data) if len(self.bat_data) > 0 else pd.DataFrame()
            else:
                self.has_pre_bat_df = False
                return pd.DataFrame(self.bat_data) if len(self.bat_data) > 0 else pd.DataFrame()
        else: # 有detect到東西
            self.has_pre_bat_df = True
            bat_info = {
                'bbox': bbox,
                'mask': mask,
                'frame_number': frame_num if frame_num is not None else -1
            }
            self.pre_bat_df = pd.DataFrame([bat_info])
            if frame_num is not None and frame_num >= 0:
                self.bat_data.append(bat_info)
                return pd.DataFrame(self.bat_data)
            return pd.DataFrame([bat_info])
        
    def mergeBallData(self, bbox, frame_num: int = None):
        """
        - frame_num is None: 重置 ball_data
        - frame_num 有值：
            * 若 bbox 有偵測到 → 直接記錄
            * 若 bbox 無偵測到（None 或空）：
                - 若 ball_data 不為空 → 沿用上一筆的 bbox
                - 若 ball_data 為空 → 不紀錄這一幀
        """
        # 初始化：清空所有紀錄
        if frame_num is None:
            self.ball_data = []
            return pd.DataFrame(self.ball_data)

        # 判斷這一幀「有沒有點」
        has_ball = bbox is not None and len(bbox) > 0

        if has_ball:
            # 正常新增一筆
            ball_info = {
                "bbox": bbox,
                "frame_number": frame_num,
            }
            self.ball_data.append(ball_info)

        return pd.DataFrame(self.ball_data)

    def getBatData(self, bbox, mask, frame_num: int = None):
        # 如果當下有偵測到
        if mask is not None:
            self.has_pre_bat_df = True
            bat_info = {
                'bbox': bbox,
                'mask': mask,
                'frame_number': frame_num if frame_num is not None else -1
            }
            # 更新「最後一次成功」的參考
            self.pre_bat_df = pd.DataFrame([bat_info])
        else:
            # 當下沒偵測到，但 pre_bat_df 裡有舊資料、而且仍在打者手上
            if self._stillHeld():
                self.has_pre_bat_df = True
                # 取得舊資料但將 frame_num 更新為當前 (模擬持續存在)
                bat_info = self.pre_bat_df.iloc[0].to_dict()
            else:
                self.has_pre_bat_df = False
                return pd.DataFrame()

        # 影片模式記錄長期列表
        if frame_num is not None and frame_num >= 0:
            self.bat_data.append(bat_info)
            return pd.DataFrame(self.bat_data)
        
        # 即時模式：回傳當前或緩衝的單幀
        return pd.DataFrame([bat_info])

    def getBallData(self, bbox, frame_num: int = None):
        """
        獲取球偵測資料。
        """
        has_ball = bbox is not None and len(bbox) > 0

        if not has_ball:
            return pd.DataFrame() # 無偵測時回傳空

        ball_info = {
            "bbox": bbox,
            "frame_number": frame_num if frame_num is not None else -1,
        }

        # 影片模式：持續累積路徑
        if frame_num is not None and frame_num >= 0:
            self.ball_data.append(ball_info)
            return pd.DataFrame(self.ball_data)

        # 即時模式：只回傳當前這一幀的座標
        return pd.DataFrame([ball_info])

    def detectBall(self, image: np.ndarray):
        """
        即時偵測球的位置。
        回傳偵測到的球的 bbox 列表。
        """
        if self.model is None or self.model.detector is None:
            return []
            
        yolo_model = self.model.detector
        # classes=[32] 為 ball
        result = yolo_model.predict(image, task='segment', classes=[32], conf=0.15, verbose=False)
        detections = result[0]
        
        ball_bboxes = []
        for det in detections:
            cls_id = int(det.boxes.cls[0])
            if cls_id == 32:  # ball
                # 取得 xyxy 格式並轉為 numpy array
                bbox = det.boxes.xyxy[0].cpu().numpy().flatten()
                ball_bboxes.append(bbox)
                
        return ball_bboxes


    def detectKpt(self, image:np.ndarray, frame_num:int = None, is_video:bool = False, is_processed:bool=False):
        if not self.is_detect:
            return image, pd.DataFrame(), 0, pd.DataFrame(), pd.DataFrame()
        fps = 0
        if is_video: 
            #影片處理方式
            if frame_num not in self.processed_frames:
                self.storeVideoResult(self.processImage(self.model, image, select_id=self.person_id), frame_num)
            if self.kpt_id is not None:
                self.kpt_buffer = self.updateKptBuffer(frame_num)
        else:
            #real time 處理方式
            pred_instances, person_ids, bat_bbox, bat_mask, ball_bboxes = self.processImage(self.model, image, select_id=self.person_id)
            self.person_df = self.mergePersonData(pred_instances, person_ids)
            self.bat_df = self.getBatData(bat_bbox, bat_mask, frame_num)
            self.ball_df = self.getBallData(ball_bboxes, frame_num)
            self.smoothKpt(person_ids, frame_num)
            self.pre_person_df = self.person_df.copy()
            if self.kpt_id:
                person_data = self.getPersonDf(is_select=True, is_kpt=True)
                if person_data is not None:
                    keypoint = person_data[self.kpt_id][:2]  # 確保取出的是有效的 [x, y] 數據
                    self.kpt_buffer.append(keypoint)

        average_time = self.fps_timer.toc()
        fps = int(1/max(average_time, 0.00001))

        return image, self.person_df, fps, self.bat_df, self.ball_df

    def storeVideoResult(self, result, frame_num):
        pred_instances, person_ids, bat_bbox, bat_mask, ball_bboxes = result
        self.person_df = self.mergePersonData(pred_instances, person_ids, frame_num)
        self.bat_df = self.mergeBatData(bat_bbox, bat_mask, frame_num)
        self.ball_df = self.mergeBallData(ball_bboxes, frame_num)
        self.smoothKpt(person_ids, frame_num)
        self.processed_frames.add(frame_num)

    @staticmethod
    def detectKptDual(side, front, img_side, img_front, frame_num, epipolar_F=None):
        """影片處理：雙相機同一幀一起偵測（processDual），結果與先 side.detectKpt、再 front.detectKpt 相同；回傳正面 fps"""
        if not (side.is_detect and front.is_detect) or frame_num in side.processed_frames or frame_num in front.processed_frames:
            # 逐台處理：側面打者框從側面 person_df 取該幀資料
            side.detectKpt(img_side, frame_num, is_video=True)
            guide = None
            side_df = side.person_df
            if epipolar_F is not None and side_df is not None and not side_df.empty and "frame_number" in side_df:
                rows = side_df[side_df["frame_number"] == frame_num]
                if not rows.empty:
                    guide = (epipolar_F, np.asarray(rows.iloc[0]["bbox"], dtype=np.float64).ravel())
            front.epipolar_guide = guide
            return front.detectKpt(img_front, frame_num, is_video=True)[2]
        side_result, front_result = PoseEstimater.processDual(side.model, side, front, img_side, img_front,
                                                              side.person_id, front.person_id, epipolar_F)
        fps = 0
        for est, result in ((side, side_result), (front, front_result)):
            est.storeVideoResult(result, frame_num)
            if est.kpt_id is not None:
                est.kpt_buffer = est.updateKptBuffer(frame_num)
            fps = int(1 / max(est.fps_timer.toc(), 0.00001))
        return fps

    def filterBatBiggestTarget(self, online_targets, masks, pred_instances=None):
        # 看有沒有內容
        if not online_targets: 
            return [], None
        
        # 把當前frame偵測到的bat加進去list
        tlwhs = []
        for target in online_targets:
            tlwhs.append(target.tlwh)

        # 將數據轉為張量並放到 GPU 上 (轉為 xyxy 格式)
        tlwhs_tensor = torch.tensor(np.array(tlwhs), device='cuda', dtype=torch.float32)
        bboxes_xyxy = torch.cat([tlwhs_tensor[:, :2], tlwhs_tensor[:, :2] + tlwhs_tensor[:, 2:4]], dim=1)
        
        # 尋找雙手部位座標 (雙手手腕：HAPLE 與 COCO 的 9, 10)
        hands_pt = None
        if pred_instances is not None:
            try:
                # 兼容 MMEngine / MMPose 的資料結構 (dict 或 InstanceData)
                if hasattr(pred_instances, 'keypoints'):
                    kpts = pred_instances.keypoints[0]
                elif isinstance(pred_instances, list) or isinstance(pred_instances, tuple):
                    kpts = pred_instances[0]['keypoints'][0] if isinstance(pred_instances[0], dict) else pred_instances[0].keypoints[0]
                else:
                    kpts = pred_instances['keypoints'][0]

                if kpts is not None and len(kpts) > 10:
                    wl = kpts[9][:2]
                    wr = kpts[10][:2]
                    # 如果有有效的雙手座標，取兩手正中間點
                    if wl[0] > 0 and wr[0] > 0:
                        hands_pt = (wl + wr) / 2.0
                    elif wl[0] > 0:
                        hands_pt = wl
                    elif wr[0] > 0:
                        hands_pt = wr
                    elif len(kpts) > 18 and kpts[18][0] > 0:
                        hands_pt = kpts[18][:2] # 真的都沒抓到手再配脖子
            except Exception:
                pass

        if hands_pt is not None:
            hx, hy = float(hands_pt[0]), float(hands_pt[1])
            
            # 使用手部座標到候選球棒 bbox 的邊界的最短距離
            # 若手在 bat 的邊界框內部，距離就會被計算為 0，這完全符合「人員雙手握住球棒」的物理限制！
            x1 = bboxes_xyxy[:, 0]
            y1 = bboxes_xyxy[:, 1]
            x2 = bboxes_xyxy[:, 2]
            y2 = bboxes_xyxy[:, 3]
            
            zeros = torch.zeros_like(x1)
            dx = torch.max(torch.max(x1 - hx, zeros), hx - x2)
            dy = torch.max(torch.max(y1 - hy, zeros), hy - y2)
            dist_to_hands_bbox = torch.sqrt(dx**2 + dy**2)
            
            # 建立安全圍欄：任何「與打者雙手距離超過 200 像素」的球棒 (例如背景牆上掛著的球棒) 都是假目標！
            valid_mask = dist_to_hands_bbox <= 200.0
            if not valid_mask.any():
                # 若完全找不到手上有球棒 (例如揮太快糊掉導致 YOLO 沒抓到真正的球棒)
                # 直接回傳找不到，不要選到背景的干擾物！
                return [], None
            
            # 若在雙手範圍內有不只一支球棒，面積越大的、離手越近的分數越高
            areas = tlwhs_tensor[:, 2] * tlwhs_tensor[:, 3]
            scores = areas / (dist_to_hands_bbox + 1.0)  # +1 避免除以 0
            scores[~valid_mask] = 0.0
            
            best_idx = torch.argmax(scores)
        else:
            # 萬一連打者手部骨架都沒偵測到，退回尋找單純面積最大
            areas = tlwhs_tensor[:, 2] * tlwhs_tensor[:, 3]
            best_idx = torch.argmax(areas)

        best_bbox_xyxy = bboxes_xyxy[best_idx]
        best_mask = None
        best_mask_area = 0

        bbox_list = best_bbox_xyxy.cpu().tolist()

        # 找對應 best_bbox_xyxy 的 mask
        for mask in enumerate(masks):
            if (np.all(mask[1][0][:, 0] >= (bbox_list[0] - 50)) and
                np.all(mask[1][0][:, 0] <= (bbox_list[2] + 50)) and
                np.all(mask[1][0][:, 1] >= (bbox_list[1] - 50)) and
                np.all(mask[1][0][:, 1] <= (bbox_list[3] + 50)) ):

                current_mask_area = cv2.contourArea(mask[1][0]) 
                if current_mask_area > best_mask_area:
                    best_mask = mask
                    best_mask_area = current_mask_area
        
        return bbox_list, best_mask

    def filterValidTargets(self, online_targets, select_id: int = None):
        """
        過濾出有效的追蹤目標。

        Args:
            online_targets (List): 所有在線追蹤的目標。
            select_id (int, optional): 選擇指定的追蹤ID。

        Returns:
            Tuple: 有效的邊界框和追蹤ID。
        """
        if not online_targets:
            return [], []

        # 將所有在線目標的邊界框和ID提取為兩個列表
        tlwhs = []
        track_ids = []

        for target in online_targets:
            tlwhs.append(target.tlwh)
            track_ids.append(target.track_id)

        # 將列表轉換為 NumPy array
        tlwhs = np.array(tlwhs)
        track_ids = np.array(track_ids)

        # 將數據轉為張量並放到 GPU 上
        tlwhs = torch.tensor(tlwhs, device='cuda')  # shape: (n, 4)
        track_ids = torch.tensor(track_ids, device='cuda')  # shape: (n,)

        # 計算面積 w * h
        areas = tlwhs[:, 2] * tlwhs[:, 3]  # w * h

        # 過濾面積大於 10 的邊界框
        valid_mask = areas > 10

        # 如果指定了 select_id，則進一步過濾
        if select_id is not None:
            valid_mask &= (track_ids == select_id)

        # 過濾有效的邊界框和追蹤ID
        valid_tlwhs = tlwhs[valid_mask]
        valid_track_ids = track_ids[valid_mask]

        # 將 (x1, y1, w, h) 轉為 (x1, y1, x2, y2)
        valid_bbox = torch.cat([valid_tlwhs[:, :2], valid_tlwhs[:, :2] + valid_tlwhs[:, 2:4]], dim=1)

        # 返回結果
        return valid_bbox.cpu().tolist(), valid_track_ids.cpu().tolist()

    def filterBiggestTargets(self, online_targets, select_id: int = None, img_shape=None, bat_bboxes=None):
        if not online_targets:
            return [], []

        tlwhs = []
        scores = []
        for target in online_targets:
            tlwhs.append(target['tlwh'])
            scores.append(target.get('score', 1.0))

        tlwhs = torch.tensor(tlwhs, device='cuda', dtype=torch.float32)
        img_w, img_h = img_shape

        # 1. 計算每個 bbox 的中心點
        cx = tlwhs[:, 0] + tlwhs[:, 2] / 2.0
        cy = tlwhs[:, 1] + tlwhs[:, 3] / 2.0

        # 2. 計算「中心權重」 (預設偏向畫面中央)
        dist_x = (cx - img_w / 2.0) / (img_w / 2.0)
        dist_y = (cy - img_h / 2.0) / (img_h / 2.0)
        center_weight = torch.exp(-(dist_x**2 + dist_y**2))

        # 3. 歷史追蹤權重 (鎖定上一幀追蹤到的打者)
        track_weight = torch.zeros_like(center_weight)
        if self.prev_bbox_center is not None:
            prev_cx, prev_cy = self.prev_bbox_center
            t_dist_x = (cx - prev_cx) / (img_w / 2.0)
            t_dist_y = (cy - prev_cy) / (img_h / 2.0)
            track_weight = torch.exp(-10.0 * (t_dist_x**2 + t_dist_y**2))

        # 4. 球棒互動權重 (判斷是否手持球棒)
        bat_weight = torch.ones_like(center_weight)
        if bat_bboxes and len(bat_bboxes) > 0:
            human_x1 = tlwhs[:, 0]
            human_y1 = tlwhs[:, 1]
            human_x2 = tlwhs[:, 0] + tlwhs[:, 2]
            human_y2 = tlwhs[:, 1] + tlwhs[:, 3]
            
            best_bat_weight = torch.ones_like(center_weight)
            for bat_b in bat_bboxes:
                bx1, by1, bx2, by2 = float(bat_b[0]), float(bat_b[1]), float(bat_b[2]), float(bat_b[3])
                dx = torch.max(torch.max(human_x1 - bx2, torch.zeros_like(human_x1)), bx1 - human_x2)
                dy = torch.max(torch.max(human_y1 - by2, torch.zeros_like(human_y1)), by1 - human_y2)
                dist = torch.sqrt(dx**2 + dy**2)
                
                # 若與球棒距離小於 100 pixel，給予 10 倍權重，強制鎖定打者
                curr_bat_w = torch.where(dist < 100.0, torch.tensor(10.0, device='cuda'), torch.tensor(1.0, device='cuda'))
                best_bat_weight = torch.max(best_bat_weight, curr_bat_w)
                
            bat_weight = best_bat_weight

        # 5. 計算面積並正規化
        areas = tlwhs[:, 2] * tlwhs[:, 3]
        max_area = torch.max(areas)
        norm_areas = areas / max_area

        # 5b. 過濾掉面積太小的人 (背景遠處的人 bbox 小，面積比例 < 0.3 → 直接排除)
        #     這樣後面站著的人不會搶分
        foreground_mask = norm_areas >= 0.3
        # 有對極幾何導引時不用面積排除：正投時投手離正面相機近、人框大，打者站得窄時面積常不到投手的 30%，
        # 會在套用對極權重前就被排除；背景人物離對極線遠，權重本來就很低
        if self.epipolar_guide is not None:
            foreground_mask = torch.ones_like(foreground_mask)
        if foreground_mask.sum() == 0:
            foreground_mask = torch.ones_like(foreground_mask)  # fallback: 全部保留

        # 6. 綜合評分 = 面積^2 * 位置權重 * 球棒權重
        #    面積平方：讓前景最大的人更強勢得分
        if self.prev_bbox_center is not None:
            pos_score = 0.2 * track_weight  + 0.8 * center_weight
        else:
            pos_score = center_weight

        final_scores = (norm_areas ** 2) * (pos_score ** 5) * bat_weight

        # 8. 對極幾何（正面相機）：側面已鎖定的打者，頭頂、腳底中點在正面畫面各對應一條對極線，
        #    同一個人的人框頭頂、腳底應落在線上（WeiShu 實測打者 3～15 px，其他人 ≥ 53 px）。
        #    正投時投手站在正面相機前、畫面最大，只靠面積會選到投手（距離 240～300 px）
        if self.epipolar_guide is not None:
            final_scores = final_scores * self._epipolar_weight(tlwhs)
        # 排除背景小人 (設為 -1 讓 argmax 不會選到)
        final_scores[~foreground_mask] = -1.0

        # 7. 找出最高分的 Index
        best_idx = torch.argmax(final_scores).item()

        # 更新歷史中心點紀錄，以便下一幀繼續鎖定同一人
        self.prev_bbox_center = (cx[best_idx].item(), cy[best_idx].item())

        res_tlwh = tlwhs[best_idx].cpu()
        max_bbox = [
            res_tlwh[0].item(), 
            res_tlwh[1].item(), 
            (res_tlwh[0] + res_tlwh[2]).item(), 
            (res_tlwh[1] + res_tlwh[3]).item()
        ]

        return [max_bbox], [1]

    EPIPOLAR_SIGMA_PX = 30.0

    def _epipolar_weight(self, tlwhs):
        """各人框頭頂、腳底中點到側面打者對極線的平均距離 d → 權重 exp(-(d / σ)²)"""
        F, side_bbox = self.epipolar_guide
        x1, y1, x2, y2 = [float(v) for v in side_bbox[:4]]
        side_pts = np.array([[(x1 + x2) / 2, y1, 1.0], [(x1 + x2) / 2, y2, 1.0]])
        lines = torch.tensor((F @ side_pts.T).T, device=tlwhs.device, dtype=torch.float32)  # (2, 3)
        lines = lines / torch.linalg.norm(lines[:, :2], dim=1, keepdim=True)
        cx = tlwhs[:, 0] + tlwhs[:, 2] / 2.0
        ones = torch.ones_like(cx)
        top = torch.stack([cx, tlwhs[:, 1], ones], dim=1)
        bottom = torch.stack([cx, tlwhs[:, 1] + tlwhs[:, 3], ones], dim=1)
        d = (torch.abs(top @ lines[0]) + torch.abs(bottom @ lines[1])) / 2.0
        return torch.exp(-(d / self.EPIPOLAR_SIGMA_PX) ** 2)

    def smoothKpt(self, person_ids: list, frame_num=None):
        # 如果是即時處理，則初始化前一幀的數據，否則依賴 frame_slider 進行處理
        if frame_num is not None:
            curr_frame = frame_num
            if curr_frame == 0:
                return  # 初始幀，無需處理
            pre_frame_num = curr_frame - 1
        
        # 用於即時處理時的前一幀數據
        if frame_num is None and self.pre_person_df.empty:
            self.pre_person_df = self.person_df.copy()

        # 當前幀無數據時，跳過處理
        if self.person_df.empty:
            return
        
        for person_id in person_ids:
            # 如果使用 frame_slider，根據前後幀數據進行處理
            if frame_num is not None:
                pre_person_data = self.person_df.loc[(self.person_df['frame_number'] == pre_frame_num) &
                                                    (self.person_df['person_id'] == person_id)]
                curr_person_data = self.person_df.loc[(self.person_df['frame_number'] == curr_frame) &
                                                    (self.person_df['person_id'] == person_id)]
            # 即時處理時，使用 self.pre_person_df 作為前幀數據
            else:
                pre_person_data = self.pre_person_df.loc[self.pre_person_df['person_id'] == person_id]
                curr_person_data = self.person_df.loc[self.person_df['person_id'] == person_id]
            
            if curr_person_data.empty or pre_person_data.empty:
                continue  # 當前幀或前幀沒有該 person_id 的數據，跳過
            
            # 用 Python float（float64）逐點計算：原本每個關節都建 CUDA 張量再 .item() 取回，每幀上百次 GPU 同步，
            # 一支 90 幀雙相機片段多花約 2.3 秒；濾波器有狀態（dx_prev 依序傳遞），呼叫順序維持不變
            pre_kpts = np.asarray(pre_person_data.iloc[0]['keypoints'], dtype=np.float64)
            curr_kpts = np.asarray(curr_person_data.iloc[0]['keypoints'], dtype=np.float64)
            smoothed_kpts = []

            for pre_kpt, curr_kpt in zip(pre_kpts, curr_kpts):
                pre_kptx, pre_kpty = float(pre_kpt[0]), float(pre_kpt[1])
                curr_kptx, curr_kpty, curr_conf, curr_label = (float(v) for v in curr_kpt[:4])

                if pre_kptx != 0 and pre_kpty != 0 and curr_kptx != 0 and curr_kpty != 0:
                    curr_kptx = self.smooth_filter(curr_kptx, pre_kptx)
                    curr_kpty = self.smooth_filter(curr_kpty, pre_kpty)

                smoothed_kpts.append([curr_kptx, curr_kpty, curr_conf, curr_label])
            
            # 更新當前幀的數據
            self.person_df.at[curr_person_data.index[0], 'keypoints'] = smoothed_kpts
           
    def processImage(self, model, img, select_id=None):
        """
        處理單張圖像，進行物件偵測、跟蹤和姿態估計。

        Args:
            model (dict): 包含檢測器、姿態估計器和跟蹤器的模型。
            img (np.ndarray): 輸入的圖像。
            select_id (int, optional): 選擇指定的追蹤ID，如果為None，則處理所有目標。

        Returns:
            Tuple: 預測的姿態實例和在線的追蹤ID。
        """
        # 三段：整張畫面 YOLO → 選人 → 骨架 → 雙手附近找球棒；雙相機一起送見 processDual
        result = model.detector.predict(img, **FULL_FRAME_KWARGS)
        targets = self.selectTargets(img, result[0], select_id)
        pose_results = inference_topdown(model.pose_estimator, img, np.array(targets["bbox"]))
        pred_instances, hands = self.readPose(pose_results)
        found = detect_bat_near_hands(model.detector, img, *hands) if hands is not None else None
        return self.finishTargets(model, img, targets, pred_instances, hands, found)

    def selectTargets(self, img, detections, select_id=None):
        """整張畫面 YOLO 結果分成人／球棒／球，挑出打者框"""
        # human
        human_bboxes = []
        humane_scores = []
        # bat
        bat_bboxes = []
        bat_masks = []
        bat_scores = []
        # ball
        ball_bboxes = []
        ball_scores = []

        for det in detections:
            cls_id = int(det.boxes.cls[0])
            conf = float(det.boxes.conf[0])
            if (cls_id == 0) :  # human
                human_bboxes.append(np.array(det.boxes.xyxy[0].cpu().numpy().flatten()))
                humane_scores.append(det.boxes.conf)
            elif (cls_id == 32) :  # ball
                ball_bboxes.append(np.array(det.boxes.xyxy[0].cpu().numpy().flatten()))
                ball_scores.append(det.boxes.conf)
            elif(cls_id == 34) : # bat
                bat_bboxes.append(np.array(det.boxes.xyxy[0].cpu().numpy().flatten()))
                bat_masks.append(np.array(det.masks.xy))
                bat_scores.append(det.boxes.conf)
        # human tracker
        # online_targets = model.tracker.update(
        # )

        online_targets = []
        track_id = 1  # 手動編號 track_id

        for bbox in human_bboxes:
            x1, y1, x2, y2 = bbox  # bbox 是 [x1, y1, x2, y2]
            w = x2 - x1
            h = y2 - y1
            tlwh = [x1, y1, w, h]  # 转换为 tlwh 格式
            
            online_targets.append({
                'tlwh': tlwh,   # 原本 tracker 提供的 tlwh
                'track_id': track_id  # 人工分配 track_id
            })
            track_id += 1  # 递增 ID

        # 過濾出有效的邊界框和追蹤ID
        img_h, img_w = img.shape[:2]
        online_bbox, online_ids = self.filterBiggestTargets(online_targets, select_id, (img_w, img_h), bat_bboxes)
        return {"bbox": online_bbox, "ids": online_ids, "bat_bboxes": bat_bboxes, "bat_masks": bat_masks,
                "ball_bboxes": ball_bboxes}

    def readPose(self, pose_results):
        """骨架結果合併成 pred_instances，取出雙手腕與身高（沒有可信手腕為 None）"""
        data_samples = merge_data_samples(pose_results)
        pred_instances = data_samples.get('pred_instances', None) if data_samples else None
        hands = None
        if pred_instances is not None and len(pred_instances) > 0:
            try:
                hands = hands_and_height(pred_instances.keypoints[0], pred_instances.keypoint_scores[0],
                                         pred_instances.bboxes[0])
            except (AttributeError, IndexError, KeyError):
                hands = None
        self.last_hands = hands   # 沒偵測到球棒時，判斷能否沿用上一幀（_stillHeld）
        return pred_instances, hands

    def finishTargets(self, model, img, targets, pred_instances, hands, found):
        """球棒：有打者手腕時用雙手附近的偵測結果 found（utils/bat_roi.py），模糊的球棒也抓得到、不會選到背景；
        沒有手腕時退回整張畫面的偵測結果。mask 格式皆為 (序號, 輪廓陣列 (1, n, 2))"""
        online_bat_bbox = []
        online_bat_mask = None
        bat_bboxes = targets["bat_bboxes"]
        if hands is not None:
            if found is not None:
                online_bat_bbox, online_bat_mask = found[0], (0, found[1][None])
        elif len(bat_bboxes):
            online_targets_bat = model.tracker.update(
                    np.hstack((np.vstack(bat_bboxes), np.full((np.vstack(bat_bboxes).shape[0], 2), [0.9, 0]))), img.copy()
            )
            online_bat_bbox, online_bat_mask = self.filterBatBiggestTarget(online_targets_bat, targets["bat_masks"], pred_instances)

        return pred_instances, targets["ids"], online_bat_bbox, online_bat_mask, targets["ball_bboxes"]

    @staticmethod
    def processDual(model, side, front, img_side, img_front, side_select_id=None, front_select_id=None, epipolar_F=None):
        """雙相機同一幀一起偵測：整張畫面 YOLO、骨架、雙手附近的球棒 YOLO 各只送一次（兩張一批），
        結果與兩邊各自 processImage 相同（側面先選人，側面打者框經對極幾何導引正面選人）。
        回傳 (側面結果, 正面結果)，格式同 processImage"""
        sides = ((side, img_side), (front, img_front))
        results = model.detector.predict([img_side, img_front], **FULL_FRAME_KWARGS)
        side_targets = side.selectTargets(img_side, results[0], side_select_id)
        # 與原本從側面 person_df 取該幀打者框相同：有選到人才有該幀資料
        front.epipolar_guide = ((epipolar_F, np.asarray(side_targets["bbox"][0], dtype=np.float64).ravel())
                                if epipolar_F is not None and len(side_targets["ids"]) and len(side_targets["bbox"]) else None)
        front_targets = front.selectTargets(img_front, results[1], front_select_id)
        targets = (side_targets, front_targets)

        poses = topdown_batch(model.pose_estimator, [(img, np.array(t["bbox"])) for (_, img), t in zip(sides, targets)])
        read = [est.readPose(pose) for (est, _), pose in zip(sides, poses)]

        jobs = [i for i, (_, hands) in enumerate(read) if hands is not None]
        found = dict(zip(jobs, detect_bats_near_hands(model.detector, [(sides[i][1], *read[i][1]) for i in jobs])))
        return tuple(est.finishTargets(model, img, t, pred, hands, found.get(i))
                     for i, ((est, img), t, (pred, hands)) in enumerate(zip(sides, targets, read)))

    def setPersonId(self, person_id):
        self.person_id = person_id

    def setKptId(self, kpt_id):
        self.kpt_id = kpt_id

    def setDetect(self, status:bool):
        self.is_detect = status

    def updateKptBuffer(self, frame_num:int, window_length=17, polyorder=2):
        filtered_df = self.person_df[
            (self.person_df['person_id'] == self.person_id) & 
            (self.person_df['frame_number'] < frame_num)
        ]
        if filtered_df.empty:
            return None
        filtered_df = filtered_df.sort_values(by='frame_number')
        kpt_buffer = []
        for kpts in filtered_df['keypoints']:
            kpt = kpts[self.kpt_id]
            if kpt is not None and len(kpt) >= 2:
                kpt_buffer.append((kpt[0], kpt[1]))
        
        # 如果緩衝區長度大於等於窗口長度，則應用Savgol濾波器進行平滑
        if len(kpt_buffer) >= window_length:
            # 確保窗口長度為奇數且不超過緩衝區長度
            if window_length > len(kpt_buffer):
                window_length = len(kpt_buffer) if len(kpt_buffer) % 2 == 1 else len(kpt_buffer) - 1
            # 確保多項式階數小於窗口長度
            current_polyorder = min(polyorder, window_length - 1)
            
            # 分別提取x和y座標
            x = np.array([point[0] for point in kpt_buffer])
            y = np.array([point[1] for point in kpt_buffer])
            
            # 應用Savgol濾波器
            x_smooth = savgol_filter(x, window_length=window_length, polyorder=current_polyorder)
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=current_polyorder)
            
            # 將平滑後的座標重新打包
            smoothed_points = list(zip(x_smooth, y_smooth))
        else:
            # 緩衝區長度不足，直接使用原始座標
            smoothed_points = kpt_buffer

        return smoothed_points
    
    def getPersonDf(self, frame_num=None, is_select=False, is_kpt=False):
        if self.person_df.empty:
            return pd.DataFrame()
        condition = pd.Series([True] * len(self.person_df))  # 初始條件設為全為 True
        if frame_num is not None:
            condition &= (self.person_df['frame_number'] == frame_num)
        
        if is_select and self.person_id is not None:
            condition &= (self.person_df['person_id'] == self.person_id)
        data = self.person_df.loc[condition].copy()
        
        if data.empty:
            return None
        
        if is_kpt:
            data = data['keypoints'].iloc[0]
        return data
    
    def getPrePersonDf(self):
        if self.person_df.empty:
            return None, None, None
        condition = pd.Series([True] * len(self.person_df))  # 初始條件設為全為 True
        condition &= (self.pre_person_df['person_id']==self.person_id)
        data = self.person_df.loc[condition].copy()
        
        if data.empty:
            return None, None, None
        
        data1 = data['keypoints'].iloc[0][9]
        data2 = data['keypoints'].iloc[0][10]
        data3 = data['keypoints'].iloc[0][18]
        return data1, data2, data3
    
    def getBatDF(self, frame_num=None):
        if self.bat_df is None or self.bat_df.empty:
            return pd.DataFrame()
        
        if frame_num is not None:
            if 'frame_number' not in self.bat_df.columns:
                return pd.DataFrame()
            return self.bat_df[self.bat_df['frame_number'] == frame_num]
        return self.bat_df

    def reset(self):
        self.person_df = pd.DataFrame()
        self.pre_person_df = pd.DataFrame()
        self.prev_bbox_center = None
        self.epipolar_guide = None
        self.person_id = None
        self.kpt_id = None
        self.fps = None
        self.person_data = []
        self.processed_frames = set()
        self.fps_timer = FPSTimer()
        self.smooth_filter = OneEuroFilter()
        self.is_detect = False
        self.kpt_buffer = []

        self.bat_df = pd.DataFrame()
        self.has_pre_bat_df = False
        self.pre_bat_df = pd.DataFrame()
        self.last_hands = None
        self.bat_data = []
        self.ball_df = pd.DataFrame()
        self.ball_data = []
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
