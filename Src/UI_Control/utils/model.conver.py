from argparse import ArgumentParser
from mmcv.transforms import Compose
from mmengine.logging import print_log
import sys
import os
import argparse
from ultralytics import YOLO
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../..", "tracker"))
from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer
from mmpose.apis import init_model as init_pose_estimator
from mmpose.utils import adapt_mmdet_pipeline
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False




class Model(object):
    def __init__(self):
        #self.detect_args = self.setDetectParser()
        self.detect_args_yolo = self.setDetectParserYolo()
        self.pose_args = self.setPoseParser()
        self.tracker_args = self.setTrackerParser()
        self.detector = YOLO(
            self.detect_args_yolo.model
        )
        # self.detector = init_detector(
        # self.detect_args.det_config, self.detect_args.det_checkpoint, device=self.detect_args.device)
        # self.detector.cfg.test_dataloader.dataset.pipeline[
        #     0].type = 'mmdet.LoadImageFromNDArray'
        # self.detector_test_pipeline = Compose(self.detector.cfg.test_dataloader.dataset.pipeline)

        self.pose_estimator = init_pose_estimator(
            self.pose_args.pose_config,
            self.pose_args.pose_checkpoint
        )
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
        self.image_size = (0,0,0)

    def setDetectParserYolo(self) -> ArgumentParser:
        parser = ArgumentParser() 
        parser = argparse.ArgumentParser(description="YOLO Model Argument Parsing")    
        parser.add_argument('--model', type=str, default=os.path.join(current_dir, '../../Db/pretrain/yolo11m-seg.pt'), help='Path to YOLO model file (.pt)')
        parser.add_argument('--conf-thres', type=float, default=0.3, help='Confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
        return parser.parse_args()

    # def setDetectParser(self) -> ArgumentParser:
    #     parser = ArgumentParser()
    #     parser.add_argument('--det-config', default='../mmyolo_main/configs/yolov8/yolov8_x_mask-refine_syncbn_fast_8xb16-500e_coco.py', help='Config file for detection')
    #     parser.add_argument('--det-checkpoint', default='../../Db/pretrain/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth', help='Checkpoint file for detection')
    #     parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    #     parser.add_argument(
    #     '--det-cat-id',
    #     type=int,
    #     default=0,
    #     help='Category id for bounding box detection model')
    #     parser.add_argument(
    #         '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    #     parser.add_argument(
    #         '--nms-thr',
    #         type=float,
    #         default=0.3,
    #         help='IoU threshold for bounding box NMS')
    #     parser.add_argument(
    #         '--kpt-thr',
    #         type=float,
    #         default=0.3,
    #         help='Visualizing keypoint thresholds')
    #     parser.add_argument(
    #         '--draw-heatmap',
    #         action='store_true',
    #         default=False,
    #         help='Draw heatmap predicted by the model')
    #     parser.add_argument(
    #         '--show-kpt-idx',
    #         action='store_true',
    #         default=False,
    #         help='Whether to show the index of keypoints')
    #     parser.add_argument(
    #         '--skeleton-style',
    #         default='mmpose',
    #         type=str,
    #         choices=['mmpose', 'openpose'],
    #         help='Skeleton style selection')
    #     parser.add_argument(
    #         '--radius',
    #         type=int,
    #         default=3,
    #         help='Keypoint radius for visualization')
    #     parser.add_argument(
    #         '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    #     parser.add_argument(
    #         '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    #     args = parser.parse_args()
    #     return args

    def setPoseParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument('--pose-config', default=os.path.join(current_dir, '../../mmpose_main/configs/body_2d_keypoint/topdown_heatmap/haple/ViTPose_base_simple_halpe_256x192.py'), help='Config file for pose')
        parser.add_argument('--pose-checkpoint', default=os.path.join(current_dir, '../../../Db/pretrain/epoch_210.pth'), help='Checkpoint file for pose')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--kpt-thr',
            type=float,
            default=0.3,
            help='Visualizing keypoint thresholds')
        parser.add_argument(
            '--show-kpt-idx',
            action='store_true',
            default=False,
            help='Whether to show the index of keypoints')
        parser.add_argument(
            '--skeleton-style',
            default='mmpose',
            type=str,
            choices=['mmpose', 'openpose'],
            help='Skeleton style selection')
        parser.add_argument(
            '--radius',
            type=int,
            default=3,
            help='Keypoint radius for visualization')
        args = parser.parse_args()
        return args

    def setTrackerParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        # tracking args
        parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
        parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
        parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
        parser.add_argument("--track_buffer", type=int, default=360, help="the frames for keep lost tracks")
        parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
        parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                            help="threshold for filtering out boxes of which aspect ratio are above the given value.")
        parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
        parser.add_argument("--fuse-score", dest="mot20", default=True, action='store_true',
                            help="fuse score and iou for association")

        # CMC
        parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

        # ReID
        parser.add_argument("--with-reid", dest="with_reid", default=False , help="with ReID module.")
        parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=os.path.join(current_dir, '../../tracker/fast_reid/configs/MOT17/sbs_S50.yml'),
                            type=str, help="reid config file path")
        parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=os.path.join(current_dir, '../../tracker/fast_reid/mot17_sbs_S50.pth'),
                            type=str, help="reid config file path")
        parser.add_argument('--proximity_thresh', type=float, default=0.5,
                            help='threshold for rejecting low overlap reid matches')
        parser.add_argument('--appearance_thresh', type=float, default=0.25,
                            help='threshold for rejecting low appearance similarity reid matches')

        tracker_args = parser.parse_args()

        tracker_args.jde = False
        tracker_args.ablation = False
        return tracker_args

    def init_tracker(self):
        self.tracker = None
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)

    def reset_tracker(self):
        self.tracker = None
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)

    def setImageSize(self, image_size:tuple):
        self.reset_tracker()
        self.image_size = image_size

import math
import torch
import torch.nn.functional as F

def patch_sdpa_for_onnx():
    # Monkey-patch F.scaled_dot_product_attention to pure math operations
    # so that ONNX tracer sees standard ops instead of aten::scaled_dot_product_attention
    original_sdpa = F.scaled_dot_product_attention

    def math_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        
        if is_causal:
            causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
            attn_weight = attn_weight.masked_fill(~causal_mask, float("-inf"))
            
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
            else:
                attn_weight = attn_weight + attn_mask
                
        attn_weight = torch.softmax(attn_weight, dim=-1)
        if dropout_p > 0.0:
            attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
            
        return attn_weight @ value

    F.scaled_dot_product_attention = math_sdpa

def convert_pose_to_onnx(pose_model, output_path="pose_model.onnx"):
    import torch
    pose_model.cuda()
    pose_model.eval()
    
    # 建立符合 ViTPose 輸入格式的 dummy input (Batch, Channel, Height, Width)
    dummy_input = torch.randn(1, 3, 256, 192).cuda()

    torch.onnx.export(
        pose_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14, # 使用 >=14 支援 SDPA 或一般 Math
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
if __name__ == "__main__": 
    try:
        print("--- 開始執行轉換流程 ---")
        patch_sdpa_for_onnx()  # 在初始化模型之前套用 monkeypatch
        
        model = Model() 
        pose_model = model.pose_estimator
        
        print(f"成功載入 Pose 模型，準備轉換...")
        
        save_path = os.path.join(current_dir, "../../../Db/pretrain/ViTPose_halpe.onnx")
        convert_pose_to_onnx(pose_model, save_path)
        
        print(f"--- 轉換完成！檔案已存至: {save_path} ---")

    except Exception as e:
        print(f"\n[錯誤報告] 程式執行失敗！")
        print(f"錯誤內容: {e}")
        import traceback
        traceback.print_exc() # 這行會印出詳細的錯誤位置