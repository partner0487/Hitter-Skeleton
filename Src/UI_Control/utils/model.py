from argparse import ArgumentParser
from mmcv.transforms import Compose
from mmengine.logging import print_log
import sys
import os
import argparse
from ultralytics import YOLO

# ==================== 動態計算專案關鍵目錄 ====================
# model.py 位於: <專案根目錄>/Src/UI_Control/utils/model.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# 往上兩層到 <專案根目錄>/Src
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
# 往上三層到 <專案根目錄>
project_root = os.path.abspath(os.path.join(src_dir, ".."))
# ==========================================================

sys.path.append(os.path.join(src_dir, "tracker"))
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
        # self.detect_args = self.setDetectParser()
        self.detect_args_yolo = self.setDetectParserYolo()
        self.pose_args = self.setPoseParser()
        self.tracker_args = self.setTrackerParser()
        self.detector = YOLO(
            self.detect_args_yolo.model
        )

        self.pose_estimator = init_pose_estimator(
            self.pose_args.pose_config,
            self.pose_args.pose_checkpoint
        )
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)
        self.image_size = (0, 0, 0)

    def setDetectParserYolo(self) -> ArgumentParser:
        parser = argparse.ArgumentParser(description="YOLO Model Argument Parsing")
        default_model_path = os.path.join(project_root, "Db", "pretrain", "yolo11m-seg.pt")
        parser.add_argument('--model', type=str, default=default_model_path, help='Path to YOLO model file (.pt)')
        parser.add_argument('--conf-thres', type=float, default=0.3, help='Confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
        return parser.parse_args([])

    def setPoseParser(self) -> ArgumentParser:
        parser = ArgumentParser()
        default_config_path = os.path.join(
            src_dir, "mmpose_main", "configs", "body_2d_keypoint",
            "topdown_heatmap", "haple", "ViTPose_base_simple_halpe_256x192.py"
        )
        default_ckpt_path = os.path.join(project_root, "Db", "pretrain", "epoch_210.pth")

        parser.add_argument('--pose-config', default=default_config_path, help='Config file for pose')
        parser.add_argument('--pose-checkpoint', default=default_ckpt_path, help='Checkpoint file for pose')
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
        args = parser.parse_args([])
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
        default_reid_config = os.path.join(src_dir, "tracker", "fast_reid", "configs", "MOT17", "sbs_S50.yml")
        default_reid_weights = os.path.join(src_dir, "tracker", "fast_reid", "mot17_sbs_S50.pth")

        parser.add_argument("--with-reid", dest="with_reid", default=False, help="with ReID module.")
        parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=default_reid_config,
                            type=str, help="reid config file path")
        parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=default_reid_weights,
                            type=str, help="reid config file path")
        parser.add_argument('--proximity_thresh', type=float, default=0.5,
                            help='threshold for rejecting low overlap reid matches')
        parser.add_argument('--appearance_thresh', type=float, default=0.25,
                            help='threshold for rejecting low appearance similarity reid matches')

        tracker_args = parser.parse_args([])

        tracker_args.jde = False
        tracker_args.ablation = False
        return tracker_args

    def init_tracker(self):
        self.tracker = None
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)

    def reset_tracker(self):
        self.tracker = None
        self.tracker = BoTSORT(self.tracker_args, frame_rate=30.0)

    def setImageSize(self, image_size: tuple):
        self.reset_tracker()
        self.image_size = image_size