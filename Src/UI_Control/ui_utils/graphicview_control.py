from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import numpy as np
from PyQt5.QtGui import QImage, QPixmap, QColor
from cv_utils.cv_control import VideoLoader
from skeleton.detect_skeleton import PoseEstimater


class FrameView:
    def __init__(self, frame_view: QGraphicsView, view_scene:QGraphicsScene, video_loader:VideoLoader):
        self.frame_view = frame_view
        self.view_scene = view_scene
        self.video_loader = video_loader
        self.is_video = False


    def updateFrame(self, frame: np.ndarray = None, frame_num:int = None):
        """Update the displayed frame with additional analysis."""
        # 更新當前的frame和frame_num
        if self.is_video and frame_num is not None:
            frame = self.video_loader.getVideoImage(frame_num)
        countdown_time = self.updateTimers() 
        drawed_img = self.image_drawer.drawInfo(frame, frame_num, self.pose_estimater.kpt_buffer, countdown_time)
        self.showImage(drawed_img, self.view_scene, self.frame_view)

    def showImage(self, image: np.ndarray, scene: QGraphicsScene, GraphicsView: QGraphicsView):
        """Display an image in the QGraphicsView."""
        scene.clear()
        h, w = image.shape[:2]
        qImg = QImage(image, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        scene.addPixmap(pixmap)
        GraphicsView.setScene(scene)
        GraphicsView.setAlignment(Qt.AlignLeft)
        GraphicsView.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)

    def setisVideo(self, is_video:bool):
        self.is_video = is_video
