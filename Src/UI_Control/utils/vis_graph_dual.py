import pyqtgraph as pg
from PyQt5.QtGui import QFont, QColor
from .analyze import PoseAnalyzer, BatAnalyzer
import numpy as np
import pandas as pd

class GraphPlotter():
    def __init__(self, pose_analyzer: PoseAnalyzer, angle_name: str = "右手肘"):
        """Initialize GraphPlotter with frame range and keypoint name."""
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.angle_name = angle_name
        self.graph = pg.PlotWidget()
        self.pose_analyzer = pose_analyzer

    def _init_graph(self, frame_range: int) -> pg.PlotWidget:
        """Set up the plot widget with labels, axis, and initial configuration."""
        title = f'<span style="color: blue; font-size: 30px">{self.angle_name}角度 (度)</span>'
        self.graph.setTitle(f'{title}')

        # Setting up fonts and styles
        font = QFont()
        font.setPixelSize(30)
        self.graph.addLegend(offset=(150, 5), labelTextSize="30pt")
        self.graph.setLabel('left', '<span style="font-size: 30px">角度 (度)</span>', color="blue")
        self.graph.setLabel('bottom', '<span style="font-size: 30px">幀 (fps: 285)</span>')
        self.graph.getAxis("bottom").setStyle(tickFont=font)
        self.graph.getAxis("left").setStyle(tickFont=font)
        self.graph.setXRange(0, frame_range - 1)
        self.graph.setYRange(0, 180)
        
        # Setting Y-axis ticks
        y_ticks = [(i, str(i)) for i in np.arange(0, 210, 30)]
        self.graph.getPlotItem().getAxis('left').setTicks([y_ticks])
        self.graph.getPlotItem().getAxis('left').setPen(color=QColor("blue"))
        self.graph.getPlotItem().getAxis('left').setTextPen(color=QColor("blue"))

    def updateGraph(self, frame_num: int):
        """Update the graph with the latest angle data for the specified frame."""
      
        self.graph.clear()

        # Fetch current angle data for the specified frame
        _, angle_info = self.pose_analyzer.get_frame_angle_data(frame_num, self.angle_name)
        try:
            angle_value = int(angle_info[0])
            title = f'<span style="color: blue; font-size: 30px">{self.angle_name}角度({int(angle_value):03}度)</span>'
        except TypeError:
            # print(f"角度數據格式錯誤: {angle_value}")
            return
        self.graph.setTitle(f'{title}')
        # Plot angle data for all frames
        kpt_times, kpt_angles = self.pose_analyzer.get_frame_angle_data(angle_name=self.angle_name)

        self.graph.plot(kpt_times, kpt_angles, pen='b')


        # Create a scatter plot for the point
        scatter = pg.ScatterPlotItem([frame_num], [angle_value], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
        
        # Add the scatter plot to the graph
        self.graph.addItem(scatter)
        
    def resize_graph(self,width,height):
        self.graph.resize(width,height)

    def setAngleName(self, angle_name):
        self.angle_name = angle_name
    
    def reset(self):
        self.graph = pg.PlotWidget()


class AngleGraphPlotter():
    def __init__(self, bat_analyzer: BatAnalyzer):
        """Initialize GraphPlotter with frame range and keypoint name."""
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.graph = pg.PlotWidget()
        self.bat_analyzer = bat_analyzer

    def _init_graph(self, frame_range: int) -> pg.PlotWidget:
        """Set up the plot widget with labels, axis, and initial configuration."""
        title = f'<span style="color: blue; font-size: 30px">角度 (度)</span>'
        self.graph.setTitle(f'{title}')

        # Setting up fonts and styles
        font = QFont()
        font.setPixelSize(30)
        self.graph.addLegend(offset=(150, 5), labelTextSize="30pt")
        self.graph.setLabel('left', '<span style="font-size: 30px">角度 (度)</span>', color="blue")
        self.graph.setLabel('bottom', '<span style="font-size: 30px">幀 (fps: 285)</span>')
        self.graph.getAxis("bottom").setStyle(tickFont=font)
        self.graph.getAxis("left").setStyle(tickFont=font)
        self.graph.setXRange(0, frame_range - 1)
        self.graph.setYRange(0, 360)
        
        # Setting Y-axis ticks
        y_ticks = [(i, str(i)) for i in np.arange(0, 360, 60)]
        self.graph.getPlotItem().getAxis('left').setTicks([y_ticks])
        self.graph.getPlotItem().getAxis('left').setPen(color=QColor("blue"))
        self.graph.getPlotItem().getAxis('left').setTextPen(color=QColor("blue"))

    def updateGraph(self, frame_num: int):
        """Update the graph with the latest angle data for the specified frame."""
      
        self.graph.clear()

        # Fetch current angle data for the specified frame
        if self.bat_analyzer.batHeadAngle_df.empty is True:
            return
        
        angles = self.bat_analyzer.batHeadAngle_df['angle'].values
        frames = self.bat_analyzer.batHeadAngle_df['curFrameNum'].values
        angle_value = self.bat_analyzer.getAngleData(frame_num-1)
        self.graph.plot(frames, angles, pen='b')
        if angle_value == None or len(angle_value)==0:
            title = f'<span style="color: blue; font-size: 30px">角度( 度)</span>'
        else: title = f'<span style="color: blue; font-size: 30px">角度({int(angle_value):03}度)</span>'

        
        # 创建一个新的散点图对象用于当前帧的数据点
        if angle_value !=None or len(angle_value) !=0 :
            scatter = pg.ScatterPlotItem([frame_num], [angle_value], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
            self.graph.addItem(scatter)
            self.graph.setTitle(f'{title}')
    
    def reset(self):
        self.graph = pg.PlotWidget()






class SpeedGraphPlotter():
    def __init__(self, bat_analyzer: BatAnalyzer, angle_name: str = "揮棒速度"):
        """Initialize GraphPlotter with frame range and keypoint name."""
        pg.setConfigOptions(foreground=QColor(113,148,116), antialias = True)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.graph = pg.PlotWidget()
        self.bat_analyzer = bat_analyzer
        self.pre_batSpeed = None

    def _init_graph(self, frame_range: int) -> pg.PlotWidget:
        """Set up the plot widget with labels, axis, and initial configuration."""
        title = f'<span style="color: blue; font-size: 30px">揮棒速度 (m/s)</span>'
        self.graph.setTitle(f'{title}')

        # Setting up fonts and styles
        font = QFont()
        font.setPixelSize(30)
        self.graph.addLegend(offset=(150, 5), labelTextSize="30pt")
        self.graph.setLabel('left', '<span style="font-size: 30px">揮棒速度 (m/s)</span>', color="blue")
        self.graph.setLabel('bottom', '<span style="font-size: 30px">幀 (fps: 285)</span>')
        self.graph.getAxis("bottom").setStyle(tickFont=font)
        self.graph.getAxis("left").setStyle(tickFont=font)
        self.graph.setXRange(0, frame_range - 1)
        # self.graph.setYRange(0, 180)
        
        # # Setting Y-axis ticks
        # y_ticks = [(i, str(i)) for i in np.arange(0, 210, 30)]

        self.graph.setYRange(0, 32)

        # Setting Y-axis ticks
        y_ticks = [(i, str(i)) for i in np.arange(0, 35, 5)]  # 更新刻度
        self.graph.getPlotItem().getAxis('left').setTicks([y_ticks])
        self.graph.getPlotItem().getAxis('left').setPen(color=QColor("blue"))
        self.graph.getPlotItem().getAxis('left').setTextPen(color=QColor("blue"))

    def updateGraph(self, frame_num: int):
        """Update the graph with the latest angle data for the specified frame."""

        self.graph.clear()
    
        # 如果没有数据，则直接返回
        if self.bat_analyzer.batHeadMovingSpeed_df.empty:
            return
        
        # 获取所有帧和速度数据
        speeds = self.bat_analyzer.batHeadMovingSpeed_df['speed'].values
        frames = self.bat_analyzer.batHeadMovingSpeed_df['endFrame'].values
        
        # 绘制所有帧的速度曲线
        self.graph.plot(frames, speeds, pen='b')
        
        # 检查是否存在当前帧号的数据
        speed_info = self.bat_analyzer.batHeadMovingSpeed_df[self.bat_analyzer.batHeadMovingSpeed_df['endFrame'] == (frame_num - 1)]
        if speed_info.empty:
            speed_info = self.pre_batSpeed
            speed_value = speed_info['speed'] 
            frame_value = speed_info['endFrame']

        else: 
        # 获取当前帧的速度值
            self.pre_batSpeed = speed_info
            speed_value = speed_info['speed'].values[0]
            frame_value = speed_info['endFrame'].values[0]
        
        # 设置图表标题
        title = f'<span style="color: blue; font-size: 30px">揮棒速度({round(float(speed_value), 2):.2f}m/s)</span>'
        self.graph.setTitle(f'{title}')
        
        # 创建一个新的散点图对象用于当前帧的数据点
        scatter = pg.ScatterPlotItem([frame_value], [speed_value], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
        self.graph.addItem(scatter)
        """
        # 将散点图添加到图表
        self.graph.addItem(scatter)
        self.graph.clear()
        # Fetch current angle data for the specified frame
        #_, angle_info = self.pose_analyzer.get_frame_angle_data(frame_num, self.angle_name)
        if self.bat_analyzer.batHeadMovingSpeed_df.empty: 
            return  # 如果沒有數據，則直接返回
        # speed, frame = self.bat_analyzer.getBatSpeedData(self, frame_num)
        frame_num = int(frame_num)
        speed_info = self.bat_analyzer.batHeadMovingSpeed_df[self.bat_analyzer.batHeadMovingSpeed_df['frame_num'] == frame_num]
        if speed_info.empty:
            return
        speed_value = speed_info['speed'].values[0]

        title = f'<span style="color: blue; font-size: 30px">揮棒速度({int(speed_value):03}公尺/秒)</span>'
       

        # try:
        #     speed_value = int(speed)
            # angle_value = int(angle_info[0])
        #     title = f'<span style="color: blue; font-size: 30px">揮棒速度({int(speed_value):03}公尺/秒)</span>'
        # except TypeError:
        #     # print(f"角度數據格式錯誤: {angle_value}")
        #     return
        self.graph.setTitle(f'{title}')
        # Plot angle data for all frames
        # kpt_times, kpt_angles = self.pose_analyzer.get_frame_angle_data(angle_name=self.angle_name)
        speed = self.bat_analyzer.batHeadMovingSpeed_df['speed'].values
        frames = self.bat_analyzer.batHeadMovingSpeed_df['frame_num'].values
        self.graph.plot(frames, speed, pen='b')


        # Create a scatter plot for the point
        scatter = pg.ScatterPlotItem([frame_num], [speed_value], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 255))
        
        # Add the scatter plot to the graph
        self.graph.addItem(scatter)
        # print('----------------------------------------------------------------')"""
    
    def reset(self):
        self.graph = pg.PlotWidget()

