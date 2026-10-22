from PyQt5.QtWidgets import QTableWidgetItem, QTableWidget
from PyQt5.QtCore import Qt
import numpy as np
from skeleton.detect_skeleton_dual import PoseEstimater


class KeypointTable:
    def __init__(self, table_widget: QTableWidget, pose_estimater:PoseEstimater):
        self.kpt_table = table_widget
        self.pose_estimater = pose_estimater
        self.kpt_dict = self.pose_estimater.joints["haple"]["keypoints"]
        self.correct_kpt_idx = None
        self.label_kpt = False
        self.clearTableView()

    def importDataToTable(self, frame_num:int):
        self.clearTableView()
        person_id = self.pose_estimater.person_id
        if person_id is None:
            return
        person_data = self.pose_estimater.getPersonDf(frame_num=frame_num, is_select=True)
        if person_data.empty:
            self.clearTableView()
            return
        kpt_dict = self.pose_estimater.joints["haple"]["keypoints"]
        num_keypoints = len(kpt_dict)
        if self.kpt_table.rowCount() < num_keypoints:
            self.kpt_table.setRowCount(num_keypoints)

        for kpt_idx, kpt in enumerate(person_data['keypoints'].iloc[0]): 
            kptx, kpty, kpt_label = kpt[0], kpt[1], kpt[3]
            kpt_name = kpt_dict[kpt_idx]
            self.setTableItem(kpt_idx, kpt_name, kptx, kpty, kpt_label)

    def setTableItem(self, row: int, kpt_name: str, kptx: float, kpty: float, kpt_label: bool):
        """Set items in the table for the given row."""
        kpt_name_item = QTableWidgetItem(str(kpt_name))
        kptx_item = QTableWidgetItem(str(np.round(kptx, 1)))
        kpty_item = QTableWidgetItem(str(np.round(kpty, 1)))
        kpt_label_item = QTableWidgetItem("Y" if kpt_label else "N")

        for item in [kpt_name_item, kptx_item, kpty_item, kpt_label_item]:
            item.setTextAlignment(Qt.AlignRight)

        if kpt_name != "":
            self.kpt_table.setItem(row, 0, kpt_name_item)
        self.kpt_table.setItem(row, 1, kptx_item)
        self.kpt_table.setItem(row, 2, kpty_item)
        self.kpt_table.setItem(row, 3, kpt_label_item)

    def clearTableView(self):
        self.kpt_table.clear()
        self.kpt_table.setColumnCount(4)
        title = ["關節點", "X", "Y", "有無更改"]
        self.kpt_table.setHorizontalHeaderLabels(title)
        header = self.kpt_table.horizontalHeader()
        for i in range(4):
            header.setDefaultAlignment(Qt.AlignLeft)

    def onCellClicked(self, row:int, column:int):
        print("row:"+str(row))
        self.correct_kpt_idx = row
        self.label_kpt = True
        
    def sendToTable(self, kptx: float, kpty: float, kpt_label: int, frame_num: int):
        """Send corrected keypoint data back to the table and update the pose estimator."""
        self.setTableItem(self.correct_kpt_idx, "", kptx, kpty, kpt_label)
        self.pose_estimater.update_person_df(kptx, kpty, frame_num, self.correct_kpt_idx)
        self.label_kpt = False