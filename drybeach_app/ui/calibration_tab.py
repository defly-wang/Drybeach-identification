"""
干滩识别系统 - 校准模块UI
相机校准功能界面，支持参考点设置和距离校准

CalibrationTab: 相机校准功能标签页
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QDoubleSpinBox)
from PyQt6.QtCore import pyqtSignal


class CalibrationTab(QWidget):
    calibration_started = pyqtSignal()
    calibration_point_clicked = pyqtSignal(tuple)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibration_done = False
        self.calibration_points = []
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        lbl_info = QLabel("通过两点校准像素与实际距离的转换比例")
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #666;")
        layout.addWidget(lbl_info)
        
        cal_layout = QHBoxLayout()
        cal_layout.addWidget(QLabel("已知距离(m):"))
        self.spin_cal_distance = QDoubleSpinBox()
        self.spin_cal_distance.setRange(0.1, 10000)
        self.spin_cal_distance.setValue(10)
        cal_layout.addWidget(self.spin_cal_distance)
        layout.addLayout(cal_layout)
        
        self.btn_calibrate = QPushButton("开始校准(点击两点)")
        self.btn_calibrate.clicked.connect(self.start_calibration)
        layout.addWidget(self.btn_calibrate)
        
        self.lbl_calibration_status = QLabel("未校准")
        self.lbl_calibration_status.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.lbl_calibration_status)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def start_calibration(self):
        self.calibration_points = []
        self.calibration_done = False
        self.btn_calibrate.setText("点击第一个校准点...")
        self.lbl_calibration_status.setText("点击第1点")
        self.calibration_started.emit()
    
    def add_calibration_point(self, point):
        self.calibration_points.append(point)
        n = len(self.calibration_points)
        if n == 1:
            self.btn_calibrate.setText("点击第二个校准点...")
            self.lbl_calibration_status.setText(f"已选第1点: {point}")
        elif n == 2:
            p1, p2 = self.calibration_points
            import numpy as np
            dist_px = int(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
            self.lbl_calibration_status.setText(f"已选第2点: {point}, 像素距离: {dist_px}px")
            self.btn_calibrate.setText("校准完成!")
            self.calibration_done = True
            self.calibration_completed.emit({
                'distance': self.spin_cal_distance.value(),
                'points': self.calibration_points
            })
    
    def reset_calibration(self):
        self.calibration_points = []
        self.calibration_done = False
        self.btn_calibrate.setText("开始校准(点击两点)")
        self.lbl_calibration_status.setText("未校准")
    
    calibration_completed = object()
