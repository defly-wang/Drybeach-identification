from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QComboBox, QCheckBox, QMessageBox,
                             QSpinBox)
from PyQt6.QtCore import pyqtSignal

from .viewers import WaterLineViewer


class WaterLineTab(QWidget):
    result_ready = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.current_image_path = None
        self.water_line_points = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.open_image)
        btn_layout.addWidget(self.btn_open)
        
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_clear.setEnabled(False)
        btn_layout.addWidget(self.btn_clear)
        
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        
        self.lbl_image_info = QLabel("提示: 打开图片后，点击添加水线点，双击完成")
        self.lbl_image_info.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_image_info)
        
        options_layout = QHBoxLayout()
        self.chk_fill = QCheckBox("填充水面区域")
        self.chk_fill.setChecked(True)
        options_layout.addWidget(self.chk_fill)
        layout.addLayout(options_layout)
        
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("线条粗细:"))
        self.spin_thickness = QSpinBox()
        self.spin_thickness.setRange(1, 10)
        self.spin_thickness.setValue(3)
        param_layout.addWidget(self.spin_thickness)
        layout.addLayout(param_layout)
        
        self.btn_draw_line = QPushButton("绘制水线")
        self.btn_draw_line.clicked.connect(self.draw_water_line)
        self.btn_draw_line.setEnabled(False)
        layout.addWidget(self.btn_draw_line)
        
        self.lbl_result = QLabel("")
        self.lbl_result.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_result.setWordWrap(True)
        layout.addWidget(self.lbl_result)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def set_viewer(self, viewer: WaterLineViewer):
        self.viewer = viewer
        self.viewer.line_drawn.connect(self.on_line_drawn)
    
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is not None:
                h, w = self.current_image.shape[:2]
                self.lbl_image_info.setText(f"已加载: {self.current_image_path.name} ({w}x{h}) | 点击添加水线点，双击完成")
                self.btn_clear.setEnabled(True)
                self.btn_draw_line.setEnabled(True)
                self.water_line_points = None
                self.lbl_result.setText("")
                
                if hasattr(self, 'viewer'):
                    self.viewer.set_image(self.current_image)
                    self.viewer.clear_line()
    
    def clear_all(self):
        self.water_line_points = None
        self.lbl_result.setText("")
        self.btn_save.setEnabled(False)
        if hasattr(self, 'viewer'):
            self.viewer.clear_line()
    
    def draw_water_line(self):
        if hasattr(self, 'viewer') and self.viewer.current_image is not None:
            self.viewer.setFocus()
            self.lbl_image_info.setText("绘制模式: 点击添加水线点，双击完成绘制")
    
    def on_line_drawn(self, points):
        if not points or len(points) < 2:
            return
        
        self.water_line_points = points
        
        result_image = self.current_image.copy()
        
        line_color = (0, 255, 255)
        thickness = self.spin_thickness.value()
        
        if self.chk_fill.isChecked():
            h, w = result_image.shape[:2]
            pts = np.array(points, dtype=np.int32)
            
            fill_pts = []
            for i in range(len(pts) - 1):
                fill_pts.append([pts[i, 0], pts[i, 1]])
                fill_pts.append([pts[i + 1, 0], pts[i + 1, 1]])
                fill_pts.append([pts[i + 1, 0], h - 1])
                fill_pts.append([pts[i, 0], h - 1])
            
            if fill_pts:
                fill_pts = np.array(fill_pts, dtype=np.int32)
                cv2.fillPoly(result_image, [fill_pts], (0, 150, 255))
        
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(result_image, [pts], False, line_color, thickness)
        
        for pt in points:
            cv2.circle(result_image, (int(pt[0]), int(pt[1])), 5, line_color, -1)
        
        self.result_image = result_image
        self.result_ready.emit(result_image)
        
        self.btn_save.setEnabled(True)
        
        y_coords = [p[1] for p in points]
        min_y, max_y = int(min(y_coords)), int(max(y_coords))
        avg_y = int(sum(y_coords) / len(y_coords))
        self.lbl_result.setText(f"水线已绘制 | Y范围: {min_y}-{max_y} | 平均Y: {avg_y}")
    
    def save_result(self):
        if not hasattr(self, 'result_image') or self.result_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果图片", "", "JPEG Files (*.jpg)"
        )
        
        if file_path:
            cv2.imwrite(file_path, self.result_image)
            QMessageBox.information(self, "完成", f"已保存到:\n{file_path}")
