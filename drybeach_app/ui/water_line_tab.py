from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QComboBox, QCheckBox, QMessageBox,
                             QSpinBox)
from PyQt6.QtCore import pyqtSignal


class WaterLineTab(QWidget):
    result_ready = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.current_image_path = None
        self.detection_region = None
        self.detected_line = None
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
        
        self.lbl_image_info = QLabel("提示: 打开图片后，点击绘制检测区域，双击完成")
        self.lbl_image_info.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_image_info)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("检测方法:"))
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["边缘检测", "颜色分割", "综合方法"])
        self.cmb_method.setCurrentText("边缘检测")
        method_layout.addWidget(self.cmb_method)
        layout.addLayout(method_layout)
        
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
        
        self.btn_draw_region = QPushButton("绘制检测区域")
        self.btn_draw_region.clicked.connect(self.start_drawing)
        self.btn_draw_region.setEnabled(False)
        layout.addWidget(self.btn_draw_region)
        
        self.btn_detect = QPushButton("检测水线")
        self.btn_detect.clicked.connect(self.detect_water_line)
        self.btn_detect.setEnabled(False)
        layout.addWidget(self.btn_detect)
        
        self.lbl_result = QLabel("")
        self.lbl_result.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_result.setWordWrap(True)
        layout.addWidget(self.lbl_result)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def set_viewer(self, viewer):
        self.viewer = viewer
        self.viewer.region_drawn.connect(self.on_region_drawn)
    
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is not None:
                h, w = self.current_image.shape[:2]
                self.lbl_image_info.setText(f"已加载: {self.current_image_path.name} ({w}x{h}) | 点击绘制检测区域，双击完成")
                self.btn_clear.setEnabled(True)
                self.btn_draw_region.setEnabled(True)
                self.btn_detect.setEnabled(False)
                self.detection_region = None
                self.detected_line = None
                self.lbl_result.setText("")
                
                if hasattr(self, 'viewer'):
                    self.viewer.set_image(self.current_image)
                    self.viewer.clear_regions()
    
    def start_drawing(self):
        if hasattr(self, 'viewer') and self.viewer.current_image is not None:
            self.viewer.set_mode('draw')
            self.viewer.current_category = '检测区域'
            self.viewer.polygon_points = []
            self.lbl_image_info.setText("绘制模式: 点击添加多边形顶点，双击完成绘制")
    
    def on_region_drawn(self, category, points):
        if category == '检测区域' and points and len(points) >= 3:
            self.detection_region = points
            self.btn_detect.setEnabled(True)
            self.lbl_image_info.setText(f"检测区域已设置 ({len(points)}个点) | 点击检测水线")
    
    def clear_all(self):
        self.detection_region = None
        self.detected_line = None
        self.lbl_result.setText("")
        self.btn_save.setEnabled(False)
        self.btn_detect.setEnabled(False)
        if hasattr(self, 'viewer'):
            self.viewer.clear_regions()
            self.viewer.set_mode('normal')
        self.lbl_image_info.setText("已清空所有设置")
    
    def detect_water_line(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        if not self.detection_region or len(self.detection_region) < 3:
            QMessageBox.warning(self, "警告", "请先绘制检测区域")
            return
        
        from drybeach_app.water_line_detector import WaterLineDetector
        
        detector = WaterLineDetector()
        method = self.cmb_method.currentText()
        
        region_points = np.array(self.detection_region, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(region_points)
        
        mask = np.zeros(self.current_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [region_points], -1, 255, -1)
        
        roi_image = self.current_image[y:y+h, x:x+w].copy()
        roi_mask = mask[y:y+h, x:x+w]
        
        roi_image_masked = cv2.bitwise_and(roi_image, roi_image, mask=roi_mask)
        
        gray = cv2.cvtColor(roi_image_masked, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edges = cv2.dilate(edges, None, iterations=2)
        edges = cv2.erode(edges, None, iterations=1)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80, minLineLength=30, maxLineGap=10)
        
        result_image = self.current_image.copy()
        
        cv2.polylines(result_image, [region_points], True, (0, 255, 0), 2)
        
        water_line_points = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 30:
                    water_line_points.append((x + x1, y + y1))
                    water_line_points.append((x + x2, y + y2))
        
        if not water_line_points:
            horizontal_profiles = np.mean(gray, axis=1)
            gradient = np.gradient(horizontal_profiles)
            threshold = np.mean(gradient) + 2 * np.std(gradient)
            significant_changes = np.where(np.abs(gradient) > threshold)[0]
            
            if len(significant_changes) > 0:
                water_y = significant_changes[0]
                water_line_points = [(x, y + water_y), (x + w, y + water_y)]
            else:
                water_y = h // 2
                water_line_points = [(x, y + water_y), (x + w, y + water_y)]
        
        if len(water_line_points) >= 2:
            pts = np.array(water_line_points, dtype=np.int32)
            pts = pts.reshape((-1, 2))
            
            if self.chk_fill.isChecked():
                fill_pts = []
                for i in range(len(pts) - 1):
                    fill_pts.append([pts[i, 0], pts[i, 1]])
                    fill_pts.append([pts[i + 1, 0], pts[i + 1, 1]])
                    fill_pts.append([pts[i + 1, 0], self.current_image.shape[0] - 1])
                    fill_pts.append([pts[i, 0], self.current_image.shape[0] - 1])
                
                if fill_pts:
                    fill_pts = np.array(fill_pts, dtype=np.int32)
                    cv2.fillPoly(result_image, [fill_pts], (0, 150, 255))
            
            thickness = self.spin_thickness.value()
            cv2.polylines(result_image, [pts], False, (0, 255, 255), thickness)
            
            for pt in water_line_points:
                cv2.circle(result_image, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
        
        self.result_image = result_image
        self.detected_line = water_line_points
        self.result_ready.emit(result_image)
        
        self.btn_save.setEnabled(True)
        
        if water_line_points:
            y_coords = [p[1] for p in water_line_points]
            min_y, max_y = int(min(y_coords)), int(max(y_coords))
            avg_y = int(sum(y_coords) / len(y_coords))
            self.lbl_result.setText(f"检测完成 | 水线Y范围: {min_y}-{max_y} | 平均Y: {avg_y}")
    
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
