from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QComboBox, QCheckBox, QMessageBox,
                             QSpinBox)
from PyQt6.QtCore import pyqtSignal


class WaterLineTab(QWidget):
    result_ready = pyqtSignal(object)
    image_loaded = pyqtSignal(str)
    roi_requested = pyqtSignal()
    detection_requested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.current_image_path = None
        self.detected_line = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.open_image)
        btn_layout.addWidget(self.btn_open)
        
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)
        
        self.lbl_image_info = QLabel("未加载图片")
        self.lbl_image_info.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_image_info)
        
        roi_layout = QHBoxLayout()
        self.btn_set_roi = QPushButton("设置检测区域")
        self.btn_set_roi.clicked.connect(self.set_roi)
        self.btn_set_roi.setEnabled(False)
        roi_layout.addWidget(self.btn_set_roi)
        
        self.btn_clear_roi = QPushButton("清除区域")
        self.btn_clear_roi.clicked.connect(self.clear_roi)
        self.btn_clear_roi.setEnabled(False)
        roi_layout.addWidget(self.btn_clear_roi)
        layout.addLayout(roi_layout)
        
        self.lbl_roi_info = QLabel("未设置检测区域")
        self.lbl_roi_info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_roi_info)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("检测方法:"))
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["边缘检测", "颜色分割", "综合方法"])
        self.cmb_method.setCurrentText("综合方法")
        method_layout.addWidget(self.cmb_method)
        layout.addLayout(method_layout)
        
        options_layout = QHBoxLayout()
        self.chk_smooth = QCheckBox("平滑曲线")
        self.chk_smooth.setChecked(True)
        options_layout.addWidget(self.chk_smooth)
        
        self.chk_fill = QCheckBox("填充水面区域")
        options_layout.addWidget(self.chk_fill)
        layout.addLayout(options_layout)
        
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("线条粗细:"))
        self.spin_thickness = QSpinBox()
        self.spin_thickness.setRange(1, 10)
        self.spin_thickness.setValue(3)
        param_layout.addWidget(self.spin_thickness)
        layout.addLayout(param_layout)
        
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
    
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is not None:
                h, w = self.current_image.shape[:2]
                self.lbl_image_info.setText(f"已加载: {self.current_image_path.name} ({w}x{h})")
                self.btn_detect.setEnabled(True)
                self.btn_set_roi.setEnabled(True)
                self.detected_line = None
                self.roi = None
                self.lbl_result.setText("")
                self.lbl_roi_info.setText("未设置检测区域")
                self.image_loaded.emit(file_path)
    
    def set_roi(self):
        self.roi_requested.emit()
    
    def on_roi_selected(self, roi):
        self.roi = roi
        x, y, w, h = roi
        self.lbl_roi_info.setText(f"已设置: ({x},{y}) {w}x{h}")
        self.btn_clear_roi.setEnabled(True)
    
    def clear_roi(self):
        self.roi = None
        self.lbl_roi_info.setText("未设置检测区域")
        self.btn_clear_roi.setEnabled(False)
    
    def detect_water_line(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        from drybeach_app.water_line_detector import WaterLineDetector, visualize_water_line
        
        detector = WaterLineDetector()
        method = self.cmb_method.currentText()
        roi = self.roi
        
        if method == "边缘检测":
            self.detected_line = detector.detect_by_edge_detection(self.current_image, roi=roi)
        elif method == "颜色分割":
            self.detected_line = detector.detect_by_color_segmentation(self.current_image, roi=roi)
        else:
            results = detector.detect_multi_method(self.current_image, roi=roi)
            self.detected_line = results['final_line']
        
        result_image = self.current_image.copy()
        
        if roi:
            x, y, w, h = roi
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if self.detected_line is not None and len(self.detected_line) > 0:
            if self.chk_fill.isChecked():
                h, w = result_image.shape[:2]
                fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                pts = self.detected_line.astype(np.int32)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                
                water_pts = []
                for i in range(len(pts) - 1):
                    water_pts.append([pts[i, 0], pts[i, 1]])
                    water_pts.append([pts[i + 1, 0], pts[i + 1, 1]])
                    water_pts.append([pts[i + 1, 0], h - 1])
                    water_pts.append([pts[i, 0], h - 1])
                
                if water_pts:
                    water_pts = np.array(water_pts, dtype=np.int32)
                    cv2.fillPoly(result_image, [water_pts], (0, 150, 255))
            
            thickness = self.spin_thickness.value()
            result_image = visualize_water_line(result_image, self.detected_line, 
                                               color=(0, 255, 255), thickness=thickness)
        
        self.result_image = result_image
        self.result_ready.emit(result_image)
        
        self.btn_save.setEnabled(True)
        
        if self.detected_line is not None and len(self.detected_line) > 0:
            y_coords = self.detected_line[:, 1]
            min_y, max_y = int(y_coords.min()), int(y_coords.max())
            avg_y = int(y_coords.mean())
            self.lbl_result.setText(
                f"检测完成 | 水线Y范围: {min_y}-{max_y} | 平均: {avg_y} | "
                f"置信度: {detector.confidence:.2f}"
            )
    
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
