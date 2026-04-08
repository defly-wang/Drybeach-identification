from pathlib import Path

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QTextCursor

import cv2


class DetectionTab(QWidget):
    model_loaded = pyqtSignal(str)
    detection_requested = pyqtSignal(dict)
    image_display_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_path = None
        self.model_info = None
        self.detection_region = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.clicked.connect(self.load_model)
        layout.addWidget(self.btn_load_model)
        
        self.lbl_model_path = QLabel("未加载模型")
        self.lbl_model_path.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_model_path)
        
        self.txt_model_info = QTextEdit()
        self.txt_model_info.setReadOnly(True)
        self.txt_model_info.setMaximumHeight(120)
        self.txt_model_info.setStyleSheet("background-color: #1e1e1e; color: #aaa; font-size: 11px;")
        layout.addWidget(self.txt_model_info)
        
        self.btn_load_image = QPushButton("打开图片")
        self.btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load_image)
        
        self.lbl_image_path = QLabel("未加载图片")
        self.lbl_image_path.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_image_path)
        
        region_layout = QHBoxLayout()
        self.btn_draw_region = QPushButton("画识别区域")
        self.btn_draw_region.clicked.connect(self.draw_detection_region)
        region_layout.addWidget(self.btn_draw_region)
        
        self.btn_clear_region = QPushButton("清除区域")
        self.btn_clear_region.clicked.connect(self.clear_detection_region)
        region_layout.addWidget(self.btn_clear_region)
        layout.addLayout(region_layout)
        
        self.lbl_region = QLabel("未设置识别区域")
        self.lbl_region.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_region)
        
        save_load_layout = QHBoxLayout()
        self.btn_save_region = QPushButton("保存区域")
        self.btn_save_region.clicked.connect(self.save_region)
        save_load_layout.addWidget(self.btn_save_region)
        
        self.btn_load_region = QPushButton("导入区域")
        self.btn_load_region.clicked.connect(self.load_region)
        save_load_layout.addWidget(self.btn_load_region)
        layout.addLayout(save_load_layout)
        
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("切片尺寸:"))
        self.cmb_detect_patch = QComboBox()
        self.cmb_detect_patch.addItems(["24", "32", "64", "96", "128"])
        self.cmb_detect_patch.setCurrentText("32")
        param_layout.addWidget(self.cmb_detect_patch)
        
        param_layout.addWidget(QLabel("步长:"))
        self.spin_detect_stride = QSpinBox()
        self.spin_detect_stride.setRange(2, 64)
        self.spin_detect_stride.setValue(16)
        param_layout.addWidget(self.spin_detect_stride)
        
        param_layout.addWidget(QLabel("置信度:"))
        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0.0, 1.0)
        self.spin_confidence.setValue(0.0)
        self.spin_confidence.setSingleStep(0.05)
        self.spin_confidence.setDecimals(2)
        param_layout.addWidget(self.spin_confidence)
        layout.addLayout(param_layout)
        
        self.btn_run_detection = QPushButton("开始识别")
        self.btn_run_detection.clicked.connect(self.run_detection)
        self.btn_run_detection.setEnabled(False)
        layout.addWidget(self.btn_run_detection)
        
        self.detect_progress = QProgressBar()
        self.detect_progress.setVisible(False)
        layout.addWidget(self.detect_progress)
        
        self.lbl_detect_result = QLabel("")
        self.lbl_detect_result.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_detect_result)
        
        self.btn_save_result = QPushButton("保存识别结果")
        self.btn_save_result.clicked.connect(self.save_result)
        self.btn_save_result.setEnabled(False)
        layout.addWidget(self.btn_save_result)
        
        self.btn_draw_boundary = QPushButton("生成分界线")
        self.btn_draw_boundary.clicked.connect(self.draw_boundary_line)
        self.btn_draw_boundary.setEnabled(False)
        layout.addWidget(self.btn_draw_boundary)
        
        self.btn_clear_boundary = QPushButton("清除分界线")
        self.btn_clear_boundary.clicked.connect(self.clear_boundary_line)
        self.btn_clear_boundary.setEnabled(False)
        layout.addWidget(self.btn_clear_boundary)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_model(self):
        from drybeach_app.recognizer import DryBeachRecognizer
        from PyQt6.QtWidgets import QMessageBox
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "Model Files (*.pt)"
        )
        
        if file_path:
            try:
                recognizer = DryBeachRecognizer(model_path=file_path)
                self.model_path = Path(file_path)
                self.model_info = recognizer.model_info
                
                self.lbl_model_path.setText(f"已加载: {self.model_path.name}")
                self._display_model_info()
                
                self.cmb_detect_patch.setCurrentText(str(recognizer.patch_size))
                self.spin_detect_stride.setValue(recognizer.stride)
                
                self.model_loaded.emit(file_path)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
    
    def _display_model_info(self):
        if not self.model_info:
            return
        
        info_text = f"""模型信息:
━━━━━━━━━━━━━━━━━━━
路径: {self.model_info.get('path', 'N/A')}
设备: {self.model_info.get('device', 'N/A')}
类别数: {self.model_info.get('num_classes', 4)}
类别: {', '.join(self.model_info.get('categories', []))}
总参数: {self.model_info.get('total_params', 0):,}
可训练参数: {self.model_info.get('trainable_params', 0):,}"""
        
        self.txt_model_info.setPlainText(info_text)
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.image_path = Path(file_path)
            img = cv2.imread(file_path)
            if img is not None:
                h, w = img.shape[:2]
                self.lbl_image_path.setText(f"已选择: {self.image_path.name} ({w}x{h})")
                self.image_display_requested.emit(file_path)
                
                if self.model_path:
                    self.btn_run_detection.setEnabled(True)
    
    def draw_detection_region(self):
        if not hasattr(self, 'image_path') or not self.image_path:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        self.draw_region_requested.emit(str(self.image_path))
    
    def clear_detection_region(self):
        self.detection_region = None
        self.lbl_region.setText("未设置识别区域")
        self.region_cleared.emit()
    
    def save_region(self):
        from PyQt6.QtWidgets import QMessageBox
        
        region_data = self.detection_region
        print(f"save_region: detection_region = {region_data}")
        
        if region_data is None:
            QMessageBox.warning(self, "警告", "没有可保存的区域")
            return
        
        if not isinstance(region_data, dict) or 'points' not in region_data:
            QMessageBox.warning(self, "警告", "没有可保存的区域")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存区域", "", "JSON Files (*.json)"
        )
        
        if file_path:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(region_data, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "完成", f"区域已保存至: {file_path}")
    
    def load_region(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择区域文件", "", "JSON Files (*.json)"
        )
        
        if file_path:
            import json
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.detection_region = json.load(f)
                
                if self.detection_region and 'points' in self.detection_region:
                    self.lbl_region.setText(f"已加载区域: {len(self.detection_region['points'])}个点")
                    self.region_loaded.emit(self.detection_region)
                else:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "警告", "区域文件格式错误")
                    
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")
    
    def set_detection_region(self, region_data):
        self.detection_region = region_data
        if region_data and 'points' in region_data:
            self.lbl_region.setText(f"识别区域: {len(region_data['points'])}个点")
    
    def run_detection(self):
        from PyQt6.QtWidgets import QMessageBox
        
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        self.detect_progress.setVisible(True)
        self.detect_progress.setValue(0)
        self.btn_run_detection.setEnabled(False)
        
        self.detection_requested.emit({
            'model_path': str(self.model_path),
            'image_path': str(self.image_path),
            'patch_size': int(self.cmb_detect_patch.currentText()),
            'stride': self.spin_detect_stride.value(),
            'detection_region': self.detection_region,
            'confidence_threshold': self.spin_confidence.value()
        })
    
    def on_detection_complete(self, annotated_image, class_counts, detection_points=None):
        self.detect_progress.setVisible(False)
        self.btn_run_detection.setEnabled(True)
        
        if annotated_image is not None:
            self.annotated_image = annotated_image
            self.original_result = annotated_image.copy()
            self.detection_points = detection_points or []
            count_text = " | ".join([f"{k}: {v}" for k, v in class_counts.items()])
            self.lbl_detect_result.setText(count_text)
            self.btn_save_result.setEnabled(True)
            self.btn_draw_boundary.setEnabled(True)
            self.result_ready.emit(annotated_image)
    
    def save_result(self):
        from PyQt6.QtWidgets import QMessageBox
        
        if not hasattr(self, 'annotated_image') or self.annotated_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的识别结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存识别结果", "", "JPEG Files (*.jpg)"
        )
        
        if file_path:
            try:
                save_image = cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, save_image)
                QMessageBox.information(self, "完成", f"结果已保存至: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def set_progress(self, value):
        self.detect_progress.setValue(value)
    
    def draw_boundary_line(self):
        from drybeach_app.recognizer import BoundaryLineGenerator
        
        if not hasattr(self, 'detection_points') or not self.detection_points:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请先进行识别")
            return
        
        if self.annotated_image is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "没有可处理的图像")
            return
        
        self.original_result = self.annotated_image.copy()
        
        stride = self.spin_detect_stride.value()
        
        boundary_lines = BoundaryLineGenerator.generate_boundary_lines(
            self.detection_points, 
            self.annotated_image.shape,
            stride=stride
        )
        
        if not boundary_lines:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "未能生成分界线，请确保有足够的边界点")
            return
        
        result = BoundaryLineGenerator.draw_boundary_lines(
            self.annotated_image, 
            boundary_lines,
            color=(0, 255, 0),
            thickness=3
        )
        
        self.annotated_image = result
        self.boundary_lines = boundary_lines
        self.btn_clear_boundary.setEnabled(True)
        self.result_ready.emit(result)
        
        line_types = [line['type'] for line in boundary_lines]
        self.lbl_detect_result.setText(f"已生成 {len(boundary_lines)} 条分界线: {', '.join(set(line_types))}")
    
    def clear_boundary_line(self):
        if hasattr(self, 'original_result'):
            self.annotated_image = self.original_result.copy()
            self.result_ready.emit(self.annotated_image)
            self.btn_clear_boundary.setEnabled(False)
    
    def update_image_with_boundary(self, image_with_boundary):
        if image_with_boundary is not None:
            self.annotated_image = image_with_boundary
            self.result_ready.emit(image_with_boundary)
    
    image_loaded = pyqtSignal(str)
    result_ready = pyqtSignal(object)
    draw_region_requested = pyqtSignal(str)
    region_cleared = pyqtSignal()
    region_loaded = pyqtSignal(object)
    boundary_draw_requested = pyqtSignal()