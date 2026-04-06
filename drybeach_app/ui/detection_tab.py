from pathlib import Path

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QComboBox, QSpinBox, QProgressBar)
from PyQt6.QtCore import pyqtSignal


class DetectionTab(QWidget):
    model_loaded = pyqtSignal(str)
    detection_requested = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_path = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.clicked.connect(self.load_model)
        layout.addWidget(self.btn_load_model)
        
        self.lbl_model_path = QLabel("未加载模型")
        self.lbl_model_path.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_model_path)
        
        self.btn_load_image = QPushButton("打开图片")
        self.btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load_image)
        
        self.lbl_image_path = QLabel("未加载图片")
        self.lbl_image_path.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_image_path)
        
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("切片尺寸:"))
        self.cmb_detect_patch = QComboBox()
        self.cmb_detect_patch.addItems(["24", "32", "64", "96", "128"])
        self.cmb_detect_patch.setCurrentText("64")
        param_layout.addWidget(self.cmb_detect_patch)
        
        param_layout.addWidget(QLabel("步长:"))
        self.spin_detect_stride = QSpinBox()
        self.spin_detect_stride.setRange(2, 64)
        self.spin_detect_stride.setValue(32)
        param_layout.addWidget(self.spin_detect_stride)
        layout.addLayout(param_layout)
        
        self.btn_run_detection = QPushButton("运行识别")
        self.btn_run_detection.clicked.connect(self.run_detection)
        layout.addWidget(self.btn_run_detection)
        
        self.detect_progress = QProgressBar()
        self.detect_progress.setVisible(False)
        layout.addWidget(self.detect_progress)
        
        btn_save_result = QPushButton("保存结果图片")
        btn_save_result.clicked.connect(self.save_result)
        layout.addWidget(btn_save_result)
        
        self.lbl_detect_result = QLabel("")
        self.lbl_detect_result.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_detect_result)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "Model Files (*.pt)"
        )
        
        if file_path:
            self.model_path = Path(file_path)
            self.lbl_model_path.setText(f"已加载: {self.model_path.name}")
            self.model_loaded.emit(file_path)
    
    def load_image(self):
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.image_path = Path(file_path)
            import cv2
            img = cv2.imread(file_path)
            if img is not None:
                h, w = img.shape[:2]
                self.lbl_image_path.setText(f"已选择: {self.image_path.name} ({w}x{h})")
                self.image_loaded.emit(file_path)
    
    def run_detection(self):
        if not self.model_path:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        self.detection_requested.emit({
            'model_path': str(self.model_path),
            'patch_size': int(self.cmb_detect_patch.currentText()),
            'stride': self.spin_detect_stride.value()
        })
    
    def save_result(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        if not hasattr(self, 'annotated_image') or self.annotated_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的标注图片")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果图片", "", "JPEG Files (*.jpg)"
        )
        
        if file_path:
            import cv2
            cv2.imwrite(file_path, self.annotated_image)
    
    def on_detection_complete(self, results):
        self.detect_progress.setVisible(False)
        self.btn_run_detection.setEnabled(True)
        
        if results:
            first_result = results[0]
            if 'annotated' in first_result:
                self.annotated_image = first_result['annotated']
                counts = first_result.get('class_counts', {})
                count_text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
                self.lbl_detect_result.setText(count_text)
                self.result_ready.emit(self.annotated_image)
    
    def set_progress(self, value):
        self.detect_progress.setValue(value)
    
    image_loaded = pyqtSignal(str)
    result_ready = pyqtSignal(object)
