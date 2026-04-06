from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QMessageBox, QComboBox, QSpinBox)
from PyQt6.QtCore import pyqtSignal

from .viewers import MarkImageViewer


class MarkSegmentationWidget(QWidget):
    image_opened = pyqtSignal(str)
    
    category_safe_names = {
        '水面': 'water',
        '摊面': 'beach',
        '分界线': 'boundary',
        '坝体': 'dam'
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_image = None
        self.current_image_path = None
        self._viewer = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("打开图片")
        self.btn_open.clicked.connect(self.open_image)
        btn_layout.addWidget(self.btn_open)
        
        self.btn_clear = QPushButton("清空")
        self.btn_clear.clicked.connect(self.clear_all)
        btn_layout.addWidget(self.btn_clear)
        layout.addLayout(btn_layout)
        
        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("类别:"))
        
        self.btn_water = QPushButton("水面")
        self.btn_water.setCheckable(True)
        self.btn_water.setStyleSheet("QPushButton:checked { background-color: #00c8ff; }")
        self.btn_water.clicked.connect(lambda: self.select_category('水面'))
        category_layout.addWidget(self.btn_water)
        
        self.btn_beach = QPushButton("摊面")
        self.btn_beach.setCheckable(True)
        self.btn_beach.setStyleSheet("QPushButton:checked { background-color: #c8ff00; }")
        self.btn_beach.clicked.connect(lambda: self.select_category('摊面'))
        category_layout.addWidget(self.btn_beach)
        
        self.btn_boundary = QPushButton("分界线")
        self.btn_boundary.setCheckable(True)
        self.btn_boundary.setStyleSheet("QPushButton:checked { background-color: #ff00ff; }")
        self.btn_boundary.clicked.connect(lambda: self.select_category('分界线'))
        category_layout.addWidget(self.btn_boundary)
        
        self.btn_dam = QPushButton("坝体")
        self.btn_dam.setCheckable(True)
        self.btn_dam.setStyleSheet("QPushButton:checked { background-color: #ff6400; }")
        self.btn_dam.clicked.connect(lambda: self.select_category('坝体'))
        category_layout.addWidget(self.btn_dam)
        
        layout.addLayout(category_layout)
        
        self.category_buttons = {
            '水面': self.btn_water,
            '摊面': self.btn_beach,
            '分界线': self.btn_boundary,
            '坝体': self.btn_dam
        }
        
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("切片尺寸:"))
        self.cmb_patch_size = QComboBox()
        self.cmb_patch_size.addItems(["24x24", "32x32", "64x64", "96x96", "128x128"])
        self.cmb_patch_size.setCurrentText("64x64")
        param_layout.addWidget(self.cmb_patch_size)
        
        param_layout.addWidget(QLabel("步长:"))
        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(2, 32)
        self.spin_stride.setValue(2)
        param_layout.addWidget(self.spin_stride)
        
        layout.addLayout(param_layout)
        
        self.btn_segment = QPushButton("图像分割")
        self.btn_segment.clicked.connect(self.segment_image)
        self.btn_segment.setEnabled(False)
        layout.addWidget(self.btn_segment)
        
        self.lbl_info = QLabel("提示: 打开图片后，用多边形标记区域，再设置切片尺寸和步长进行分割")
        self.lbl_info.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)
        
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
                self.image_opened.emit(file_path)
                self.btn_segment.setEnabled(True)
                self.lbl_info.setText(f"已加载: {Path(file_path).name}")
    
    def select_category(self, category: str):
        for cat, btn in self.category_buttons.items():
            if cat != category:
                btn.setChecked(False)
        
        self._viewer.mode = 'draw'
        self._viewer.current_category = category if self.category_buttons[category].isChecked() else None
        
        if self.category_buttons[category].isChecked():
            self._viewer.polygon_points = []
            self.lbl_info.setText(f"当前类别: {category}，点击添加多边形顶点，双击完成选区")
        else:
            self._viewer.mode = 'normal'
            self._viewer.polygon_points = []
            self.lbl_info.setText("提示: 打开图片后，选择类别，点击添加多边形顶点，双击完成")
    
    def clear_all(self):
        if self._viewer:
            self._viewer.clear_regions()
        for btn in self.category_buttons.values():
            btn.setChecked(False)
        self.lbl_info.setText("已清空所有标记")
    
    def segment_image(self):
        if self._viewer is None or self._viewer.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        regions = self._viewer.regions
        has_regions = any(regions.values())
        
        if not has_regions:
            QMessageBox.warning(self, "警告", "请先标记至少一个区域")
            return
        
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        
        if not output_dir:
            return
        
        patch_size_str = self.cmb_patch_size.currentText()
        patch_size = int(patch_size_str.split('x')[0])
        stride = self.spin_stride.value()
        
        output_dir = Path(output_dir)
        original_name = Path(self.current_image_path).stem
        
        img = self._viewer.current_image
        img_h, img_w = img.shape[:2]
        
        all_masks = {}
        for category, polygons in regions.items():
            if not polygons:
                continue
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for polygon_data in polygons:
                pts = np.array([[int(px), int(py)] for px, py in polygon_data['points']], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            all_masks[category] = mask
        
        counts = {cat: 0 for cat in all_masks.keys()}
        
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                center_x = x + patch_size // 2
                center_y = y + patch_size // 2
                
                for category, mask in all_masks.items():
                    if mask[center_y, center_x] > 0:
                        safe_name = self.category_safe_names.get(category, category)
                        category_dir = output_dir / safe_name
                        category_dir.mkdir(parents=True, exist_ok=True)
                        
                        patch = img[y:y+patch_size, x:x+patch_size]
                        output_file = category_dir / f"{original_name}_{y}_{x}.jpg"
                        cv2.imwrite(str(output_file), patch)
                        counts[category] += 1
                        break
        
        total_saved = sum(counts.values())
        self.lbl_info.setText(f"已分割保存 {total_saved} 张图片: 水面{counts['水面']} 滩面{counts['摊面']} 分界线{counts['分界线']} 坝体{counts['坝体']}")
        QMessageBox.information(self, "完成", f"已分割保存 {total_saved} 张图片到:\n{output_dir}")
    
    def set_image_viewer(self, viewer: MarkImageViewer):
        self._viewer = viewer
