from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QMessageBox)
from PyQt6.QtCore import pyqtSignal

from .viewers import MarkImageViewer


class MarkSegmentationWidget(QWidget):
    image_opened = pyqtSignal(str)
    
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
        
        self.btn_segment = QPushButton("图像分割")
        self.btn_segment.clicked.connect(self.segment_image)
        self.btn_segment.setEnabled(False)
        layout.addWidget(self.btn_segment)
        
        self.lbl_info = QLabel("提示: 打开图片后，选择类别，用鼠标拖动画矩形区域标记")
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
        
        output_dir = Path(output_dir)
        original_name = Path(self.current_image_path).stem
        
        total_saved = 0
        for category, polygons in regions.items():
            if not polygons:
                continue
            
            category_dir = output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            for i, polygon_data in enumerate(polygons):
                disp_points = polygon_data['points']
                if len(disp_points) < 3:
                    continue
                
                img_points = []
                for px, py in disp_points:
                    ix = int(px / self._viewer._zoom_factor)
                    iy = int(py / self._viewer._zoom_factor)
                    img_points.append([ix, iy])
                
                pts = np.array(img_points, dtype=np.int32)
                
                x, y, w, h = cv2.boundingRect(pts)
                x = max(0, x)
                y = max(0, y)
                w = min(w, self._viewer.current_image.shape[1] - x)
                h = min(h, self._viewer.current_image.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue
                
                mask = np.zeros(self._viewer.current_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                
                roi = self._viewer.current_image[y:y+h, x:x+w]
                mask_roi = mask[y:y+h, x:x+w]
                
                if roi.size > 0:
                    roi_masked = cv2.bitwise_and(roi, roi, mask=mask_roi)
                    
                    output_file = category_dir / f"{original_name}_{category}_{i+1}.jpg"
                    cv2.imwrite(str(output_file), roi_masked)
                    total_saved += 1
        
        self.lbl_info.setText(f"已分割保存 {total_saved} 张图片到 {output_dir}")
        QMessageBox.information(self, "完成", f"已分割保存 {total_saved} 张图片到:\n{output_dir}")
    
    def set_image_viewer(self, viewer: MarkImageViewer):
        self._viewer = viewer
