from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QMessageBox, QComboBox, QSpinBox, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal, Qt

from .viewers import MarkImageViewer


class SegmentationThread(QThread):
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(dict, object)
    error = pyqtSignal(str)
    
    def __init__(self, img, regions, output_dir, original_name, patch_size, stride, category_safe_names, region_colors):
        super().__init__()
        self._original_img = img
        self.img = img.copy()
        self.regions = regions
        self.output_dir = output_dir
        self.original_name = original_name
        self.patch_size = patch_size
        self.stride = stride
        self.category_safe_names = category_safe_names
        self.region_colors = region_colors
    
    def run(self):
        try:
            img_h, img_w = self.img.shape[:2]
            
            all_masks = {}
            for category, polygons in self.regions.items():
                if not polygons:
                    continue
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                for polygon_data in polygons:
                    pts = np.array([[int(px), int(py)] for px, py in polygon_data['points']], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                all_masks[category] = mask
            
            counts = {cat: 0 for cat in all_masks.keys()}
            
            total_steps = ((img_h - self.patch_size) // self.stride + 1) * ((img_w - self.patch_size) // self.stride + 1)
            current_step = 0
            patch_area = self.patch_size * self.patch_size
            
            display_img = self.img.copy()
            
            for y in range(0, img_h - self.patch_size + 1, self.stride):
                for x in range(0, img_w - self.patch_size + 1, self.stride):
                    patch_y1, patch_y2 = y, y + self.patch_size
                    patch_x1, patch_x2 = x, x + self.patch_size
                    
                    best_category = None
                    best_overlap = 0
                    
                    for category, mask in all_masks.items():
                        patch_mask = mask[patch_y1:patch_y2, patch_x1:patch_x2]
                        overlap_pixels = np.count_nonzero(patch_mask)
                        overlap_ratio = overlap_pixels / patch_area
                        
                        if overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_category = category
                    
                    if best_category and best_overlap > 0.1:
                        safe_name = self.category_safe_names.get(best_category, best_category)
                        category_dir = self.output_dir / safe_name
                        category_dir.mkdir(parents=True, exist_ok=True)
                        
                        patch = self._original_img[patch_y1:patch_y2, patch_x1:patch_x2]
                        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                        output_file = category_dir / f"{self.original_name}_{y}_{x}.jpg"
                        cv2.imwrite(str(output_file), patch_bgr)
                        counts[best_category] += 1
                        
                        color = self.region_colors.get(best_category, (255, 255, 0))
                        overlay_patch = display_img[patch_y1:patch_y2, patch_x1:patch_x2]
                        color_filled = np.full((self.patch_size, self.patch_size, 3), color, dtype=np.uint8)
                        cv2.addWeighted(color_filled, 0.3, overlay_patch, 0.7, 0, overlay_patch)
                    
                    current_step += 1
                    self.progress.emit(current_step, total_steps)
            
            self.finished.emit(counts, display_img)
        except Exception as e:
            self.error.emit(str(e))


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
        
        self.btn_save_regions = QPushButton("保存区域")
        self.btn_save_regions.clicked.connect(self.save_regions)
        btn_layout.addWidget(self.btn_save_regions)
        
        self.btn_load_regions = QPushButton("导入区域")
        self.btn_load_regions.clicked.connect(self.load_regions)
        btn_layout.addWidget(self.btn_load_regions)
        
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
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
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
                self._viewer.set_image(self.current_image, Path(file_path).name)
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
    
    def save_regions(self):
        if self._viewer is None:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        regions = self._viewer.regions
        has_regions = any(regions.values())
        
        if not has_regions:
            QMessageBox.warning(self, "警告", "没有可保存的区域")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存区域", "", "JSON Files (*.json)"
        )
        
        if file_path:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(regions, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "完成", f"区域已保存至: {file_path}")
    
    def load_regions(self):
        if self._viewer is None or self._viewer.current_image is None:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择区域文件", "", "JSON Files (*.json)"
        )
        
        if file_path:
            import json
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    regions = json.load(f)
                
                valid_categories = {'水面', '摊面', '分界线', '坝体'}
                for cat in valid_categories:
                    if cat not in regions:
                        regions[cat] = []
                    elif not isinstance(regions[cat], list):
                        regions[cat] = []
                
                self._viewer.regions = regions
                self._viewer._update_display()
                QMessageBox.information(self, "完成", "区域已导入")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")
    
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
        total_steps = ((img_h - patch_size) // stride + 1) * ((img_w - patch_size) // stride + 1)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(total_steps)
        self.progress_bar.setValue(0)
        self.btn_segment.setEnabled(False)
        
        self._segment_thread = SegmentationThread(
            img, regions, output_dir, original_name, patch_size, stride, self.category_safe_names, self._viewer.region_colors
        )
        self._segment_thread.progress.connect(self._on_segment_progress)
        self._segment_thread.finished.connect(self._on_segment_finished)
        self._segment_thread.error.connect(self._on_segment_error)
        self._segment_thread.start()
    
    def _on_segment_progress(self, current, total):
        self.progress_bar.setValue(current)
        self.lbl_info.setText(f"正在分割图像... {current}/{total}")
    
    def _on_segment_finished(self, counts, result_img):
        self.progress_bar.setVisible(False)
        self.btn_segment.setEnabled(True)
        total_saved = sum(counts.values())
        self.lbl_info.setText(f"已分割保存 {total_saved} 张图片: 水面{counts.get('水面',0)} 滩面{counts.get('摊面',0)} 分界线{counts.get('分界线',0)} 坝体{counts.get('坝体',0)}")
        
        if result_img is not None:
            display_name = f"{Path(self.current_image_path).stem}_segmented.jpg"
            self._viewer.set_image(result_img, display_name)
            self._viewer.current_image = result_img
        
        QMessageBox.information(self, "完成", f"已分割保存 {total_saved} 张图片")
    
    def _on_segment_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.btn_segment.setEnabled(True)
        QMessageBox.critical(self, "错误", f"分割失败: {error_msg}")
    
    def set_image_viewer(self, viewer: MarkImageViewer):
        self._viewer = viewer
