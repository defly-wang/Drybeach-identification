import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSpinBox,
                             QDoubleSpinBox, QComboBox, QTabWidget, QScrollArea,
                             QMessageBox, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction

from .viewers import ImageViewer, MarkImageViewer, WaterLineViewer
from .threads import ProcessingThread
from .video_tab import VideoExtractWidget
from .mark_tab import MarkSegmentationWidget
from .detection_tab import DetectionTab
from .calibration_tab import CalibrationTab
from .training_tab import TrainingTab
from .water_line_tab import WaterLineTab


class DryBeachGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_image_path = None
        self.roi = None
        self.roi_start = None
        self.roi_end = None
        self.roi_mode = False
        self.roi_mode_context = None
        self.calibration_done = False
        self.calibration_points = []
        self.training_data_path = None
        self.recognizer = None
        self.annotated_image = None
        
        self.processing_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("干滩识别系统 - Dry Beach Identification System")
        self.setGeometry(100, 100, 1400, 900)
        
        self._create_menu_bar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, 3)
        
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        self.mark_tab.set_image_viewer(self.mark_viewer)
        
        self.statusBar().showMessage("就绪")
    
    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("文件")
        
        open_image_action = QAction("打开图片", self)
        open_image_action.triggered.connect(self.load_image)
        file_menu.addAction(open_image_action)
        
        save_action = QAction("保存结果", self)
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        
        export_action = QAction("导出报告", self)
        export_action.triggered.connect(self.export_report)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        tools_menu = menubar.addMenu("工具")
        
        detect_action = QAction("运行识别", self)
        detect_action.triggered.connect(self.run_detection)
        tools_menu.addAction(detect_action)
        
        train_action = QAction("训练模型", self)
        train_action.triggered.connect(self.train_model)
        tools_menu.addAction(train_action)
    
    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout()
        
        self.tabs = QTabWidget()
        
        self.video_extract_widget = VideoExtractWidget()
        self.video_extract_widget.thumbnail_clicked.connect(self.load_image_from_path)
        self.video_extract_widget.status_updated.connect(self.statusBar().showMessage)
        self.tabs.addTab(self.video_extract_widget, "视频提取")
        
        self.water_line_tab = self._create_water_line_tab()
        self.tabs.addTab(self.water_line_tab, "水线检测")
        
        self.mark_tab = self._create_mark_segment_tab()
        self.tabs.addTab(self.mark_tab, "标记分割")
        
        self.training_tab = self._create_training_tab()
        self.tabs.addTab(self.training_tab, "模型训练")
        
        self.detection_tab = self._create_detection_tab()
        self.tabs.addTab(self.detection_tab, "识别")
        
        self.calibration_tab = self._create_calibration_tab()
        self.tabs.addTab(self.calibration_tab, "校准")
        
        layout.addWidget(self.tabs)
        
        panel.setLayout(layout)
        return panel
    
    def _create_detection_tab(self) -> DetectionTab:
        tab = DetectionTab()
        tab.model_loaded.connect(self.on_model_loaded)
        tab.detection_requested.connect(self.on_detection_requested)
        tab.image_loaded.connect(self.on_detection_image_loaded)
        return tab
    
    def _create_calibration_tab(self) -> CalibrationTab:
        tab = CalibrationTab()
        tab.calibration_started.connect(self.on_calibration_started)
        return tab
    
    def _create_mark_segment_tab(self) -> MarkSegmentationWidget:
        tab = MarkSegmentationWidget()
        tab.image_opened.connect(self.on_mark_image_opened)
        return tab
    
    def _create_training_tab(self) -> TrainingTab:
        tab = TrainingTab()
        tab.training_requested.connect(self.on_training_requested)
        return tab
    
    def _create_water_line_tab(self) -> WaterLineTab:
        tab = WaterLineTab()
        tab.result_ready.connect(self.on_water_line_result)
        tab.set_viewer(self.water_line_viewer)
        return tab
    
    def on_water_line_image_loaded(self, image_path: str):
        self.current_image_path = Path(image_path)
        self.current_image = cv2.imread(image_path)
        if self.current_image is not None:
            self.image_viewer.set_image(self.current_image)
            h, w = self.current_image.shape[:2]
            self.statusBar().showMessage(f"已加载: {self.current_image_path.name} ({w}x{h})")
    
    def on_water_line_result(self, result_image):
        self.annotated_image = result_image
        self.image_viewer.set_image(result_image)
    
    def on_mark_image_opened(self, image_path: str):
        self.mark_viewer.set_image(cv2.imread(image_path))
        self.tabs.setCurrentIndex(2)
    
    def _create_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.image_viewer = ImageViewer()
        self.image_viewer.roi_selected.connect(self.on_roi_selected)
        self.image_viewer.calibration_point_clicked.connect(self.on_calibration_point)
        scroll_area.setWidget(self.image_viewer)
        
        self.mark_viewer = MarkImageViewer()
        self.mark_viewer.hide()
        
        self.water_line_viewer = WaterLineViewer()
        self.water_line_viewer.hide()
        
        layout.addWidget(scroll_area)
        layout.addWidget(self.mark_viewer)
        layout.addWidget(self.water_line_viewer)
        
        self.scroll_area = scroll_area
        
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        return panel
    
    def on_tab_changed(self, index):
        self.scroll_area.hide()
        self.mark_viewer.hide()
        self.water_line_viewer.hide()
        
        if index == 1:
            self.water_line_viewer.show()
        elif index == 2:
            self.mark_viewer.show()
        else:
            self.scroll_area.show()
            self.image_viewer.show()
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is not None:
                self.image_viewer.set_image(self.current_image)
                h, w = self.current_image.shape[:2]
                self.statusBar().showMessage(f"已加载: {self.current_image_path.name} ({w}x{h})")
    
    def load_image_from_path(self, file_path: str):
        self.current_image_path = Path(file_path)
        self.current_image = cv2.imread(file_path)
        
        if self.current_image is not None:
            self.image_viewer.set_image(self.current_image)
            h, w = self.current_image.shape[:2]
            self.statusBar().showMessage(f"已加载: {self.current_image_path.name} ({w}x{h})")
    
    def on_model_loaded(self, model_path: str):
        self.statusBar().showMessage(f"已加载模型: {Path(model_path).name}")
    
    def on_detection_image_loaded(self, image_path: str):
        self.current_image_path = Path(image_path)
        self.current_image = cv2.imread(image_path)
        if self.current_image is not None:
            h, w = self.current_image.shape[:2]
            self.statusBar().showMessage(f"已加载: {self.current_image_path.name} ({w}x{h})")
    
    def on_detection_requested(self, params: dict):
        self.detection_tab.detect_progress.setVisible(True)
        self.detection_tab.detect_progress.setValue(0)
        self.detection_tab.btn_run_detection.setEnabled(False)
        self.statusBar().showMessage("正在识别...")
        
        image_paths = [self.current_image_path] if self.current_image_path else []
        
        if not image_paths:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        self.processing_thread = ProcessingThread('detection', {
            'image_paths': image_paths,
            'model_path': params['model_path'],
            'patch_size': params['patch_size'],
            'stride': params['stride']
        })
        
        self.processing_thread.progress_updated.connect(self.detection_tab.set_progress)
        self.processing_thread.finished.connect(self.on_detection_complete)
        self.processing_thread.error_occurred.connect(self.on_error)
        
        self.processing_thread.start()
    
    def on_detection_complete(self, results: list):
        self.statusBar().showMessage(f"识别完成: {len(results)} 张图片")
        
        if results:
            first_result = results[0]
            if 'annotated' in first_result:
                self.annotated_image = first_result['annotated']
                self.image_viewer.set_image(self.annotated_image)
                
                counts = first_result.get('class_counts', {})
                count_text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
                self.detection_tab.lbl_detect_result.setText(count_text)
    
    def on_error(self, error_msg: str):
        self.statusBar().showMessage("处理出错")
        QMessageBox.critical(self, "错误", error_msg)
    
    def on_roi_selected(self, roi_tuple: Tuple[int, int, int, int]):
        self.roi = roi_tuple
        x, y, w, h = roi_tuple
        self.image_viewer.set_mode('normal')
        
        if self.roi_mode_context == 'water_line':
            self.water_line_tab.on_roi_selected(roi_tuple)
            self.tabs.setCurrentIndex(1)
            self.roi_mode_context = None
        else:
            self.statusBar().showMessage(f"ROI已设置: ({x},{y}) {w}x{h}")
    
    def on_calibration_started(self):
        self.image_viewer.set_mode('calibration')
        self.calibration_points = []
        self.statusBar().showMessage("校准模式: 请在图像上点击两个校准点")
    
    def on_calibration_point(self, point: Tuple[int, int]):
        if self.calibration_tab.calibration_points is None:
            self.calibration_tab.calibration_points = []
        self.calibration_tab.add_calibration_point(point)
    
    def on_training_requested(self, params: dict):
        self.training_tab.training_progress.setVisible(True)
        self.training_tab.training_progress.setMaximum(100)
        self.training_tab.training_progress.setValue(0)
        self.training_tab.btn_start_train.setEnabled(False)
        self.statusBar().showMessage("正在训练模型...")
        
        self.processing_thread = ProcessingThread('training', params)
        
        self.processing_thread.progress_updated.connect(self.training_tab.set_progress)
        self.processing_thread.finished.connect(self.on_training_complete)
        self.processing_thread.error_occurred.connect(self.on_error)
        
        self.processing_thread.start()
    
    def on_training_complete(self, results: list):
        self.training_tab.on_training_complete(results[0] if results else None)
        self.statusBar().showMessage(f"训练完成: {results[0] if results else '失败'}")
    
    def save_result(self):
        if not hasattr(self, 'annotated_image') or self.annotated_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的标注图片")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果图片", "", "JPEG Files (*.jpg)"
        )
        
        if file_path:
            cv2.imwrite(file_path, self.annotated_image)
            self.statusBar().showMessage(f"已保存: {Path(file_path).name}")
    
    def export_report(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", "", "Text Files (*.txt)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.statusBar().currentMessage())
            self.statusBar().showMessage(f"已导出: {Path(file_path).name}")
    
    def run_detection(self):
        self.tabs.setCurrentIndex(4)
        if hasattr(self.detection_tab, 'btn_run_detection'):
            self.detection_tab.btn_run_detection.click()
    
    def train_model(self):
        self.tabs.setCurrentIndex(3)


def launch_gui():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DryBeachGUI()
    window.show()
    
    sys.exit(app.exec())
