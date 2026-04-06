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

from .viewers import ImageViewer, MarkImageViewer
from .threads import ProcessingThread
from .video_tab import VideoExtractWidget
from .mark_tab import MarkSegmentationWidget


class DryBeachGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_image_path = None
        self.roi = None
        self.roi_start = None
        self.roi_end = None
        self.roi_mode = False
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
        
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        center_panel = self._create_center_panel()
        main_layout.addWidget(center_panel, 3)
        
        self.mark_widget.set_image_viewer(self.mark_viewer)
        
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
        
        detection_tab = self._create_detection_tab()
        self.tabs.addTab(detection_tab, "识别")
        
        calibration_tab = self._create_calibration_tab()
        self.tabs.addTab(calibration_tab, "校准")
        
        mark_tab = self._create_mark_segment_tab()
        self.tabs.addTab(mark_tab, "标记分割")
        
        training_tab = self._create_training_tab()
        self.tabs.addTab(training_tab, "训练")
        
        layout.addWidget(self.tabs)
        
        panel.setLayout(layout)
        return panel
    
    def _create_detection_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        
        btn_load_model = QPushButton("加载模型")
        btn_load_model.clicked.connect(self.load_model)
        layout.addWidget(btn_load_model)
        
        self.lbl_model_path = QLabel("未加载模型")
        self.lbl_model_path.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_model_path)
        
        btn_load_image = QPushButton("打开图片")
        btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(btn_load_image)
        
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
        
        btn_run_detection = QPushButton("运行识别")
        btn_run_detection.clicked.connect(self.run_detection)
        self.btn_run_detection = btn_run_detection
        layout.addWidget(btn_run_detection)
        
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
        tab.setLayout(layout)
        return tab
    
    def _create_calibration_tab(self) -> QWidget:
        tab = QWidget()
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
        
        btn_calibrate = QPushButton("开始校准(点击两点)")
        btn_calibrate.clicked.connect(self.start_calibration)
        self.btn_calibrate = btn_calibrate
        layout.addWidget(btn_calibrate)
        
        self.lbl_calibration_status = QLabel("未校准")
        self.lbl_calibration_status.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.lbl_calibration_status)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def _create_mark_segment_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        
        self.mark_widget = MarkSegmentationWidget()
        self.mark_widget.image_opened.connect(self.on_mark_image_opened)
        layout.addWidget(self.mark_widget)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
    def on_mark_image_opened(self, image_path: str):
        self.mark_viewer.set_image(cv2.imread(image_path))
        self.tabs.setCurrentIndex(3)
    
    def _create_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        
        btn_select_data = QPushButton("选择数据目录")
        btn_select_data.clicked.connect(self.select_training_data)
        layout.addWidget(btn_select_data)
        
        self.lbl_data_path = QLabel("未选择数据目录")
        self.lbl_data_path.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_data_path)
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("训练轮数:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        epochs_layout.addWidget(self.spin_epochs)
        layout.addLayout(epochs_layout)
        
        self.btn_start_train = QPushButton("开始训练")
        self.btn_start_train.clicked.connect(self.train_model)
        self.btn_start_train.setEnabled(False)
        layout.addWidget(self.btn_start_train)
        
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        layout.addWidget(self.training_progress)
        
        self.lbl_model_info = QLabel("")
        self.lbl_model_info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_model_info)
        
        layout.addStretch()
        tab.setLayout(layout)
        return tab
    
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
        
        layout.addWidget(scroll_area)
        layout.addWidget(self.mark_viewer)
        
        self.scroll_area = scroll_area
        
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        return panel
    
    def on_tab_changed(self, index):
        if index == 3:
            self.scroll_area.hide()
            self.mark_viewer.show()
        else:
            self.scroll_area.show()
            self.mark_viewer.hide()
    
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
                self.lbl_image_path.setText(f"已选择: {self.current_image_path.name} ({w}x{h})")
                self.statusBar().showMessage(f"已加载: {self.current_image_path.name} ({w}x{h})")
    
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "Model Files (*.pt)"
        )
        
        if file_path:
            try:
                from drybeach_app.recognizer import DryBeachRecognizer
                self.recognizer = DryBeachRecognizer(model_path=file_path)
                self.model_path = Path(file_path)
                self.lbl_model_path.setText(f"已加载: {self.model_path.name}")
                self.statusBar().showMessage(f"已加载模型: {self.model_path.name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败:\n{str(e)}")
    
    def load_image_from_path(self, file_path: str):
        self.current_image_path = Path(file_path)
        self.current_image = cv2.imread(file_path)
        
        if self.current_image is not None:
            self.image_viewer.set_image(self.current_image)
            h, w = self.current_image.shape[:2]
            self.statusBar().showMessage(f"已加载: {self.current_image_path.name} ({w}x{h})")
    
    def toggle_roi_mode(self):
        self.roi_mode = not self.roi_mode
        if self.roi_mode:
            self.image_viewer.set_mode('roi')
            self.btn_set_roi.setText("取消ROI")
            self.statusBar().showMessage("ROI模式: 点击拖动选择区域")
        else:
            self.image_viewer.set_mode('normal')
            self.btn_set_roi.setText("设置ROI区域")
            self.roi = None
            self.statusBar().showMessage("就绪")
    
    def start_calibration(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        self.image_viewer.set_mode('calibration')
        self.calibration_points = []
        self.btn_calibrate.setText("点击第一个校准点...")
        self.lbl_calibration_status.setText("点击第1点")
        self.statusBar().showMessage("校准模式: 请在图像上点击两个校准点")
    
    def on_roi_selected(self, roi_tuple: Tuple[int, int, int, int]):
        self.roi = roi_tuple
        x, y, w, h = roi_tuple
        self.statusBar().showMessage(f"ROI已设置: ({x},{y}) {w}x{h}")
        self.roi_mode = False
        self.btn_set_roi.setText("设置ROI区域")
        self.image_viewer.set_mode('normal')
    
    def on_calibration_point(self, point: Tuple[int, int]):
        self.calibration_points.append(point)
        n = len(self.calibration_points)
        if n == 1:
            self.btn_calibrate.setText("点击第二个校准点...")
            self.lbl_calibration_status.setText(f"已选第1点: {point}")
        elif n == 2:
            p1, p2 = self.calibration_points
            dist_px = int(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))
            self.lbl_calibration_status.setText(f"已选第2点: {point}, 像素距离: {dist_px}px")
            self.btn_calibrate.setText("校准完成!")
            self.statusBar().showMessage(f"校准点: {p1} -> {p2}, 距离: {dist_px}px")
            self.calibration_done = True
            self.calibration_mode = False
            self.image_viewer.set_mode('normal')
    
    def run_detection(self):
        if self.recognizer is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        image_paths = []
        
        if self.current_image_path and self.current_image is not None:
            image_paths = [self.current_image_path]
        else:
            folder = QFileDialog.getExistingDirectory(self, "选择图片目录")
            if folder:
                folder_path = Path(folder)
                image_paths = list(folder_path.glob('*.jpg')) + list(folder_path.glob('*.png'))
        
        if not image_paths:
            QMessageBox.warning(self, "警告", "请先打开图片或选择图片目录")
            return
        
        patch_size = int(self.cmb_detect_patch.currentText())
        stride = self.spin_detect_stride.value()
        
        self.detect_progress.setVisible(True)
        self.detect_progress.setValue(0)
        self.btn_run_detection.setEnabled(False)
        self.statusBar().showMessage("正在识别...")
        
        self.processing_thread = ProcessingThread('detection', {
            'image_paths': image_paths,
            'model_path': str(self.model_path) if hasattr(self, 'model_path') else None,
            'patch_size': patch_size,
            'stride': stride
        })
        
        self.processing_thread.progress_updated.connect(self.detect_progress.setValue)
        self.processing_thread.finished.connect(self.on_detection_complete)
        self.processing_thread.error_occurred.connect(self.on_error)
        
        self.processing_thread.start()
    
    def select_training_data(self):
        data_dir = QFileDialog.getExistingDirectory(self, "选择数据目录")
        
        if not data_dir:
            return
        
        self.training_data_path = Path(data_dir)
        
        categories = ['water', 'beach', 'boundary', 'dam']
        counts = {}
        total = 0
        
        for cat in categories:
            cat_dir = self.training_data_path / cat
            if cat_dir.exists():
                count = len(list(cat_dir.glob('*.jpg'))) + len(list(cat_dir.glob('*.png')))
                counts[cat] = count
                total += count
        
        if total == 0:
            QMessageBox.warning(self, "警告", "数据目录为空，请检查目录结构")
            self.training_data_path = None
            self.lbl_data_path.setText("未选择数据目录")
            self.btn_start_train.setEnabled(False)
            return
        
        self.lbl_data_path.setText(f"已选择: {self.training_data_path.name} (共{total}张)")
        self.lbl_model_info.setText(
            f"类别统计: 水面{counts.get('water',0)} 滩面{counts.get('beach',0)} "
            f"分界线{counts.get('boundary',0)} 坝体{counts.get('dam',0)}"
        )
        self.btn_start_train.setEnabled(True)
    
    def train_model(self):
        if not self.training_data_path:
            QMessageBox.warning(self, "警告", "请先选择数据目录")
            return
        
        model_save_dir = QFileDialog.getExistingDirectory(self, "选择模型保存目录")
        
        if not model_save_dir:
            return
        
        self.training_progress.setVisible(True)
        self.training_progress.setMaximum(100)
        self.training_progress.setValue(0)
        self.btn_start_train.setEnabled(False)
        self.statusBar().showMessage("正在训练模型...")
        
        self.processing_thread = ProcessingThread('training', {
            'data_path': self.training_data_path,
            'epochs': self.spin_epochs.value(),
            'model_save': Path(model_save_dir)
        })
        
        self.processing_thread.progress_updated.connect(self.training_progress.setValue)
        self.processing_thread.finished.connect(lambda x: self.on_training_complete(x, model_save_dir))
        self.processing_thread.error_occurred.connect(self.on_error)
        
        self.processing_thread.start()
    
    def on_training_complete(self, results: list, model_save_dir: str):
        self.training_progress.setVisible(False)
        self.btn_start_train.setEnabled(True)
        model_path = results[0] if results else None
        self.statusBar().showMessage(f"训练完成: {model_path}")
        if model_path:
            QMessageBox.information(self, "完成", f"模型已保存到:\n{model_path}")
    
    def on_processing_complete(self, results: list, message: str):
        self.statusBar().showMessage(f"{message} - {len(results)} 个文件")
    
    def on_detection_complete(self, results: list):
        self.detect_progress.setVisible(False)
        self.btn_run_detection.setEnabled(True)
        
        self.statusBar().showMessage(f"识别完成: {len(results)} 张图片")
        
        if results:
            first_result = results[0]
            if 'annotated' in first_result:
                self.annotated_image = first_result['annotated']
                self.image_viewer.set_image(self.annotated_image)
                
                counts = first_result.get('class_counts', {})
                count_text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
                self.lbl_detect_result.setText(count_text)
            else:
                result_img = cv2.imread(str(first_result))
                if result_img is not None:
                    self.annotated_image = result_img
                    self.image_viewer.set_image(result_img)
    
    def on_error(self, error_msg: str):
        self.statusBar().showMessage("处理出错")
        QMessageBox.critical(self, "错误", error_msg)
    
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
    
    def clear_results(self):
        self.image_viewer.clear_viewer()
        self.statusBar().showMessage("就绪")


def launch_gui():
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DryBeachGUI()
    window.show()
    
    sys.exit(app.exec())
