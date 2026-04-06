import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QPushButton, QLabel, QFileDialog,
                                 QSpinBox, QDoubleSpinBox, QGroupBox, QTextEdit,
                                 QProgressBar, QCheckBox, QComboBox, QMessageBox,
                                 QSizePolicy, QScrollArea, QRadioButton, QListWidget,
                                 QListWidgetItem, QAbstractItemView)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
    from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logging.warning("PyQt6 not available, GUI disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #2a2a2a;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
    def set_image(self, image: np.ndarray):
        if image is None:
            return
        
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, c = rgb_image.shape
        bytes_per_line = c * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        
    def clear_viewer(self):
        self.clear()


class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    
    def __init__(self, task_type: str, params: dict):
        super().__init__()
        self.task_type = task_type
        self.params = params
    
    def run(self):
        try:
            if self.task_type == 'video_extract':
                self._process_video_extraction()
            elif self.task_type == 'detection':
                self._process_detection()
            elif self.task_type == 'training':
                self._process_training()
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _process_video_extraction(self):
        from .video_capture import VideoFrameExtractor
        
        video_path = self.params['video_path']
        output_dir = Path(self.params['output_dir'])
        interval = self.params['interval']
        
        frames = []
        with VideoFrameExtractor(video_path) as extractor:
            total = max(1, extractor.total_frames // interval)
            for i, frame_num in enumerate(range(0, extractor.total_frames, interval)):
                frame = extractor.extract_frame(frame_num)
                if frame is not None:
                    frames.append((frame_num, frame))
                    self.progress_updated.emit(int((i + 1) / total * 100))
        
        extractor_save = VideoFrameExtractor(video_path)
        saved_paths = extractor_save.save_frames(frames, output_dir)
        
        self.finished.emit(saved_paths)
    
    def _process_detection(self):
        from .recognizer import DryBeachRecognizer
        from .image_annotator import RegionOfInterest
        
        image_paths = self.params['image_paths']
        output_dir = Path(self.params['output_dir'])
        calibration = self.params.get('calibration')
        
        recognizer = DryBeachRecognizer()
        
        if calibration:
            recognizer.calibrate(calibration['distance'], calibration['points'])
        
        results = []
        for i, img_path in enumerate(image_paths):
            image = cv2.imread(str(img_path))
            if image is not None:
                result, annotated = recognizer.detect_and_visualize(image)
                
                output_path = output_dir / f"result_{img_path.stem}.jpg"
                cv2.imwrite(str(output_path), annotated)
                results.append(output_path)
                
                self.progress_updated.emit(int((i + 1) / len(image_paths) * 100))
        
        self.finished.emit(results)
    
    def _process_training(self):
        from .model_trainer import ModelTrainer
        
        config_path = self.params['config_path']
        epochs = self.params['epochs']
        model_save = Path(self.params['model_save'])
        
        trainer = ModelTrainer(model_save_path=model_save)
        model_path = trainer.train_with_yolo(config_path, epochs=epochs)
        
        self.finished.emit([model_path])


class VideoExtractThread(QThread):
    progress_updated = pyqtSignal(int)
    frame_extracted = pyqtSignal(int, object)
    finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path: str, mode: str, value: int, output_dir: Path):
        super().__init__()
        self.video_path = video_path
        self.mode = mode
        self.value = value
        self.output_dir = output_dir
    
    def run(self):
        try:
            from .video_capture import VideoFrameExtractor
            
            saved_paths = []
            with VideoFrameExtractor(self.video_path) as extractor:
                if self.mode == 'count':
                    frames = extractor.extract_frames_by_count(self.value)
                else:
                    frames = extractor.extract_frames_at_interval(self.value)
                
                total = len(frames)
                for i, (frame_num, frame) in enumerate(frames):
                    self.frame_extracted.emit(frame_num, frame)
                    self.progress_updated.emit(int((i + 1) / total * 100))
                
                saved_paths = extractor.save_frames(frames, self.output_dir)
            
            self.finished.emit(saved_paths)
        except Exception as e:
            self.error_occurred.emit(str(e))


class VideoExtractWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = None
        self.video_info = {}
        self.extract_thread = None
        self.extracted_frames = []
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        btn_layout = QHBoxLayout()
        self.btn_open_video = QPushButton("打开视频")
        self.btn_open_video.clicked.connect(self.open_video)
        btn_layout.addWidget(self.btn_open_video)
        
        self.btn_save_dir = QPushButton("选择保存目录")
        self.btn_save_dir.clicked.connect(self.select_save_dir)
        btn_layout.addWidget(self.btn_save_dir)
        layout.addLayout(btn_layout)
        
        self.lbl_video_info = QLabel("未加载视频")
        self.lbl_video_info.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.lbl_video_info)
        
        mode_layout = QHBoxLayout()
        self.radio_count = QRadioButton("按张数")
        self.radio_count.setChecked(True)
        self.radio_count.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.radio_count)
        
        self.radio_interval = QRadioButton("按时间间隔")
        self.radio_interval.toggled.connect(self.on_mode_changed)
        mode_layout.addWidget(self.radio_interval)
        layout.addLayout(mode_layout)
        
        input_layout = QHBoxLayout()
        
        self.lbl_count = QLabel("提取张数:")
        self.spin_count = QSpinBox()
        self.spin_count.setRange(1, 10000)
        self.spin_count.setValue(10)
        input_layout.addWidget(self.lbl_count)
        input_layout.addWidget(self.spin_count)
        
        self.lbl_interval = QLabel("时间间隔(秒):")
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.1, 3600)
        self.spin_interval.setValue(1.0)
        self.spin_interval.setSingleStep(0.5)
        self.spin_interval.setSuffix(" s")
        self.lbl_interval.hide()
        self.spin_interval.hide()
        input_layout.addWidget(self.lbl_interval)
        input_layout.addWidget(self.spin_interval)
        
        layout.addLayout(input_layout)
        
        self.btn_extract = QPushButton("提取帧")
        self.btn_extract.clicked.connect(self.extract_frames)
        self.btn_extract.setEnabled(False)
        layout.addWidget(self.btn_extract)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        lbl_preview = QLabel("提取预览:")
        lbl_preview.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_preview)
        
        self.list_thumbnails = QListWidget()
        self.list_thumbnails.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_thumbnails.setIconSize(QSize(120, 90))
        self.list_thumbnails.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_thumbnails.setMaximumHeight(200)
        self.list_thumbnails.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.list_thumbnails)
        
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #888;")
        layout.addWidget(self.lbl_status)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def on_mode_changed(self):
        if self.radio_count.isChecked():
            self.lbl_count.show()
            self.spin_count.show()
            self.lbl_interval.hide()
            self.spin_interval.hide()
        else:
            self.lbl_count.hide()
            self.spin_count.hide()
            self.lbl_interval.show()
            self.spin_interval.show()
    
    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.flv)"
        )
        
        if file_path:
            self.video_path = file_path
            self.load_video_info()
    
    def load_video_info(self):
        from .video_capture import VideoFrameExtractor
        
        try:
            with VideoFrameExtractor(self.video_path) as extractor:
                self.video_info = {
                    'width': extractor.width,
                    'height': extractor.height,
                    'fps': extractor.fps,
                    'total_frames': extractor.total_frames,
                    'duration': extractor.duration
                }
            
            info_text = (f"分辨率: {self.video_info['width']}x{self.video_info['height']} | "
                        f"FPS: {self.video_info['fps']:.2f} | "
                        f"帧数: {self.video_info['total_frames']} | "
                        f"时长: {self.video_info['duration']:.2f}s")
            self.lbl_video_info.setText(info_text)
            self.btn_extract.setEnabled(True)
            self.list_thumbnails.clear()
            self.extracted_frames = []
            
        except Exception as e:
            self.lbl_video_info.setText(f"加载失败: {str(e)}")
            self.btn_extract.setEnabled(False)
    
    def select_save_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if dir_path:
            self.save_dir = Path(dir_path)
            self.btn_save_dir.setText(f"保存至: {self.save_dir.name}")
    
    def extract_frames(self):
        if not self.video_path:
            QMessageBox.warning(self, "警告", "请先打开视频文件")
            return
        
        if not hasattr(self, 'save_dir'):
            self.select_save_dir()
            if not hasattr(self, 'save_dir'):
                return
        
        mode = 'count' if self.radio_count.isChecked() else 'interval'
        value = self.spin_count.value() if mode == 'count' else int(self.spin_interval.value() * self.video_info['fps'])
        
        self.progress_bar.setValue(0)
        self.btn_extract.setEnabled(False)
        self.list_thumbnails.clear()
        self.extracted_frames = []
        
        self.extract_thread = VideoExtractThread(
            self.video_path, mode, value, self.save_dir
        )
        self.extract_thread.progress_updated.connect(self.progress_bar.setValue)
        self.extract_thread.frame_extracted.connect(self.add_thumbnail)
        self.extract_thread.finished.connect(self.on_extraction_complete)
        self.extract_thread.error_occurred.connect(self.on_error)
        self.extract_thread.start()
    
    def add_thumbnail(self, frame_num: int, frame: np.ndarray):
        self.extracted_frames.append((frame_num, frame))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        bytes_per_line = 3 * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        icon = pixmap.scaled(120, 90, Qt.AspectRatioMode.KeepAspectRatio, 
                            Qt.TransformationMode.SmoothTransformation)
        
        item = QListWidgetItem()
        item.setIcon(icon)
        item.setText(f"帧 {frame_num}")
        item.setToolTip(f"帧号: {frame_num}")
        self.list_thumbnails.addItem(item)
        
        self.lbl_status.setText(f"已提取 {len(self.extracted_frames)} 帧")
    
    def on_extraction_complete(self, saved_paths: List[Path]):
        self.progress_bar.setValue(100)
        self.btn_extract.setEnabled(True)
        self.lbl_status.setText(f"提取完成! 已保存 {len(saved_paths)} 帧到 {self.save_dir}")
        QMessageBox.information(self, "完成", f"已提取 {len(saved_paths)} 帧\n保存至: {self.save_dir}")
    
    def on_error(self, error_msg: str):
        self.btn_extract.setEnabled(True)
        self.lbl_status.setText(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)


class DryBeachGUI(QMainWindow if PYQT_AVAILABLE else object):
    def __init__(self):
        if not PYQT_AVAILABLE:
            raise RuntimeError("PyQt6 is not available. Please install it to use the GUI.")
        
        super().__init__()
        self.current_image = None
        self.current_image_path = None
        self.roi = None
        self.roi_start = None
        self.roi_end = None
        self.roi_mode = False
        self.calibration_done = False
        self.calibration_points = []
        
        self.processing_thread = None
        
        self.init_ui()
        
        logger.info("GUI initialized")
    
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
        
        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)
    
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
        
        self.video_extract_widget = VideoExtractWidget()
        layout.addWidget(self.video_extract_widget)
        
        line_sep = QLabel("<hr>")
        line_sep.setStyleSheet("color: #888;")
        layout.addWidget(line_sep)
        
        lbl_main = QLabel("主控制")
        lbl_main.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_main)
        
        btn_load_image = QPushButton("打开图片")
        btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(btn_load_image)
        
        line1 = QLabel("<hr>")
        line1.setStyleSheet("color: #888;")
        layout.addWidget(line1)
        
        lbl_recognition = QLabel("识别设置")
        lbl_recognition.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_recognition)
        
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("识别方法:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(['multi', 'edge', 'color'])
        method_layout.addWidget(self.combo_method)
        layout.addLayout(method_layout)
        
        btn_run_detection = QPushButton("运行识别")
        btn_run_detection.clicked.connect(self.run_detection)
        layout.addWidget(btn_run_detection)
        
        btn_set_roi = QPushButton("设置ROI区域")
        btn_set_roi.clicked.connect(self.toggle_roi_mode)
        self.btn_set_roi = btn_set_roi
        layout.addWidget(btn_set_roi)
        
        line2 = QLabel("<hr>")
        line2.setStyleSheet("color: #888;")
        layout.addWidget(line2)
        
        lbl_calibration = QLabel("校准设置")
        lbl_calibration.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_calibration)
        
        cal_layout = QHBoxLayout()
        cal_layout.addWidget(QLabel("距离(m):"))
        self.spin_cal_distance = QDoubleSpinBox()
        self.spin_cal_distance.setRange(0.1, 10000)
        self.spin_cal_distance.setValue(10)
        cal_layout.addWidget(self.spin_cal_distance)
        layout.addLayout(cal_layout)
        
        btn_calibrate = QPushButton("开始校准(点击两点)")
        btn_calibrate.clicked.connect(self.start_calibration)
        self.btn_calibrate = btn_calibrate
        layout.addWidget(btn_calibrate)
        
        line3 = QLabel("<hr>")
        line3.setStyleSheet("color: #888;")
        layout.addWidget(line3)
        
        lbl_training = QLabel("模型训练")
        lbl_training.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_training)
        
        btn_train_model = QPushButton("训练模型")
        btn_train_model.clicked.connect(self.train_model)
        layout.addWidget(btn_train_model)
        
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("训练轮数:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        epochs_layout.addWidget(self.spin_epochs)
        layout.addLayout(epochs_layout)
        
        layout.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.lbl_status)
        
        panel.setLayout(layout)
        return panel
    
    def _create_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.image_viewer = ImageViewer()
        scroll_area.setWidget(self.image_viewer)
        
        layout.addWidget(scroll_area)
        
        return panel
    
    def _create_right_panel(self) -> QGroupBox:
        panel = QGroupBox("识别结果")
        layout = QVBoxLayout()
        
        self.text_results = QTextEdit()
        self.text_results.setReadOnly(True)
        self.text_results.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.text_results)
        
        btn_save_result = QPushButton("保存结果图片")
        btn_save_result.clicked.connect(self.save_result)
        layout.addWidget(btn_save_result)
        
        btn_export_report = QPushButton("导出报告")
        btn_export_report.clicked.connect(self.export_report)
        layout.addWidget(btn_export_report)
        
        btn_clear_results = QPushButton("清除结果")
        btn_clear_results.clicked.connect(self.clear_results)
        layout.addWidget(btn_clear_results)
        
        panel.setLayout(layout)
        return panel
    
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.current_image_path = Path(file_path)
            self.current_image = cv2.imread(file_path)
            
            if self.current_image is not None:
                self.image_viewer.set_image(self.current_image)
                self.lbl_status.setText(f"已加载图片: {self.current_image_path.name}")
                h, w = self.current_image.shape[:2]
                self.text_results.append(f"[INFO] 已加载图片: {file_path} ({w}x{h})")
    
    def toggle_roi_mode(self):
        self.roi_mode = not self.roi_mode
        if self.roi_mode:
            self.btn_set_roi.setText("取消ROI")
            self.text_results.append("[INFO] ROI模式: 点击拖动选择区域")
        else:
            self.btn_set_roi.setText("设置ROI区域")
            self.roi = None
    
    def start_calibration(self):
        self.calibration_mode = True
        self.calibration_points = []
        self.btn_calibrate.setText("点击第一个校准点...")
        self.text_results.append("[INFO] 校准模式: 请点击两个已知距离的点")
    
    def run_detection(self):
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
        
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录")
        
        if not output_dir:
            return
        
        self.progress_bar.setValue(0)
        self.lbl_status.setText("正在识别...")
        
        calibration_data = None
        if self.calibration_done:
            calibration_data = {
                'distance': self.spin_cal_distance.value(),
                'points': [(0, 100), (100, 100)]
            }
        
        self.processing_thread = ProcessingThread('detection', {
            'image_paths': image_paths,
            'output_dir': output_dir,
            'calibration': calibration_data
        })
        
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.finished.connect(lambda x: self.on_detection_complete(x, output_dir))
        self.processing_thread.error_occurred.connect(self.on_error)
        
        self.processing_thread.start()
    
    def train_model(self):
        config_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集配置文件", "", "YAML Files (*.yaml *.yml)"
        )
        
        if not config_path:
            return
        
        model_save_dir = QFileDialog.getExistingDirectory(self, "选择模型保存目录")
        
        if not model_save_dir:
            return
        
        self.progress_bar.setValue(0)
        self.lbl_status.setText("正在训练模型...")
        
        self.processing_thread = ProcessingThread('training', {
            'config_path': config_path,
            'epochs': self.spin_epochs.value(),
            'model_save': Path(model_save_dir) / 'best.pt'
        })
        
        self.processing_thread.progress_updated.connect(self.progress_bar.setValue)
        self.processing_thread.finished.connect(lambda x: self.on_processing_complete(x, "训练完成"))
        self.processing_thread.error_occurred.connect(self.on_error)
        
        self.processing_thread.start()
    
    def on_processing_complete(self, results: list, message: str):
        self.progress_bar.setValue(100)
        self.lbl_status.setText(message)
        self.text_results.append(f"\n[SUCCESS] {message}")
        self.text_results.append(f"处理了 {len(results)} 个文件")
    
    def on_detection_complete(self, results: list, output_dir: str):
        self.progress_bar.setValue(100)
        self.lbl_status.setText("识别完成")
        self.text_results.append(f"\n[SUCCESS] 识别完成")
        self.text_results.append(f"处理了 {len(results)} 张图片")
        self.text_results.append(f"结果保存在: {output_dir}")
        
        if results:
            result_img = cv2.imread(str(results[0]))
            if result_img is not None:
                self.image_viewer.set_image(result_img)
    
    def on_error(self, error_msg: str):
        self.lbl_status.setText("处理出错")
        self.text_results.append(f"\n[ERROR] {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)
    
    def save_result(self):
        if self.image_viewer.pixmap() is None:
            QMessageBox.warning(self, "警告", "没有可保存的图片")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果图片", "", "JPEG Files (*.jpg)"
        )
        
        if file_path:
            pixmap = self.image_viewer.pixmap()
            pixmap.save(file_path)
            self.text_results.append(f"[INFO] 结果已保存: {file_path}")
    
    def export_report(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", "", "Text Files (*.txt)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.text_results.toPlainText())
            self.text_results.append(f"[INFO] 报告已导出: {file_path}")
    
    def clear_results(self):
        self.text_results.clear()
        self.image_viewer.clear_viewer()
        self.lbl_status.setText("就绪")


def launch_gui():
    if not PYQT_AVAILABLE:
        print("PyQt6 is not available. Please install it: pip install PyQt6")
        return None
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = DryBeachGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    launch_gui()
