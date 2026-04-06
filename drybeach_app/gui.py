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
                                 QListWidgetItem, QAbstractItemView, QTabWidget)
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
    from PyQt6.QtGui import QImage, QPixmap, QAction, QIcon, QPainter, QPen, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logging.warning("PyQt6 not available, GUI disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageViewer(QLabel):
    roi_selected = pyqtSignal(tuple)
    calibration_point_clicked = pyqtSignal(tuple)
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #2a2a2a;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        
        self.mode = 'normal'
        self.drag_start = None
        self.drag_end = None
        self.calibration_points = []
        self.roi_rect = None
        
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.displayed_size = QSize()
    
    def set_mode(self, mode: str):
        self.mode = mode
        self.calibration_points = []
        self.roi_rect = None
        self.drag_start = None
        self.drag_end = None
        self.update_overlay()
    
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
        self.current_pixmap = pixmap
        
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        self.displayed_size = scaled_pixmap.size()
        
        self.offset_x = (self.size().width() - self.displayed_size.width()) // 2
        self.offset_y = (self.size().height() - self.displayed_size.height()) // 2
        
        self.scale_factor = w / self.displayed_size.width() if self.displayed_size.width() > 0 else 1.0
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.setPixmap(scaled_pixmap)
        self.current_image = image
    
    def _create_overlay(self) -> Optional[QPixmap]:
        if self.displayed_size.isEmpty():
            return None
        
        overlay = QPixmap(self.displayed_size)
        overlay.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(overlay)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        
        if self.mode == 'roi' and self.drag_start and self.drag_end:
            x1 = min(self.drag_start.x(), self.drag_end.x())
            y1 = min(self.drag_start.y(), self.drag_end.y())
            x2 = max(self.drag_start.x(), self.drag_end.x())
            y2 = max(self.drag_start.y(), self.drag_end.y())
            
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            
            text = f"ROI: ({int(x1*self.scale_factor)},{int(y1*self.scale_factor)})-({int(x2*self.scale_factor)},{int(y2*self.scale_factor)})"
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(x1 + 5, y1 + 15, text)
        
        elif self.mode == 'calibration' and self.calibration_points:
            for i, pt in enumerate(self.calibration_points):
                painter.setPen(QPen(Qt.GlobalColor.yellow, 3))
                painter.drawEllipse(int(pt.x()) - 5, int(pt.y()) - 5, 10, 10)
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(int(pt.x()) + 8, int(pt.y()) + 8, f"P{i+1}")
            
            if len(self.calibration_points) == 2:
                p1 = self.calibration_points[0]
                p2 = self.calibration_points[1]
                painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
                painter.drawLine(int(p1.x()), int(p1.y()), int(p2.x()), int(p2.y()))
                
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                dist = int(np.sqrt(dx*dx + dy*dy) * self.scale_factor)
                mid_x = int((p1.x() + p2.x()) / 2)
                mid_y = int((p1.y() + p2.y()) / 2)
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(mid_x + 5, mid_y - 5, f"{dist}px")
        
        painter.end()
        return overlay
    
    def update_overlay(self):
        if self.current_pixmap is None:
            return
        
        scaled_pixmap = self.current_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                                  Qt.TransformationMode.SmoothTransformation)
        self.displayed_size = scaled_pixmap.size()
        self.offset_x = (self.size().width() - self.displayed_size.width()) // 2
        self.offset_y = (self.size().height() - self.displayed_size.height()) // 2
        self.scale_factor = self.current_pixmap.width() / self.displayed_size.width() if self.displayed_size.width() > 0 else 1.0
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.setPixmap(scaled_pixmap)
    
    def _widget_to_image(self, widget_x: int, widget_y: int) -> Tuple[int, int]:
        img_x = int((widget_x - self.offset_x) * self.scale_factor)
        img_y = int((widget_y - self.offset_y) * self.scale_factor)
        
        img_w = self.current_pixmap.width() if self.current_pixmap else 0
        img_h = self.current_pixmap.height() if self.current_pixmap else 0
        
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))
        
        return img_x, img_y
    
    def mousePressEvent(self, event):
        if self.current_pixmap is None:
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == 'roi':
                self.drag_start = event.position().toPoint()
                self.drag_end = self.drag_start
            elif self.mode == 'calibration':
                if len(self.calibration_points) < 2:
                    self.calibration_points.append(event.position().toPoint())
                    self.update_overlay()
                    img_x, img_y = self._widget_to_image(
                        event.position().x(), event.position().y()
                    )
                    self.calibration_point_clicked.emit((img_x, img_y))
    
    def mouseMoveEvent(self, event):
        if self.current_pixmap is None or self.mode != 'roi':
            return
        
        if self.drag_start is not None:
            self.drag_end = event.position().toPoint()
            self.update_overlay()
    
    def mouseReleaseEvent(self, event):
        if self.current_pixmap is None or self.mode != 'roi':
            return
        
        if event.button() == Qt.MouseButton.LeftButton and self.drag_start is not None:
            self.drag_end = event.position().toPoint()
            
            x1 = min(self.drag_start.x(), self.drag_end.x())
            y1 = min(self.drag_start.y(), self.drag_end.y())
            x2 = max(self.drag_start.x(), self.drag_end.x())
            y2 = max(self.drag_start.y(), self.drag_end.y())
            
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                ix1, iy1 = self._widget_to_image(x1, y1)
                ix2, iy2 = self._widget_to_image(x2, y2)
                
                self.roi_rect = (min(ix1, ix2), min(iy1, iy2), 
                                abs(ix2 - ix1), abs(iy2 - iy1))
                self.roi_selected.emit(self.roi_rect)
            
            self.drag_start = None
            self.drag_end = None
    
    def clear_viewer(self):
        self.current_image = None
        self.current_pixmap = None
        self.calibration_points = []
        self.roi_rect = None
        self.drag_start = None
        self.drag_end = None
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
        roi_data = self.params.get('roi')
        method = self.params.get('method', 'multi')
        
        recognizer = DryBeachRecognizer()
        
        if calibration:
            recognizer.calibrate(calibration['distance'], calibration['points'])
        
        roi = None
        if roi_data:
            x, y, w, h = roi_data
            roi = RegionOfInterest(x, y, w, h)
        
        results = []
        for i, img_path in enumerate(image_paths):
            image = cv2.imread(str(img_path))
            if image is not None:
                result, annotated = recognizer.detect_and_visualize(image, roi=roi, method=method)
                
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
    
    def __init__(self, video_path: str, mode: str, value: int, output_dir: Path, video_info: dict):
        super().__init__()
        self.video_path = video_path
        self.mode = mode
        self.value = value
        self.output_dir = output_dir
        self.video_info = video_info
    
    def run(self):
        import subprocess
        import tempfile
        import shutil
        
        try:
            video_path = Path(self.video_path)
            output_dir = self.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            temp_dir = Path(tempfile.mkdtemp(prefix='drybeach_'))
            
            try:
                saved_paths = []
                fps = self.video_info.get('fps', 25)
                total_frames = self.video_info.get('total_frames', 0)
                
                if self.mode == 'count':
                    frame_interval = total_frames // self.value if total_frames > 0 else 1
                    frame_interval = max(1, frame_interval)
                    select_filter = rf"not(mod(n\,{frame_interval}))"
                    cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-vf', rf"select='{select_filter}'",
                        '-vsync', '0',
                        '-q:v', '2',
                        '-frames:v', str(self.value),
                        '-y',
                        str(temp_dir / "frame_%06d.jpg")
                    ]
                else:
                    seconds_interval = self.value
                    frame_interval = int(seconds_interval * fps)
                    frame_interval = max(1, frame_interval)
                    select_filter = rf"not(mod(n\,{frame_interval}))"
                    cmd = [
                        'ffmpeg',
                        '-i', str(video_path),
                        '-vf', rf"select='{select_filter}',scale=iw/2:ih/2",
                        '-vsync', '0',
                        '-q:v', '2',
                        '-y',
                        str(temp_dir / "frame_%06d.jpg")
                    ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    error_msg = f"FFmpeg错误: {stderr[-500:]}"
                    self.error_occurred.emit(error_msg)
                    return
                
                temp_files = sorted(temp_dir.glob("frame_*.jpg"))
                
                if not temp_files:
                    self.error_occurred.emit("未提取到任何帧，请检查视频文件")
                    return
                
                for i, temp_file in enumerate(temp_files):
                    dest_file = output_dir / f"{video_path.stem}_{temp_file.name}"
                    shutil.copy2(temp_file, dest_file)
                    saved_paths.append(dest_file)
                    
                    frame_num = i * frame_interval if self.mode == 'count' else int(i * seconds_interval * fps)
                    self.frame_extracted.emit(frame_num, str(dest_file))
                    self.progress_updated.emit(int((i + 1) / len(temp_files) * 100))
                
                self.finished.emit(saved_paths)
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
            
        except FileNotFoundError:
            self.error_occurred.emit("未找到FFmpeg，请确保已安装并添加到PATH环境变量")
        except Exception as e:
            self.error_occurred.emit(str(e))


class VideoExtractWidget(QWidget):
    thumbnail_clicked = pyqtSignal(str)
    
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
        
        lbl_preview = QLabel("提取预览:")
        lbl_preview.setStyleSheet("font-weight: bold;")
        layout.addWidget(lbl_preview)
        
        self.list_thumbnails = QListWidget()
        self.list_thumbnails.setViewMode(QListWidget.ViewMode.IconMode)
        self.list_thumbnails.setIconSize(QSize(120, 90))
        self.list_thumbnails.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list_thumbnails.setMaximumHeight(200)
        self.list_thumbnails.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_thumbnails.itemClicked.connect(self.on_thumbnail_clicked)
        self._thumbnail_paths = []
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
        import subprocess
        import json
        
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
                '-of', 'json',
                self.video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise Exception(result.stderr)
            
            data = json.loads(result.stdout)
            stream = data.get('streams', [{}])[0]
            
            fps_str = stream.get('r_frame_rate', '25/1')
            if '/' in fps_str:
                fps = eval(fps_str)
            else:
                fps = float(fps_str)
            
            duration = float(stream.get('duration', 0))
            nb_frames = stream.get('nb_frames')
            
            if nb_frames is None and duration > 0:
                nb_frames = int(duration * fps)
            
            self.video_info = {
                'width': stream.get('width', 0),
                'height': stream.get('height', 0),
                'fps': fps,
                'total_frames': int(nb_frames) if nb_frames else 0,
                'duration': duration
            }
            
            info_text = (f"分辨率: {self.video_info['width']}x{self.video_info['height']} | "
                        f"FPS: {self.video_info['fps']:.2f} | "
                        f"帧数: {self.video_info['total_frames']} | "
                        f"时长: {self.video_info['duration']:.2f}s")
            self.lbl_video_info.setText(info_text)
            self.btn_extract.setEnabled(True)
            self.list_thumbnails.clear()
            self.extracted_frames = []
            
        except subprocess.TimeoutExpired:
            self.lbl_video_info.setText("获取视频信息超时")
            self.btn_extract.setEnabled(False)
        except Exception as e:
            error_msg = str(e)
            self.lbl_video_info.setText(f"加载失败: {error_msg[:50]}")
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
        
        self.btn_extract.setEnabled(False)
        self.list_thumbnails.clear()
        self.extracted_frames = []
        self._thumbnail_paths = []
        
        self.extract_thread = VideoExtractThread(
            self.video_path, mode, value, self.save_dir, self.video_info
        )
        self.extract_thread.frame_extracted.connect(self.add_thumbnail)
        self.extract_thread.finished.connect(self.on_extraction_complete)
        self.extract_thread.error_occurred.connect(self.on_error)
        self.extract_thread.start()
    
    def add_thumbnail(self, frame_num: int, frame_data):
        self.extracted_frames.append((frame_num, frame_data))
        
        if isinstance(frame_data, str):
            pixmap = QPixmap(frame_data)
            self._thumbnail_paths.append(frame_data)
        else:
            rgb_frame = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]
            bytes_per_line = 3 * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self._thumbnail_paths.append(None)
        
        icon_pixmap = pixmap.scaled(120, 90, Qt.AspectRatioMode.KeepAspectRatio, 
                            Qt.TransformationMode.SmoothTransformation)
        icon = QIcon(icon_pixmap)
        
        item = QListWidgetItem()
        item.setIcon(icon)
        item.setText(f"帧 {frame_num}")
        item.setToolTip(f"帧号: {frame_num}")
        self.list_thumbnails.addItem(item)
        
        self.lbl_status.setText(f"已提取 {len(self.extracted_frames)} 帧")
    
    def on_thumbnail_clicked(self, item):
        row = self.list_thumbnails.row(item)
        if row < len(self._thumbnail_paths):
            path = self._thumbnail_paths[row]
            if path:
                self.thumbnail_clicked.emit(path)
    
    def on_extraction_complete(self, saved_paths: List[Path]):
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
        self.tabs.addTab(self.video_extract_widget, "视频提取")
        
        detection_tab = self._create_detection_tab()
        self.tabs.addTab(detection_tab, "识别")
        
        calibration_tab = self._create_calibration_tab()
        self.tabs.addTab(calibration_tab, "校准")
        
        training_tab = self._create_training_tab()
        self.tabs.addTab(training_tab, "训练")
        
        layout.addWidget(self.tabs)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.lbl_status)
        
        panel.setLayout(layout)
        return panel
    
    def _create_detection_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        
        btn_load_image = QPushButton("打开图片")
        btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(btn_load_image)
        
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
        
        sep = QLabel("<hr>")
        sep.setStyleSheet("color: #888;")
        layout.addWidget(sep)
        
        btn_save_result = QPushButton("保存结果图片")
        btn_save_result.clicked.connect(self.save_result)
        layout.addWidget(btn_save_result)
        
        btn_export_report = QPushButton("导出报告")
        btn_export_report.clicked.connect(self.export_report)
        layout.addWidget(btn_export_report)
        
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
    
    def _create_training_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout()
        
        btn_train_model = QPushButton("选择数据集配置文件")
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
        
        layout.addWidget(scroll_area)
        
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
                h, w = self.current_image.shape[:2]
                self.lbl_status.setText(f"已加载: {self.current_image_path.name} ({w}x{h})")
    
    def load_image_from_path(self, file_path: str):
        self.current_image_path = Path(file_path)
        self.current_image = cv2.imread(file_path)
        
        if self.current_image is not None:
            self.image_viewer.set_image(self.current_image)
            h, w = self.current_image.shape[:2]
            self.lbl_status.setText(f"已加载: {self.current_image_path.name} ({w}x{h})")
    
    def toggle_roi_mode(self):
        self.roi_mode = not self.roi_mode
        if self.roi_mode:
            self.image_viewer.set_mode('roi')
            self.btn_set_roi.setText("取消ROI")
            self.lbl_status.setText("ROI模式: 点击拖动选择区域")
        else:
            self.image_viewer.set_mode('normal')
            self.btn_set_roi.setText("设置ROI区域")
            self.roi = None
            self.lbl_status.setText("就绪")
    
    def start_calibration(self):
        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
        self.image_viewer.set_mode('calibration')
        self.calibration_points = []
        self.btn_calibrate.setText("点击第一个校准点...")
        self.lbl_calibration_status.setText("点击第1点")
        self.lbl_status.setText("校准模式: 请在图像上点击两个校准点")
    
    def on_roi_selected(self, roi_tuple: Tuple[int, int, int, int]):
        self.roi = roi_tuple
        x, y, w, h = roi_tuple
        self.lbl_status.setText(f"ROI已设置: ({x},{y}) {w}x{h}")
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
            self.lbl_status.setText(f"校准点: {p1} -> {p2}, 距离: {dist_px}px")
            self.calibration_done = True
            self.calibration_mode = False
            self.image_viewer.set_mode('normal')
    
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
        if self.calibration_done and len(self.calibration_points) == 2:
            calibration_data = {
                'distance': self.spin_cal_distance.value(),
                'points': self.calibration_points
            }
        
        roi_data = self.roi
        method = self.combo_method.currentText()
        
        self.processing_thread = ProcessingThread('detection', {
            'image_paths': image_paths,
            'output_dir': output_dir,
            'calibration': calibration_data,
            'roi': roi_data,
            'method': method
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
        self.lbl_status.setText(f"{message} - {len(results)} 个文件")
    
    def on_detection_complete(self, results: list, output_dir: str):
        self.progress_bar.setValue(100)
        self.lbl_status.setText(f"识别完成: {len(results)} 张图片")
        
        if results:
            result_img = cv2.imread(str(results[0]))
            if result_img is not None:
                self.image_viewer.set_image(result_img)
    
    def on_error(self, error_msg: str):
        self.lbl_status.setText("处理出错")
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
            self.lbl_status.setText(f"已保存: {Path(file_path).name}")
    
    def export_report(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出报告", "", "Text Files (*.txt)"
        )
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.lbl_status.text())
            self.lbl_status.setText(f"已导出: {Path(file_path).name}")
    
    def clear_results(self):
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
