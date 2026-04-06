from pathlib import Path
from typing import List

import cv2
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QSpinBox, QDoubleSpinBox, QRadioButton,
                             QProgressBar, QListWidget, QListWidgetItem, QAbstractItemView,
                             QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QIcon, QImage

from .threads import VideoExtractThread


class VideoExtractWidget(QWidget):
    thumbnail_clicked = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    
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
        self.list_thumbnails.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_thumbnails.itemClicked.connect(self.on_thumbnail_clicked)
        self._thumbnail_paths = []
        layout.addWidget(self.list_thumbnails)
        
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
        self.progress_bar.setValue(0)
        self.list_thumbnails.clear()
        self.extracted_frames = []
        self._thumbnail_paths = []
        
        self.extract_thread = VideoExtractThread(
            self.video_path, mode, value, self.save_dir, self.video_info
        )
        self.extract_thread.progress_updated.connect(self.progress_bar.setValue)
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
        
        self.status_updated.emit(f"已提取 {len(self.extracted_frames)} 帧")
    
    def on_thumbnail_clicked(self, item):
        row = self.list_thumbnails.row(item)
        if row < len(self._thumbnail_paths):
            path = self._thumbnail_paths[row]
            if path:
                self.thumbnail_clicked.emit(path)
    
    def on_extraction_complete(self, saved_paths: List[Path]):
        self.btn_extract.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_updated.emit(f"提取完成! 已保存 {len(saved_paths)} 帧到 {self.save_dir}")
    
    def on_error(self, error_msg: str):
        self.btn_extract.setEnabled(True)
        self.status_updated.emit(f"错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)
