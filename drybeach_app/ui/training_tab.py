from pathlib import Path

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QSpinBox, QProgressBar, QMessageBox)
from PyQt6.QtCore import pyqtSignal


class TrainingTab(QWidget):
    training_requested = pyqtSignal(dict)
    training_progress = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_data_path = None
        self.init_ui()
    
    def init_ui(self):
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
        self.btn_start_train.clicked.connect(self.start_training)
        self.btn_start_train.setEnabled(False)
        layout.addWidget(self.btn_start_train)
        
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        layout.addWidget(self.training_progress)
        
        self.lbl_model_info = QLabel("")
        self.lbl_model_info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_model_info)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def select_training_data(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据目录")
        
        if not dir_path:
            return
        
        self.training_data_path = Path(dir_path)
        
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
    
    def start_training(self):
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
        
        self.training_requested.emit({
            'data_path': self.training_data_path,
            'epochs': self.spin_epochs.value(),
            'model_save': Path(model_save_dir)
        })
    
    def on_training_complete(self, model_path):
        self.training_progress.setVisible(False)
        self.btn_start_train.setEnabled(True)
        if model_path:
            QMessageBox.information(self, "完成", f"模型已保存到:\n{model_path}")
    
    def set_progress(self, value):
        self.training_progress.setValue(value)
