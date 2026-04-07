from pathlib import Path

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QLabel, QFileDialog, QSpinBox, QProgressBar, QMessageBox,
                             QDoubleSpinBox, QComboBox, QGroupBox, QGridLayout)
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
        
        self.param_group = QGroupBox("训练参数 (点击展开)")
        self.param_group.setCheckable(True)
        self.param_group.setChecked(False)
        param_layout = QGridLayout()
        
        param_layout.addWidget(QLabel("训练轮数:"), 0, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(50)
        param_layout.addWidget(self.spin_epochs, 0, 1)
        
        param_layout.addWidget(QLabel("批量大小:"), 0, 2)
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(16)
        param_layout.addWidget(self.spin_batch, 0, 3)
        
        param_layout.addWidget(QLabel("图像尺寸:"), 1, 0)
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 512)
        self.spin_imgsz.setValue(64)
        self.spin_imgsz.setSingleStep(32)
        param_layout.addWidget(self.spin_imgsz, 1, 1)
        
        param_layout.addWidget(QLabel("学习率:"), 1, 2)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 0.1)
        self.spin_lr.setValue(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setSingleStep(0.0001)
        param_layout.addWidget(self.spin_lr, 1, 3)
        
        param_layout.addWidget(QLabel("优化器:"), 2, 0)
        self.cmb_optimizer = QComboBox()
        self.cmb_optimizer.addItems(["Adam", "SGD", "AdamW"])
        self.cmb_optimizer.setCurrentText("Adam")
        param_layout.addWidget(self.cmb_optimizer, 2, 1)
        
        param_layout.addWidget(QLabel("早停轮数:"), 2, 2)
        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(0, 50)
        self.spin_patience.setValue(15)
        param_layout.addWidget(self.spin_patience, 2, 3)
        
        param_layout.addWidget(QLabel("动量:"), 3, 0)
        self.spin_momentum = QDoubleSpinBox()
        self.spin_momentum.setRange(0.0, 0.99)
        self.spin_momentum.setValue(0.9)
        self.spin_momentum.setDecimals(2)
        param_layout.addWidget(self.spin_momentum, 3, 1)
        
        param_layout.addWidget(QLabel("权重衰减:"), 3, 2)
        self.spin_weight_decay = QDoubleSpinBox()
        self.spin_weight_decay.setRange(0.0, 0.01)
        self.spin_weight_decay.setValue(0.0001)
        self.spin_weight_decay.setDecimals(5)
        param_layout.addWidget(self.spin_weight_decay, 3, 3)
        
        param_layout.addWidget(QLabel("Dropout:"), 4, 0)
        self.spin_dropout = QDoubleSpinBox()
        self.spin_dropout.setRange(0.0, 0.8)
        self.spin_dropout.setValue(0.5)
        self.spin_dropout.setDecimals(2)
        self.spin_dropout.setSingleStep(0.05)
        param_layout.addWidget(self.spin_dropout, 4, 1)
        
        param_layout.addWidget(QLabel("LR衰减:"), 4, 2)
        self.spin_lr_decay = QDoubleSpinBox()
        self.spin_lr_decay.setRange(0.1, 0.99)
        self.spin_lr_decay.setValue(0.5)
        self.spin_lr_decay.setDecimals(2)
        param_layout.addWidget(self.spin_lr_decay, 4, 3)
        
        self.param_group.setLayout(param_layout)
        layout.addWidget(self.param_group)
        
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
            'batch_size': self.spin_batch.value(),
            'image_size': self.spin_imgsz.value(),
            'learning_rate': self.spin_lr.value(),
            'optimizer': self.cmb_optimizer.currentText(),
            'patience': self.spin_patience.value(),
            'momentum': self.spin_momentum.value(),
            'weight_decay': self.spin_weight_decay.value(),
            'dropout': self.spin_dropout.value(),
            'lr_decay': self.spin_lr_decay.value(),
            'model_save': Path(model_save_dir)
        })
    
    def on_training_complete(self, model_path):
        self.training_progress.setVisible(False)
        self.btn_start_train.setEnabled(True)
        if model_path:
            QMessageBox.information(self, "完成", f"模型已保存到:\n{model_path}")
    
    def set_progress(self, value):
        self.training_progress.setValue(value)
