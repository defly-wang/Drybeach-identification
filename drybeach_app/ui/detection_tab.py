"""
干滩识别系统 - 识别模块UI
图像识别功能界面，包括模型加载、区域设置、识别执行和结果显示

DetectionTab: 识别功能标签页
"""

from pathlib import Path

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QTextCursor

import cv2


class DetectionTab(QWidget):
    """
    识别功能标签页
    
    提供以下功能:
    - 加载CNN分类模型
    - 打开待识别图片
    - 设置识别区域(可选)
    - 配置识别参数(切片尺寸、步长、置信度)
    - 执行图像识别
    - 生成分界线
    - 保存识别结果
    """
    
    # ======== 信号定义 ========
    model_loaded = pyqtSignal(str)              # 模型加载完成时发出，参数为模型路径
    detection_requested = pyqtSignal(dict)      # 请求执行识别时发出，参数为识别参数字典
    image_display_requested = pyqtSignal(str)  # 请求显示图片时发出，参数为图片路径
    result_ready = pyqtSignal(object)           # 识别结果准备好时发出，参数为带标注的图片
    draw_region_requested = pyqtSignal(str)     # 请求绘制区域时发出，参数为图片路径
    region_cleared = pyqtSignal()               # 区域被清除时发出
    region_loaded = pyqtSignal(object)         # 区域被加载时发出，参数为区域数据
    boundary_draw_requested = pyqtSignal()     # 请求绘制分界线时发出
    
    def __init__(self, parent=None):
        """
        初始化识别标签页
        
        Args:
            parent: 父 widget
        """
        super().__init__(parent)
        self.model_path = None           # 当前加载的模型路径
        self.model_info = None            # 模型信息字典
        self.detection_region = None     # 识别区域数据(多边形点列表)
        self.image_path = None           # 当前打开的图片路径
        self.annotated_image = None      # 带标注的识别结果图片
        self.original_result = None      # 原始识别结果(生成边界线前)
        self.detection_points = []        # 识别到的所有检测点列表
        self.boundary_lines = []         # 生成分界线的列表
        
        # 初始化UI
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面组件"""
        layout = QVBoxLayout()
        
        # ======== 模型加载区域 ========
        # 加载模型按钮
        self.btn_load_model = QPushButton("加载模型")
        self.btn_load_model.clicked.connect(self.load_model)
        layout.addWidget(self.btn_load_model)
        
        # 模型路径显示标签
        self.lbl_model_path = QLabel("未加载模型")
        self.lbl_model_path.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_model_path)
        
        # 模型信息显示文本框
        self.txt_model_info = QTextEdit()
        self.txt_model_info.setReadOnly(True)
        self.txt_model_info.setMaximumHeight(120)
        self.txt_model_info.setStyleSheet("background-color: #1e1e1e; color: #aaa; font-size: 11px;")
        layout.addWidget(self.txt_model_info)
        
        # ======== 图片加载区域 ========
        # 打开图片按钮
        self.btn_load_image = QPushButton("打开图片")
        self.btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load_image)
        
        # 图片路径显示标签
        self.lbl_image_path = QLabel("未加载图片")
        self.lbl_image_path.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_image_path)
        
        # ======== 识别区域设置 ========
        region_layout = QHBoxLayout()
        
        # 绘制识别区域按钮
        self.btn_draw_region = QPushButton("画识别区域")
        self.btn_draw_region.clicked.connect(self.draw_detection_region)
        region_layout.addWidget(self.btn_draw_region)
        
        # 清除识别区域按钮
        self.btn_clear_region = QPushButton("清除区域")
        self.btn_clear_region.clicked.connect(self.clear_detection_region)
        region_layout.addWidget(self.btn_clear_region)
        layout.addLayout(region_layout)
        
        # 识别区域显示标签
        self.lbl_region = QLabel("未设置识别区域")
        self.lbl_region.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_region)
        
        # ======== 区域保存/加载 ========
        save_load_layout = QHBoxLayout()
        
        # 保存区域按钮
        self.btn_save_region = QPushButton("保存区域")
        self.btn_save_region.clicked.connect(self.save_region)
        save_load_layout.addWidget(self.btn_save_region)
        
        # 加载区域按钮
        self.btn_load_region = QPushButton("导入区域")
        self.btn_load_region.clicked.connect(self.load_region)
        save_load_layout.addWidget(self.btn_load_region)
        layout.addLayout(save_load_layout)
        
        # ======== 识别参数设置 ========
        param_layout = QHBoxLayout()
        
        # 切片尺寸选择(滑动窗口大小)
        param_layout.addWidget(QLabel("切片尺寸:"))
        self.cmb_detect_patch = QComboBox()
        self.cmb_detect_patch.addItems(["24", "32", "64", "96", "128"])
        self.cmb_detect_patch.setCurrentText("32")
        param_layout.addWidget(self.cmb_detect_patch)
        
        # 步长设置(滑动窗口移动距离)
        param_layout.addWidget(QLabel("步长:"))
        self.spin_detect_stride = QSpinBox()
        self.spin_detect_stride.setRange(2, 64)
        self.spin_detect_stride.setValue(16)
        param_layout.addWidget(self.spin_detect_stride)
        
        # 置信度阈值设置(过滤低置信度结果)
        param_layout.addWidget(QLabel("置信度:"))
        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(0.0, 1.0)
        self.spin_confidence.setValue(0.0)
        self.spin_confidence.setSingleStep(0.05)
        self.spin_confidence.setDecimals(2)
        param_layout.addWidget(self.spin_confidence)
        layout.addLayout(param_layout)
        
        # ======== 识别执行 ========
        # 开始识别按钮
        self.btn_run_detection = QPushButton("开始识别")
        self.btn_run_detection.clicked.connect(self.run_detection)
        self.btn_run_detection.setEnabled(False)  # 未加载模型时禁用
        layout.addWidget(self.btn_run_detection)
        
        # 识别进度条
        self.detect_progress = QProgressBar()
        self.detect_progress.setVisible(False)
        layout.addWidget(self.detect_progress)
        
        # 识别结果显示标签
        self.lbl_detect_result = QLabel("")
        self.lbl_detect_result.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_detect_result)
        
        # ======== 结果保存 ========
        # 保存识别结果按钮
        self.btn_save_result = QPushButton("保存识别结果")
        self.btn_save_result.clicked.connect(self.save_result)
        self.btn_save_result.setEnabled(False)
        layout.addWidget(self.btn_save_result)
        
        # ======== 分界线生成 ========
        # 生成分界线按钮
        self.btn_draw_boundary = QPushButton("生成分界线")
        self.btn_draw_boundary.clicked.connect(self.draw_boundary_line)
        self.btn_draw_boundary.setEnabled(False)
        layout.addWidget(self.btn_draw_boundary)
        
        # 清除分界线按钮
        self.btn_clear_boundary = QPushButton("清除分界线")
        self.btn_clear_boundary.clicked.connect(self.clear_boundary_line)
        self.btn_clear_boundary.setEnabled(False)
        layout.addWidget(self.btn_clear_boundary)
        
        # 弹性空间
        layout.addStretch()
        self.setLayout(layout)
    
    def load_model(self):
        """
        加载CNN分类模型
        
        打开文件选择对话框，让用户选择模型文件(.pt)，
        然后创建DryBeachRecognizer实例并加载模型
        """
        from drybeach_app.recognizer import DryBeachRecognizer
        from PyQt6.QtWidgets import QMessageBox
        
        # 打开文件选择对话框，过滤.pt模型文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "Model Files (*.pt)"
        )
        
        if file_path:
            try:
                # 创建识别器并加载模型
                recognizer = DryBeachRecognizer(model_path=file_path)
                self.model_path = Path(file_path)
                self.model_info = recognizer.model_info
                
                # 更新界面显示
                self.lbl_model_path.setText(f"已加载: {self.model_path.name}")
                self._display_model_info()
                
                # 同步识别器参数到界面
                self.cmb_detect_patch.setCurrentText(str(recognizer.patch_size))
                self.spin_detect_stride.setValue(recognizer.stride)
                
                # 发出模型加载信号
                self.model_loaded.emit(file_path)
                
            except Exception as e:
                # 加载失败显示错误对话框
                QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
    
    def _display_model_info(self):
        """
        在文本框中显示模型信息
        
        显示模型路径、设备、类别数、类别名称、参数数量等信息
        """
        if not self.model_info:
            return
        
        # 格式化模型信息文本
        info_text = f"""模型信息:
━━━━━━━━━━━━━━━━━━━━
路径: {self.model_info.get('path', 'N/A')}
设备: {self.model_info.get('device', 'N/A')}
类别数: {self.model_info.get('num_classes', 4)}
类别: {', '.join(self.model_info.get('categories', []))}
总参数: {self.model_info.get('total_params', 0):,}
可训练参数: {self.model_info.get('trainable_params', 0):,}"""
        
        self.txt_model_info.setPlainText(info_text)
    
    def load_image(self):
        """
        打开待识别的图片
        
        打开文件选择对话框，让用户选择图片文件，
        然后显示图片并启用识别按钮(如果模型已加载)
        """
        # 打开文件选择对话框，过滤常见图片格式
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            self.image_path = Path(file_path)
            img = cv2.imread(file_path)
            if img is not None:
                # 获取图片尺寸并显示
                h, w = img.shape[:2]
                self.lbl_image_path.setText(f"已选择: {self.image_path.name} ({w}x{h})")
                
                # 发出显示图片请求
                self.image_display_requested.emit(file_path)
                
                # 如果模型已加载，启用识别按钮
                if self.model_path:
                    self.btn_run_detection.setEnabled(True)
    
    def draw_detection_region(self):
        """
        请求绘制识别区域
        
        发出信号请求在图片上绘制多边形识别区域
        需要先打开图片才能绘制区域
        """
        # 检查是否已打开图片
        if not hasattr(self, 'image_path') or not self.image_path:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        # 发出绘制区域请求信号
        self.draw_region_requested.emit(str(self.image_path))
    
    def clear_detection_region(self):
        """
        清除已设置的识别区域
        
        将区域数据设为None，更新显示标签，发出区域清除信号
        """
        self.detection_region = None
        self.lbl_region.setText("未设置识别区域")
        self.region_cleared.emit()
    
    def save_region(self):
        """
        保存识别区域到JSON文件
        
        将当前设置的多边形识别区域保存为JSON格式文件
        """
        from PyQt6.QtWidgets import QMessageBox
        
        region_data = self.detection_region
        print(f"save_region: detection_region = {region_data}")
        
        # 检查是否有可保存的区域
        if region_data is None:
            QMessageBox.warning(self, "警告", "没有可保存的区域")
            return
        
        if not isinstance(region_data, dict) or 'points' not in region_data:
            QMessageBox.warning(self, "警告", "没有可保存的区域")
            return
        
        # 打开保存文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存区域", "", "JSON Files (*.json)"
        )
        
        if file_path:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(region_data, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "完成", f"区域已保存至: {file_path}")
    
    def load_region(self):
        """
        从JSON文件加载识别区域
        
        打开文件选择对话框，选择JSON文件并加载区域数据
        """
        # 打开文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择区域文件", "", "JSON Files (*.json)"
        )
        
        if file_path:
            import json
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.detection_region = json.load(f)
                
                # 验证并显示加载结果
                if self.detection_region and 'points' in self.detection_region:
                    self.lbl_region.setText(f"已加载区域: {len(self.detection_region['points'])}个点")
                    self.region_loaded.emit(self.detection_region)
                else:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "警告", "区域文件格式错误")
                    
            except Exception as e:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "错误", f"导入失败: {str(e)}")
    
    def set_detection_region(self, region_data):
        """
        设置识别区域数据(供外部调用)
        
        Args:
            region_data: 区域数据字典，应包含'points'键
        """
        self.detection_region = region_data
        if region_data and 'points' in region_data:
            self.lbl_region.setText(f"识别区域: {len(region_data['points'])}个点")
    
    def run_detection(self):
        """
        执行图像识别
        
        检查模型和图片是否已加载，然后发出识别请求信号
        """
        from PyQt6.QtWidgets import QMessageBox
        
        # 检查模型是否已加载
        if not self.model_path:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return
        
        # 检查图片是否已加载
        if not hasattr(self, 'image_path') or not self.image_path:
            QMessageBox.warning(self, "警告", "请先打开图片")
            return
        
        # 显示进度条，禁用识别按钮
        self.detect_progress.setVisible(True)
        self.detect_progress.setValue(0)
        self.btn_run_detection.setEnabled(False)
        
        # 发出识别请求信号，传递识别参数
        self.detection_requested.emit({
            'model_path': str(self.model_path),
            'image_path': str(self.image_path),
            'patch_size': int(self.cmb_detect_patch.currentText()),  # 切片尺寸
            'stride': self.spin_detect_stride.value(),                # 步长
            'detection_region': self.detection_region,              # 识别区域
            'confidence_threshold': self.spin_confidence.value()      # 置信度阈值
        })
    
    def on_detection_complete(self, annotated_image, class_counts, detection_points=None):
        """
        识别完成回调函数
        
        Args:
            annotated_image: 带标注的识别结果图片
            class_counts: 各类别的计数字典
            detection_points: 检测点列表(可选)
        """
        # 隐藏进度条，恢复识别按钮
        self.detect_progress.setVisible(False)
        self.btn_run_detection.setEnabled(True)
        
        if annotated_image is not None:
            # 保存识别结果
            self.annotated_image = annotated_image
            self.original_result = annotated_image.copy()
            self.detection_points = detection_points or []
            
            # 显示识别结果统计
            count_text = " | ".join([f"{k}: {v}" for k, v in class_counts.items()])
            self.lbl_detect_result.setText(count_text)
            
            # 启用结果保存和分界线生成按钮
            self.btn_save_result.setEnabled(True)
            self.btn_draw_boundary.setEnabled(True)
            
            # 发出结果准备好信号
            self.result_ready.emit(annotated_image)
    
    def save_result(self):
        """
        保存识别结果图片
        
        将带标注的识别结果保存为JPEG格式图片
        """
        from PyQt6.QtWidgets import QMessageBox
        
        # 检查是否有可保存的结果
        if not hasattr(self, 'annotated_image') or self.annotated_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的识别结果")
            return
        
        # 打开保存文件对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存识别结果", "", "JPEG Files (*.jpg)"
        )
        
        if file_path:
            try:
                # 转换颜色空间并保存
                save_image = cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, save_image)
                QMessageBox.information(self, "完成", f"结果已保存至: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
    
    def set_progress(self, value):
        """
        设置识别进度条值(供外部调用)
        
        Args:
            value: 进度值(0-100)
        """
        self.detect_progress.setValue(value)
    
    def draw_boundary_line(self):
        """
        生成分界线
        
        使用BoundaryLineGenerator根据识别结果中的分界点生成分界线
        对每个连通区域分别处理，生成平滑的中心线
        """
        from drybeach_app.recognizer import BoundaryLineGenerator
        
        # 检查是否有识别结果
        if not hasattr(self, 'detection_points') or not self.detection_points:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "请先进行识别")
            return
        
        if self.annotated_image is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "没有可处理的图像")
            return
        
        # 保存原始结果(以便清除分界线时恢复)
        self.original_result = self.annotated_image.copy()
        
        # 获取步长参数
        stride = self.spin_detect_stride.value()
        
        # 生成分界线
        boundary_lines = BoundaryLineGenerator.generate_boundary_lines(
            self.detection_points, 
            self.annotated_image.shape,
            stride=stride
        )
        
        if not boundary_lines:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "警告", "未能生成分界线，请确保有足够的边界点")
            return
        
        # 绘制分界线到图片上
        result = BoundaryLineGenerator.draw_boundary_lines(
            self.annotated_image, 
            boundary_lines,
            color=(0, 255, 0),  # 绿色
            thickness=3
        )
        
        # 保存结果
        self.annotated_image = result
        self.boundary_lines = boundary_lines
        self.btn_clear_boundary.setEnabled(True)
        self.result_ready.emit(result)
        
        # 显示生成的分界线信息
        line_types = [line['type'] for line in boundary_lines]
        self.lbl_detect_result.setText(f"已生成 {len(boundary_lines)} 条分界线: {', '.join(set(line_types))}")
    
    def clear_boundary_line(self):
        """
        清除分界线
        
        恢复原始识别结果图片
        """
        if hasattr(self, 'original_result'):
            self.annotated_image = self.original_result.copy()
            self.result_ready.emit(self.annotated_image)
            self.btn_clear_boundary.setEnabled(False)
    
    def update_image_with_boundary(self, image_with_boundary):
        """
        更新显示带分界线的图片(供外部调用)
        
        Args:
            image_with_boundary: 带分界线的图片
        """
        if image_with_boundary is not None:
            self.annotated_image = image_with_boundary
            self.result_ready.emit(image_with_boundary)