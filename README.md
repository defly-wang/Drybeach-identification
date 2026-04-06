# 干滩识别系统 (Dry Beach Identification System)

基于Python和PyQt6的计算机视觉系统，用于水面分界线与坝体识别及距离测量。

## 功能特性

- **视频帧提取**: 从视频文件或视频流中按指定间隔截取图片
- **水面分界线识别**: 多种算法检测水面与干滩的分界线
- **坝体识别**: 自动检测并定位坝体边界
- **距离测量计算**: 基于校准的像素到实际距离转换
- **ROI区域划定**: 支持用户自定义分析区域
- **图片切分**: 支持大图切分成小块便于处理
- **图片标注**: 支持多种标注方式(多边形、线段、矩形、点)
- **模型训练**: 支持YOLO等目标检测模型训练
- **结果可视化**: 自动标注并生成分界线可视化
- **PyQt6 GUI**: 友好的图形用户界面

## 项目结构

```
Drybeach-identification/
├── drybeach_app/           # 核心功能模块
│   ├── video_capture.py    # 视频帧提取
│   ├── image_annotator.py  # 图片标注与切分
│   ├── water_line_detector.py  # 水面分界线检测
│   ├── dam_detector.py     # 坝体检测
│   ├── distance_calculator.py  # 距离测量计算
│   ├── model_trainer.py    # 模型训练
│   ├── recognizer.py       # 识别主模块
│   └── gui.py             # PyQt6 GUI界面
├── utils/                  # 工具函数
│   └── generate_data.py    # 测试数据生成
├── data/                   # 数据目录
├── models/                # 模型存储
├── outputs/               # 输出结果
├── config.py              # 配置文件
├── main.py               # 主程序入口
└── requirements.txt      # 依赖列表
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 启动GUI界面

```bash
python main.py gui
```

### 命令行使用

#### 提取视频帧
```bash
python main.py extract --video input.mp4 --output frames/ --interval 30
```

#### 识别单张图片
```bash
python main.py detect --image input.jpg --output result.jpg --calibrate 10
```

#### 批量识别
```bash
python main.py batch --input images/ --output results/
```

#### 训练模型
```bash
python main.py train --config dataset.yaml --epochs 100
```

### Python API使用

```python
from drybeach_app import DryBeachRecognizer, VideoFrameExtractor

# 识别单张图片
recognizer = DryBeachRecognizer()
recognizer.calibrate(known_distance_meters=10, 
                    reference_points=[(0, 100), (500, 100)])

result, annotated = recognizer.detect_and_visualize(image, method='multi')
print(f"最短干滩距离: {result.distance_meters:.2f} 米")

# 从视频提取帧
with VideoFrameExtractor('video.mp4') as extractor:
    frame = extractor.extract_frame(30)
    result, _ = recognizer.detect_and_visualize(frame)
```

## 核心模块说明

### video_capture.py - 视频帧提取
- `VideoFrameExtractor`: 视频文件帧提取器
- `VideoStreamGrabber`: 视频流捕获
- `extract_frames_from_video()`: 批量提取函数

### image_annotator.py - 图片标注与切分
- `ImageAnnotator`: 图片标注工具，支持多边形、线段、矩形、点标注
- `BatchImageSlicer`: 批量图片切分器
- `RegionOfInterest`: ROI区域定义

### water_line_detector.py - 水面分界线检测
- `WaterLineDetector`: 基础检测器
  - `detect_by_edge_detection()`: 边缘检测方法
  - `detect_by_color_segmentation()`: 颜色分割方法
  - `detect_multi_method()`: 多方法融合
- `AdaptiveWaterLineDetector`: 自适应检测器

### dam_detector.py - 坝体检测
- `DamDetector`: 坝体边缘检测
- `DamEdgeDetector`: 坝体纹理检测

### distance_calculator.py - 距离测量
- `DistanceCalculator`: 距离计算器
  - `calibrate()`: 系统校准
  - `calculate_shortest_distance()`: 计算最短距离
  - `calculate_water_line_properties()`: 水面线属性计算
- `MeasurementReporter`: 测量报告生成

### model_trainer.py - 模型训练
- `ModelTrainer`: 模型训练器
  - `train_with_yolo()`: YOLO模型训练
  - `train_simple_model()`: 自定义模型训练
- `DryBeachDataset`: 数据集类

### recognizer.py - 识别主模块
- `DryBeachRecognizer`: 综合识别器
  - `calibrate()`: 系统校准
  - `detect()`: 检测识别
  - `detect_and_visualize()`: 检测并可视化
  - `batch_detect()`: 批量检测

### gui.py - PyQt6 GUI
- 菜单栏：文件、工具
- 左侧控制面板
- 中心图片显示区
- 右侧结果输出区

## 算法原理

### 水面分界线检测算法

1. **边缘检测法 (Edge Detection)**
   - 使用Canny边缘检测
   - Hough变换检测直线
   - 筛选水平方向的线段

2. **颜色分割法 (Color Segmentation)**
   - HSV色彩空间分析
   - 水体颜色范围mask
   - 边缘提取确定分界线

3. **强度变化法 (Intensity Change)**
   - 分析水平方向的像素强度变化
   - 梯度检测显著变化点

### 距离测量算法

1. **像素距离计算**
   - 点到线段距离
   - 曲线到边界最短距离

2. **校准转换**
   - 参考点校准
   - 像素/米转换比例

## 生成测试数据

```bash
python utils/generate_data.py --output data/synthetic --num 100 --video
```

## GUI使用说明

1. **打开文件**: 点击"打开视频"或"打开图片"加载媒体文件
2. **提取帧**: 选择视频文件，点击"提取视频帧"保存帧
3. **设置ROI**: 点击"设置ROI区域"，在图片上拖动选择分析区域
4. **校准**: 输入已知距离，点击"开始校准"，在图片上点击两个点
5. **运行识别**: 选择识别方法，点击"运行识别"
6. **查看结果**: 识别结果和测量距离显示在右侧面板
7. **保存结果**: 点击"保存结果图片"或"导出报告"

## 依赖库

- opencv-python >= 4.8.0
- numpy >= 1.24.0
- pillow >= 10.0.0
- matplotlib >= 3.7.0
- scikit-learn >= 1.3.0
- torch >= 2.0.0
- torchvision >= 0.15.0
- ultralytics >= 8.0.0
- PyQt6 >= 6.5.0
- scipy >= 1.11.0

## 许可证

MIT License
