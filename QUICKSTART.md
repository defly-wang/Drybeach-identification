# 干滩识别系统 - 快速入门指南

## 虚拟环境状态

✅ **虚拟环境**: `venv\` 已创建  
✅ **依赖安装**: 核心依赖已安装  
✅ **测试数据**: 已生成5张测试图片  
✅ **功能测试**: 所有测试通过

## 快速启动

### 方法1: 使用启动脚本（推荐）

```batch
双击运行 start.bat
```

### 方法2: 手动启动

```bash
# 激活虚拟环境
.\venv\Scripts\activate

# 启动GUI
python main.py gui

# 或运行测试
python test_project.py
```

## 功能测试

### 1. 测试识别功能
```bash
python main.py detect --image "data/test/images/drybeach_0000.jpg" --output outputs/result.jpg
```

### 2. 批量识别
```bash
python main.py batch --input data/test/images --output outputs/
```

### 3. 生成更多测试数据
```bash
python utils/generate_data.py --num 50 --output data/custom --video
```

## 项目结构

```
Drybeach-identification/
├── venv/                    # 虚拟环境
├── drybeach_app/           # 核心功能模块
│   ├── video_capture.py    # 视频帧提取
│   ├── image_annotator.py  # ROI、标注、切分
│   ├── water_line_detector.py  # 水面分界线识别
│   ├── dam_detector.py     # 坝体识别
│   ├── distance_calculator.py  # 距离测量
│   ├── model_trainer.py    # 模型训练
│   ├── recognizer.py       # 综合识别
│   └── gui.py             # PyQt6 GUI
├── data/
│   └── test/images/       # 测试图片
├── outputs/               # 输出结果
├── models/               # 模型存储
├── utils/generate_data.py  # 数据生成
├── main.py              # 主程序入口
├── config.py           # 配置文件
├── test_project.py     # 测试脚本
├── start.bat          # 启动器
└── README.md          # 详细文档
```

## 已实现功能

| 功能 | 状态 | 描述 |
|------|------|------|
| 视频帧提取 | ✅ | 按间隔/数量提取帧 |
| ROI区域划定 | ✅ | 自定义分析区域 |
| 图片切分 | ✅ | 大图切分 |
| 图片标注 | ✅ | 多边形/线段/矩形/点 |
| 水面线识别 | ✅ | 边缘/颜色/强度方法 |
| 坝体识别 | ✅ | 边缘检测 |
| 距离测量 | ✅ | 像素-米转换 |
| 模型训练 | ⚠️ | 需要安装ultralytics |
| GUI界面 | ✅ | PyQt6图形界面 |

## 注意事项

1. **YOLO训练**: 需要安装 `ultralytics` 库
   ```bash
   .\venv\Scripts\pip install ultralytics
   ```

2. **GPU加速**: 如需GPU支持，安装CUDA版本的PyTorch
   ```bash
   .\venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Windows编码**: 部分输出可能显示乱码，不影响功能

## 故障排除

### 问题: 模块导入失败
**解决**: 重新安装依赖
```bash
.\venv\Scripts\pip install -r requirements.txt
```

### 问题: PyQt6界面无法启动
**解决**: 检查PyQt6是否正确安装
```bash
.\venv\Scripts\python -c "from PyQt6.QtWidgets import QApplication"
```

### 问题: 识别结果为0
**解决**: 需要校准系统
```bash
python main.py detect --image test.jpg --calibrate 10
```

## 下一步

1. 查看 `README.md` 了解详细功能
2. 准备自己的视频/图片数据
3. 使用GUI进行交互式识别
4. 训练自定义模型提高精度
