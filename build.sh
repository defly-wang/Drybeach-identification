#!/bin/bash
# 打包脚本 - 使用 PyInstaller 将干滩识别系统打包为可执行文件
# 
# 使用方法: 
#   bash build.sh          # 标准打包
#   bash build.sh --clean  # 清理后打包
#   bash build.sh --debug  # 调试模式打包
#
# 打包时间可能较长（5-15分钟），请耐心等待
#
# 输出: dist/DryBeach

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查虚拟环境
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "========================================="
echo "  干滩识别系统 - PyInstaller 打包脚本"
echo "========================================="
echo ""

# 检查 PyInstaller
if ! $PYTHON -c "import PyInstaller" 2>/dev/null; then
    echo "[1/4] 安装 PyInstaller..."
    pip install pyinstaller
fi

# 清理选项
CLEAN=""
if [ "$1" = "--clean" ]; then
    echo "[2/4] 清理旧文件..."
    rm -rf build dist *.spec
fi

echo "[3/4] 开始打包（这可能需要5-15分钟）..."
echo "      - 处理 PyQt6"
echo "      - 处理 OpenCV"
echo "      - 处理 PyTorch"
echo ""

# 打包命令
$PYTHON -m PyInstaller main.py \
    --name DryBeach \
    --onefile \
    --windowed \
    --noconfirm \
    --add-data "config:config" \
    --hidden-import cv2 \
    --hidden-import numpy \
    --hidden-import torch \
    --hidden-import ultralytics \
    --hidden-import PyQt6 \
    --hidden-import PyQt6.QtCore \
    --hidden-import PyQt6.QtGui \
    --hidden-import PyQt6.QtWidgets \
    --collect-all cv2 \
    --collect-all torch \
    --collect-all ultralytics

echo ""
echo "[4/4] 打包完成！"
echo ""
echo "输出文件: dist/DryBeach"
echo ""
echo "提示: 如果打包失败，可尝试以下操作:"
echo "  1. 确保虚拟环境 .venv 已激活"
echo "  2. 运行: pip install -r requirements.txt"
echo "  3. 确保 main.py 可以正常运行"
echo ""