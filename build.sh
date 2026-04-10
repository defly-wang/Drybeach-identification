#!/bin/bash
# 打包脚本 - 使用 PyInstaller 将干滩识别系统打包为可执行文件

# 确保在项目根目录
cd "$(dirname "$0")"

echo "开始打包干滩识别系统..."

# 检查并安装依赖
if ! command -v pyinstaller &> /dev/null; then
    echo "安装 PyInstaller..."
    pip install pyinstaller
fi

# 创建 spec 文件
cat > drybeach.spec << 'EOF'
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('config/*.json', 'config'),
    ],
    hiddenimports=[
        'cv2',
        'numpy',
        'torch',
        'ultralytics',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='DryBeachIdentification',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DryBeachIdentification',
)
EOF

echo "创建 PyInstaller spec 文件完成"

# 执行打包
echo "开始打包..."
pyinstaller drybeach.spec --clean

# 清理临时文件
echo "清理临时文件..."
rm -f drybeach.spec

echo "打包完成！"
echo "输出目录: dist/DryBeachIdentification"