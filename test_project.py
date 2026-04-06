#!/usr/bin/env python3
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("干滩识别系统测试")
print("=" * 60)

try:
    print("\n[1/6] 测试模块导入...")
    from drybeach_app import (
        DryBeachRecognizer, 
        VideoFrameExtractor,
        WaterLineDetector,
        DamDetector,
        DistanceCalculator
    )
    print("   ✓ 所有核心模块导入成功")
except Exception as e:
    print(f"   ✗ 模块导入失败: {e}")
    sys.exit(1)

try:
    print("\n[2/6] 测试识别器初始化...")
    recognizer = DryBeachRecognizer()
    print("   ✓ 识别器初始化成功")
except Exception as e:
    print(f"   ✗ 识别器初始化失败: {e}")
    sys.exit(1)

try:
    print("\n[3/6] 测试合成图像生成...")
    import numpy as np
    import cv2
    
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:240] = [135, 206, 235]
    test_image[240:] = [194, 178, 128]
    print("   ✓ 测试图像创建成功")
except Exception as e:
    print(f"   ✗ 测试图像创建失败: {e}")
    sys.exit(1)

try:
    print("\n[4/6] 测试水面分界线检测...")
    water_detector = WaterLineDetector()
    water_line = water_detector.detect_by_edge_detection(test_image)
    print(f"   ✓ 检测到水面线，包含 {len(water_line)} 个点")
except Exception as e:
    print(f"   ✗ 水面线检测失败: {e}")
    sys.exit(1)

try:
    print("\n[5/6] 测试坝体检测...")
    dam_detector = DamDetector()
    dam_results = dam_detector.detect_dam_edges(test_image)
    print(f"   ✓ 坝体检测完成，边界框: {dam_results.get('bbox')}")
except Exception as e:
    print(f"   ✗ 坝体检测失败: {e}")
    sys.exit(1)

try:
    print("\n[6/6] 测试距离计算...")
    calculator = DistanceCalculator()
    calculator.calibrate(10.0, 100.0)
    distance = calculator.pixel_to_meters(50.0)
    print(f"   ✓ 校准完成，50像素 = {distance:.2f}米")
except Exception as e:
    print(f"   ✗ 距离计算失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("所有测试通过！系统准备就绪。")
print("=" * 60)
print("\n使用方法:")
print("  启动GUI: python main.py gui")
print("  命令行识别: python main.py detect -i test.jpg")
print("=" * 60)
