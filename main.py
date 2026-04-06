#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='干滩识别系统 - Dry Beach Identification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 1. 提取视频帧
  python main.py extract --video input.mp4 --output frames/ --interval 30

  # 2. 识别单张图片
  python main.py detect --image input.jpg --output result.jpg --calibrate 10

  # 3. 批量识别
  python main.py batch --input images/ --output results/

  # 4. 训练模型
  python main.py train --config dataset.yaml --epochs 100

  # 5. 启动GUI
  python main.py gui
        """
    )
    
    parser.add_argument('command', nargs='?', default='gui',
                       choices=['extract', 'detect', 'batch', 'train', 'gui'],
                       help='命令: extract(提取帧), detect(识别), batch(批量), train(训练), gui(界面)')
    
    parser.add_argument('--video', '-v', type=str, help='输入视频文件路径')
    parser.add_argument('--image', '-i', type=str, help='输入图片文件路径')
    parser.add_argument('--input', type=str, help='输入目录路径')
    parser.add_argument('--output', '-o', type=str, help='输出路径')
    parser.add_argument('--interval', type=int, default=30, help='帧提取间隔(默认30)')
    parser.add_argument('--calibrate', type=float, help='校准距离(米)')
    parser.add_argument('--method', type=str, default='multi', choices=['multi', 'edge', 'color'],
                        help='识别方法')
    parser.add_argument('--config', type=str, help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数(默认100)')
    parser.add_argument('--model', type=str, help='模型文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'gui':
        run_gui()
    elif args.command == 'extract':
        run_extraction(args)
    elif args.command == 'detect':
        run_detection(args)
    elif args.command == 'batch':
        run_batch_detection(args)
    elif args.command == 'train':
        run_training(args)


def run_gui():
    logger.info("启动GUI界面...")
    from drybeach_app.gui import launch_gui
    launch_gui()


def run_extraction(args):
    from drybeach_app.video_capture import extract_frames_from_video
    
    if not args.video:
        logger.error("请提供视频文件路径 (--video)")
        return
    
    if not args.output:
        args.output = 'extracted_frames'
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"从视频 {args.video} 提取帧, 间隔={args.interval}")
    
    saved_files = extract_frames_from_video(
        args.video, output_dir,
        mode='interval',
        value=args.interval
    )
    
    logger.info(f"提取完成, 共 {len(saved_files)} 帧")


def run_detection(args):
    from drybeach_app.recognizer import DryBeachRecognizer
    from drybeach_app.image_annotator import RegionOfInterest
    import cv2
    
    if not args.image:
        logger.error("请提供图片文件路径 (--image)")
        return
    
    image = cv2.imread(args.image)
    if image is None:
        logger.error(f"无法读取图片: {args.image}")
        return
    
    recognizer = DryBeachRecognizer()
    
    if args.calibrate:
        calibrator = recognizer.distance_calculator
        calibrator.calibrate(args.calibrate, 100)
        logger.info(f"系统已校准: {args.calibrate}米")
    
    logger.info(f"识别图片: {args.image}, 方法={args.method}")
    
    result, annotated = recognizer.detect_and_visualize(image, method=args.method)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.image).with_name(f"{Path(args.image).stem}_result.jpg")
    
    cv2.imwrite(str(output_path), annotated)
    recognizer.save_result(output_path.with_suffix('.json'), result)
    
    logger.info(f"结果已保存: {output_path}")
    
    if result.distance_meters is not None:
        logger.info(f"测量结果: {result.distance_meters:.2f} 米")
    else:
        logger.warning("未能计算出距离")


def run_batch_detection(args):
    from drybeach_app.recognizer import DryBeachRecognizer
    import cv2
    
    if not args.input:
        logger.error("请提供输入目录路径 (--input)")
        return
    
    input_dir = Path(args.input)
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    if not image_files:
        logger.error(f"目录 {input_dir} 中未找到图片文件")
        return
    
    if not args.output:
        args.output = 'batch_results'
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    recognizer = DryBeachRecognizer()
    
    logger.info(f"批量识别: {len(image_files)} 张图片")
    
    for i, img_path in enumerate(image_files, 1):
        logger.info(f"处理 ({i}/{len(image_files)}): {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"无法读取: {img_path}")
            continue
        
        result, annotated = recognizer.detect_and_visualize(image, method=args.method)
        
        output_path = output_dir / f"{img_path.stem}_result.jpg"
        cv2.imwrite(str(output_path), annotated)
        
        if result.distance_meters is not None:
            logger.info(f"  -> 距离: {result.distance_meters:.2f} m")
    
    logger.info(f"批量识别完成, 结果保存在: {output_dir}")


def run_training(args):
    from drybeach_app.model_trainer import ModelTrainer
    
    if not args.config:
        logger.error("请提供数据集配置文件路径 (--config)")
        return
    
    model_save = Path(args.model) if args.model else Path('models/drybeach_model.pt')
    model_save.parent.mkdir(parents=True, exist_ok=True)
    
    trainer = ModelTrainer(model_save_path=model_save)
    
    logger.info(f"开始训练, 配置: {args.config}, 轮数: {args.epochs}")
    
    model_path = trainer.train_with_yolo(
        Path(args.config),
        epochs=args.epochs
    )
    
    logger.info(f"训练完成, 模型保存于: {model_path}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
