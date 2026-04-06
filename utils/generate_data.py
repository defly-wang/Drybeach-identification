#!/usr/bin/env python3
import numpy as np
import cv2
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_drybeach_image(width: int = 1280, height: int = 720,
                                      water_level: float = 0.4,
                                      dam_position: float = 0.8,
                                      noise_level: float = 0.05) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    sky_color = np.array([135, 206, 235])
    water_color = np.array([30, 100, 150])
    beach_color = np.array([194, 178, 128])
    dam_color = np.array([100, 80, 70])
    
    image[:int(height * 0.3)] = sky_color
    
    water_line_y = int(height * water_level)
    
    image[water_line_y:int(height * 0.6)] = water_color
    
    beach_start = int(height * 0.6)
    beach_end = int(height * dam_position)
    image[beach_start:beach_end] = beach_color
    
    dam_start = int(height * dam_position)
    dam_end = height
    
    for y in range(dam_start, dam_end):
        dam_height = (y - dam_start) / (dam_end - dam_start)
        shade = 1.0 - dam_height * 0.3
        color = (dam_color * shade).astype(np.uint8)
        image[y] = color
    
    noise = np.random.randint(0, 255, image.shape, dtype=np.uint8)
    image = cv2.addWeighted(image, 1 - noise_level, noise, noise_level, 0)
    
    points = []
    for x in range(0, width, 10):
        wave_offset = np.sin(x * 0.02) * 5 + np.sin(x * 0.05) * 3
        y = water_line_y + int(wave_offset)
        points.append((x, y))
    
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    for offset in [-2, 0, 2]:
        pts_shifted = pts.copy()
        pts_shifted[:, 0, 1] += offset
        cv2.polylines(image, [pts_shifted], False, (0, 255, 255), 1)
    
    dam_top_y = dam_start
    
    for x in range(0, width, 20):
        slope = np.random.uniform(-0.1, 0.1)
        dx = int(slope * (dam_end - dam_top_y))
        cv2.line(image, (x + dx, dam_top_y), (x, dam_end), (int(dam_color[0]), int(dam_color[1]), int(dam_color[2])), 2)
    
    cv2.line(image, (0, dam_top_y), (width, dam_top_y), (80, 60, 50), 4)
    cv2.line(image, (0, dam_end - 10), (width, dam_end - 10), (80, 60, 50), 4)
    
    return image


def generate_dataset(output_dir: Path, num_images: int = 100,
                    width: int = 1280, height: int = 720):
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"生成 {num_images} 张合成干滩图片...")
    
    for i in range(num_images):
        water_level = np.random.uniform(0.3, 0.5)
        dam_position = np.random.uniform(0.7, 0.9)
        noise_level = np.random.uniform(0.02, 0.08)
        
        image = generate_synthetic_drybeach_image(
            width=width,
            height=height,
            water_level=water_level,
            dam_position=dam_position,
            noise_level=noise_level
        )
        
        image_filename = f"drybeach_{i:04d}.jpg"
        cv2.imwrite(str(images_dir / image_filename), image)
        
        water_line_y = int(height * water_level)
        dam_top_y = int(height * dam_position)
        
        label_filename = f"drybeach_{i:04d}.txt"
        with open(labels_dir / label_filename, 'w') as f:
            f.write(f"0 0.5 {water_line_y / height:.6f} 1.0 {0.02:.6f}\n")
            f.write(f"1 0.5 {dam_top_y / height:.6f} 1.0 {0.2:.6f}\n")
        
        if (i + 1) % 10 == 0:
            logger.info(f"已生成 {i + 1}/{num_images} 张图片")
    
    logger.info(f"数据集生成完成保存在: {output_dir}")
    
    create_dataset_yaml(output_dir)


def create_dataset_yaml(dataset_dir: Path):
    yaml_content = f"""
path: {dataset_dir}
train: images
val: images

nc: 2
names:
  0: water_line
  1: dam
"""
    
    yaml_path = dataset_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"数据集配置文件已创建: {yaml_path}")


def generate_test_video(output_path: Path, num_frames: int = 300,
                       fps: int = 30, width: int = 1280, height: int = 720):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    logger.info(f"生成测试视频: {output_path}")
    
    for frame_idx in range(num_frames):
        water_level = 0.4 + np.sin(frame_idx * 0.02) * 0.05
        dam_position = 0.8 + np.sin(frame_idx * 0.01) * 0.02
        
        image = generate_synthetic_drybeach_image(
            width=width,
            height=height,
            water_level=water_level,
            dam_position=dam_position,
            noise_level=0.05
        )
        
        cv2.putText(image, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        writer.write(image)
        
        if (frame_idx + 1) % 30 == 0:
            logger.info(f"已生成 {frame_idx + 1}/{num_frames} 帧")
    
    writer.release()
    logger.info(f"视频生成完成: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成合成干滩数据')
    parser.add_argument('--output', '-o', type=str, default='data/synthetic',
                       help='输出目录')
    parser.add_argument('--num', '-n', type=int, default=100,
                       help='生成图片数量')
    parser.add_argument('--video', action='store_true',
                       help='同时生成测试视频')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    generate_dataset(output_dir, num_images=args.num)
    
    if args.video:
        video_path = output_dir / 'test_video.mp4'
        generate_test_video(video_path)
