import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


class DetectionResult:
    def __init__(self):
        self.class_map = None
        self.annotated_image = None
        self.class_counts = {}
        self.detection_points = []


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DryBeachRecognizer:
    CATEGORY_COLORS = {
        0: (0, 200, 255),
        1: (200, 255, 0),
        2: (255, 0, 255),
        3: (255, 100, 0)
    }
    CATEGORY_NAMES = ['water', 'beach', 'boundary', 'dam']
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.device = None
        self.patch_size = 32
        self.stride = 16
        self.result = DetectionResult()
        self.model_info = {}
        
        if model_path and TORCH_AVAILABLE:
            self.load_model(model_path)
        
        logger.info("DryBeachRecognizer initialized")
    
    def load_model(self, model_path: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CNNClassifier(num_classes=4)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.model_info = {
            'path': model_path,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'num_classes': 4,
            'categories': self.CATEGORY_NAMES,
            'device': str(self.device)
        }
        
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Device: {self.device}")
        
        return self.model_info
    
    def detect(self, image: np.ndarray, progress_callback=None) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        img_h, img_w = image.shape[:2]
        
        self.result = DetectionResult()
        class_counts = {name: 0 for name in self.CATEGORY_NAMES}
        detection_points = []
        
        total_patches = 0
        for y in range(0, img_h - self.patch_size + 1, self.stride):
            for x in range(0, img_w - self.patch_size + 1, self.stride):
                total_patches += 1
        
        processed = 0
        print(f"\n{'='*50}")
        print(f"开始识别: 图像尺寸 {img_w}x{img_h}, 切片尺寸 {self.patch_size}, 步长 {self.stride}")
        print(f"总切片数: {total_patches}")
        print(f"{'='*50}")
        
        for y in range(0, img_h - self.patch_size + 1, self.stride):
            for x in range(0, img_w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_float = patch_rgb.astype(np.float32) / 255.0
                patch_tensor = np.transpose(patch_float, (2, 0, 1))
                patch_tensor = torch.from_numpy(patch_tensor).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(patch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    class_id = int(probs[0].argmax())
                    confidence = float(probs[0].max())
                
                center_x = x + self.patch_size // 2
                center_y = y + self.patch_size // 2
                
                detection_points.append({
                    'x': center_x,
                    'y': center_y,
                    'class_id': class_id,
                    'confidence': confidence
                })
                
                class_counts[self.CATEGORY_NAMES[class_id]] += 1
                
                processed += 1
                if processed % 100 == 0 or processed == total_patches:
                    pct = processed / total_patches * 100
                    print(f"进度: {processed}/{total_patches} ({pct:.1f}%)")
                    if progress_callback:
                        progress_callback(int(pct))
        
        self.result.class_counts = class_counts
        self.result.detection_points = detection_points
        
        print(f"\n{'='*50}")
        print("识别结果统计:")
        for name, count in class_counts.items():
            pct = count / total_patches * 100 if total_patches > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")
        print(f"{'='*50}\n")
        
        return self.result
    
    def detect_and_visualize(self, image: np.ndarray, progress_callback=None) -> Tuple[DetectionResult, np.ndarray]:
        self.detect(image, progress_callback=progress_callback)
        
        annotated = self._create_annotated_image(image)
        self.result.annotated_image = annotated
        
        return self.result, annotated
    
    def _create_annotated_image(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        
        point_radius = 5
        
        for point in self.result.detection_points:
            x = point['x']
            y = point['y']
            class_id = point['class_id']
            confidence = point['confidence']
            
            color = self.CATEGORY_COLORS.get(class_id, (255, 255, 255))
            
            cv2.circle(result, (x, y), point_radius, color, -1)
            
            cv2.circle(result, (x, y), point_radius + 2, (255, 255, 255), 1)
        
        h, w = result.shape[:2]
        legend_x, legend_y = 10, 10
        line_height = 25
        
        cv2.rectangle(result, (legend_x - 5, legend_y - 5), 
                     (legend_x + 180, legend_y + len(self.CATEGORY_NAMES) * line_height + 5), 
                     (0, 0, 0), -1)
        
        for i, (name, color) in enumerate(zip(self.CATEGORY_NAMES, 
                                             [self.CATEGORY_COLORS[i] for i in range(4)])):
            count = self.result.class_counts.get(name, 0)
            text = f"{name}: {count}"
            cv2.putText(result, text, (legend_x, legend_y + i * line_height + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return result
    
    def batch_detect(self, images: List[np.ndarray]) -> List[DetectionResult]:
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.detect(image)
            results.append(result)
        
        return results
    
    def save_result(self, output_path: Path, result: DetectionResult,
                   image: Optional[np.ndarray] = None):
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = result.to_dict()
        
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        if image is not None and result.annotated_image is not None:
            cv2.imwrite(str(output_path.with_suffix('.jpg')), result.annotated_image)
        
        logger.info(f"Result saved to {output_path}")


def load_recognizer_model(model_path: str) -> DryBeachRecognizer:
    recognizer = DryBeachRecognizer(model_path=model_path)
    logger.info(f"Loaded model from {model_path}")
    return recognizer


def process_video_frames(video_path: str, output_dir: Path,
                        recognizer: DryBeachRecognizer,
                        frame_interval: int = 30) -> List[Path]:
    from .video_capture import VideoFrameExtractor
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_images = []
    
    with VideoFrameExtractor(video_path) as extractor:
        for frame_num in range(0, extractor.total_frames, frame_interval):
            frame = extractor.extract_frame(frame_num)
            
            if frame is not None:
                result, annotated = recognizer.detect_and_visualize(frame)
                
                output_filename = f"frame_{frame_num:06d}_result.jpg"
                output_path = output_dir / output_filename
                
                cv2.imwrite(str(output_path), annotated)
                output_images.append(output_path)
                
                result_filename = f"frame_{frame_num:06d}_data.json"
                recognizer.save_result(output_dir / result_filename, result)
                
                logger.info(f"Processed frame {frame_num}")
    
    logger.info(f"Processed {len(output_images)} frames")
    
    return output_images