import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics not available")


class DetectionResult:
    def __init__(self):
        self.class_map = None
        self.annotated_image = None
        self.class_counts = {}
    
    def to_dict(self) -> Dict:
        return {
            'class_map': self.class_map.tolist() if self.class_map is not None else None,
            'class_counts': self.class_counts
        }


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
        self.patch_size = 64
        self.stride = 32
        self.result = DetectionResult()
        
        if model_path and ULTRALYTICS_AVAILABLE:
            self.load_model(model_path)
        
        logger.info("DryBeachRecognizer initialized")
    
    def load_model(self, model_path: str):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics not available")
        
        self.model = YOLO(model_path)
        self.patch_size = self.model.overrides.get('imgsz', 64)
        logger.info(f"Loaded model from {model_path}, patch_size={self.patch_size}")
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        img_h, img_w = image.shape[:2]
        
        self.result.class_map = np.zeros((img_h, img_w), dtype=np.uint8)
        class_counts = {name: 0 for name in self.CATEGORY_NAMES}
        
        for y in range(0, img_h - self.patch_size + 1, self.stride):
            for x in range(0, img_w - self.patch_size + 1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                
                results = self.model(patch, verbose=False)
                
                probs = results[0].probs
                if probs is not None and probs.data is not None:
                    class_id = int(probs.data.argmax())
                else:
                    class_id = 0
                
                self.result.class_map[y:y+self.patch_size, x:x+self.patch_size] = class_id
                class_counts[self.CATEGORY_NAMES[class_id]] += 1
        
        self.result.class_counts = class_counts
        return self.result
    
    def detect_and_visualize(self, image: np.ndarray) -> Tuple[DetectionResult, np.ndarray]:
        self.detect(image)
        
        annotated = self._create_annotated_image(image)
        self.result.annotated_image = annotated
        
        return self.result, annotated
    
    def _create_annotated_image(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        
        for class_id, color in self.CATEGORY_COLORS.items():
            mask = (self.result.class_map == class_id)
            
            overlay = np.zeros_like(result)
            overlay[mask] = color
            
            alpha = 0.4
            result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
        
        h, w = self.result.class_map.shape[:2]
        legend_x, legend_y = 10, 10
        line_height = 25
        
        cv2.rectangle(result, (legend_x - 5, legend_y - 5), (legend_x + 150, legend_y + len(self.CATEGORY_NAMES) * line_height + 5), (0, 0, 0), -1)
        
        for i, (name, color) in enumerate(zip(self.CATEGORY_NAMES, [self.CATEGORY_COLORS[i] for i in range(4)])):
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
