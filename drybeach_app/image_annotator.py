import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegionOfInterest:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    def contains(self, point: Tuple[int, int]) -> bool:
        px, py = point
        return (self.x <= px <= self.x + self.width and 
                self.y <= py <= self.y + self.height)


class Annotation:
    def __init__(self, label: str, points: List[Tuple[int, int]], 
                 annotation_type: str = 'polygon'):
        self.label = label
        self.points = points
        self.annotation_type = annotation_type
    
    def to_dict(self) -> Dict:
        return {
            'label': self.label,
            'points': self.points,
            'type': self.annotation_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Annotation':
        return cls(
            label=data['label'],
            points=[tuple(p) for p in data['points']],
            annotation_type=data.get('type', 'polygon')
        )


class ImageAnnotator:
    def __init__(self, image_path: str):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(image_path))
        
        if self.image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        self.annotations: List[Annotation] = []
        self.current_points: List[Tuple[int, int]] = []
        self.roi = None
        self.display_image = self.image.copy()
        
    def set_roi(self, x: int, y: int, width: int, height: int):
        self.roi = RegionOfInterest(x, y, width, height)
        logger.info(f"ROI set: ({x}, {y}, {width}, {height})")
    
    def add_point(self, x: int, y: int):
        self.current_points.append((x, y))
        logger.debug(f"Added point: ({x}, {y})")
    
    def finish_polygon(self, label: str):
        if len(self.current_points) >= 3:
            annotation = Annotation(label, self.current_points.copy(), 'polygon')
            self.annotations.append(annotation)
            logger.info(f"Added polygon annotation: {label}")
            self.current_points = []
        else:
            logger.warning("Polygon needs at least 3 points")
    
    def add_line(self, start: Tuple[int, int], end: Tuple[int, int], label: str):
        annotation = Annotation(label, [start, end], 'line')
        self.annotations.append(annotation)
        logger.info(f"Added line annotation: {label}")
    
    def add_box(self, x: int, y: int, width: int, height: int, label: str):
        points = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        annotation = Annotation(label, points, 'box')
        self.annotations.append(annotation)
        logger.info(f"Added box annotation: {label}")
    
    def add_point_annotation(self, x: int, y: int, label: str):
        annotation = Annotation(label, [(x, y)], 'point')
        self.annotations.append(annotation)
        logger.info(f"Added point annotation: {label}")
    
    def get_annotated_image(self) -> np.ndarray:
        result = self.image.copy()
        
        if self.roi:
            cv2.rectangle(result, 
                         (self.roi.x, self.roi.y),
                         (self.roi.x + self.roi.width, self.roi.y + self.roi.height),
                         (0, 255, 255), 2)
        
        for ann in self.annotations:
            color = self._get_color_for_label(ann.label)
            
            if ann.annotation_type == 'polygon':
                pts = np.array(ann.points, np.int32)
                cv2.polylines(result, [pts], True, color, 2)
                
            elif ann.annotation_type == 'line':
                cv2.line(result, ann.points[0], ann.points[1], color, 3)
                
            elif ann.annotation_type == 'box':
                x, y = ann.points[0]
                w = ann.points[1][0] - x
                h = ann.points[2][1] - y
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                
            elif ann.annotation_type == 'point':
                cv2.circle(result, ann.points[0], 5, color, -1)
            
            label_pos = ann.points[0]
            cv2.putText(result, ann.label, (label_pos[0] + 10, label_pos[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if self.current_points and len(self.current_points) > 0:
            for pt in self.current_points:
                cv2.circle(result, pt, 5, (0, 255, 0), -1)
            
            if len(self.current_points) > 1:
                for i in range(len(self.current_points) - 1):
                    cv2.line(result, self.current_points[i], 
                            self.current_points[i + 1], (0, 255, 0), 2)
            
            cv2.line(result, self.current_points[-1], self.current_points[0], 
                    (0, 255, 0), 2)
        
        return result
    
    def _get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        colors = {
            'water_line': (0, 255, 255),
            'dam': (255, 0, 0),
            'reference': (0, 255, 0),
            'measurement': (255, 255, 0)
        }
        return colors.get(label, (128, 128, 128))
    
    def save_annotations(self, output_path: Optional[Path] = None):
        if output_path is None:
            output_path = self.image_path.with_suffix('.json').name
            
        data = {
            'image_path': str(self.image_path),
            'annotations': [ann.to_dict() for ann in self.annotations],
            'roi': self.roi.to_tuple() if self.roi else None
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Annotations saved to {output_path}")
    
    def load_annotations(self, annotation_path: Path):
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.annotations = [Annotation.from_dict(ann) for ann in data['annotations']]
        
        if data.get('roi'):
            roi_data = data['roi']
            self.roi = RegionOfInterest(*roi_data)
        
        logger.info(f"Loaded {len(self.annotations)} annotations from {annotation_path}")


class BatchImageSlicer:
    def __init__(self, slice_width: int = 512, slice_height: int = 512, 
                 overlap: int = 0):
        self.slice_width = slice_width
        self.slice_height = slice_height
        self.overlap = overlap
    
    def slice_image(self, image: np.ndarray, 
                   save_dir: Optional[Path] = None) -> List[Tuple[str, np.ndarray]]:
        h, w = image.shape[:2]
        slices = []
        
        step_x = self.slice_width - self.overlap
        step_y = self.slice_height - self.overlap
        
        slice_info = []
        
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                end_x = min(x + self.slice_width, w)
                end_y = min(y + self.slice_height, h)
                
                slice_img = image[y:end_y, x:end_x]
                
                slice_name = f"slice_y{y:04d}_x{x:04d}.png"
                slice_info.append({
                    'name': slice_name,
                    'bbox': (x, y, end_x - x, end_y - y),
                    'original_shape': slice_img.shape
                })
                
                if save_dir:
                    cv2.imwrite(str(save_dir / slice_name), slice_img)
                
                slices.append((slice_name, slice_img))
        
        if save_dir:
            with open(save_dir / 'slice_info.json', 'w') as f:
                json.dump(slice_info, f, indent=2)
        
        logger.info(f"Sliced image into {len(slices)} pieces")
        return slices
    
    def slice_image_with_roi(self, image: np.ndarray, roi: RegionOfInterest,
                            save_dir: Optional[Path] = None) -> List[Tuple[str, np.ndarray]]:
        roi_x, roi_y = roi.x, roi.y
        roi_w, roi_h = roi.width, roi.height
        
        roi_image = image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        slices = self.slice_image(roi_image, save_dir)
        
        for i in range(len(slices)):
            name, _ = slices[i]
            new_name = f"roi_{name}"
            slices[i] = (new_name, slices[i][1])
        
        return slices


def split_dataset(image_dir: Path, label_dir: Path, 
                 train_ratio: float = 0.8) -> Tuple[List, List]:
    import shutil
    
    train_img_dir = image_dir.parent / 'train' / 'images'
    val_img_dir = image_dir.parent / 'val' / 'images'
    train_lbl_dir = image_dir.parent / 'train' / 'labels'
    val_lbl_dir = image_dir.parent / 'val' / 'labels'
    
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
    
    import random
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    for f in train_files:
        shutil.copy(f, train_img_dir / f.name)
        lbl_file = label_dir / f.with_suffix('.json').name
        if lbl_file.exists():
            shutil.copy(lbl_file, train_lbl_dir / lbl_file.name)
    
    for f in val_files:
        shutil.copy(f, val_img_dir / f.name)
        lbl_file = label_dir / f.with_suffix('.json').name
        if lbl_file.exists():
            shutil.copy(lbl_file, val_lbl_dir / lbl_file.name)
    
    logger.info(f"Dataset split: {len(train_files)} train, {len(val_files)} val")
    return train_files, val_files
