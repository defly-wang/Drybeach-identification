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
        self.boundary_lines = []
    
    def to_dict(self) -> Dict:
        return {
            'class_map': self.class_map,
            'class_counts': self.class_counts,
            'detection_points': self.detection_points,
            'boundary_lines': self.boundary_lines
        }


class BoundaryLineGenerator:
    CATEGORY_NAMES = ['water', 'beach', 'boundary', 'dam']
    BOUNDARY_CLASS_ID = 2
    
    @staticmethod
    def generate_boundary_lines(detection_points: List[Dict], image_shape: Tuple[int, int], 
                                stride: int = 16) -> List[Dict]:
        if not detection_points:
            return []
        
        boundary_points = [p for p in detection_points if p['class_id'] == BoundaryLineGenerator.BOUNDARY_CLASS_ID]
        
        if len(boundary_points) < 3:
            logger.warning("Not enough boundary points")
            return []
        
        points = np.array([[p['x'], p['y']] for p in boundary_points])
        
        regions = BoundaryLineGenerator._group_by_adjacency(points, stride)
        
        boundary_lines = []
        for region_points in regions:
            if len(region_points) < 5:
                continue
            
            stroke_path = BoundaryLineGenerator._extract_brush_stroke(region_points)
            
            if stroke_path and len(stroke_path) > 2:
                boundary_lines.append({
                    'type': 'stroke',
                    'points': stroke_path,
                    'num_points': len(region_points)
                })
        
        logger.info(f"Generated {len(boundary_lines)} stroke paths from {len(boundary_points)} boundary points")
        return boundary_lines
    
    @staticmethod
    def _group_by_adjacency(points: np.ndarray, stride: int) -> List[np.ndarray]:
        if len(points) == 0:
            return []
        
        eps = stride * 1.5
        n = len(points)
        used = np.zeros(n, dtype=bool)
        regions = []
        
        for i in range(n):
            if used[i]:
                continue
            
            cluster = [i]
            used[i] = True
            queue = [i]
            
            while queue:
                idx = queue.pop(0)
                for j in range(n):
                    if not used[j]:
                        dist = np.sqrt((points[idx, 0] - points[j, 0])**2 + 
                                      (points[idx, 1] - points[j, 1])**2)
                        if dist < eps:
                            queue.append(j)
                            cluster.append(j)
                            used[j] = True
            
            regions.append(points[cluster])
        
        regions.sort(key=lambda r: np.min(r[:, 1]))
        
        return regions
    
    @staticmethod
    def _extract_brush_stroke(points: np.ndarray) -> List[List[int]]:
        min_x, min_y = int(np.min(points[:, 0])), int(np.min(points[:, 1]))
        max_x, max_y = int(np.max(points[:, 0])) + 1, int(np.max(points[:, 1])) + 1
        
        h, w = max_y - min_y + 10, max_x - min_x + 10
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for p in points:
            x, y = int(p[0]) - min_x, int(p[1]) - min_y
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(mask, (x, y), 2, 255, -1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(mask, kernel, iterations=1)
        
        skeleton = np.zeros_like(dilated)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False
        size = img_size = dilated.shape
        
        while not done:
            eroded = cv2.erode(dilated, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(dilated, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            dilated = eroded.copy()
            done = (dilated.sum() == 0)
        
        skeleton_points = np.argwhere(skeleton > 0)
        
        if len(skeleton_points) < 3:
            skeleton_points = points - np.array([min_x, min_y])
        
        path = BoundaryLineGenerator._order_skeleton_path(skeleton_points)
        
        path = [[int(p[0] + min_x), int(p[1] + min_y)] for p in path]
        
        return path
    
    @staticmethod
    def _order_skeleton_path(points: np.ndarray) -> List[np.ndarray]:
        if len(points) < 2:
            return points.tolist()
        
        points = points.astype(float)
        
        start_idx = np.argmin(points[:, 1])
        start = points[start_idx]
        
        path = [start]
        remaining = list(range(len(points)))
        remaining.remove(start_idx)
        current = start.copy()
        
        while remaining:
            min_dist = float('inf')
            nearest_idx = -1
            
            for idx in remaining:
                d = np.linalg.norm(current - points[idx])
                if d < min_dist:
                    min_dist = d
                    nearest_idx = idx
            
            if nearest_idx == -1 or min_dist > 50:
                break
            
            path.append(points[nearest_idx])
            remaining.remove(nearest_idx)
            current = points[nearest_idx].copy()
        
        return path
    
    @staticmethod
    def draw_boundary_lines(image: np.ndarray, boundary_lines: List[Dict], 
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 3) -> np.ndarray:
        result = image.copy()
        
        for line in boundary_lines:
            points = line['points']
            if len(points) < 2:
                continue
            
            for i in range(len(points) - 1):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[i+1][0]), int(points[i+1][1]))
                cv2.line(result, pt1, pt2, color, thickness)
            
            for pt in points[::max(1, len(points) // 10)]:
                cv2.circle(result, (int(pt[0]), int(pt[1])), 5, color, -1)
        
        return result


class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, input_size: int = 64):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(),
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
        self.detection_region = None
        self.result = DetectionResult()
        self.model_info = {}
        
        if model_path and TORCH_AVAILABLE:
            self.load_model(model_path)
        
        logger.info("DryBeachRecognizer initialized")
    
    def load_model(self, model_path: str):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CNNClassifier(num_classes=4, input_size=64)
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
        
        self._original_image = image.copy()
        
        img_h, img_w = image.shape[:2]
        
        self.result = DetectionResult()
        class_counts = {name: 0 for name in self.CATEGORY_NAMES}
        detection_points = []
        
        region_mask = None
        if self.detection_region and 'points' in self.detection_region:
            region_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            points = np.array([[int(p['x']), int(p['y'])] for p in self.detection_region['points']], dtype=np.int32)
            cv2.fillPoly(region_mask, [points], 1)
        
        total_patches = 0
        for y in range(0, img_h - self.patch_size + 1, self.stride):
            for x in range(0, img_w - self.patch_size + 1, self.stride):
                if region_mask is not None:
                    patch_mask = region_mask[y:y+self.patch_size, x:x+self.patch_size]
                    if np.sum(patch_mask) < (self.patch_size * self.patch_size * 0.3):
                        continue
                total_patches += 1
        
        processed = 0
        print(f"\n{'='*50}")
        print(f"开始识别: 图像尺寸 {img_w}x{img_h}, 切片尺寸 {self.patch_size}, 步长 {self.stride}")
        print(f"总切片数: {total_patches}")
        if self.detection_region:
            print(f"识别区域: 已设置")
        print(f"{'='*50}")
        
        for y in range(0, img_h - self.patch_size + 1, self.stride):
            for x in range(0, img_w - self.patch_size + 1, self.stride):
                if region_mask is not None:
                    patch_mask = region_mask[y:y+self.patch_size, x:x+self.patch_size]
                    if np.sum(patch_mask) < (self.patch_size * self.patch_size * 0.3):
                        continue
                
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
        
        self._generate_boundary_line()
        
        print(f"\n{'='*50}")
        print("识别结果统计:")
        for name, count in class_counts.items():
            pct = count / total_patches * 100 if total_patches > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")
        print(f"{'='*50}\n")
        
        return self.result
    
    def detect_and_visualize(self, image: np.ndarray, progress_callback=None, draw_boundary: bool = False) -> Tuple[DetectionResult, np.ndarray]:
        self.detect(image, progress_callback=progress_callback)
        
        annotated = self._create_annotated_image(image, draw_boundary=draw_boundary)
        self.result.annotated_image = annotated
        
        return self.result, annotated
    
    def _create_annotated_image(self, image: np.ndarray, draw_boundary: bool = False, draw_points: bool = True) -> np.ndarray:
        result = image.copy()
        
        if draw_points:
            for point in self.result.detection_points:
                x = point['x']
                y = point['y']
                class_id = point['class_id']
                confidence = point['confidence']
                
                if confidence < 0.7:
                    continue
                
                color = self.CATEGORY_COLORS.get(class_id, (255, 255, 255))
                
                if 0 <= x < result.shape[1] and 0 <= y < result.shape[0]:
                    cv2.circle(result, (x, y), 2, color, -1)
        
        if draw_boundary:
            self._draw_boundary_line(result)
        
        return result
    
    def _generate_boundary_line(self):
        boundary_points = [
            (p['x'], p['y']) 
            for p in self.result.detection_points 
            if p['class_id'] == 2 and p['confidence'] >= 0.7
        ]
        
        logger.info(f"分界线点数量: {len(boundary_points)}")
        
        if not boundary_points:
            self.result.boundary_lines = []
            return
        
        points_array = np.array(boundary_points)
        x_min, x_max = points_array[:, 0].min(), points_array[:, 0].max()
        y_min, y_max = points_array[:, 1].min(), points_array[:, 1].max()
        
        stride = self.stride if hasattr(self, 'stride') else 16
        cell_size = stride
        
        grid_cols = int((x_max - x_min) / cell_size) + 1
        grid_rows = int((y_max - y_min) / cell_size) + 1
        
        grid = [[set() for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        for x, y in boundary_points:
            col = int((x - x_min) / cell_size)
            row = int((y - y_min) / cell_size)
            col = min(col, grid_cols - 1)
            row = min(row, grid_rows - 1)
            grid[row][col].add((x, y))
        
        visited = [[False for _ in range(grid_cols)] for _ in range(grid_rows)]
        clusters = []
        
        def flood_fill(r, c, cluster):
            stack = [(r, c)]
            
            while stack:
                cr, cc = stack.pop()
                if cr < 0 or cr >= grid_rows or cc < 0 or cc >= grid_cols:
                    continue
                if visited[cr][cc] or not grid[cr][cc]:
                    continue
                
                visited[cr][cc] = True
                cluster.update(grid[cr][cc])
                
                stack.append((cr-1, cc))
                stack.append((cr+1, cc))
                stack.append((cr, cc-1))
                stack.append((cr, cc+1))
                stack.append((cr-1, cc-1))
                stack.append((cr-1, cc+1))
                stack.append((cr+1, cc-1))
                stack.append((cr+1, cc+1))
        
        for r in range(grid_rows):
            for c in range(grid_cols):
                if not visited[r][c] and grid[r][c]:
                    cluster = set()
                    flood_fill(r, c, cluster)
                    if cluster:
                        clusters.append(list(cluster))
        
        logger.info(f"连通区域数量: {len(clusters)}")
        
        self.result.boundary_lines = []
        
        for cluster in clusters:
            cluster.sort(key=lambda p: p[0])
            
            x_coords = [p[0] for p in cluster]
            y_coords = [p[1] for p in cluster]
            
            if len(x_coords) < 2:
                self.result.boundary_lines.append(cluster)
                continue
            
            try:
                from scipy.interpolate import UnivariateSpline
                
                head_count = min(5, len(x_coords) // 3)
                tail_count = min(5, len(x_coords) // 3)
                
                head_x = x_coords[:head_count]
                head_y = y_coords[:head_count]
                tail_x = x_coords[-tail_count:]
                tail_y = y_coords[-tail_count:]
                
                head_coeffs = np.polyfit(head_x, head_y, 1)
                tail_coeffs = np.polyfit(tail_x, tail_y, 1)
                
                min_x = min(x_coords)
                max_x = max(x_coords)
                
                head_trend_y = head_coeffs[0] * min_x + head_coeffs[1]
                tail_trend_y = tail_coeffs[0] * max_x + tail_coeffs[1]
                
                start_x = min_x
                end_x = max_x
                
                unique_x = sorted(set(x_coords))
                if len(unique_x) < 4:
                    k = len(unique_x) - 1
                else:
                    k = 3
                k = min(k, len(x_coords) - 1)
                
                spline = UnivariateSpline(x_coords, y_coords, k=k, s=0)
                
                x_range = np.arange(start_x, end_x + 1, 1)
                y_interpolated = spline(x_range).astype(int)
                
                line = [(int(x), int(max(0, y))) for x, y in zip(x_range, y_interpolated)]
                self.result.boundary_lines.append(line)
                
                logger.info(f"生成分界线: {len(line)} 个点, 起点({start_x},{int(head_trend_y)}), 终点({end_x},{int(tail_trend_y)})")
            except Exception as e:
                logger.warning(f"Failed to interpolate boundary line: {e}")
                self.result.boundary_lines.append(cluster)
    
    def _draw_boundary_line(self, image: np.ndarray):
        if not hasattr(self.result, 'boundary_lines') or not self.result.boundary_lines:
            logger.warning("没有分界线数据")
            return
        
        logger.info(f"开始绘制分界线，数量: {len(self.result.boundary_lines)}")
        
        for idx, boundary_line in enumerate(self.result.boundary_lines):
            logger.info(f"  线 {idx}: {len(boundary_line)} 个点, 前3点: {boundary_line[:3]}")
        
        boundary_color = (0, 255, 255)
        
        for boundary_line in self.result.boundary_lines:
            for i in range(len(boundary_line) - 1):
                pt1 = boundary_line[i]
                pt2 = boundary_line[i + 1]
                cv2.line(image, pt1, pt2, boundary_color, 2)
    
    def draw_boundary_only(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        
        if not hasattr(self.result, 'boundary_lines') or not self.result.boundary_lines:
            return result
        
        boundary_color = (0, 255, 255)
        
        for boundary_line in self.result.boundary_lines:
            for i in range(len(boundary_line) - 1):
                pt1 = boundary_line[i]
                pt2 = boundary_line[i + 1]
                cv2.line(result, pt1, pt2, boundary_color, 2)
        
        return result
    
    def draw_boundary_only_from_original(self) -> np.ndarray:
        if not hasattr(self, '_original_image'):
            return None
        
        result = self._original_image.copy()
        return self.draw_boundary_only(result)
    
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