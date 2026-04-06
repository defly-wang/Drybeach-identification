import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

from .water_line_detector import WaterLineDetector, AdaptiveWaterLineDetector
from .dam_detector import DamDetector
from .distance_calculator import DistanceCalculator, MeasurementReporter
from .image_annotator import RegionOfInterest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionResult:
    def __init__(self):
        self.water_line = None
        self.water_line_confidence = 0.0
        self.dam_bbox = None
        self.dam_edges = None
        self.shortest_distance = None
        self.distance_meters = None
        self.annotated_image = None
        self.metadata = {}
    
    def to_dict(self) -> Dict:
        import json
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.intc, np.intp, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif obj is None or isinstance(obj, (int, float, str, bool)):
                return obj
            else:
                return str(obj)
        
        return {
            'water_line': convert_to_serializable(self.water_line),
            'water_line_confidence': float(self.water_line_confidence),
            'dam_bbox': convert_to_serializable(self.dam_bbox),
            'dam_edges': convert_to_serializable(self.dam_edges),
            'shortest_distance': convert_to_serializable(self.shortest_distance),
            'distance_meters': convert_to_serializable(self.distance_meters),
            'metadata': convert_to_serializable(self.metadata)
        }


class DryBeachRecognizer:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.water_detector = WaterLineDetector()
        self.dam_detector = DamDetector()
        self.distance_calculator = DistanceCalculator()
        self.measurement_reporter = MeasurementReporter()
        self.calibrated = False
        
        self.result = DetectionResult()
        
        logger.info("DryBeachRecognizer initialized")
    
    def calibrate(self, known_distance_meters: float,
                  reference_points: List[Tuple[int, int]]):
        p1, p2 = reference_points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        self.distance_calculator.calibrate(known_distance_meters, pixel_distance)
        self.calibrated = True
        logger.info(f"Calibrated with reference: {known_distance_meters}m = {pixel_distance:.2f}px")
    
    def detect(self, image: np.ndarray,
              roi: Optional[RegionOfInterest] = None,
              method: str = 'multi') -> DetectionResult:
        roi_tuple = None
        if roi:
            roi_tuple = (roi.x, roi.y, roi.width, roi.height)
        
        if method == 'multi':
            water_results = self.water_detector.detect_multi_method(image, roi_tuple)
            self.result.water_line = water_results['final_line']
            self.result.water_line_confidence = water_results['confidence']
        elif method == 'edge':
            self.result.water_line = self.water_detector.detect_by_edge_detection(image, roi_tuple)
        elif method == 'color':
            self.result.water_line = self.water_detector.detect_by_color_segmentation(image, roi_tuple)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        dam_results = self.dam_detector.detect_dam_edges(image, roi_tuple)
        self.result.dam_bbox = dam_results['bbox']
        self.result.dam_edges = {
            'left': dam_results['left_edge'],
            'right': dam_results['right_edge'],
            'top': dam_results['top_edge'],
            'bottom': dam_results['bottom_edge']
        }
        
        if self.result.water_line is not None and self.result.dam_bbox is not None:
            dam_boundary = self.dam_detector.get_dam_boundary()
            
            distance_result = self.distance_calculator.calculate_shortest_distance(
                self.result.water_line, dam_boundary
            )
            
            self.result.shortest_distance = distance_result['distance_pixels']
            self.result.distance_meters = distance_result['distance_meters']
            
            self.measurement_reporter.add_measurement(
                'shortest_distance',
                distance_result['distance_meters'],
                'meters',
                {
                    'water_point': distance_result['point_water'],
                    'dam_point': distance_result['point_dam']
                }
            )
            
            line_properties = self.distance_calculator.calculate_water_line_properties(
                self.result.water_line
            )
            self.result.metadata['water_line_properties'] = line_properties
        
        return self.result
    
    def detect_and_visualize(self, image: np.ndarray,
                           roi: Optional[RegionOfInterest] = None,
                           method: str = 'multi') -> Tuple[DetectionResult, np.ndarray]:
        self.detect(image, roi, method)
        
        annotated = self._create_annotated_image(image)
        self.result.annotated_image = annotated
        
        return self.result, annotated
    
    def _create_annotated_image(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        
        if self.result.water_line is not None and len(self.result.water_line) > 0:
            pts = self.result.water_line.astype(np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(result, [pts], False, (0, 255, 255), 3)
            
            for i in range(0, len(pts), max(1, len(pts) // 5)):
                cv2.circle(result, tuple(pts[i][0]), 5, (0, 255, 0), -1)
        
        if self.result.dam_bbox is not None:
            x, y, w, h = self.result.dam_bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result, "Dam", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if self.result.dam_edges:
            edges = self.result.dam_edges
            if edges.get('left'):
                x1, y1, x2, y2 = edges['left']
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if edges.get('right'):
                x1, y1, x2, y2 = edges['right']
                cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        if self.result.shortest_distance is not None and self.distance_calculator.pixels_per_meter:
            mid_x = int(image.shape[1] / 2)
            
            cv2.line(result, 
                    (int(image.shape[1] / 2), 0),
                    (int(image.shape[1] / 2), image.shape[0]),
                    (0, 255, 0), 1)
            
            dist_text = f"Distance: {self.result.distance_meters:.2f} m"
            cv2.putText(result, dist_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            conf_text = f"Confidence: {self.result.water_line_confidence:.2%}"
            cv2.putText(result, conf_text, (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result
    
    def batch_detect(self, images: List[np.ndarray],
                    roi: Optional[RegionOfInterest] = None,
                    method: str = 'multi') -> List[DetectionResult]:
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing image {i+1}/{len(images)}")
            result = self.detect(image, roi, method)
            results.append(result)
        
        return results
    
    def save_result(self, output_path: Path, result: DetectionResult,
                   image: Optional[np.ndarray] = None):
        import json
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.intc, np.intp, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, tuple):
                    return list(obj)
                return super().default(obj)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = result.to_dict()
        
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        if image is not None and result.annotated_image is not None:
            cv2.imwrite(str(output_path.with_suffix('.jpg')), 
                       result.annotated_image)
        
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
