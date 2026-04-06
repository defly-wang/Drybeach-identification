import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.spatial.distance import cdist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistanceCalculator:
    def __init__(self):
        self.pixels_per_meter = None
        self.reference_distance = None
        self.calibration_matrix = None
        
    def calibrate(self, known_distance_meters: float,
                  pixel_distance: float):
        self.pixels_per_meter = pixel_distance / known_distance_meters
        self.reference_distance = known_distance_meters
        logger.info(f"Calibrated: {self.pixels_per_meter:.2f} pixels/meter")
    
    def calibrate_with_points(self, point1: Tuple[float, float],
                             point2: Tuple[float, float],
                             known_distance: float):
        pixel_dist = np.sqrt((point2[0] - point1[0])**2 + 
                            (point2[1] - point1[1])**2)
        self.calibrate(known_distance, pixel_dist)
    
    def pixel_to_meters(self, pixel_distance: float) -> float:
        if self.pixels_per_meter is None:
            logger.warning("Calculator not calibrated, returning pixel distance")
            return pixel_distance
        return pixel_distance / self.pixels_per_meter
    
    def meters_to_pixels(self, meters: float) -> float:
        if self.pixels_per_meter is None:
            logger.warning("Calculator not calibrated, returning meters distance")
            return meters
        return meters * self.pixels_per_meter
    
    def calculate_point_to_line_distance(self, point: Tuple[float, float],
                                         line_points: np.ndarray) -> float:
        point = np.array(point)
        
        min_distance = float('inf')
        for i in range(len(line_points) - 1):
            p1 = line_points[i]
            p2 = line_points[i + 1]
            
            distance = self._point_to_segment_distance(point, p1, p2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _point_to_segment_distance(self, point: np.ndarray,
                                   p1: np.ndarray,
                                   p2: np.ndarray) -> float:
        line_vec = p2 - p1
        point_vec = point - p1
        
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return np.linalg.norm(point - p1)
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        
        proj_length = np.clip(proj_length, 0, line_len)
        
        nearest = p1 + proj_length * line_unitvec
        
        return np.linalg.norm(point - nearest)
    
    def _get_closest_point_on_segment(self, point: np.ndarray,
                                     p1: np.ndarray,
                                     p2: np.ndarray) -> np.ndarray:
        line_vec = p2 - p1
        point_vec = point - p1
        
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return p1
        
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        
        proj_length = np.clip(proj_length, 0, line_len)
        
        return p1 + proj_length * line_unitvec
    
    def calculate_shortest_distance(self, water_line: np.ndarray,
                                   dam_boundary: List[Tuple[float, float]]) -> Dict:
        if len(water_line) == 0 or len(dam_boundary) == 0:
            return {'distance': None, 'point_water': None, 'point_dam': None}
        
        min_distance = float('inf')
        closest_water_point = None
        closest_dam_point = None
        
        for water_point in water_line:
            for i in range(len(dam_boundary)):
                p1 = np.array(dam_boundary[i])
                p2 = np.array(dam_boundary[(i + 1) % len(dam_boundary)])
                
                distance = self._point_to_segment_distance(
                    np.array(water_point), p1, p2
                )
                
                if distance < min_distance:
                    min_distance = distance
                    closest_water_point = water_point
                    closest_dam_point = self._get_closest_point_on_segment(
                        np.array(water_point), p1, p2
                    )
        
        pixel_distance = min_distance
        
        result = {
            'distance_pixels': pixel_distance,
            'point_water': tuple(closest_water_point) if closest_water_point is not None else None,
            'point_dam': tuple(closest_dam_point) if closest_dam_point is not None else None,
            'distance_meters': self.pixel_to_meters(pixel_distance)
        }
        
        return result
    
    def calculate_water_line_properties(self, water_line: np.ndarray) -> Dict:
        if len(water_line) < 2:
            return {}
        
        x_coords = water_line[:, 0]
        y_coords = water_line[:, 1]
        
        line_length_pixels = np.sum(np.sqrt(np.diff(x_coords)**2 + np.diff(y_coords)**2))
        
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        
        angles = np.arctan2(np.diff(y_coords), np.diff(x_coords))
        mean_angle = np.mean(angles) * 180 / np.pi
        
        curvature = self._calculate_curvature(water_line)
        
        return {
            'length_pixels': line_length_pixels,
            'length_meters': self.pixel_to_meters(line_length_pixels),
            'x_range': x_range,
            'y_range': y_range,
            'mean_angle': mean_angle,
            'curvature_mean': np.mean(curvature),
            'curvature_max': np.max(curvature)
        }
    
    def _calculate_curvature(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            return np.zeros(len(points))
        
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
        
        curvature = np.nan_to_num(curvature, nan=0.0)
        
        return curvature


class MeasurementReporter:
    def __init__(self):
        self.measurements = []
        
    def add_measurement(self, measurement_type: str, value: float,
                       unit: str = 'meters', metadata: Optional[Dict] = None):
        self.measurements.append({
            'type': measurement_type,
            'value': value,
            'unit': unit,
            'metadata': metadata or {}
        })
    
    def get_summary(self) -> Dict:
        if not self.measurements:
            return {}
        
        summary = {
            'total_measurements': len(self.measurements),
            'measurement_types': list(set(m['type'] for m in self.measurements)),
            'measurements': self.measurements
        }
        
        return summary
    
    def export_to_text(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("Dry Beach Measurement Report\n")
            f.write("=" * 50 + "\n\n")
            
            for i, m in enumerate(self.measurements, 1):
                f.write(f"{i}. {m['type']}: {m['value']:.4f} {m['unit']}\n")
                
                if m['metadata']:
                    for key, value in m['metadata'].items():
                        f.write(f"   - {key}: {value}\n")
            
            f.write("\n" + "=" * 50 + "\n")


def draw_measurement_annotations(image: np.ndarray,
                                 water_line: np.ndarray,
                                 dam_boundary: List[Tuple],
                                 measurement_result: Dict,
                                 calculator: DistanceCalculator) -> np.ndarray:
    result = image.copy()
    
    if len(water_line) > 0:
        pts = water_line.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result, [pts], False, (0, 255, 255), 3)
    
    for i, point in enumerate(dam_boundary):
        next_point = dam_boundary[(i + 1) % len(dam_boundary)]
        cv2.line(result, tuple(point), tuple(next_point), (255, 0, 0), 3)
    
    if measurement_result.get('point_water') and measurement_result.get('point_dam'):
        pt_water = tuple(map(int, measurement_result['point_water']))
        pt_dam = tuple(map(int, measurement_result['point_dam']))
        
        cv2.line(result, pt_water, pt_dam, (0, 255, 0), 3)
        cv2.circle(result, pt_water, 8, (0, 0, 255), -1)
        cv2.circle(result, pt_dam, 8, (255, 0, 0), -1)
        
        mid_point = ((pt_water[0] + pt_dam[0]) // 2,
                     (pt_water[1] + pt_dam[1]) // 2)
        
        distance_text = f"{measurement_result['distance_meters']:.2f} m"
        cv2.putText(result, distance_text, mid_point,
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return result
