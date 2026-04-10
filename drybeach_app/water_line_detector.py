"""
干滩识别系统 - 水线检测模块
基于边缘检测和图像处理的水线检测算法

Classes:
    WaterLineDetector: 水线检测器，使用边缘检测方法
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy import ndimage
from scipy.interpolate import interp1d
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaterLineDetector:
    def __init__(self):
        self.detected_line = None
        self.confidence = 0.0
        
    def detect_by_edge_detection(self, image: np.ndarray, 
                                  roi: Optional[Tuple] = None) -> np.ndarray:
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges = cv2.Canny(blurred, 50, 150)
        
        edges = cv2.dilate(edges, None, iterations=2)
        edges = cv2.erode(edges, None, iterations=1)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            line_points = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 30:
                    line_points.extend([(x1, y1), (x2, y2)])
            
            if line_points:
                line_points = np.array(line_points)
                if roi:
                    line_points[:, 0] += x
                    line_points[:, 1] += y
                
                self.detected_line = self._fit_curve_to_points(line_points)
                self.confidence = 0.8
                return self.detected_line
        
        self.detected_line = self._detect_by_intensity_change(image, roi)
        return self.detected_line
    
    def _detect_by_intensity_change(self, image: np.ndarray,
                                    roi: Optional[Tuple] = None) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        h, w = gray.shape
        
        horizontal_profiles = np.mean(gray, axis=1)
        
        gradient = np.gradient(horizontal_profiles)
        
        threshold = np.mean(gradient) + 2 * np.std(gradient)
        
        significant_changes = np.where(np.abs(gradient) > threshold)[0]
        
        if len(significant_changes) > 0:
            water_line_y = significant_changes[0]
            
            if roi:
                water_line_y += roi[1]
            
            line_points = [(x, water_line_y) for x in range(w)]
            self.detected_line = np.array(line_points)
            self.confidence = 0.7
        else:
            water_line_y = h // 2
            self.detected_line = np.array([(x, water_line_y) for x in range(w)])
            self.confidence = 0.5
        
        return self.detected_line
    
    def _fit_curve_to_points(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return points
        
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        x = sorted_points[:, 0]
        y = sorted_points[:, 1]
        
        try:
            from scipy.interpolate import UnivariateSpline
            spline = UnivariateSpline(x, y, k=min(3, len(x) - 1), s=len(x))
            
            x_new = np.linspace(x.min(), x.max(), 100)
            y_new = spline(x_new)
            
            self.detected_line = np.column_stack([x_new, y_new])
        except:
            from scipy.interpolate import interp1d
            f = interp1d(x, y, kind='linear', fill_value='extrapolate')
            x_new = np.linspace(x.min(), x.max(), 100)
            y_new = f(x_new)
            self.detected_line = np.column_stack([x_new, y_new])
        
        return self.detected_line
    
    def detect_by_color_segmentation(self, image: np.ndarray,
                                     roi: Optional[Tuple] = None) -> np.ndarray:
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_water = np.array([90, 30, 50])
        upper_water = np.array([130, 255, 255])
        
        water_mask = cv2.inRange(hsv, lower_water, upper_water)
        
        kernel = np.ones((5, 5), np.uint8)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            water_line_points = []
            
            for i in range(water_mask.shape[1]):
                column = water_mask[:, i]
                water_pixels = np.where(column > 0)[0]
                
                if len(water_pixels) > 0:
                    water_line_points.append((i, water_pixels[0]))
            
            if water_line_points:
                water_line_points = np.array(water_line_points)
                if roi:
                    water_line_points[:, 0] += x
                    water_line_points[:, 1] += y
                
                self.detected_line = self._smooth_line(water_line_points)
                self.confidence = 0.85
            else:
                self.detected_line = self._detect_by_intensity_change(image, roi)
                self.confidence = 0.6
        else:
            self.detected_line = self._detect_by_intensity_change(image, roi)
            self.confidence = 0.5
        
        return self.detected_line
    
    def _smooth_line(self, points: np.ndarray) -> np.ndarray:
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        x = sorted_points[:, 0]
        y = sorted_points[:, 1]
        
        from scipy.ndimage import gaussian_filter1d
        y_smooth = gaussian_filter1d(y, sigma=3)
        
        return np.column_stack([x, y_smooth])
    
    def detect_multi_method(self, image: np.ndarray,
                           roi: Optional[Tuple] = None) -> Dict:
        results = {}
        
        results['edge'] = self.detect_by_edge_detection(image, roi)
        edge_confidence = self.confidence
        
        results['color'] = self.detect_by_color_segmentation(image, roi)
        color_confidence = self.confidence
        
        if edge_confidence > color_confidence:
            self.detected_line = results['edge']
            self.confidence = edge_confidence
            results['best_method'] = 'edge'
        else:
            self.detected_line = results['color']
            self.confidence = color_confidence
            results['best_method'] = 'color'
        
        results['final_line'] = self.detected_line
        results['confidence'] = self.confidence
        
        return results


class AdaptiveWaterLineDetector:
    def __init__(self):
        self.base_detector = WaterLineDetector()
        self.calibration_params = {}
        
    def calibrate_with_reference(self, reference_image: np.ndarray,
                                 known_distance: float,
                                 reference_points: List[Tuple]):
        self.calibration_params['known_distance'] = known_distance
        self.calibration_params['reference_points'] = reference_points
        
        x1, y1 = reference_points[0]
        x2, y2 = reference_points[1]
        pixel_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        self.calibration_params['pixels_per_meter'] = pixel_distance / known_distance
        logger.info(f"Calibrated: {pixel_distance} pixels = {known_distance} meters")
    
    def detect_and_measure(self, image: np.ndarray,
                          roi: Optional[Tuple] = None) -> Dict:
        results = self.base_detector.detect_multi_method(image, roi)
        
        if self.calibration_params:
            pixels_per_meter = self.calibration_params['pixels_per_meter']
            results['pixels_per_meter'] = pixels_per_meter
        
        return results


def visualize_water_line(image: np.ndarray, water_line: np.ndarray,
                        color: Tuple[int, int, int] = (0, 255, 255),
                        thickness: int = 3) -> np.ndarray:
    result = image.copy()
    
    if len(water_line) > 0:
        pts = water_line.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(result, [pts], False, color, thickness)
        
        cv2.circle(result, tuple(pts[0][0]), 5, (0, 0, 255), -1)
        cv2.circle(result, tuple(pts[-1][0]), 5, (0, 255, 0), -1)
    
    return result
