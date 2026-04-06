import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DamEdgeDetector:
    def __init__(self):
        self.detected_edges = None
        self.dam_region = None
        
    def detect_by_texture(self, image: np.ndarray,
                         roi: Optional[Tuple] = None) -> np.ndarray:
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        dam_edges = self._find_vertical_structures(edges)
        
        if roi and len(dam_edges) > 0:
            dam_edges[:, 0] += x
            dam_edges[:, 1] += y
        
        self.detected_edges = dam_edges
        return dam_edges
    
    def _find_vertical_structures(self, edges: np.ndarray) -> np.ndarray:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                               minLineLength=30, maxLineGap=20)
        
        vertical_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle > 80 or angle < 10:
                    vertical_lines.append((x1, y1, x2, y2))
        
        if vertical_lines:
            vertical_lines = np.array(vertical_lines)
            left_edge = vertical_lines[vertical_lines[:, 0].argmin()]
            right_edge = vertical_lines[vertical_lines[:, 2].argmax()]
            
            return np.array([[left_edge[0], left_edge[1], left_edge[2], left_edge[3]],
                           [right_edge[0], right_edge[1], right_edge[2], right_edge[3]]])
        
        return np.array([])
    
    def detect_by_contour(self, image: np.ndarray,
                         roi: Optional[Tuple] = None) -> Dict:
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        dam_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                dam_contours.append(contour)
        
        if dam_contours:
            dam_contour = max(dam_contours, key=cv2.contourArea)
            
            x_dam, y_dam, w_dam, h_dam = cv2.boundingRect(dam_contour)
            
            if roi:
                x_dam += x
                y_dam += y
            
            self.dam_region = (x_dam, y_dam, w_dam, h_dam)
            
            return {
                'contour': dam_contour,
                'bbox': self.dam_region,
                'area': cv2.contourArea(dam_contour)
            }
        
        return {}


class DamDetector:
    def __init__(self):
        self.left_edge = None
        self.right_edge = None
        self.top_edge = None
        self.bottom_edge = None
        self.dam_bbox = None
        
    def detect_dam_edges(self, image: np.ndarray,
                         roi: Optional[Tuple] = None) -> Dict:
        if roi:
            x, y, w, h = roi
            image = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 30, 100)
        
        edges = cv2.dilate(edges, None, iterations=2)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 80,
                               minLineLength=50, maxLineGap=30)
        
        vertical_lines = []
        horizontal_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                angle = np.arctan2(abs(y2 - y1), abs(x2 - x1)) * 180 / np.pi
                
                if angle > 70:
                    vertical_lines.append((min(x1, x2), y1, max(x1, x2), y2))
                elif angle < 20:
                    horizontal_lines.append((x1, min(y1, y2), x2, max(y1, y2)))
        
        h, w = gray.shape
        
        if vertical_lines:
            left_line = min(vertical_lines, key=lambda l: l[0])
            right_line = max(vertical_lines, key=lambda l: l[2])
            
            self.left_edge = (left_line[0], left_line[1], left_line[2], left_line[3])
            self.right_edge = (right_line[0], right_line[1], right_line[2], right_line[3])
        else:
            self.left_edge = (0, 0, 0, h)
            self.right_edge = (w, 0, w, h)
        
        if horizontal_lines:
            top_line = min(horizontal_lines, key=lambda l: l[1])
            bottom_line = max(horizontal_lines, key=lambda l: l[3])
            
            self.top_edge = (top_line[0], top_line[1], top_line[2], top_line[3])
            self.bottom_edge = (bottom_line[0], bottom_line[1], bottom_line[2], bottom_line[3])
        else:
            self.top_edge = (0, 0, w, 0)
            self.bottom_edge = (0, h, w, h)
        
        if roi:
            self.left_edge = (self.left_edge[0] + x, self.left_edge[1] + y,
                             self.left_edge[2] + x, self.left_edge[3] + y)
            self.right_edge = (self.right_edge[0] + x, self.right_edge[1] + y,
                              self.right_edge[2] + x, self.right_edge[3] + y)
            self.top_edge = (self.top_edge[0] + x, self.top_edge[1] + y,
                           self.top_edge[2] + x, self.top_edge[3] + y)
            self.bottom_edge = (self.bottom_edge[0] + x, self.bottom_edge[1] + y,
                               self.bottom_edge[2] + x, self.bottom_edge[3] + y)
        
        dam_left_x = self.left_edge[0]
        dam_right_x = self.right_edge[2]
        dam_top_y = min(self.top_edge[1], self.top_edge[3])
        dam_bottom_y = max(self.bottom_edge[1], self.bottom_edge[3])
        
        self.dam_bbox = (dam_left_x, dam_top_y, 
                        dam_right_x - dam_left_x, 
                        dam_bottom_y - dam_top_y)
        
        return {
            'left_edge': self.left_edge,
            'right_edge': self.right_edge,
            'top_edge': self.top_edge,
            'bottom_edge': self.bottom_edge,
            'bbox': self.dam_bbox
        }
    
    def get_dam_boundary(self) -> List[Tuple[int, int]]:
        if self.dam_bbox is None:
            return []
        
        x, y, w, h = self.dam_bbox
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def visualize_dam_detection(image: np.ndarray, detector: DamDetector,
                            line_color: Tuple[int, int, int] = (255, 0, 0),
                            thickness: int = 3) -> np.ndarray:
    result = image.copy()
    
    if detector.left_edge:
        x1, y1, x2, y2 = detector.left_edge
        cv2.line(result, (x1, y1), (x2, y2), line_color, thickness)
    
    if detector.right_edge:
        x1, y1, x2, y2 = detector.right_edge
        cv2.line(result, (x1, y1), (x2, y2), line_color, thickness)
    
    if detector.top_edge:
        x1, y1, x2, y2 = detector.top_edge
        cv2.line(result, (x1, y1), (x2, y2), line_color, thickness)
    
    if detector.bottom_edge:
        x1, y1, x2, y2 = detector.bottom_edge
        cv2.line(result, (x1, y1), (x2, y2), line_color, thickness)
    
    if detector.dam_bbox:
        x, y, w, h = detector.dam_bbox
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(result, "Dam", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result
