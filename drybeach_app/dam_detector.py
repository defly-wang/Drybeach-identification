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
        ih, iw = gray.shape
        
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
                
                if angle > 60:
                    vertical_lines.append((min(x1, x2), y1, max(x1, x2), y2))
                elif angle < 30:
                    horizontal_lines.append((x1, min(y1, y2), x2, max(y1, y2)))
        
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        grad_strength = np.mean(np.abs(grad_y), axis=0)
        
        threshold = np.percentile(grad_strength, 75)
        significant_mask = grad_strength > threshold
        
        dam_edges_found = False
        
        if vertical_lines and len(vertical_lines) < 50:
            clustered_left_xs = []
            clustered_right_xs = []
            
            for vl in vertical_lines:
                lx1, _, lx2, _ = vl
                if lx1 < iw * 0.3:
                    clustered_left_xs.append(lx1)
                elif lx2 > iw * 0.7:
                    clustered_right_xs.append(lx2)
            
            if clustered_left_xs:
                left_x = int(np.median(clustered_left_xs))
                dam_edges_found = True
            else:
                left_x = 0
            
            if clustered_right_xs:
                right_x = int(np.median(clustered_right_xs))
                dam_edges_found = True
            else:
                right_x = iw
            
            if left_x >= right_x:
                left_x = 0
                right_x = iw
                dam_edges_found = False
        else:
            left_x = 0
            right_x = iw
        
        if dam_edges_found:
            self.left_edge = (left_x, 0, left_x, ih)
            self.right_edge = (right_x, 0, right_x, ih)
        elif significant_mask.any():
            edge_cols = np.where(significant_mask)[0]
            if len(edge_cols) > 0:
                left_x = int(np.percentile(edge_cols, 10))
                right_x = int(np.percentile(edge_cols, 90))
                self.left_edge = (left_x, 0, left_x, ih)
                self.right_edge = (right_x, 0, right_x, ih)
                dam_edges_found = True
            else:
                self.left_edge = (0, 0, 0, ih)
                self.right_edge = (iw, 0, iw, ih)
        else:
            self.left_edge = (0, 0, 0, ih)
            self.right_edge = (iw, 0, iw, ih)
        
        if horizontal_lines and len(horizontal_lines) < 30:
            top_ys = [min(l[1], l[3]) for l in horizontal_lines if l[1] < ih * 0.4 or l[3] < ih * 0.4]
            bottom_ys = [max(l[1], l[3]) for l in horizontal_lines if l[1] > ih * 0.5 or l[3] > ih * 0.5]
            
            top_y = int(np.median(top_ys)) if top_ys else 0
            bottom_y = int(np.median(bottom_ys)) if bottom_ys else ih
            
            if top_y >= bottom_y:
                top_y = 0
                bottom_y = ih
        else:
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            row_strength = np.mean(np.abs(grad_x), axis=1)
            threshold_row = np.percentile(row_strength, 60)
            
            top_candidates = np.where(row_strength[:ih//2] > threshold_row)[0]
            bottom_candidates = np.where(row_strength[ih//2:] > threshold_row)[0] + ih // 2
            
            top_y = int(np.median(top_candidates)) if len(top_candidates) > 0 else 0
            bottom_y = int(np.median(bottom_candidates)) if len(bottom_candidates) > 0 else ih
            
            if top_y >= bottom_y:
                top_y = 0
                bottom_y = ih
        
        self.top_edge = (0, top_y, iw, top_y)
        self.bottom_edge = (0, bottom_y, iw, bottom_y)
        
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
        
        if dam_right_x <= dam_left_x:
            dam_right_x = dam_left_x + max(1, iw // 4)
        if dam_bottom_y <= dam_top_y:
            dam_bottom_y = dam_top_y + max(1, ih // 4)
        
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
