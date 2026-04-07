import cv2
import numpy as np
from typing import Optional, Tuple

from PyQt6.QtWidgets import QLabel, QSizePolicy, QWidget, QScrollArea, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor


class ImageViewer(QLabel):
    roi_selected = pyqtSignal(tuple)
    calibration_point_clicked = pyqtSignal(tuple)
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #2a2a2a;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        
        self.mode = 'normal'
        self.drag_start = None
        self.drag_end = None
        self.calibration_points = []
        self.roi_rect = None
        
        self.detection_region = None
        self.detection_region_polygon = []
        
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.displayed_size = QSize()
        
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
    
    def set_mode(self, mode: str):
        self.mode = mode
        self.calibration_points = []
        self.roi_rect = None
        self.drag_start = None
        self.drag_end = None
        if mode != 'draw':
            self.detection_region_polygon = []
        self.update_overlay()
    
    def set_detection_region(self, region_data):
        if region_data and 'points' in region_data:
            self.detection_region_polygon = [(float(p['x']), float(p['y'])) for p in region_data['points']]
            self.detection_region = region_data
            self._update_display()
    
    def _update_display(self):
        if self.current_pixmap is None:
            return
        
        scaled_size = self.current_pixmap.size() * self._zoom_factor
        scaled_pixmap = self.current_pixmap.scaled(
            scaled_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.displayed_size = scaled_pixmap.size()
        
        self.setFixedSize(self.displayed_size)
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.setPixmap(scaled_pixmap)
    
    def set_image(self, image: np.ndarray):
        if image is None:
            return
        
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, c = rgb_image.shape
        bytes_per_line = c * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        self.current_pixmap = pixmap
        
        self._update_display()
        
        self.current_image = image
    
    def _update_display(self):
        if self.current_pixmap is None:
            return
        
        scaled_size = self.current_pixmap.size() * self._zoom_factor
        scaled_pixmap = self.current_pixmap.scaled(
            scaled_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.displayed_size = scaled_pixmap.size()
        
        self.setFixedSize(self.displayed_size)
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.setPixmap(scaled_pixmap)
    
    def wheelEvent(self, event):
        if self.current_pixmap is None:
            return
        
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_factor *= 1.1
        else:
            self._zoom_factor /= 1.1
        
        self._zoom_factor = max(self._min_zoom, min(self._max_zoom, self._zoom_factor))
        self._update_display()
    
    def _create_overlay(self) -> Optional[QPixmap]:
        if self.displayed_size.isEmpty():
            return None
        
        overlay = QPixmap(self.displayed_size)
        overlay.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(overlay)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        
        if self.mode == 'roi' and self.drag_start and self.drag_end:
            x1 = min(self.drag_start.x(), self.drag_end.x())
            y1 = min(self.drag_start.y(), self.drag_end.y())
            x2 = max(self.drag_start.x(), self.drag_end.x())
            y2 = max(self.drag_start.y(), self.drag_end.y())
            
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            
            text = f"ROI: ({int(x1*self.scale_factor)},{int(y1*self.scale_factor)})-({int(x2*self.scale_factor)},{int(y2*self.scale_factor)})"
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(x1 + 5, y1 + 15, text)
        
        elif self.mode == 'calibration' and self.calibration_points:
            for i, pt in enumerate(self.calibration_points):
                painter.setPen(QPen(Qt.GlobalColor.yellow, 3))
                painter.drawEllipse(int(pt.x()) - 5, int(pt.y()) - 5, 10, 10)
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(int(pt.x()) + 8, int(pt.y()) + 8, f"P{i+1}")
            
            if len(self.calibration_points) == 2:
                p1 = self.calibration_points[0]
                p2 = self.calibration_points[1]
                painter.setPen(QPen(Qt.GlobalColor.yellow, 2))
                painter.drawLine(int(p1.x()), int(p1.y()), int(p2.x()), int(p2.y()))
                
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                dist = int(np.sqrt(dx*dx + dy*dy) * self.scale_factor)
                mid_x = int((p1.x() + p2.x()) / 2)
                mid_y = int((p1.y() + p2.y()) / 2)
                painter.setPen(QPen(Qt.GlobalColor.white, 1))
                painter.drawText(mid_x + 5, mid_y - 5, f"{dist}px")
        
        if hasattr(self, 'detection_region_polygon') and self.detection_region_polygon:
            painter.setPen(QPen(Qt.GlobalColor.cyan, 2))
            for i in range(len(self.detection_region_polygon)):
                p1 = self.detection_region_polygon[i]
                p2 = self.detection_region_polygon[(i + 1) % len(self.detection_region_polygon)]
                painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            
            if self.detection_region:
                color = QColor(0, 255, 255, 50)
                painter.setBrush(color)
                painter.setPen(QPen(Qt.GlobalColor.cyan, 1))
                polygon = [QPoint(int(p[0]), int(p[1])) for p in self.detection_region_polygon]
                painter.drawPolygon(polygon)
        
        painter.end()
        return overlay
    
    def update_overlay(self):
        if self.current_pixmap is None:
            return
        
        scaled_pixmap = self.current_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                                  Qt.TransformationMode.SmoothTransformation)
        self.displayed_size = scaled_pixmap.size()
        self.offset_x = (self.size().width() - self.displayed_size.width()) // 2
        self.offset_y = (self.size().height() - self.displayed_size.height()) // 2
        self.scale_factor = self.current_pixmap.width() / self.displayed_size.width() if self.displayed_size.width() > 0 else 1.0
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.setPixmap(scaled_pixmap)
    
    def _widget_to_image(self, widget_x: int, widget_y: int) -> Tuple[int, int]:
        img_x = int((widget_x - self.offset_x) * self.scale_factor)
        img_y = int((widget_y - self.offset_y) * self.scale_factor)
        
        img_w = self.current_pixmap.width() if self.current_pixmap else 0
        img_h = self.current_pixmap.height() if self.current_pixmap else 0
        
        img_x = max(0, min(img_x, img_w - 1))
        img_y = max(0, min(img_y, img_h - 1))
        
        return img_x, img_y
    
    def mousePressEvent(self, event):
        if self.current_pixmap is None:
            return
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.mode == 'roi':
                self.drag_start = event.position().toPoint()
                self.drag_end = self.drag_start
            elif self.mode == 'calibration':
                if len(self.calibration_points) < 2:
                    self.calibration_points.append(event.position().toPoint())
                    self.update_overlay()
                    img_x, img_y = self._widget_to_image(
                        event.position().x(), event.position().y()
                    )
                    self.calibration_point_clicked.emit((img_x, img_y))
            elif self.mode == 'draw' and self.current_category == 'detection_region':
                if not hasattr(self, 'detection_region_polygon'):
                    self.detection_region_polygon = []
                self.detection_region_polygon.append((
                    event.position().x(),
                    event.position().y()
                ))
                self.update()
    
    def mouseDoubleClickEvent(self, event):
        if self.mode == 'draw' and self.current_category == 'detection_region':
            if len(self.detection_region_polygon) >= 3:
                self.detection_region = {
                    'points': [{'x': x, 'y': y} for x, y in self.detection_region_polygon]
                }
                self.mode = 'normal'
                self.update()
    
    def mouseMoveEvent(self, event):
        if self.current_pixmap is None or self.mode != 'roi':
            return
        
        if self.drag_start is not None:
            self.drag_end = event.position().toPoint()
            self.update_overlay()
    
    def mouseReleaseEvent(self, event):
        if self.current_pixmap is None or self.mode != 'roi':
            return
        
        if event.button() == Qt.MouseButton.LeftButton and self.drag_start is not None:
            self.drag_end = event.position().toPoint()
            
            x1 = min(self.drag_start.x(), self.drag_end.x())
            y1 = min(self.drag_start.y(), self.drag_end.y())
            x2 = max(self.drag_start.x(), self.drag_end.x())
            y2 = max(self.drag_start.y(), self.drag_end.y())
            
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                ix1, iy1 = self._widget_to_image(x1, y1)
                ix2, iy2 = self._widget_to_image(x2, y2)
                
                self.roi_rect = (min(ix1, ix2), min(iy1, iy2), 
                                abs(ix2 - ix1), abs(iy2 - iy1))
                self.roi_selected.emit(self.roi_rect)
            
            self.drag_start = None
            self.drag_end = None
    
    def clear_viewer(self):
        self.current_image = None
        self.current_pixmap = None
        self.calibration_points = []
        self.roi_rect = None
        self.drag_start = None
        self.drag_end = None
        self.clear()


class MarkImageViewer(QWidget):
    region_drawn = pyqtSignal(str, tuple)
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_pixmap = None
        self.image_filename = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setStyleSheet("border: 2px solid #ccc; background-color: #2a2a2a;")
        
        self.canvas = QLabel()
        self.canvas.setMouseTracking(True)
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setStyleSheet("background-color: #2a2a2a;")
        
        self.scroll_area.setWidget(self.canvas)
        layout.addWidget(self.scroll_area)
        
        self.displayed_size = QSize()
        
        self.mode = 'normal'
        self.current_category = None
        self.regions = {'水面': [], '摊面': [], '分界线': [], '坝体': []}
        self.region_colors = {
            '水面': (0, 200, 255),
            '摊面': (200, 255, 0),
            '分界线': (255, 0, 255),
            '坝体': (255, 100, 0)
        }
        
        self.polygon_points = []
        self._mouse_pos = None
        
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        
        self.canvas.mousePressEvent = self._canvas_mouse_press
        self.canvas.mouseMoveEvent = self._canvas_mouse_move
        self.canvas.mouseReleaseEvent = self._canvas_mouse_release
        self.canvas.mouseDoubleClickEvent = self._canvas_double_click
    
    def set_image(self, image: np.ndarray, filename: str = None):
        if image is None:
            return
        
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.current_image = rgb_image.copy()
        h, w = rgb_image.shape[:2]
        bytes_per_line = 3 * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qt_image)
        
        self.image_filename = filename
        self._zoom_factor = 1.0
        self._update_display()
    
    def _update_display(self):
        if self.current_pixmap is None:
            return
        
        scaled_size = self.current_pixmap.size() * self._zoom_factor
        scaled_pixmap = self.current_pixmap.scaled(
            scaled_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.displayed_size = scaled_pixmap.size()
        
        self.canvas.setFixedSize(self.displayed_size)
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.canvas.setPixmap(scaled_pixmap)
    
    def _create_overlay(self) -> Optional[QPixmap]:
        if self.displayed_size.isEmpty():
            return None
        
        overlay = QPixmap(self.displayed_size)
        overlay.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(overlay)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        
        scale_x = self.displayed_size.width() / self.current_pixmap.width() if self.current_pixmap.width() > 0 else 1.0
        scale_y = self.displayed_size.height() / self.current_pixmap.height() if self.current_pixmap.height() > 0 else 1.0
        
        for category, polygons in self.regions.items():
            color = self.region_colors.get(category, (255, 255, 0))
            qt_color = QColor(*reversed(color))
            painter.setPen(QPen(qt_color, 2))
            
            for polygon in polygons:
                points = polygon['points']
                if len(points) >= 3:
                    for i in range(len(points)):
                        p1 = points[i]
                        p2 = points[(i + 1) % len(points)]
                        painter.drawLine(int(p1[0] * scale_x), int(p1[1] * scale_y),
                                        int(p2[0] * scale_x), int(p2[1] * scale_y))
                    
                    min_x = min(p[0] * scale_x for p in points)
                    min_y = min(p[1] * scale_y for p in points)
                    painter.setPen(QPen(qt_color, 1))
                    font = painter.font()
                    font.setPointSize(10)
                    painter.setFont(font)
                    painter.drawText(int(min_x) + 5, int(min_y) + 15, category)
        
        if self.mode == 'draw' and self.current_category and self.polygon_points:
            color = self.region_colors.get(self.current_category, (255, 255, 0))
            qt_color = QColor(*reversed(color))
            
            disp_points = [(int(px * scale_x), int(py * scale_y)) for px, py in self.polygon_points]
            
            for i in range(len(disp_points) - 1):
                p1 = disp_points[i]
                p2 = disp_points[i + 1]
                painter.setPen(QPen(qt_color, 2))
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])
            
            if self._mouse_pos and len(disp_points) > 0:
                last_pt = disp_points[-1]
                preview_color = QColor(255, 255, 255)
                painter.setPen(QPen(preview_color, 1))
                painter.drawLine(last_pt[0], last_pt[1], int(self._mouse_pos[0]), int(self._mouse_pos[1]))
            
            for pt in disp_points:
                painter.setPen(QPen(qt_color, 2))
                painter.setBrush(qt_color)
                painter.drawEllipse(pt[0] - 5, pt[1] - 5, 10, 10)
        
        if self.image_filename:
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            font = painter.font()
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(10, 20, self.image_filename)
        
        painter.end()
        return overlay
    
    def wheelEvent(self, event):
        if self.current_pixmap is None:
            return
        
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_factor *= 1.1
        else:
            self._zoom_factor /= 1.1
        
        self._zoom_factor = max(self._min_zoom, min(self._max_zoom, self._zoom_factor))
        self._update_display()
    
    def _canvas_mouse_press(self, event):
        if self.current_pixmap is None:
            return
        
        pos = event.position().toPoint()
        
        if event.button() == Qt.MouseButton.RightButton:
            if self.mode == 'draw' and self.current_category:
                if self.polygon_points:
                    self.polygon_points.pop()
                    self._mouse_pos = None
                    self._update_display()
            else:
                self.scroll_area.wheelEvent(event)
            return
        
        if self.mode == 'draw' and self.current_category:
            if event.button() == Qt.MouseButton.LeftButton:
                if 0 <= pos.x() < self.displayed_size.width() and 0 <= pos.y() < self.displayed_size.height():
                    scale_x = self.current_pixmap.width() / self.displayed_size.width() if self.displayed_size.width() > 0 else 1.0
                    scale_y = self.current_pixmap.height() / self.displayed_size.height() if self.displayed_size.height() > 0 else 1.0
                    img_x = int(pos.x() * scale_x)
                    img_y = int(pos.y() * scale_y)
                    self.polygon_points.append((img_x, img_y))
                    self._update_display()
    
    def _canvas_mouse_move(self, event):
        if self.current_pixmap is None:
            return
        
        pos = event.position().toPoint()
        
        if self.mode == 'draw' and self.current_category and self.polygon_points:
            if 0 <= pos.x() < self.displayed_size.width() and 0 <= pos.y() < self.displayed_size.height():
                self._mouse_pos = (pos.x(), pos.y())
            else:
                self._mouse_pos = None
            self._update_display()
    
    def _canvas_mouse_release(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            pass
    
    def _canvas_double_click(self, event):
        if self.current_pixmap is None:
            return
        
        if event.button() == Qt.MouseButton.LeftButton and self.mode == 'draw' and self.current_category:
            if len(self.polygon_points) >= 3:
                self.regions[self.current_category].append({
                    'points': list(self.polygon_points)
                })
                self.region_drawn.emit(self.current_category, tuple(self.polygon_points))
            
            self.polygon_points = []
            self._mouse_pos = None
            self._update_display()
    
    def clear_regions(self):
        self.regions = {'水面': [], '摊面': [], '分界线': [], '坝体': []}
        self.polygon_points = []
        self._mouse_pos = None
        self._update_display()


class WaterLineViewer(QWidget):
    region_drawn = pyqtSignal(str, tuple)
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_pixmap = None
        self.displayed_size = QSize()
        
        self.mode = 'normal'
        self.polygon_points = []
        self.regions = {'检测区域': []}
        self.region_colors = {
            '检测区域': (0, 255, 0)
        }
        self.current_category = None
        self._mouse_pos = None
        
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        
        self._setup_ui()
        
        self.canvas.mousePressEvent = self._canvas_mouse_press
        self.canvas.mouseMoveEvent = self._canvas_mouse_move
        self.canvas.mouseReleaseEvent = self._canvas_mouse_release
        self.canvas.mouseDoubleClickEvent = self._canvas_double_click
    
    def set_mode(self, mode: str):
        self.mode = mode
        self.polygon_points = []
        self._mouse_pos = None
        self._update_display()
    
    def clear_regions(self):
        self.regions = {'检测区域': []}
        self.polygon_points = []
        self._mouse_pos = None
        self._update_display()
    
    def _setup_ui(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setStyleSheet("background-color: #2a2a2a;")
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMouseTracking(True)
        
        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)
        
        self.scroll_area.setWidget(self.canvas)
    
    def set_image(self, image: np.ndarray):
        if image is None:
            return
        
        if len(image.shape) == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        self.current_image = rgb_image.copy()
        h, w = rgb_image.shape[:2]
        bytes_per_line = 3 * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qt_image)
        
        self._zoom_factor = 1.0
        self._update_display()
    
    def _update_display(self):
        if self.current_pixmap is None:
            return
        
        scaled_size = self.current_pixmap.size() * self._zoom_factor
        scaled_pixmap = self.current_pixmap.scaled(
            scaled_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.displayed_size = scaled_pixmap.size()
        
        self.canvas.setFixedSize(self.displayed_size)
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.canvas.setPixmap(scaled_pixmap)
    
    def _create_overlay(self) -> Optional[QPixmap]:
        if self.displayed_size.isEmpty():
            return None
        
        overlay = QPixmap(self.displayed_size)
        overlay.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(overlay)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        
        scale_x = self.displayed_size.width() / self.current_pixmap.width() if self.current_pixmap.width() > 0 else 1.0
        scale_y = self.displayed_size.height() / self.current_pixmap.height() if self.current_pixmap.height() > 0 else 1.0
        
        for category, polygons in self.regions.items():
            color = self.region_colors.get(category, (0, 255, 0))
            qt_color = QColor(*reversed(color))
            painter.setPen(QPen(qt_color, 2))
            
            for polygon in polygons:
                points = polygon['points']
                if len(points) >= 3:
                    for i in range(len(points)):
                        p1 = points[i]
                        p2 = points[(i + 1) % len(points)]
                        painter.drawLine(int(p1[0] * scale_x), int(p1[1] * scale_y),
                                        int(p2[0] * scale_x), int(p2[1] * scale_y))
                    
                    min_x = min(p[0] * scale_x for p in points)
                    min_y = min(p[1] * scale_y for p in points)
                    painter.setPen(QPen(qt_color, 1))
                    font = painter.font()
                    font.setPointSize(10)
                    painter.setFont(font)
                    painter.drawText(int(min_x) + 5, int(min_y) + 15, category)
        
        if self.mode == 'draw' and self.current_category and self.polygon_points:
            color = self.region_colors.get(self.current_category, (0, 255, 0))
            qt_color = QColor(*reversed(color))
            
            disp_points = [(int(px * scale_x), int(py * scale_y)) for px, py in self.polygon_points]
            
            for i in range(len(disp_points) - 1):
                p1 = disp_points[i]
                p2 = disp_points[i + 1]
                painter.setPen(QPen(qt_color, 2))
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])
            
            if self._mouse_pos and len(disp_points) > 0:
                last_pt = disp_points[-1]
                preview_color = QColor(255, 255, 255)
                painter.setPen(QPen(preview_color, 1))
                painter.drawLine(last_pt[0], last_pt[1], int(self._mouse_pos[0]), int(self._mouse_pos[1]))
            
            for pt in disp_points:
                painter.setPen(QPen(qt_color, 2))
                painter.setBrush(qt_color)
                painter.drawEllipse(pt[0] - 5, pt[1] - 5, 10, 10)
        
        painter.end()
        return overlay
    
    def wheelEvent(self, event):
        if self.current_pixmap is None:
            return
        
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_factor *= 1.1
        else:
            self._zoom_factor /= 1.1
        
        self._zoom_factor = max(self._min_zoom, min(self._max_zoom, self._zoom_factor))
        self._update_display()
    
    def _canvas_mouse_press(self, event):
        if self.current_pixmap is None:
            return
        
        pos = event.position().toPoint()
        
        if event.button() == Qt.MouseButton.RightButton:
            if self.mode == 'draw' and self.current_category:
                if self.polygon_points:
                    self.polygon_points.pop()
                    self._mouse_pos = None
                    self._update_display()
            return
        
        if self.mode == 'draw' and self.current_category:
            if event.button() == Qt.MouseButton.LeftButton:
                if 0 <= pos.x() < self.displayed_size.width() and 0 <= pos.y() < self.displayed_size.height():
                    scale_x = self.current_pixmap.width() / self.displayed_size.width() if self.displayed_size.width() > 0 else 1.0
                    scale_y = self.current_pixmap.height() / self.displayed_size.height() if self.displayed_size.height() > 0 else 1.0
                    img_x = int(pos.x() * scale_x)
                    img_y = int(pos.y() * scale_y)
                    self.polygon_points.append((img_x, img_y))
                    self._update_display()
    
    def _canvas_mouse_move(self, event):
        if self.current_pixmap is None:
            return
        
        pos = event.position().toPoint()
        
        if self.mode == 'draw' and self.current_category and self.polygon_points:
            if 0 <= pos.x() < self.displayed_size.width() and 0 <= pos.y() < self.displayed_size.height():
                self._mouse_pos = (pos.x(), pos.y())
            else:
                self._mouse_pos = None
            self._update_display()
    
    def _canvas_mouse_release(self, event):
        pass
    
    def _canvas_double_click(self, event):
        if self.current_pixmap is None:
            return
        
        if event.button() == Qt.MouseButton.LeftButton and self.mode == 'draw' and self.current_category:
            if len(self.polygon_points) >= 3:
                self.regions[self.current_category].append({
                    'points': list(self.polygon_points)
                })
                self.region_drawn.emit(self.current_category, tuple(self.polygon_points))
            
            self.polygon_points = []
            self._mouse_pos = None
            self._update_display()
    
    def clear_line(self):
        self.polygon_points = []
        self._mouse_pos = None
        self._update_display()
