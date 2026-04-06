import cv2
import numpy as np
from typing import Optional, Tuple

from PyQt6.QtWidgets import QLabel, QSizePolicy
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
        
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.displayed_size = QSize()
    
    def set_mode(self, mode: str):
        self.mode = mode
        self.calibration_points = []
        self.roi_rect = None
        self.drag_start = None
        self.drag_end = None
        self.update_overlay()
    
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
        
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        self.displayed_size = scaled_pixmap.size()
        
        self.offset_x = (self.size().width() - self.displayed_size.width()) // 2
        self.offset_y = (self.size().height() - self.displayed_size.height()) // 2
        
        self.scale_factor = w / self.displayed_size.width() if self.displayed_size.width() > 0 else 1.0
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.setPixmap(scaled_pixmap)
        self.current_image = image
    
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


class MarkImageViewer(QLabel):
    region_drawn = pyqtSignal(str, tuple)
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 2px solid #ccc; background-color: #2a2a2a;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
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
        
        self.draw_start = None
        self.draw_end = None
        self.is_panning = False
        self.pan_start = None
        
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
    
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
        
        self.offset_x = (self.size().width() - self.displayed_size.width()) // 2
        self.offset_y = (self.size().height() - self.displayed_size.height()) // 2
        
        overlay = self._create_overlay()
        if overlay:
            painter = QPainter(scaled_pixmap)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
        
        self.setPixmap(scaled_pixmap)
    
    def _create_overlay(self) -> Optional[QPixmap]:
        if self.displayed_size.isEmpty():
            return None
        
        overlay = QPixmap(self.displayed_size)
        overlay.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(overlay)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        
        for category, regions in self.regions.items():
            color = self.region_colors.get(category, (255, 255, 0))
            qt_color = QColor(*reversed(color))
            painter.setPen(QPen(qt_color, 2))
            
            for rect in regions:
                x, y, w, h = rect
                painter.drawRect(int(x), int(y), int(w), int(h))
                
                painter.setPen(QPen(qt_color, 1))
                font = painter.font()
                font.setPointSize(10)
                painter.setFont(font)
                painter.drawText(int(x) + 5, int(y) + 15, category)
        
        if self.draw_start and self.draw_end and self.mode == 'draw':
            x1 = min(self.draw_start.x(), self.draw_end.x())
            y1 = min(self.draw_start.y(), self.draw_end.y())
            x2 = max(self.draw_start.x(), self.draw_end.x())
            y2 = max(self.draw_start.y(), self.draw_end.y())
            
            if self.current_category:
                color = self.region_colors.get(self.current_category, (255, 255, 0))
                qt_color = QColor(*reversed(color))
                painter.setPen(QPen(qt_color, 2))
                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
        
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
    
    def mousePressEvent(self, event):
        if self.current_pixmap is None:
            return
        
        pos = event.position().toPoint()
        
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = True
            self.pan_start = pos
            return
        
        if self.mode == 'draw' and self.current_category:
            self.draw_start = pos
            self.draw_end = pos
    
    def mouseMoveEvent(self, event):
        if self.current_pixmap is None:
            return
        
        pos = event.position().toPoint()
        
        if self.is_panning and self.pan_start:
            dx = pos.x() - self.pan_start.x()
            dy = pos.y() - self.pan_start.y()
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = pos
            self._update_display()
            return
        
        if self.mode == 'draw' and self.draw_start:
            self.draw_end = pos
            self._update_display()
    
    def mouseReleaseEvent(self, event):
        if self.current_pixmap is None:
            return
        
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = False
            self.pan_start = None
            return
        
        if self.mode == 'draw' and self.draw_start and self.draw_end and self.current_category:
            x1 = min(self.draw_start.x(), self.draw_end.x())
            y1 = min(self.draw_start.y(), self.draw_end.y())
            x2 = max(self.draw_start.x(), self.draw_end.x())
            y2 = max(self.draw_start.y(), self.draw_end.y())
            
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                img_x1 = int((x1 - self.offset_x) / self._zoom_factor)
                img_y1 = int((y1 - self.offset_y) / self._zoom_factor)
                img_x2 = int((x2 - self.offset_x) / self._zoom_factor)
                img_y2 = int((y2 - self.offset_y) / self._zoom_factor)
                
                img_x1 = max(0, img_x1)
                img_y1 = max(0, img_y1)
                img_x2 = min(self.current_pixmap.width(), img_x2)
                img_y2 = min(self.current_pixmap.height(), img_y2)
                
                rect = (img_x1, img_y1, img_x2 - img_x1, img_y2 - img_y1)
                self.regions[self.current_category].append(rect)
                self.region_drawn.emit(self.current_category, rect)
            
            self.draw_start = None
            self.draw_end = None
            self._update_display()
    
    def clear_regions(self):
        self.regions = {'水面': [], '摊面': [], '分界线': [], '坝体': []}
        self._update_display()
