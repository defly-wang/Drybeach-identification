"""
干滩识别系统 - 视频捕获模块
用于从视频文件中提取帧图像

Classes:
    VideoFrameExtractor: 视频帧提取器
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.duration = 0
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"Resolution: {self.width}x{self.height}, FPS: {self.fps}, "
                   f"Frames: {self.total_frames}, Duration: {self.duration:.2f}s")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
    
    def extract_frame(self, frame_number: int) -> Optional[np.ndarray]:
        if frame_number < 0 or frame_number >= self.total_frames:
            logger.warning(f"Frame number {frame_number} out of range")
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            logger.info(f"Extracted frame {frame_number}")
            return frame
        return None
    
    def extract_frames_at_interval(self, interval: int) -> List[Tuple[int, np.ndarray]]:
        frames = []
        for i in range(0, self.total_frames, interval):
            frame = self.extract_frame(i)
            if frame is not None:
                frames.append((i, frame))
        return frames
    
    def extract_frames_by_count(self, num_frames: int) -> List[Tuple[int, np.ndarray]]:
        if num_frames <= 0:
            return []
        
        interval = max(1, self.total_frames // num_frames)
        return self.extract_frames_at_interval(interval)
    
    def extract_frames_by_time(self, time_points: List[float]) -> List[Tuple[int, np.ndarray]]:
        frames = []
        for t in time_points:
            frame_number = int(t * self.fps)
            frame = self.extract_frame(frame_number)
            if frame is not None:
                frames.append((frame_number, frame))
        return frames
    
    def save_frames(self, frames: List[Tuple[int, np.ndarray]], output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for frame_num, frame in frames:
            filename = f"{self.video_path.stem}_frame_{frame_num:06d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_paths.append(filepath)
            logger.info(f"Saved frame to {filepath}")
        
        return saved_paths


class VideoStreamGrabber:
    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self.cap = None
        self.is_connected = False
        
    def connect(self) -> bool:
        self.cap = cv2.VideoCapture(self.stream_url)
        self.is_connected = self.cap.isOpened()
        
        if self.is_connected:
            logger.info(f"Connected to stream: {self.stream_url}")
        else:
            logger.error(f"Failed to connect to stream: {self.stream_url}")
        
        return self.is_connected
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame if ret else None
    
    def release(self):
        if self.cap:
            self.cap.release()
        self.is_connected = False


def extract_frames_from_video(video_path: str, output_dir: Path, 
                             mode: str = 'interval', 
                             value: int = 30) -> List[Path]:
    with VideoFrameExtractor(video_path) as extractor:
        if mode == 'interval':
            frames = extractor.extract_frames_at_interval(value)
        elif mode == 'count':
            frames = extractor.extract_frames_by_count(value)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return extractor.save_frames(frames, output_dir)
