from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

import cv2


class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    
    def __init__(self, task_type: str, params: dict):
        super().__init__()
        self.task_type = task_type
        self.params = params
    
    def run(self):
        try:
            if self.task_type == 'video_extract':
                self._process_video_extraction()
            elif self.task_type == 'detection':
                self._process_detection()
            elif self.task_type == 'training':
                self._process_training()
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _process_video_extraction(self):
        from drybeach_app.video_capture import VideoFrameExtractor
        
        video_path = self.params['video_path']
        output_dir = Path(self.params['output_dir'])
        interval = self.params['interval']
        
        frames = []
        with VideoFrameExtractor(video_path) as extractor:
            total = max(1, extractor.total_frames // interval)
            for i, frame_num in enumerate(range(0, extractor.total_frames, interval)):
                frame = extractor.extract_frame(frame_num)
                if frame is not None:
                    frames.append((frame_num, frame))
                    self.progress_updated.emit(int((i + 1) / total * 100))
        
        extractor_save = VideoFrameExtractor(video_path)
        saved_paths = extractor_save.save_frames(frames, output_dir)
        
        self.finished.emit(saved_paths)
    
    def _process_detection(self):
        from drybeach_app.recognizer import DryBeachRecognizer
        from drybeach_app.image_annotator import RegionOfInterest
        
        image_paths = self.params['image_paths']
        output_dir = Path(self.params['output_dir'])
        calibration = self.params.get('calibration')
        roi_data = self.params.get('roi')
        method = self.params.get('method', 'multi')
        
        recognizer = DryBeachRecognizer()
        
        if calibration:
            recognizer.calibrate(calibration['distance'], calibration['points'])
        
        roi = None
        if roi_data:
            x, y, w, h = roi_data
            roi = RegionOfInterest(x, y, w, h)
        
        results = []
        for i, img_path in enumerate(image_paths):
            image = cv2.imread(str(img_path))
            if image is not None:
                result, annotated = recognizer.detect_and_visualize(image, roi=roi, method=method)
                
                output_path = output_dir / f"result_{img_path.stem}.jpg"
                cv2.imwrite(str(output_path), annotated)
                results.append(output_path)
                
                self.progress_updated.emit(int((i + 1) / len(image_paths) * 100))
        
        self.finished.emit(results)
    
    def _process_training(self):
        from drybeach_app.model_trainer import ModelTrainer
        
        config_path = self.params['config_path']
        epochs = self.params['epochs']
        model_save = Path(self.params['model_save'])
        
        trainer = ModelTrainer(model_save_path=model_save)
        model_path = trainer.train_with_yolo(config_path, epochs=epochs)
        
        self.finished.emit([model_path])


class VideoExtractThread(QThread):
    progress_updated = pyqtSignal(int)
    frame_extracted = pyqtSignal(int, object)
    finished = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, video_path: str, mode: str, value: int, output_dir: Path, video_info: dict):
        super().__init__()
        self.video_path = video_path
        self.mode = mode
        self.value = value
        self.output_dir = output_dir
        self.video_info = video_info
        self._stop = False
    
    def run(self):
        import subprocess
        
        try:
            video_path = Path(self.video_path)
            output_dir = self.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            fps = self.video_info.get('fps', 25)
            duration = self.video_info.get('duration', 0)
            
            if self.mode == 'count':
                target_count = self.value
                interval = duration / target_count if duration > 0 else 1.0
                timestamps = [i * interval for i in range(target_count)]
            else:
                seconds_interval = self.value
                timestamps = [i * seconds_interval for i in range(int(duration / seconds_interval) + 1)]
            
            saved_paths = []
            total = len(timestamps)
            
            for i, timestamp in enumerate(timestamps):
                if self._stop:
                    break
                
                output_file = output_dir / f"{video_path.stem}_{i+1:06d}.jpg"
                
                cmd = [
                    'ffmpeg',
                    '-ss', f'{timestamp:.3f}',
                    '-i', str(video_path),
                    '-vframes', '1',
                    '-q:v', '2',
                    '-threads', '2',
                    '-y',
                    str(output_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                if result.returncode == 0 and output_file.exists():
                    saved_paths.append(output_file)
                    frame_num = int(timestamp * fps)
                    self.frame_extracted.emit(frame_num, str(output_file))
                
                self.progress_updated.emit(int((i + 1) / total * 100))
            
            if not saved_paths:
                self.error_occurred.emit("未提取到任何帧")
                return
            
            self.finished.emit(saved_paths)
            
        except FileNotFoundError:
            self.error_occurred.emit("未找到FFmpeg")
        except Exception as e:
            self.error_occurred.emit(str(e))
