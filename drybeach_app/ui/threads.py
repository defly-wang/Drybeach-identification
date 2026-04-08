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
        
        image_paths = self.params['image_paths']
        model_path = self.params.get('model_path')
        patch_size = self.params.get('patch_size', 64)
        stride = self.params.get('stride', 32)
        
        recognizer = DryBeachRecognizer(model_path=model_path)
        recognizer.patch_size = patch_size
        recognizer.stride = stride
        
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"\n处理图片: {img_path}")
            image = cv2.imread(str(img_path))
            if image is not None:
                result, annotated = recognizer.detect_and_visualize(
                    image, 
                    progress_callback=self.progress_updated.emit
                )
                results.append({
                    'annotated': annotated,
                    'class_counts': result.class_counts,
                    'detection_points': result.detection_points,
                    'image_path': img_path
                })
                
                self.progress_updated.emit(int((i + 1) / len(image_paths) * 100))
        
        self.finished.emit(results)
    
    def _process_training(self):
        from drybeach_app.model_trainer import ModelTrainer
        import cv2
        import random
        
        data_path = self.params['data_path']
        epochs = self.params.get('epochs', 100)
        batch_size = self.params.get('batch_size', 16)
        image_size = self.params.get('image_size', 32)
        learning_rate = self.params.get('learning_rate', 0.001)
        optimizer = self.params.get('optimizer', 'Adam')
        patience = self.params.get('patience', 50)
        momentum = self.params.get('momentum', 0.9)
        weight_decay = self.params.get('weight_decay', 0.0001)
        warmup_epochs = self.params.get('warmup_epochs', 1.0)
        lrf = self.params.get('lrf', 0.01)
        model_save = Path(self.params['model_save'])
        
        categories = ['water', 'beach', 'boundary', 'dam']
        class_names = ['water', 'beach', 'boundary', 'dam']
        
        temp_dataset = model_save / 'dataset'
        train_img_dir = temp_dataset / 'images' / 'train'
        train_lbl_dir = temp_dataset / 'labels' / 'train'
        val_img_dir = temp_dataset / 'images' / 'val'
        val_lbl_dir = temp_dataset / 'labels' / 'val'
        
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        all_images = []
        sample_img_h, sample_img_w = None, None
        for cat_idx, cat in enumerate(categories):
            cat_dir = data_path / cat
            if cat_dir.exists():
                for img_file in cat_dir.glob('*.jpg'):
                    all_images.append((img_file, cat_idx))
                    if sample_img_h is None:
                        sample_img = cv2.imread(str(img_file))
                        if sample_img is not None:
                            sample_img_h, sample_img_w = sample_img.shape[:2]
                for img_file in cat_dir.glob('*.png'):
                    all_images.append((img_file, cat_idx))
                    if sample_img_h is None:
                        sample_img = cv2.imread(str(img_file))
                        if sample_img is not None:
                            sample_img_h, sample_img_w = sample_img.shape[:2]
        
        if not all_images:
            raise ValueError("未找到任何训练图片")
        
        random.shuffle(all_images)
        split_idx = int(len(all_images) * 0.8)
        train_files = all_images[:split_idx]
        val_files = all_images[split_idx:]
        
        total = len(all_images)
        processed = 0
        
        for img_path, cat_idx in train_files:
            self._copy_and_label(img_path, cat_idx, train_img_dir, train_lbl_dir)
            processed += 1
            self.progress_updated.emit(int(processed / total * 50))
        
        for img_path, cat_idx in val_files:
            self._copy_and_label(img_path, cat_idx, val_img_dir, val_lbl_dir)
            processed += 1
            self.progress_updated.emit(int(processed / total * 50))
        
        config_path = temp_dataset / 'data.yaml'
        config_content = f"""path: {temp_dataset}
train: images/train
val: images/val

nc: {len(class_names)}
names: {class_names}
"""
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        self.status_updated.emit("正在训练分类模型...")
        trainer = ModelTrainer(model_save_path=model_save / 'best.pt')
        
        dropout = self.params.get('dropout', 0.5)
        lr_decay = self.params.get('lr_decay', 0.5)
        resume_model_path = self.params.get('resume_model_path')
        
        def progress_callback(pct):
            self.progress_updated.emit(pct)
        
        model_path = trainer.train_classification_model(
            data_path=data_path,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            lr=learning_rate,
            optimizer_name=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            patience=patience if patience > 0 else 15,
            dropout=dropout,
            lr_decay=lr_decay,
            resume_model_path=resume_model_path,
            progress_callback=progress_callback
        )
        
        self.finished.emit([model_path])
    
    def _copy_and_label(self, img_path, cat_idx, img_dir, lbl_dir):
        import cv2
        
        img = cv2.imread(str(img_path))
        if img is None:
            return
        
        new_img_path = img_dir / f"{img_path.stem}_{cat_idx}.jpg"
        cv2.imwrite(str(new_img_path), img)
        
        lbl_path = lbl_dir / f"{img_path.stem}_{cat_idx}.txt"
        with open(lbl_path, 'w') as f:
            f.write(f"{cat_idx} 0.5 0.5 1.0 1.0\n")


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
