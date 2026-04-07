import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    torch = None
    logger = logging.getLogger(__name__)
    logger.warning(f"Torch not available: {e}. Training features will be disabled.")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Ultralytics not available, YOLO training disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DryBeachDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, 
                 transform=None, image_size: Tuple[int, int] = (640, 640)):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        self.image_size = image_size
        
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                 list(self.image_dir.glob('*.png')))
        
        logger.info(f"Dataset initialized with {len(self.image_files)} images")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        img_path = self.image_files[idx]
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        image = cv2.resize(image, self.image_size)
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        
        label_path = self.label_dir / (img_path.stem + '.txt')
        label = self._load_label(label_path, w, h)
        
        return image, label


class ClassificationDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], image_size: int = 32):
        self.image_paths = image_paths
        self.labels = labels
        self.image_size = image_size
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        
        if image is None:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label
    
    def _load_label(self, label_path: Path, img_w: int, img_h: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        if not label_path.exists():
            return torch.zeros((0, 5))
        
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls, x, y, w, h = map(float, parts[:5])
                    labels.append([cls, x, y, w, h])
        
        if not labels:
            return torch.zeros((0, 5))
        return torch.tensor(labels, dtype=torch.float32)


class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, input_size: int = 32):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ModelTrainer:
    def __init__(self, model_save_path: Optional[Path] = None):
        self.model_save_path = model_save_path
        self.model = None
        self.device = None
        self.use_amp = False
        self.scaler = None
        if TORCH_AVAILABLE:
            self._setup_cuda()
    
    def _setup_cuda(self):
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            logger.info(f"CUDA version: {cuda_version}")
            
            self.device = torch.device('cuda')
            
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            self.use_amp = True
            self.scaler = GradScaler()
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}, Total Memory: {gpu_memory:.1f}GB")
            logger.info("Mixed precision training (AMP) enabled")
            self._log_gpu_memory("Initial")
        else:
            self.device = torch.device('cpu')
            logger.info("CUDA not available, using CPU")
    
    def _log_gpu_memory(self, prefix: str = ""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"{prefix} GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    
    def train_with_yolo(self, data_yaml: Path, epochs: int = 100,
                       batch_size: int = 16, model_size: str = 'yolov8n',
                       image_size: int = 32) -> str:
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics library not available")
        
        logger.info(f"Loading YOLO model {model_size} for {image_size}x{image_size} images...")
        self._log_gpu_memory("After model load")
        
        model = YOLO(f'{model_size}.pt')
        
        device = 0 if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Starting training on device: {'GPU' if device == 0 else 'CPU'}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self._log_gpu_memory("After cache clear")
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            project=str(self.model_save_path.parent) if self.model_save_path else 'runs',
            name=self.model_save_path.stem if self.model_save_path else 'train',
            exist_ok=True,
            patience=50,
            save=True,
            plots=False,
            amp=False,
            workers=0,
            cache=False,
            verbose=False,
            deterministic=True,
            optimizer='Adam',
            lr0=0.001,
            lrf=0.01,
            momentum=0.9,
            weight_decay=0.0001,
            warmup_epochs=1.0,
            warmup_momentum=0.8,
            box=0.05,
            cls=0.5,
            dfl=0.5,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            flipud=0.0,
            fliplr=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._log_gpu_memory("After training")
        
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        logger.info(f"Training completed. Best model saved to {best_model_path}")
        
        return str(best_model_path)
    
    def train_classification_model(self, data_path: Path, epochs: int = 100,
                                   batch_size: int = 32, lr: float = 0.001) -> str:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        logger.info(f"Training classification model on {data_path}")
        
        categories = ['water', 'beach', 'boundary', 'dam']
        num_classes = len(categories)
        
        image_size = 32
        
        train_images = []
        train_labels = []
        val_images = []
        val_labels = []
        
        for cat_idx, cat in enumerate(categories):
            cat_dir = data_path / cat
            if not cat_dir.exists():
                continue
            
            images = list(cat_dir.glob('*.jpg')) + list(cat_dir.glob('*.png'))
            import random
            random.shuffle(images)
            
            split_idx = int(len(images) * 0.8)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]
            
            for img_path in train_imgs:
                train_images.append(str(img_path))
                train_labels.append(cat_idx)
            
            for img_path in val_imgs:
                val_images.append(str(img_path))
                val_labels.append(cat_idx)
        
        if not train_images:
            raise ValueError("No training images found")
        
        logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}")
        
        train_dataset = ClassificationDataset(train_images, train_labels, image_size)
        val_dataset = ClassificationDataset(val_images, val_labels, image_size)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available()
        )
        
        self.model = SimpleCNNClassifier(num_classes=num_classes, input_size=image_size)
        self.model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = train_correct / train_total
            
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = val_correct / val_total
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.4f} - "
                       f"Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if self.model_save_path:
                    torch.save(self.model.state_dict(), self.model_save_path)
        
        logger.info(f"Training completed. Best val accuracy: {best_val_acc:.4f}")
        
        return str(self.model_save_path)
        
        num_workers = 4 if torch.cuda.is_available() else 0
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        training_history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            
            avg_loss = epoch_loss / len(train_loader)
            training_history['loss'].append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        if self.model_save_path and self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epochs,
            }, self.model_save_path)
            logger.info(f"Model saved to {self.model_save_path}")
        
        return training_history
    
    def convert_to_yolo_format(self, annotations: List[Dict],
                              output_dir: Path,
                              image_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        yolo_data = []
        
        for ann in annotations:
            image_path = ann.get('image_path')
            if not image_path:
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            yolo_labels = []
            
            for detection in ann.get('detections', []):
                label = detection['label']
                class_id = 0 if label == 'water_line' else 1
                
                bbox = detection['bbox']
                x, y, bw, bh = bbox
                
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                norm_w = bw / w
                norm_h = bh / h
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} "
                                 f"{norm_w:.6f} {norm_h:.6f}")
            
            image_filename = Path(image_path).name
            output_filename = Path(image_path).stem + '.txt'
            
            with open(output_dir / output_filename, 'w') as f:
                f.write('\n'.join(yolo_labels))
            
            yolo_data.append({
                'image': image_filename,
                'labels': output_filename,
                'num_objects': len(yolo_labels)
            })
        
        logger.info(f"Converted {len(yolo_data)} annotations to YOLO format")
        
        return yolo_data


def create_yolo_dataset_config(train_images: Path, val_images: Path,
                               train_labels: Path, val_labels: Path,
                               class_names: List[str]) -> Path:
    config = f"""
train: {train_images}
val: {val_images}

nc: {len(class_names)}
names: {class_names}
"""
    
    config_path = Path('dataset_config.yaml')
    with open(config_path, 'w') as f:
        f.write(config)
    
    logger.info(f"YOLO dataset config saved to {config_path}")
    
    return config_path
