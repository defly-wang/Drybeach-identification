import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import cv2

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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def _load_label(self, label_path: Path, img_w: int, img_h: int) -> torch.Tensor:
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


class SimpleDetectionModel(nn.Module):
    def __init__(self, num_classes: int = 2, input_size: Tuple[int, int] = (640, 640)):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes * 5)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        x = x.view(-1, self.num_classes, 5)
        return x


class ModelTrainer:
    def __init__(self, model_save_path: Optional[Path] = None):
        self.model_save_path = model_save_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {self.device}")
    
    def train_with_yolo(self, data_yaml: Path, epochs: int = 100,
                       batch_size: int = 16, model_size: str = 'yolov8n') -> str:
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics library not available")
        
        model = YOLO(f'{model_size}.pt')
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            project=str(self.model_save_path.parent) if self.model_save_path else 'runs',
            name=self.model_save_path.stem if self.model_save_path else 'train',
            exist_ok=True,
            patience=20,
            save=True,
            plots=True
        )
        
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        logger.info(f"Training completed. Best model saved to {best_model_path}")
        
        return str(best_model_path)
    
    def train_simple_model(self, train_dataset: Dataset,
                          val_dataset: Optional[Dataset] = None,
                          epochs: int = 50,
                          lr: float = 0.001) -> Dict:
        self.model = SimpleDetectionModel()
        self.model.to(self.device)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        training_history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            training_history['loss'].append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if self.model_save_path and self.model:
            torch.save(self.model.state_dict(), self.model_save_path)
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
