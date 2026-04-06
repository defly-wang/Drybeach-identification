import os
from pathlib import Path

class Config:
    PROJECT_ROOT = Path(__file__).parent
    DATA_ROOT = PROJECT_ROOT / 'data'
    IMAGES_DIR = DATA_ROOT / 'images'
    LABELS_DIR = DATA_ROOT / 'labels'
    MODELS_DIR = PROJECT_ROOT / 'models'
    OUTPUTS_DIR = PROJECT_ROOT / 'outputs'
    
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    DEFAULT_FRAME_INTERVAL = 30
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_IOU_THRESHOLD = 0.45
    
    MODEL_CONFIGS = {
        'water_line': {
            'name': 'Water Line Detection',
            'class_names': ['water_line'],
            'input_size': (640, 640)
        },
        'dam': {
            'name': 'Dam Detection',
            'class_names': ['dam'],
            'input_size': (640, 640)
        },
        'combined': {
            'name': 'Combined Detection',
            'class_names': ['water_line', 'dam'],
            'input_size': (640, 640)
        }
    }
    
    DISTANCE_CALCULATION = {
        'unit': 'meters',
        'calibration_mode': 'auto',
        'reference_points': 2
    }
    
    ANNOTATION = {
        'line_color': (0, 255, 0),
        'line_thickness': 3,
        'box_color': (255, 0, 0),
        'box_thickness': 2,
        'text_color': (255, 255, 255),
        'text_scale': 0.8,
        'text_thickness': 2
    }
    
    @classmethod
    def ensure_dirs(cls):
        for dir_path in [cls.DATA_ROOT, cls.IMAGES_DIR, cls.LABELS_DIR, 
                        cls.MODELS_DIR, cls.OUTPUTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
