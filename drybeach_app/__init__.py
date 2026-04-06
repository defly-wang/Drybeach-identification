from .video_capture import VideoFrameExtractor, VideoStreamGrabber, extract_frames_from_video
from .image_annotator import ImageAnnotator, BatchImageSlicer, RegionOfInterest, Annotation
from .water_line_detector import WaterLineDetector, AdaptiveWaterLineDetector, visualize_water_line
from .dam_detector import DamDetector, DamEdgeDetector, visualize_dam_detection
from .distance_calculator import DistanceCalculator, MeasurementReporter, draw_measurement_annotations
from .model_trainer import ModelTrainer, DryBeachDataset, SimpleDetectionModel
from .recognizer import DryBeachRecognizer, DetectionResult, load_recognizer_model, process_video_frames

__version__ = '1.0.0'

__all__ = [
    'VideoFrameExtractor',
    'VideoStreamGrabber', 
    'extract_frames_from_video',
    'ImageAnnotator',
    'BatchImageSlicer',
    'RegionOfInterest',
    'Annotation',
    'WaterLineDetector',
    'AdaptiveWaterLineDetector',
    'visualize_water_line',
    'DamDetector',
    'DamEdgeDetector',
    'visualize_dam_detection',
    'DistanceCalculator',
    'MeasurementReporter',
    'draw_measurement_annotations',
    'ModelTrainer',
    'DryBeachDataset',
    'SimpleDetectionModel',
    'DryBeachRecognizer',
    'DetectionResult',
    'load_recognizer_model',
    'process_video_frames'
]
