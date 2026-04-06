import logging

try:
    from PyQt6.QtWidgets import QApplication
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    logging.warning("PyQt6 not available, GUI disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .viewers import ImageViewer, MarkImageViewer
from .threads import ProcessingThread, VideoExtractThread
from .video_tab import VideoExtractWidget
from .mark_tab import MarkSegmentationWidget
from .main_window import DryBeachGUI, launch_gui

__all__ = [
    'ImageViewer',
    'MarkImageViewer',
    'ProcessingThread',
    'VideoExtractThread',
    'VideoExtractWidget',
    'MarkSegmentationWidget',
    'DryBeachGUI',
    'launch_gui',
    'PYQT_AVAILABLE',
]
