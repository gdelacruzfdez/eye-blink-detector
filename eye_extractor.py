import cv2
import numpy as np
from typing import Protocol, Tuple, Optional
from PIL import Image

from eye_detector import EyeDetector
from frame_processor import FrameProcessor


class EyeExtractor(Protocol):
    def extract(self, frame: np.ndarray) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Return left and right eye images from a frame."""
        ...


class DlibEyeExtractor:
    """Extracts eyes from frames using dlib-based detection."""

    def __init__(self) -> None:
        self._eye_detector = EyeDetector()
        self._frame_processor = FrameProcessor()
        self.last_eye_boxes: list[dict[str, dict[str, np.ndarray]]] | None = None
        self.last_frame_with_boxes: Optional[Image.Image] = None

    def extract(self, frame: np.ndarray) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        eye_boxes = self._eye_detector.calculate_eye_boxes(frame)
        self.last_eye_boxes = eye_boxes
        self.last_frame_with_boxes = self._frame_processor.visualize_eye_boxes(frame, eye_boxes)
        left_eye_imgs, right_eye_imgs = self._frame_processor.extract_eye_images(frame, eye_boxes)
        left_eye = left_eye_imgs[0] if left_eye_imgs else None
        right_eye = right_eye_imgs[0] if right_eye_imgs else None
        return left_eye, right_eye


class SingleEyeExtractor:
    """Extractor for sources already cropped to a single eye."""

    def extract(self, frame: np.ndarray) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        left_eye = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return left_eye, None
