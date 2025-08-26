from __future__ import annotations

from typing import Protocol
import numpy as np
from PIL import Image
import cv2

from eye_detector import EyeDetector
from frame_processor import FrameProcessor


class EyeExtractor(Protocol):
    """Protocol for eye extractors."""

    def extract(
        self, frame: np.ndarray
    ) -> tuple[list[Image.Image], list[Image.Image], Image.Image, list]:
        """Extract eyes from a frame."""
        ...


class DlibEyeExtractor:
    """Extractor that uses dlib to detect eyes in a frame."""

    def __init__(self) -> None:
        self.eye_detector = EyeDetector()
        self.frame_processor = FrameProcessor()

    def extract(
        self, frame: np.ndarray
    ) -> tuple[list[Image.Image], list[Image.Image], Image.Image, list]:
        eye_boxes = self.eye_detector.calculate_eye_boxes(frame)
        frame_with_boxes = self.frame_processor.visualize_eye_boxes(frame, eye_boxes)
        left_eye_images, right_eye_images = self.frame_processor.extract_eye_images(
            frame, eye_boxes
        )
        return left_eye_images, right_eye_images, frame_with_boxes, eye_boxes


class SingleEyeExtractor:
    """Extractor that treats the whole frame as a single eye."""

    def __init__(self, eye: str) -> None:
        if eye not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        self.eye = eye

    def extract(
        self, frame: np.ndarray
    ) -> tuple[list[Image.Image], list[Image.Image], Image.Image, list]:
        eye_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        left_eye_images = [eye_img] if self.eye == "left" else []
        right_eye_images = [eye_img] if self.eye == "right" else []
        return left_eye_images, right_eye_images, eye_img, []
