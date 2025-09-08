from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from eye_detector import EyeDetector
from frame_processor import FrameProcessor
from irislandmarks import IrisLandmarks


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
        logging.info("Initializing DlibEyeExtractor.")
        self.eye_detector = EyeDetector()
        self.frame_processor = FrameProcessor()

    def extract(
        self, frame: np.ndarray
    ) -> tuple[list[Image.Image], list[Image.Image], Image.Image, list]:
        logging.debug("Extracting eyes using DlibEyeExtractor.")
        eye_boxes = self.eye_detector.calculate_eye_boxes(frame)
        frame_with_boxes = self.frame_processor.visualize_eye_boxes(frame, eye_boxes)
        left_eye_images, right_eye_images = self.frame_processor.extract_eye_images(
            frame, eye_boxes
        )
        logging.debug(
            f"Extracted {len(left_eye_images)} left eyes and {len(right_eye_images)} right eyes."
        )
        return left_eye_images, right_eye_images, frame_with_boxes, eye_boxes


class SingleEyeExtractor:
    """Extractor that uses IrisLandmarks to detect landmarks from an eye crop."""

    def __init__(self, eye: str) -> None:
        logging.info(f"Initializing SingleEyeExtractor for {eye} eye.")
        if eye not in ("left", "right"):
            raise ValueError("eye must be 'left' or 'right'")
        self.eye = eye
        self.model = IrisLandmarks()
        self.model.load_weights("irislandmarks.pth")

    def extract(
        self, frame: np.ndarray
    ) -> tuple[list[Image.Image], list[Image.Image], Image.Image, list]:
        logging.debug(f"Extracting single eye ({self.eye}) using SingleEyeExtractor.")

        h, w, _ = frame.shape
        frame_resized = cv2.resize(frame, (64, 64))

        eye_landmarks, _ = self.model.predict_on_image(frame_resized)
        eye_landmarks = eye_landmarks.squeeze(0).numpy()

        # The landmarks are for a 64x64 image, scale them to the original frame size
        eye_landmarks[:, 0] *= w / 64.0
        eye_landmarks[:, 1] *= h / 64.0

        landmark_points = eye_landmarks[:, :2].astype(int).tolist()

        # Extract the eye from the image using the landmarks
        x_coords = [p[0] for p in landmark_points]
        y_coords = [p[1] for p in landmark_points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Add some padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        eye_img = Image.fromarray(cv2.cvtColor(frame[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB))

        left_eye_images = [eye_img] if self.eye == "left" else []
        right_eye_images = [eye_img] if self.eye == "right" else []

        frame_with_landmarks = frame.copy()
        for point in landmark_points:
            cv2.circle(frame_with_landmarks, tuple(point), 1, (0, 255, 0), -1)

        frame_with_landmarks_pil = Image.fromarray(
            cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)
        )

        return (
            left_eye_images,
            right_eye_images,
            frame_with_landmarks_pil,
            landmark_points,
        )
