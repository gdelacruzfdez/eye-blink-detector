"""Frame source implementation that reads frames from a video file."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from frame_source import FrameSource


class VideoFileCapture(FrameSource):
    """Provide frames by reading a video file using OpenCV."""

    def __init__(self, path: str) -> None:
        self.cap = cv2.VideoCapture(path)

    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

    def release(self) -> None:
        self.cap.release()

    def get_frame_width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps != 0 else 30

