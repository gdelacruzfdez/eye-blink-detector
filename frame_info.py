from PIL import Image
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


@dataclass
class EyeData:
    """Container for per-eye information."""
    img: Image
    pred: Optional[int] = None
    blink_prob: Optional[float] = None
    closed_prob: Optional[float] = None


class Eye(Enum):
    LEFT = "left"
    RIGHT = "right"

    def __str__(self) -> str:  # pragma: no cover - simple delegation
        return self.value


class FrameInfo:
    """
    Holds all relevant information for a single video frame.  Eye related data is
    stored in :attr:`eyes` which maps :class:`Eye` enums to :class:`EyeData` objects.
    This makes it possible to work with frames that contain only a subset of eyes
    while avoiding stringly-typed keys.
    """

    def __init__(self, frame_num: int, frame_img: Image, frame_with_boxes: Image,
                 eye_boxes: list[dict[str, dict[str, np.ndarray]]],
                 eyes: Optional[Dict[Eye, EyeData]] = None,
                 left_eye_img: Optional[Image] = None,
                 right_eye_img: Optional[Image] = None,
                 left_eye_pred: Optional[int] = None,
                 right_eye_pred: Optional[int] = None,
                 left_eye_blink_prob: Optional[float] = None,
                 right_eye_blink_prob: Optional[float] = None,
                 left_eye_closed_prob: Optional[float] = None,
                 right_eye_closed_prob: Optional[float] = None) -> None:
        self.frame_num = frame_num
        self.frame_img = frame_img
        self.frame_with_boxes = frame_with_boxes
        self.eye_boxes = eye_boxes

        # Initialize eye dictionary and populate from provided data
        self.eyes: Dict[Eye, EyeData] = eyes.copy() if eyes else {}
        if left_eye_img is not None or any(v is not None for v in [left_eye_pred, left_eye_blink_prob, left_eye_closed_prob]):
            self.eyes[Eye.LEFT] = EyeData(
                img=left_eye_img if left_eye_img is not None else Image.new("RGB", (1, 1), (0, 0, 0)),
                pred=left_eye_pred,
                blink_prob=left_eye_blink_prob,
                closed_prob=left_eye_closed_prob,
            )
        if right_eye_img is not None or any(v is not None for v in [right_eye_pred, right_eye_blink_prob, right_eye_closed_prob]):
            self.eyes[Eye.RIGHT] = EyeData(
                img=right_eye_img if right_eye_img is not None else Image.new("RGB", (1, 1), (0, 0, 0)),
                pred=right_eye_pred,
                blink_prob=right_eye_blink_prob,
                closed_prob=right_eye_closed_prob,
            )

    # --- Convenience properties for backward compatibility ---

    @property
    def left_eye_img(self) -> Image:
        eye = self.eyes.get(Eye.LEFT)
        if eye is not None:
            return eye.img
        return Image.new("RGB", (1, 1), (0, 0, 0))

    @property
    def right_eye_img(self) -> Image:
        eye = self.eyes.get(Eye.RIGHT)
        if eye is not None:
            return eye.img
        return Image.new("RGB", (1, 1), (0, 0, 0))

    @property
    def left_eye_pred(self) -> Optional[int]:
        eye = self.eyes.get(Eye.LEFT)
        return eye.pred if eye is not None else None

    @left_eye_pred.setter
    def left_eye_pred(self, value: Optional[int]) -> None:
        if Eye.LEFT in self.eyes:
            self.eyes[Eye.LEFT].pred = value

    @property
    def right_eye_pred(self) -> Optional[int]:
        eye = self.eyes.get(Eye.RIGHT)
        return eye.pred if eye is not None else None

    @right_eye_pred.setter
    def right_eye_pred(self, value: Optional[int]) -> None:
        if Eye.RIGHT in self.eyes:
            self.eyes[Eye.RIGHT].pred = value

    @property
    def left_eye_blink_prob(self) -> Optional[float]:
        eye = self.eyes.get(Eye.LEFT)
        return eye.blink_prob if eye is not None else None

    @left_eye_blink_prob.setter
    def left_eye_blink_prob(self, value: Optional[float]) -> None:
        if Eye.LEFT in self.eyes:
            self.eyes[Eye.LEFT].blink_prob = value

    @property
    def right_eye_blink_prob(self) -> Optional[float]:
        eye = self.eyes.get(Eye.RIGHT)
        return eye.blink_prob if eye is not None else None

    @right_eye_blink_prob.setter
    def right_eye_blink_prob(self, value: Optional[float]) -> None:
        if Eye.RIGHT in self.eyes:
            self.eyes[Eye.RIGHT].blink_prob = value

    @property
    def left_eye_closed_prob(self) -> Optional[float]:
        eye = self.eyes.get(Eye.LEFT)
        return eye.closed_prob if eye is not None else None

    @left_eye_closed_prob.setter
    def left_eye_closed_prob(self, value: Optional[float]) -> None:
        if Eye.LEFT in self.eyes:
            self.eyes[Eye.LEFT].closed_prob = value

    @property
    def right_eye_closed_prob(self) -> Optional[float]:
        eye = self.eyes.get(Eye.RIGHT)
        return eye.closed_prob if eye is not None else None

    @right_eye_closed_prob.setter
    def right_eye_closed_prob(self, value: Optional[float]) -> None:
        if Eye.RIGHT in self.eyes:
            self.eyes[Eye.RIGHT].closed_prob = value
