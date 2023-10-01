from PIL import Image
import numpy as np
from typing import Optional


class FrameInfo:
    def __init__(self, frame_num: int, frame_img: Image, frame_with_boxes: Image, eye_boxes: list[dict[str, dict[str, np.ndarray]]],
                 left_eye_img: Image, right_eye_img: Image,
                 left_eye_pred: Optional[int], right_eye_pred: Optional[int],
                 left_eye_prob: Optional[float], right_eye_prob: Optional[float]):
        self.frame_num = frame_num
        self.frame_img = frame_img
        self.frame_with_boxes = frame_with_boxes
        self.eye_boxes = eye_boxes
        self.left_eye_img = left_eye_img
        self.right_eye_img = right_eye_img
        self.left_eye_pred = left_eye_pred
        self.right_eye_pred = right_eye_pred
        self.left_eye_prob = left_eye_prob
        self.right_eye_prob = right_eye_prob
