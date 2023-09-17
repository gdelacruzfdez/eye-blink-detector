import cv2
from PIL import Image
import numpy as np

class FrameProcessor:
    def __init__(self):
        self.box_color: tuple[int, int, int] = (0, 255, 0)  # Green color for the bounding boxes
        self.box_thickness: int = 2

    def visualize_eye_boxes(self, frame: np.ndarray, eye_boxes: list[dict[str, dict[str, np.ndarray]]], is_blinking: bool = False) -> Image.Image:
        """
        Visualize the eye boxes on the given frame.

        Args:
            frame (numpy.ndarray): Input frame.
            eye_boxes (list): List of eye boxes.

        Returns:
            PIL.Image.Image: Frame with eye boxes visualized.
        """
        frame_with_boxes = frame.copy()

        if is_blinking:
            box_color = (0, 0, 255)  # RGB for Red
        else:
            box_color = self.box_color  # Assuming self.box_color is a predefined color

        for eye_box_pair in eye_boxes:
            left_eye_box = eye_box_pair['left_eye']['box']
            right_eye_box = eye_box_pair['right_eye']['box']

            cv2.drawContours(frame_with_boxes, [left_eye_box], 0, box_color, self.box_thickness)
            cv2.drawContours(frame_with_boxes, [right_eye_box], 0, box_color, self.box_thickness)

        return Image.fromarray(cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB))


    def extract_eye_images(self, frame: np.ndarray, eye_boxes: list[dict[str, dict[str, np.ndarray]]]) -> tuple[list[Image.Image], list[Image.Image]]:
        """
        Extract the eye images from the given frame based on the eye boxes.

        Args:
            frame (numpy.ndarray): Input frame.
            eye_boxes (list): List of eye boxes.

        Returns:
            tuple: Tuple containing left and right eye images as PIL.Image.Image objects.
        """
        left_eye_images = []
        right_eye_images = []

        for eye_box_pair in eye_boxes:
            left_eye_box = eye_box_pair['left_eye']['box']
            left_eye_angle = eye_box_pair['left_eye']['angle']
            right_eye_box = eye_box_pair['right_eye']['box']
            right_eye_angle = eye_box_pair['right_eye']['angle']

            left_eye_image = self._extract_eye_image(frame, left_eye_box, left_eye_angle)
            right_eye_image = self._extract_eye_image(frame, right_eye_box, right_eye_angle)

            left_eye_images.append(left_eye_image)
            right_eye_images.append(right_eye_image)

        return left_eye_images, right_eye_images

    def _extract_eye_image(self, frame: np.ndarray, eye_box: np.ndarray, angle: float) -> Image.Image:
        """
        Extract the eye image from the given frame using the eye box and rotation angle.

        Args:
            frame (numpy.ndarray): Input frame.
            eye_box (numpy.ndarray): Eye box coordinates.
            angle (float): Rotation angle.

        Returns:
            PIL.Image.Image: Extracted eye image.
        """
        if len(eye_box) == 0:
            return None

        rows, cols = frame.shape[0], frame.shape[1]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_rot = cv2.warpAffine(frame, M, (cols, rows))

        pts = cv2.transform(np.array([eye_box]), M)[0]
        pts[pts < 0] = 0

        img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

        return Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))

    def extract_rotated_box(self, img: np.ndarray, box: np.ndarray, angle: float) -> np.ndarray:
        """
        Extract a rotated box region from the given image.

        Args:
            img (numpy.ndarray): Input image.
            box (numpy.ndarray): Box coordinates.
            angle (float): Rotation angle.

        Returns:
            numpy.ndarray: Extracted rotated box region.
        """
        if len(box) == 0:
            return None

        rows, cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_rot = cv2.warpAffine(img, M, (cols, rows))

        pts = np.int0(cv2.transform(np.array([box]), M))[0]
        pts[pts < 0] = 0

        img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

        return img_crop
