import dlib
import math
import numpy as np
import cv2
import os

SHAPE_PREDICTOR_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_68_face_landmarks.dat")
BOX_COLOR: tuple[int, int, int] = (0, 255, 0)  # Green color for the bounding boxes
BOX_THICKNESS: int = 2


class FaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    def detect_faces(self, frame: np.ndarray) -> list[dlib.rectangle]:
        """
        Detect faces in the given frame.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            list: List of face rectangles.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces


class EyeDetector:
    def __init__(self):
        self.face_detector = FaceDetector()

    def calculate_distance(self, landmark1: tuple[int, int], landmark2: tuple[int, int]) -> float:
        """
        Calculate the distance between two landmarks.

        Args:
            landmark1 (tuple): First landmark coordinates (x, y).
            landmark2 (tuple): Second landmark coordinates (x, y).

        Returns:
            float: Distance between the landmarks.
        """
        x1, y1 = landmark1
        x2, y2 = landmark2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_angle(self, landmark1: tuple[int, int], landmark2: tuple[int, int]) -> float:
        """
        Calculate the angle between two landmarks.

        Args:
            landmark1 (tuple): First landmark coordinates (x, y).
            landmark2 (tuple): Second landmark coordinates (x, y).

        Returns:
            float: Angle between the landmarks in degrees.
        """
        x1, y1 = landmark1
        x2, y2 = landmark2
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

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

    def get_eye_landmarks(self, landmarks: dlib.full_object_detection, start: int, end: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Get the eye landmarks from the given set of landmarks.

        Args:
            landmarks (dlib.full_object_detection): Detected landmarks.
            start (int): Start index of the eye landmarks.
            end (int): End index of the eye landmarks.

        Returns:
            tuple: Tuple containing the coordinates of the eye landmarks.
        """
        eye_landmark1 = (landmarks.part(start).x, landmarks.part(start).y)
        eye_landmark2 = (landmarks.part(end).x, landmarks.part(end).y)
        return eye_landmark1, eye_landmark2

    def get_eye_box(self, eye_landmark1: tuple[int, int], eye_landmark2: tuple[int, int], eye_distance: float) -> tuple[np.ndarray, float]:
        """
        Get the bounding box and angle for an eye based on the given landmarks and distance.

        Args:
            eye_landmark1 (tuple): First eye landmark coordinates (x, y).
            eye_landmark2 (tuple): Second eye landmark coordinates (x, y).
            eye_distance (float): Distance between the eye landmarks.

        Returns:
            tuple: Tuple containing the eye box and angle.
        """
        eye_center = ((eye_landmark1[0] + eye_landmark2[0]) // 2, (eye_landmark1[1] + eye_landmark2[1]) // 2)
        eye_width = 2 * eye_distance
        eye_height = eye_distance
        eye_angle = self.calculate_angle(eye_landmark1, eye_landmark2)
        eye_box = cv2.boxPoints(((eye_center[0], eye_center[1]), (eye_width, eye_height), eye_angle))
        return np.int0(eye_box), eye_angle

    def calculate_eye_boxes(self, frame: np.ndarray) -> list[dict[str, dict[str, np.ndarray]]]:
        """
        Calculate the eye boxes for each face in the given frame.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            list: List of dictionaries containing eye box and angle information for each eye.
        """
        faces = self.face_detector.detect_faces(frame)

        eye_boxes = []

        for face in faces:
            landmarks = self.face_detector.predictor(frame, face)

            left_eye_landmark1, left_eye_landmark2 = self.get_eye_landmarks(landmarks, 36, 39)
            right_eye_landmark1, right_eye_landmark2 = self.get_eye_landmarks(landmarks, 42, 45)

            left_eye_distance = self.calculate_distance(left_eye_landmark1, left_eye_landmark2)
            right_eye_distance = self.calculate_distance(right_eye_landmark1, right_eye_landmark2)

            left_eye_box, left_eye_angle = self.get_eye_box(left_eye_landmark1, left_eye_landmark2, left_eye_distance)
            right_eye_box, right_eye_angle = self.get_eye_box(right_eye_landmark1, right_eye_landmark2, right_eye_distance)

            eye_boxes.append({
                'left_eye': {'box': left_eye_box, 'angle': left_eye_angle},
                'right_eye': {'box': right_eye_box, 'angle': right_eye_angle}
            })

        return eye_boxes
