from typing import Optional
import cv2
import numpy as np

class WebcamCapture:
    def __init__(self, device=0):
        self.cap = cv2.VideoCapture(device)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the webcam.

        Returns:
            numpy.ndarray or None: The captured frame, or None if there was an error.
        """
        ret, frame = self.cap.read()

        if ret:
            return frame
        return None

    def release(self):
        """
        Release the webcam capture.
        """
        self.cap.release()

    def get_frame_width(self) -> int:
        """
        Get the width of the captured frames.

        Returns:
            int: Frame width.
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_frame_height(self) -> int:
        """
        Get the height of the captured frames.

        Returns:
            int: Frame height.
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_fps(self) -> float:
        """
        Get the frames per second (FPS) of the captured frames.

        Returns:
            float: Frames per second.
        """
        return self.cap.get(cv2.CAP_PROP_FPS) if self.cap.get(cv2.CAP_PROP_FPS) != 0 else 30

    @staticmethod        
    def get_available_cameras(max_range):
        cameras = []
        for i in range(max_range):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                cameras.append(str(i))
            cap.release()
        return cameras

