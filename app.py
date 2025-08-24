from controller import EyeDetectionController
from ui import EyeDetectionUI
from webcam_capture import WebcamCapture


class EyeDetectionApp:
    def __init__(self):
        frame_source = WebcamCapture()
        self.controller = EyeDetectionController(frame_source)
        self.ui = EyeDetectionUI(self.controller)

    def start(self):
        self.ui.start()

