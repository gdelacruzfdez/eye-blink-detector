from controller import EyeDetectionController
from ui import EyeDetectionUI
from webcam_capture import WebcamCapture
from eye_extractor import DlibEyeExtractor, EyeExtractor
from frame_info import Eye


class EyeDetectionApp:
    def __init__(
        self,
        eye_extractor: EyeExtractor | None = None,
        eyes: list[Eye] | None = None,
    ):
        frame_source = WebcamCapture()
        eye_extractor = eye_extractor or DlibEyeExtractor()
        self.controller = EyeDetectionController(frame_source, eye_extractor, eyes)
        self.ui = EyeDetectionUI(self.controller)

    def start(self):
        self.ui.start()

