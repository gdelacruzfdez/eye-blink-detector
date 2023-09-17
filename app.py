from controller import EyeDetectionController
from ui import EyeDetectionUI


class EyeDetectionApp:
    def __init__(self):
        self.controller = EyeDetectionController()
        self.ui = EyeDetectionUI(self.controller)

    def start(self):
        self.ui.start()

