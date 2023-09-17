import cv2
import numpy as np
from typing import Callable, Optional
from webcam_capture import WebcamCapture
import time

class VideoRecorder:
    def __init__(self, webcam_capture: WebcamCapture):
        self.webcam_capture = webcam_capture
        self.recording = False
        self.video_writer = None
        self.on_recording_start: Optional[Callable[[], None]] = None
        self.on_recording_end: Optional[Callable[[], None]] = None
        self.start_recording_time = None

    def set_recording_start_callback(self, callback: Callable[[], None]):
        """
        Set the callback function to be called when recording starts.

        Args:
            callback (Callable[[], None]): The callback function.
        """
        self.on_recording_start = callback

    def set_recording_end_callback(self, callback: Callable[[], None]):
        """
        Set the callback function to be called when recording ends.

        Args:
            callback (Callable[[], None]): The callback function.
        """
        self.on_recording_end = callback

    def start_recording(self):
        """
        Start recording the frames from the webcam.
        """
        self.recording = True
        self.start_recording_time = time.time()
        print('fps ' + str(self.webcam_capture.get_fps()))
        self.video_writer = cv2.VideoWriter(
            'output.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.webcam_capture.get_fps(),
            (self.webcam_capture.get_frame_width(), self.webcam_capture.get_frame_height())
        )
        if self.on_recording_start:
            self.on_recording_start()

    def stop_recording(self):
        """
        Stop recording and release the video writer.
        """
        self.recording = False
        self.start_recording_time = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        if self.on_recording_end:
            self.on_recording_end()

    def process_frame(self, frame: np.ndarray):
        """
        Process a frame and write it to the video if recording is enabled.

        Args:
            frame (numpy.ndarray): The frame to be processed.
        """
        if self.recording:
            if self.video_writer is not None:
                self.video_writer.write(frame)
