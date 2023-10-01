import cv2
import numpy as np
from typing import Callable, Optional

from blink_predictor import BlinkPredictor
from webcam_capture import WebcamCapture
import time
import os

class VideoRecorder:
    def __init__(self, webcam_capture: WebcamCapture, blink_predictor: BlinkPredictor):
        self.webcam_capture = webcam_capture
        self.recording = False
        self.video_writer = None
        self.on_recording_start: Optional[Callable[[], None]] = None
        self.on_recording_end: Optional[Callable[[], None]] = None
        self.start_recording_time = None
        self.blink_predictor = blink_predictor

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
        if self.blink_predictor.export_recording_data:
            video_path = os.path.join(self.blink_predictor.session_save_dir, 'video.mp4')  # Save video in session directory
            self.video_writer = cv2.VideoWriter(
                video_path,
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
        if self.recording and self.blink_predictor.export_recording_data:
            if self.video_writer is not None:
                self.video_writer.write(frame)
