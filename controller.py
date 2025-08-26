import threading
from queue import Queue
import time

from blink_data_exporter import BlinkDataExporter
from eye_detector import EyeDetector
from frame_info import FrameInfo, EyeData, Eye
from frame_processor import FrameProcessor
from video_recorder import VideoRecorder
from blink_predictor import BlinkPredictor
from frame_source import FrameSource
from webcam_capture import WebcamCapture
from PIL import Image
import cv2
import os


class EyeDetectionController:
    def __init__(self, frame_source: FrameSource) -> None:
        self.cameras = WebcamCapture.get_available_cameras(5)
        self.frame_source = frame_source
        self.frame_queue = Queue()
        self.eye_detector = EyeDetector()
        self.frame_processor = FrameProcessor()
        self.blink_predictor = BlinkPredictor(self.frame_source)
        self.video_recorder = VideoRecorder(self.frame_source, self.blink_predictor)
        self.frame_count = 0

        self.recording_thread = threading.Thread(target=self.record_frames)
        self.stop_recording_flag = threading.Event()

    def start(self):
        """
        Start the eye detection and recording application.
        """
        self.recording_thread.start()
        self.blink_predictor.start()

    def stop(self):
        """
        Stop the eye detection and recording application.
        """
        self.blink_predictor.stop()
        self.video_recorder.stop_recording()  # Stop the recording explicitly
        self.stop_recording_flag.set()  # Signal the recording thread to stop
        self.recording_thread.join()  # Wait for the recording thread to finish
        self.frame_source.release()

    def toggle_recording(self):
        """
        Toggle the recording state.
        """
        if not self.video_recorder.recording:
            self.frame_count = 0
            self.blink_predictor.start_new_recording_session()
            self.video_recorder.start_recording()
        else:
            self.video_recorder.stop_recording()
            self.blink_predictor.end_recording_session()

    def record_frames(self):
        frame_count = 0
        prev_time = time.time()
        while not self.stop_recording_flag.is_set():
            frame = self.frame_source.get_frame()
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - prev_time
            if elapsed_time > 2:
                real_fps = frame_count / elapsed_time
                print("Real Frame Rate: {:.2f} fps".format(real_fps))
                frame_count = 0
                prev_time = current_time

            if frame is not None:
                if self.video_recorder.recording:
                    self.video_recorder.process_frame(frame)
                self.frame_queue.put(frame)
            time.sleep(0.01)  # Introduce a delay of 10 milliseconds

    def switch_camera(self, selection):
        """
        Switch the webcam based on the dropdown selection.
        """
        self.frame_source.release()
        self.frame_source = WebcamCapture(int(selection))
        self.blink_predictor.frame_source = self.frame_source
        self.video_recorder.frame_source = self.frame_source

    def set_export_recording_data(self, value: bool):
        self.blink_predictor.set_export_recording_data(value)

    def get_latest_frame_info(self):
        """
        Fetch the latest frame, process it, and return a FrameInfo object with all relevant details.
        """
        frame = None
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()

        if frame is not None:
            self.frame_count += 1
            frame_width = self.frame_source.get_frame_width()
            frame_height = self.frame_source.get_frame_height()

            eye_boxes = self.eye_detector.calculate_eye_boxes(frame)
            frame_with_boxes = self.frame_processor.visualize_eye_boxes(frame, eye_boxes)
            left_eye_images, right_eye_images = self.frame_processor.extract_eye_images(frame, eye_boxes)

            eyes: dict[Eye, EyeData] = {}
            if len(left_eye_images) > 0:
                eyes[Eye.LEFT] = EyeData(img=left_eye_images[0])
            if len(right_eye_images) > 0:
                eyes[Eye.RIGHT] = EyeData(img=right_eye_images[0])

            frame_info = FrameInfo(
                frame_num=self.frame_count,
                frame_img=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                frame_with_boxes=frame_with_boxes,
                eye_boxes=eye_boxes,
                eyes=eyes,
            )
            if self.video_recorder.recording and eyes:
                self.blink_predictor.add_frame_to_processing_queue(frame_info)

            return frame_info
        return None

    def get_recording_duration(self):
        if self.video_recorder.recording and self.video_recorder.start_recording_time is not None:
            return int(time.time() - self.video_recorder.start_recording_time)
        return None

    def get_left_eye_blink_count(self):
        return self.blink_predictor.left_eye_stats.blink_count

    def get_right_eye_blink_count(self):
        return self.blink_predictor.right_eye_stats.blink_count

    def get_frame_width(self):
        return self.frame_source.get_frame_width()

    def get_frame_height(self):
        return self.frame_source.get_frame_height()

    @staticmethod
    def generate_report_from_csv(csv_file_path: str, frame_rate: int):
        csv_directory = os.path.dirname(csv_file_path)
        exporter = BlinkDataExporter(csv_directory, frame_rate)
        exporter.generate_report_from_csv(csv_file_path)
