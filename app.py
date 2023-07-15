import cv2
import tkinter as tk
from PIL import Image, ImageTk
from eye_detector import EyeDetector
from frame_processor import FrameProcessor
from video_recorder import VideoRecorder
from webcam_capture import WebcamCapture

class EyeDetectionApp:
    def __init__(self):
        self.eye_detector = EyeDetector()
        self.webcam_capture = WebcamCapture()
        self.video_recorder = VideoRecorder(self.webcam_capture)
        self.frame_processor = FrameProcessor()
        self.root = tk.Tk()
        self.video_label = tk.Label(self.root)
        self.left_eye_label = tk.Label(self.root)
        self.right_eye_label = tk.Label(self.root)
        self.record_button = tk.Button(self.root, text="Start Recording", command=self.toggle_recording)
        self.recording_indicator = tk.Label(self.root, text="Not Recording", fg="red", font=("Arial", 12, "bold"))

    def setup_ui(self):
        """
        Set up the user interface elements.
        """
        self.root.title("Eye Detection and Recording")

        # Video label
        self.video_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Left eye label
        self.left_eye_label.grid(row=1, column=0, padx=10, pady=5)

        # Right eye label
        self.right_eye_label.grid(row=1, column=1, padx=10, pady=5)

        # Left eye label separator
        left_eye_separator = tk.Label(self.root, text="Left Eye", font=("Arial", 12, "bold"), pady=5)
        left_eye_separator.grid(row=2, column=0)

        # Right eye label separator
        right_eye_separator = tk.Label(self.root, text="Right Eye", font=("Arial", 12, "bold"), pady=5)
        right_eye_separator.grid(row=2, column=1)

        # Recording button
        self.record_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # Recording indicator
        self.recording_indicator.grid(row=4, column=0, columnspan=2, pady=5)

    def start(self):
        """
        Start the eye detection and recording application.
        """
        self.setup_ui()
        self.update_frame()
        self.root.mainloop()

    def stop(self):
        """
        Stop the eye detection and recording application.
        """
        self.eye_detector.stop()
        self.webcam_capture.release()

    def toggle_recording(self):
        """
        Toggle the recording state and update the UI accordingly.
        """
        if not self.video_recorder.recording:
            self.video_recorder.start_recording()
            self.record_button.configure(text="End Recording")
            self.recording_indicator.configure(text="Recording", fg="green")
        else:
            self.video_recorder.stop_recording()
            self.record_button.configure(text="Start Recording")
            self.recording_indicator.configure(text="Not Recording", fg="red")

    def update_frame(self):
        """
        Update the frame and UI labels with the latest webcam frame.
        """
        frame = self.webcam_capture.get_frame()

        if frame is not None:
            frame_width = self.webcam_capture.get_frame_width()
            frame_height = self.webcam_capture.get_frame_height()

            eye_boxes = self.eye_detector.calculate_eye_boxes(frame)
            frame_with_boxes = self.frame_processor.visualize_eye_boxes(frame, eye_boxes)
            left_eye_images, right_eye_images = self.frame_processor.extract_eye_images(frame, eye_boxes)
            left_eye_image = left_eye_images[0] if len(left_eye_images) > 0 else Image.new("RGB", (1, 1), (0, 0, 0))  # Placeholder black image
            right_eye_image = right_eye_images[0] if len(right_eye_images) > 0 else Image.new("RGB", (1, 1), (0, 0, 0))  # Placeholder black image

            # Resize the images to fit the display
            frame_with_boxes = frame_with_boxes.resize((frame_width, frame_height), Image.BILINEAR)
            left_eye_image = left_eye_image.resize(
                (int(frame_width / 2) - 20, int(frame_height / 3) - 20),
                Image.BILINEAR,
            )
            right_eye_image = right_eye_image.resize(
                (int(frame_width / 2) - 20, int(frame_height / 3) - 20),
                Image.BILINEAR,
            )

            # Convert the PIL Images to Tkinter PhotoImage format
            tk_frame = ImageTk.PhotoImage(frame_with_boxes)
            tk_left_eye = ImageTk.PhotoImage(left_eye_image)
            tk_right_eye = ImageTk.PhotoImage(right_eye_image)

            # Update the video and eye labels with the new images
            self.video_label.configure(image=tk_frame)
            self.video_label.image = tk_frame
            self.left_eye_label.configure(image=tk_left_eye)
            self.left_eye_label.image = tk_left_eye
            self.right_eye_label.configure(image=tk_right_eye)
            self.right_eye_label.image = tk_right_eye

        if self.video_recorder.recording:
            self.video_recorder.process_frame(frame)

        # Schedule the next frame update
        self.video_label.after(10, self.update_frame)
