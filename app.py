import cv2
import time
import threading
from queue import Queue
import tkinter as tk
from PIL import Image, ImageTk
from eye_detector import EyeDetector
from frame_processor import FrameProcessor
from video_recorder import VideoRecorder
from webcam_capture import WebcamCapture
from screeninfo import get_monitors

class EyeDetectionApp:
    def __init__(self):
        self.frame_queue = Queue()
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
        self.stop_recording_flag = threading.Event()
        

    def setup_ui(self):
        """
        Set up the user interface elements.
        """
        self.root.title("Eye Detection and Recording")

        frame_width = self.webcam_capture.get_frame_width()
        frame_height = self.webcam_capture.get_frame_height()

        aspect_ratio = frame_width / frame_height

        monitor = get_monitors()[0]
        screen_width = monitor.width

        # Set a fixed window width and height
        self.window_width = int(screen_width / 3)
        window_height_aspect_ratio = int(self.window_width / aspect_ratio)
        self.window_height = int(1.8 * window_height_aspect_ratio)

        # Set the window size and make it non-resizable
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.resizable(False, False)
        

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

        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def start(self):
        """
        Start the eye detection and recording application.
        """
        self.setup_ui()
        self.recording_thread = threading.Thread(target=self.record_frames)
        self.recording_thread.start()
        self.update_frame()
        self.root.mainloop()

    def stop(self):
        """
        Stop the eye detection and recording application.
        """
        self.video_recorder.stop_recording() # Stop the recording explicitly
        self.stop_recording_flag.set()  # Signal the recording thread to stop
        self.recording_thread.join()  # Wait for the recording thread to finish
        self.webcam_capture.release()
        
        # Close the Tkinter main window gracefully
        self.root.quit()
        self.root.destroy()

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

    def record_frames(self):
        frame_count = 0
        prev_time = time.time()
        while not self.stop_recording_flag.is_set():
            frame = self.webcam_capture.get_frame()
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


    def update_frame(self):
        """
        Update the frame and UI labels with the latest webcam frame.
        """
        frame = None
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()

        if frame is not None:
            frame_width = self.webcam_capture.get_frame_width()
            frame_height = self.webcam_capture.get_frame_height()

            aspect_ratio = frame_width / frame_height
            
            eye_boxes = self.eye_detector.calculate_eye_boxes(frame)
            frame_with_boxes = self.frame_processor.visualize_eye_boxes(frame, eye_boxes)
            left_eye_images, right_eye_images = self.frame_processor.extract_eye_images(frame, eye_boxes)
            left_eye_image = left_eye_images[0] if len(left_eye_images) > 0 else Image.new("RGB", (1, 1), (0, 0, 0))  # Placeholder black image
            right_eye_image = right_eye_images[0] if len(right_eye_images) > 0 else Image.new("RGB", (1, 1), (0, 0, 0))  # Placeholder black image

            frame_new_width = self.window_width - 40
            frame_new_height = int(self.window_width / aspect_ratio)

            # Resize the frame to full window width and adjusted height
            frame_with_boxes = frame_with_boxes.resize((frame_new_width , frame_new_height), Image.BILINEAR)

            # Resize the eye images to half window width and adjusted height
            left_eye_image = left_eye_image.resize((frame_new_width // 2, frame_new_height // 3), Image.BILINEAR)
            right_eye_image = right_eye_image.resize((frame_new_width // 2, frame_new_height // 3), Image.BILINEAR)

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

        # Schedule the next frame update
        self.video_label.after(10, self.update_frame)
