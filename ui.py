import tkinter as tk
from PIL import Image, ImageTk
from screeninfo import get_monitors
import threading

from controller import EyeDetectionController

FONT_BOLD = ("Arial", 12, "bold")


class EyeDetectionUI:
    def __init__(self, controller: EyeDetectionController):
        self.controller = controller
        self.root = tk.Tk()
        self.initialize_ui()
        self.refresh_display_event = threading.Event()

    def initialize_ui(self):
        self.selected_camera = tk.StringVar(self.root, value=self.controller.cameras[0])
        self.setup_window_properties()
        self.setup_video_display()
        self.setup_eye_displays()
        self.setup_eye_labels()
        self.setup_controls()
        self.setup_dropdown()
        self.setup_blink_counter()
        self.setup_menu()
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def setup_window_properties(self):
        self.root.title("Eye Detection and Recording")

        frame_width = self.controller.get_frame_width()
        frame_height = self.controller.get_frame_height()

        aspect_ratio = frame_width / frame_height

        monitor = get_monitors()[0]
        screen_width = monitor.width

        self.window_width = int(screen_width / 3)
        window_height_aspect_ratio = int(self.window_width / aspect_ratio)
        self.window_height = int(1.9 * window_height_aspect_ratio)

        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.resizable(False, False)

    def setup_video_display(self):
        self.video_label = tk.Label(self.root)
        self.video_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    def setup_eye_displays(self):
        self.left_eye_label = tk.Label(self.root)
        self.right_eye_label = tk.Label(self.root)
        self.left_eye_label.grid(row=1, column=0, padx=10, pady=5)
        self.right_eye_label.grid(row=1, column=1, padx=10, pady=5)

    def setup_eye_labels(self):
        self.left_eye_separator = tk.Label(self.root, text="Left Eye", font=FONT_BOLD, pady=5)
        self.left_eye_separator.grid(row=2, column=0)

        self.right_eye_separator = tk.Label(self.root, text="Right Eye", font=FONT_BOLD, pady=5)
        self.right_eye_separator.grid(row=2, column=1)

    def setup_controls(self):
        self.record_button = tk.Button(self.root, text="Start Recording", command=self.toggle_recording)
        self.recording_indicator = tk.Label(self.root, text="Not Recording", fg="red", font=("Arial", 12, "bold"))
        self.recording_time_label = tk.Label(self.root, text="00:00:00", font=("Arial", 12, "bold"))
        self.record_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)
        self.recording_indicator.grid(row=4, column=0, columnspan=2, pady=5)
        self.recording_time_label.grid(row=5, column=0, columnspan=2, pady=5)

    def setup_dropdown(self):
        self.camera_selection_label = tk.Label(self.root, text="Select Camera:", font=FONT_BOLD)
        self.camera_selector = tk.OptionMenu(self.root, self.selected_camera, *self.controller.cameras, command=self.switch_camera)

        self.camera_selection_label.grid(row=6, column=0, pady=5)
        self.camera_selector.grid(row=6, column=1, pady=5)

    def setup_blink_counter(self):
        self.left_blinks_label = tk.Label(self.root, text="Left Eye Blink Count: 0", font=FONT_BOLD)
        self.left_blinks_label.grid(row=7, column=0, columnspan=1, pady=5)

        self.right_blinks_label = tk.Label(self.root, text="Right Eye Blink Count: 0", font=FONT_BOLD)
        self.right_blinks_label.grid(row=7, column=1, columnspan=1, pady=5)

    def setup_menu(self):
        self.export_recording_data_var = tk.BooleanVar(value=True)

        menubar = tk.Menu(self.root)

        # Create the options menu
        options_menu = tk.Menu(menubar, tearoff=0)

        # Associate the checkbox state change with handle_export_data_toggle
        options_menu.add_checkbutton(label="Export recording data", variable=self.export_recording_data_var,
                                     onvalue=True, offvalue=False, command=self.handle_export_data_toggle)
        menubar.add_cascade(label="Options", menu=options_menu)

        self.root.config(menu=menubar)

    def handle_export_data_toggle(self):
        self.controller.set_export_recording_data(self.export_recording_data_var.get())

    def toggle_recording(self):
        """
        Toggle the recording state and update the UI accordingly.
        """
        self.controller.toggle_recording()
        if self.controller.video_recorder.recording:
            self.record_button.configure(text="End Recording")
            self.recording_indicator.configure(text="Recording", fg="green")
        else:
            self.record_button.configure(text="Start Recording")
            self.recording_indicator.configure(text="Not Recording", fg="red")
            self.recording_time_label.configure(text="00:00:00")

    def switch_camera(self, selection):
        """
        Switch the webcam based on the dropdown selection.
        """
        self.controller.switch_camera(selection)

    def refresh_display(self):
        """
        Refresh the display, which includes updating the frames, images, and recording duration.
        """
        if not self.refresh_display_event.is_set():
            self.display_recording_duration()
            self.display_frames_and_images()
            self.display_blink_counts()
            self.video_label.after(10, self.refresh_display)

    def display_recording_duration(self):
        """
        Display the recording duration if recording.
        """
        elapsed_time = self.controller.get_recording_duration()
        if elapsed_time:
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.recording_time_label.configure(text=f"{hours:02}:{minutes:02}:{seconds:02}")

    def display_blink_counts(self):
        left_blinks = self.controller.get_left_eye_blink_count()
        right_blinks = self.controller.get_right_eye_blink_count()
        self.left_blinks_label.configure(text=f"Left Eye Blink Count: {left_blinks}")
        self.right_blinks_label.configure(text=f"Right Eye Blink Count: {right_blinks}")

    def display_frames_and_images(self):
        """
        Fetch the processed frame information and update the UI labels with the corresponding images.
        """
        frame_info = self.controller.get_latest_frame_info()

        if frame_info:
            frame_with_boxes = frame_info.frame_with_boxes
            left_eye_image = frame_info.left_eye_img
            right_eye_image = frame_info.right_eye_img

            frame_width, frame_height = frame_with_boxes.size
            aspect_ratio = frame_width / frame_height

            frame_new_width = self.window_width - 40
            frame_new_height = int(frame_new_width / aspect_ratio)

            # Resize the frame to fit the UI window
            frame_with_boxes = frame_with_boxes.resize((frame_new_width, frame_new_height), Image.BILINEAR)

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

    def start(self):
        """
        Start the eye detection and recording application.
        """
        self.controller.start()
        self.refresh_display()
        self.root.mainloop()

    def stop(self):
        """
        Stop the eye detection and recording application.
        """
        self.refresh_display_event.set()
        self.controller.stop()
        self.root.quit()
        self.root.destroy()
