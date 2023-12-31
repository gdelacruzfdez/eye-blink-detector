import threading
import torch
from torchvision import transforms

from frame_info import FrameInfo
from model.cnn_transformer import get_blink_predictor
from PIL import Image
from queue import Queue
import time
import os
from datetime import datetime
import csv

# Constant defining the size of the window for processing
WINDOW_SIZE = 32


class BlinkPredictor:
    """
    Responsible for managing and orchestrating the process of predicting blinks
    in incoming frames. This class uses a separate thread to handle the processing
    to ensure that frames are processed without blocking the main thread.
    """

    def __init__(self, batch_size: int = 5, base_save_dir: str = 'recordings'):
        self.batch_size = batch_size
        # Initialize image processor and blink prediction model
        self.image_processor = ImageProcessor()
        self.blink_model_left = BlinkModel(batch_size)
        self.blink_model_right = BlinkModel(batch_size)

        # Separate statistics for left and right eyes
        self.left_eye_stats = BlinkStatistics()
        self.right_eye_stats = BlinkStatistics()

        # Queue for processing frames
        self.processing_queue = Queue()

        # Thread for blink prediction
        self.eye_predictor_thread = threading.Thread(target=self.predict_blinks)

        # Buffers for left and right eyes
        self.left_eye_buffer = BufferHandler(self.image_processor, self.blink_model_left, self.left_eye_stats,
                                             self.batch_size, 'left')
        self.right_eye_buffer = BufferHandler(self.image_processor, self.blink_model_right, self.right_eye_stats,
                                              self.batch_size, 'right')

        # List to store processed frames
        self.processed_frames: list[FrameInfo] = []

        self.stop_signal = threading.Event()

        # Initialize export flag as True
        self.export_recording_data = True

        # Create base directory if it doesn't exist
        if not os.path.exists(base_save_dir):
            os.makedirs(base_save_dir)
        self.base_save_dir = base_save_dir
        self.session_save_dir = None  # Will be set when starting a new recording

    def start(self) -> None:
        """Start the blink prediction thread."""
        self.eye_predictor_thread.start()

    def stop(self) -> None:
        """Stop the blink prediction thread."""
        self.stop_signal.set()
        self.eye_predictor_thread.join()

    def set_export_recording_data(self, value: bool):
        self.export_recording_data = value

    def initialize_recording_directory(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_save_dir = os.path.join(self.base_save_dir, timestamp)
        os.makedirs(os.path.join(self.session_save_dir, 'left_eyes'))
        os.makedirs(os.path.join(self.session_save_dir, 'right_eyes'))

    def add_frame_to_processing_queue(self, frame_info: FrameInfo) -> None:
        """Add a frame to the processing queue."""
        self.processing_queue.put(frame_info)

    def predict_blinks(self) -> None:
        """Predict blinks for the frames in the processing queue."""
        while not self.stop_signal.is_set():
            if self.processing_queue.empty():
                time.sleep(0.1)
                continue

            start_time = time.time()
            frame_info = self.processing_queue.get()

            # Handle processing for left and right eyes separately
            self.left_eye_buffer.handle(frame_info.left_eye_img, frame_info)
            self.right_eye_buffer.handle(frame_info.right_eye_img, frame_info)

            if self.export_recording_data:
                # Save frame images
                self.save_frame_data(frame_info)

            # Add the processed frame to the list
            self.processed_frames.append(frame_info)
            # Mark the task as done
            self.processing_queue.task_done()

            end_time = time.time()
            elapsed_time = end_time - start_time

    def start_new_recording_session(self) -> None:
        """Resets the blink predictor by clearing statistics, the processing queue, and recreating the prediction models."""
        if self.export_recording_data:
            self.initialize_recording_directory()
        # 1. Reset blink statistics
        self.left_eye_stats = BlinkStatistics()
        self.right_eye_stats = BlinkStatistics()

        # 2. Clear the processing queue
        while not self.processing_queue.empty():
            self.processing_queue.get()
            self.processing_queue.task_done()

        # 3. Recreate the blink prediction models
        self.blink_model_left = BlinkModel(self.batch_size)
        self.blink_model_right = BlinkModel(self.batch_size)

        # 4. Clear the processed frames list
        self.processed_frames.clear()

        # 5. Reset buffers
        self.left_eye_buffer = BufferHandler(self.image_processor, self.blink_model_left, self.left_eye_stats,
                                             self.batch_size, 'left')
        self.right_eye_buffer = BufferHandler(self.image_processor, self.blink_model_right, self.right_eye_stats,
                                              self.batch_size, 'right')

    def save_frame_data(self, frame_info: FrameInfo):
        left_eye_path = os.path.join(self.session_save_dir, 'left_eyes', f"left_eye_{frame_info.frame_num}.jpg")
        right_eye_path = os.path.join(self.session_save_dir, 'right_eyes', f"right_eye_{frame_info.frame_num}.jpg")

        frame_info.left_eye_img.save(left_eye_path)
        frame_info.right_eye_img.save(right_eye_path)

    import csv

    # ... (rest of the code in BlinkPredictor)

    def generate_csv_from_processed_frames(self):
        csv_file_path = os.path.join(self.session_save_dir, 'blink_data.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            fieldnames = ['Frame Number', 'Left Eye Path', 'Right Eye Path',
                          'Left Eye Blink Prediction', 'Left Eye Blink Probability', 'Left Eye Closed Probability', 'Right Eye Blink Prediction',
                          'Right Eye Blink Probability', 'Right Eye Closed Probability', 'Average Blink Probability', 'Frame Blink Prediction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

            writer.writeheader()  # write the headers
            for frame_info in self.processed_frames:
                left_eye_path = os.path.join('left_eyes', f"left_eye_{frame_info.frame_num}.jpg")
                right_eye_path = os.path.join('right_eyes', f"right_eye_{frame_info.frame_num}.jpg")

                avg_blink_probability = (frame_info.left_eye_blink_prob + frame_info.right_eye_blink_prob) / 2
                frame_blink_prediction = 1 if avg_blink_probability > 0.5 else 0

                writer.writerow({
                    'Frame Number': frame_info.frame_num,
                    'Left Eye Path': left_eye_path,
                    'Right Eye Path': right_eye_path,
                    'Left Eye Blink Prediction': frame_info.left_eye_pred,
                    'Left Eye Blink Probability': round(frame_info.left_eye_blink_prob, 4),
                    'Left Eye Closed Probability': round(frame_info.left_eye_closed_prob, 4),
                    'Right Eye Blink Prediction': frame_info.right_eye_pred,
                    'Right Eye Blink Probability': round(frame_info.right_eye_blink_prob, 4),
                    'Right Eye Closed Probability': round(frame_info.right_eye_closed_prob, 4),
                    'Average Blink Probability': round(avg_blink_probability, 4),
                    'Frame Blink Prediction': frame_blink_prediction
                })

    def end_recording_session(self) -> None:
        # Wait until the queue is completely processed
        self.processing_queue.join()
        # Process the remaining frames in the buffers
        self.left_eye_buffer.process_remaining()
        self.right_eye_buffer.process_remaining()

        """Processes any final tasks at the end of a recording session."""
        if self.export_recording_data and self.session_save_dir and self.processed_frames:  # Only if we have a valid session directory and frames
            self.generate_csv_from_processed_frames()


class BufferHandler:
    """
    Manages the buffer for storing processed images until there are enough images
    to make a batch prediction. The buffer size is determined by WINDOW_SIZE and
    the provided batch size.
    """

    def __init__(self, image_processor: 'ImageProcessor', blink_model: 'BlinkModel', blink_stats: 'BlinkStatistics',
                 batch_size: int, eye: str):
        # Initialize image processor, blink model, and blink statistics
        self.image_processor = image_processor
        self.blink_model = blink_model
        self.blink_stats = blink_stats
        self.batch_size = batch_size
        self.eye = eye

        # List buffer to store processed images and frame references
        self.buffer: list[tuple[torch.Tensor, FrameInfo]] = []
        self.pad_initial_frames()

    def handle(self, image: Image.Image, frame_info: FrameInfo) -> None:
        """Process the image and add to the buffer, then handle buffer processing."""
        processed = self.image_processor.process_image(image)
        self.buffer.append((processed, frame_info))
        self.process_buffer()

    def process_buffer(self) -> None:
        """Process the buffer if it reaches a certain size."""
        if len(self.buffer) < WINDOW_SIZE + self.batch_size:
            return

        stacked_frames, frame_info_refs = zip(*self.buffer)
        stacked_frames = torch.stack(stacked_frames)
        blink_predictions = self.blink_model.predict(stacked_frames)

        for i, (is_blinking, blink_probability, closed_eye_probability) in enumerate(blink_predictions):
            self.blink_stats.update_blink_stats(is_blinking)
            if frame_info_refs[WINDOW_SIZE // 2 + i] is not None:
                if self.eye == 'left':
                    frame_info_refs[WINDOW_SIZE // 2 + i].left_eye_pred = is_blinking
                    frame_info_refs[WINDOW_SIZE // 2 + i].left_eye_blink_prob = blink_probability
                    frame_info_refs[WINDOW_SIZE // 2 + i].left_eye_closed_prob = closed_eye_probability
                else:
                    frame_info_refs[WINDOW_SIZE // 2 + i].right_eye_pred = is_blinking
                    frame_info_refs[WINDOW_SIZE // 2 + i].right_eye_blink_prob = blink_probability
                    frame_info_refs[WINDOW_SIZE // 2 + i].right_eye_closed_prob = closed_eye_probability

        # Reset the buffer by removing batch_size elements
        self.buffer = self.buffer[self.batch_size:]

    def process_remaining(self) -> None:
        """Process whatever is left in the buffer, padding with zero tensors using a sliding window approach."""
        zero_tensor = torch.zeros_like(self.buffer[0][0])

        # Add zero tensors so we have at least WINDOW_SIZE + batch_size tensors
        while len(self.buffer) < WINDOW_SIZE + self.batch_size:
            self.buffer.append((zero_tensor, None))

        # Now process the buffer until the last frames are processed
        while any(item[1] is not None for item in self.buffer[-(WINDOW_SIZE // 2 + self.batch_size):]):
            self.process_buffer()
            for _ in range(self.batch_size):
                self.buffer.append((zero_tensor, None))

    def pad_initial_frames(self) -> None:
        """Pad the buffer with zero tensors for initial frames."""
        zero_tensor = torch.zeros((3, 64, 64))
        for _ in range(WINDOW_SIZE // 2):  # Padding half the window size, as it is centered around the current frame
            self.buffer.append((zero_tensor, None))


class ImageProcessor:
    """
    A helper class that processes incoming images, i.e., it applies transformations
    like resizing and normalization to prepare the images for the prediction model.
    """

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((64, 64), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class BlinkModel:
    """
    Wraps the underlying blink prediction model (obtained from `get_blink_predictor`)
    and provides a method to make predictions for a batch of processed images.
    """

    def __init__(self, batch_size: int):
        self.model = get_blink_predictor(batch_size)

    def predict(self, batch: torch.Tensor) -> list[tuple[int, float, float]]:
        with torch.no_grad():
            predictions = self.model(batch)
            _, predicted_classes = torch.max(predictions.data, 1)

        return [(item, predictions[i][1].item() + predictions[i][2].item(), predictions[i][2].item()) for i, item in enumerate(predicted_classes.tolist())]


class BlinkStatistics:
    """
    Keeps track of blink-related statistics such as the number of blinks and the
    current state (whether the subject is blinking or not).
    """

    def __init__(self):
        self._blink_count = 0
        self._is_blinking = False

    @property
    def is_blinking(self):
        return self._is_blinking

    @property
    def blink_count(self):
        return self._blink_count

    def update_blink_stats(self, is_blinking: bool):
        if is_blinking and not self._is_blinking:
            self._blink_count += 1
        self._is_blinking = is_blinking
