import logging
import os
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms

from blink_data_exporter import BlinkDataExporter
from frame_info import Eye, FrameInfo
from frame_source import FrameSource
from model.cnn_transformer import get_blink_predictor

# Constant defining the size of the window for processing
WINDOW_SIZE = 32


class BlinkPredictor:
    """
    Responsible for managing and orchestrating the process of predicting blinks
    in incoming frames. This class uses a separate thread to handle the processing
    to ensure that frames are processed without blocking the main thread.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        eyes: List[Eye] | None = None,
        batch_size: int = 5,
        base_save_dir: str = "recordings",
    ):
        logging.info("Initializing BlinkPredictor.")
        self.frame_source = frame_source
        self.batch_size = batch_size
        self.eyes = eyes if eyes is not None else [Eye.LEFT, Eye.RIGHT]

        # Initialize image processor
        self.image_processor = ImageProcessor()

        # Create blink models, statistics and buffers for each eye
        self.blink_models: Dict[Eye, BlinkModel] = {
            eye: BlinkModel(batch_size) for eye in self.eyes
        }
        self.eye_stats: Dict[Eye, BlinkStatistics] = {
            eye: BlinkStatistics() for eye in self.eyes
        }
        self.eye_buffers: Dict[Eye, BufferHandler] = {
            eye: BufferHandler(
                self.image_processor,
                self.blink_models[eye],
                self.eye_stats[eye],
                self.batch_size,
                eye,
            )
            for eye in self.eyes
        }

        # Backwards compatible attributes
        self.left_eye_stats = self.eye_stats.get(Eye.LEFT, BlinkStatistics())
        self.right_eye_stats = self.eye_stats.get(Eye.RIGHT, BlinkStatistics())

        # Queue for processing frames
        self.processing_queue = Queue()

        # Thread for blink prediction
        self.eye_predictor_thread = threading.Thread(target=self.predict_blinks)

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
        logging.info("Starting blink prediction thread.")
        self.eye_predictor_thread.start()

    def stop(self) -> None:
        """Stop the blink prediction thread."""
        logging.info("Stopping blink prediction thread.")
        self.stop_signal.set()
        self.eye_predictor_thread.join()
        logging.info("Blink prediction thread stopped.")

    def set_export_recording_data(self, value: bool):
        logging.info(f"Setting export_recording_data to {value}.")
        self.export_recording_data = value

    def set_session_save_dir(self, path: str):
        self.session_save_dir = path
        if self.export_recording_data:
            for eye in self.eyes:
                os.makedirs(os.path.join(self.session_save_dir, f"{eye.value}_eyes"), exist_ok=True)

    def initialize_recording_directory(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_save_dir = os.path.join(self.base_save_dir, timestamp)
        for eye in self.eyes:
            os.makedirs(os.path.join(self.session_save_dir, f"{eye.value}_eyes"))
        logging.info(f"Initialized recording directory: {self.session_save_dir}")

    def add_frame_to_processing_queue(self, frame_info: FrameInfo) -> None:
        """Add a frame to the processing queue."""
        logging.debug(f"Adding frame {frame_info.frame_num} to processing queue.")
        self.processing_queue.put(frame_info)

    def predict_blinks(self) -> None:
        """Predict blinks for the frames in the processing queue."""
        while not self.stop_signal.is_set():
            if self.processing_queue.empty():
                time.sleep(0.1)
                continue

            frame_info = self.processing_queue.get()
            logging.debug(f"Processing frame {frame_info.frame_num} from queue.")

            # Handle processing for available eyes
            for eye, buffer in self.eye_buffers.items():
                eye_data = frame_info.eyes.get(eye)
                if eye_data is not None:
                    buffer.handle(eye_data.img, frame_info)

            if self.export_recording_data:
                # Save frame images
                self.save_frame_data(frame_info)

            # Add the processed frame to the list
            self.processed_frames.append(frame_info)
            # Mark the task as done
            self.processing_queue.task_done()

    def start_new_recording_session(self) -> None:
        """Resets the blink predictor by clearing statistics, the processing queue, and recreating the prediction models."""
        logging.info("Starting new recording session.")
        if self.export_recording_data:
            self.initialize_recording_directory()
        # 1. Reset blink statistics
        self.eye_stats = {eye: BlinkStatistics() for eye in self.eyes}
        self.left_eye_stats = self.eye_stats.get(Eye.LEFT, BlinkStatistics())
        self.right_eye_stats = self.eye_stats.get(Eye.RIGHT, BlinkStatistics())

        # 2. Clear the processing queue
        while not self.processing_queue.empty():
            self.processing_queue.get()
            self.processing_queue.task_done()

        # 3. Recreate the blink prediction models
        self.blink_models = {eye: BlinkModel(self.batch_size) for eye in self.eyes}

        # 4. Clear the processed frames list
        self.processed_frames.clear()

        # 5. Reset buffers
        self.eye_buffers = {
            eye: BufferHandler(
                self.image_processor,
                self.blink_models[eye],
                self.eye_stats[eye],
                self.batch_size,
                eye,
            )
            for eye in self.eyes
        }
        logging.info("New recording session started.")

    def save_frame_data(self, frame_info: FrameInfo):
        if self.session_save_dir:
            for eye, eye_info in frame_info.eyes.items():
                eye_path = os.path.join(
                    self.session_save_dir,
                    f"{eye.value}_eyes",
                    f"{eye.value}_eye_{frame_info.frame_num}.jpg",
                )
                eye_info.img.save(eye_path)

    def end_recording_session(self) -> None:
        """Processes any final tasks at the end of a recording session."""
        logging.info("Ending recording session.")
        # Wait until the queue is completely processed
        self.processing_queue.join()
        # Process the remaining frames in the buffers
        for buffer in self.eye_buffers.values():
            buffer.process_remaining()

        if self.export_recording_data and self.session_save_dir and self.processed_frames:
            exporter = BlinkDataExporter(
                self.session_save_dir, self.frame_source.get_fps(), self.eyes
            )
            exporter.export_all_blink_data_to_excel(self.processed_frames)
        logging.info("Recording session ended.")


class BufferHandler:
    """
    Manages the buffer for storing processed images until there are enough images
    to make a batch prediction. The buffer size is determined by WINDOW_SIZE and
    the provided batch size.
    """

    def __init__(
        self,
        image_processor: "ImageProcessor",
        blink_model: "BlinkModel",
        blink_stats: "BlinkStatistics",
        batch_size: int,
        eye: Eye,
    ):
        logging.debug(f"Initializing BufferHandler for {eye} eye.")
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

        logging.debug(f"Processing buffer for {self.eye} eye.")
        stacked_frames, frame_info_refs = zip(*self.buffer)
        stacked_frames = torch.stack(stacked_frames)
        blink_predictions = self.blink_model.predict(stacked_frames)

        for i, (is_blinking, blink_probability, closed_eye_probability) in enumerate(
            blink_predictions
        ):
            self.blink_stats.update_blink_stats(is_blinking)
            frame_ref = frame_info_refs[WINDOW_SIZE // 2 + i]
            if frame_ref is not None:
                eye_data = frame_ref.eyes.get(self.eye)
                if eye_data is not None:
                    eye_data.pred = is_blinking
                    eye_data.blink_prob = blink_probability
                    eye_data.closed_prob = closed_eye_probability

        # Reset the buffer by removing batch_size elements
        self.buffer = self.buffer[self.batch_size :]

    def process_remaining(self) -> None:
        """Process whatever is left in the buffer, padding with zero tensors using a sliding window approach."""
        logging.debug(f"Processing remaining buffer for {self.eye} eye.")
        zero_tensor = torch.zeros_like(self.buffer[0][0])

        # Add zero tensors so we have at least WINDOW_SIZE + batch_size tensors
        while len(self.buffer) < WINDOW_SIZE + self.batch_size:
            self.buffer.append((zero_tensor, None))

        # Now process the buffer until the last frames are processed
        while any(
            item[1] is not None
            for item in self.buffer[-(WINDOW_SIZE // 2 + self.batch_size) :]
        ):
            self.process_buffer()
            for _ in range(self.batch_size):
                self.buffer.append((zero_tensor, None))

    def pad_initial_frames(self) -> None:
        """Pad the buffer with zero tensors for initial frames."""
        logging.debug(f"Padding initial frames for {self.eye} eye.")
        zero_tensor = torch.zeros((3, 64, 64))
        for _ in range(
            WINDOW_SIZE // 2
        ):  # Padding half the window size, as it is centered around the current frame
            self.buffer.append((zero_tensor, None))


class ImageProcessor:
    """
    A helper class that processes incoming images, i.e., it applies transformations
    like resizing and normalization to prepare the images for the prediction model.
    """

    def __init__(self):
        logging.debug("Initializing ImageProcessor.")
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def process_image(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class BlinkModel:
    """
    Wraps the underlying blink prediction model (obtained from `get_blink_predictor`)
    and provides a method to make predictions for a batch of processed images.
    """

    def __init__(self, batch_size: int):
        logging.debug("Initializing BlinkModel.")
        self.model = get_blink_predictor(batch_size)

    def predict(self, batch: torch.Tensor) -> list[tuple[int, float, float]]:
        logging.debug("Making a prediction with the blink model.")
        with torch.no_grad():
            predictions = self.model(batch)
            _, predicted_classes = torch.max(predictions.data, 1)

        return [
            (
                item,
                predictions[i][1].item() + predictions[i][2].item(),
                predictions[i][2].item(),
            )
            for i, item in enumerate(predicted_classes.tolist())
        ]


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
            logging.debug(f"Blink detected. Total blinks: {self._blink_count}")
        self._is_blinking = is_blinking
