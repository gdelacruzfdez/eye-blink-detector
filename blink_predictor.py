import threading
import torch
from torchvision import transforms

from frame_info import FrameInfo
from model.cnn_transformer import get_blink_predictor
from PIL import Image
from queue import Queue
import time

# Constant defining the size of the window for processing
WINDOW_SIZE = 32


class BlinkPredictor:
    """
    Responsible for managing and orchestrating the process of predicting blinks
    in incoming frames. This class uses a separate thread to handle the processing
    to ensure that frames are processed without blocking the main thread.
    """

    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        # Initialize image processor and blink prediction model
        self.image_processor = ImageProcessor()
        self.blink_model = BlinkModel(batch_size)

        # Separate statistics for left and right eyes
        self.left_eye_stats = BlinkStatistics()
        self.right_eye_stats = BlinkStatistics()

        # Queue for processing frames
        self.processing_queue = Queue()

        # Thread for blink prediction
        self.eye_predictor_thread = threading.Thread(target=self.predict_blinks)

        # Buffers for left and right eyes
        self.left_eye_buffer = BufferHandler(self.image_processor, self.blink_model, self.left_eye_stats,
                                             self.batch_size)
        self.right_eye_buffer = BufferHandler(self.image_processor, self.blink_model, self.right_eye_stats,
                                              self.batch_size)

        # List to store processed frames
        self.processed_frames: list[FrameInfo] = []

        self.stop_signal = threading.Event()

    def start(self) -> None:
        """Start the blink prediction thread."""
        self.eye_predictor_thread.start()

    def stop(self) -> None:
        """Stop the blink prediction thread."""
        self.stop_signal.set()
        self.eye_predictor_thread.join()

    def add_frame_to_processing_queue(self, frame_info: FrameInfo) -> None:
        """Add a frame to the processing queue."""
        self.processing_queue.put(frame_info)

    def predict_blinks(self) -> None:
        """Predict blinks for the frames in the processing queue."""
        while not self.stop_signal.is_set():
            start_time = time.time()
            frame_info = self.processing_queue.get()

            # Handle processing for left and right eyes separately
            self.left_eye_buffer.handle(frame_info.left_eye_img, frame_info)
            self.right_eye_buffer.handle(frame_info.right_eye_img, frame_info)

            # Add the processed frame to the list
            self.processed_frames.append(frame_info)

            # Mark the task as done
            self.processing_queue.task_done()

            end_time = time.time()
            elapsed_time = end_time - start_time


class BufferHandler:
    """
    Manages the buffer for storing processed images until there are enough images
    to make a batch prediction. The buffer size is determined by WINDOW_SIZE and
    the provided batch size.
    """

    def __init__(self, image_processor: 'ImageProcessor', blink_model: 'BlinkModel', blink_stats: 'BlinkStatistics',
                 batch_size: int):
        # Initialize image processor, blink model, and blink statistics
        self.image_processor = image_processor
        self.blink_model = blink_model
        self.blink_stats = blink_stats
        self.batch_size = batch_size

        # List buffer to store processed images and frame references
        self.buffer: list[tuple[torch.Tensor, FrameInfo]] = []

    def handle(self, image: Image.Image, frame_info: FrameInfo) -> None:
        """Process the image and add to the buffer, then handle buffer processing."""
        processed = self.image_processor.process_image(image)
        self.buffer.append((processed, frame_info))
        self.process_buffer()

    def process_buffer(self) -> None:
        """Process the buffer if it reaches a certain size."""
        if len(self.buffer) < WINDOW_SIZE + self.batch_size:
            return

        stacked_frames, frame_info_refs = zip(*self.buffer[:WINDOW_SIZE + self.batch_size])
        stacked_frames = torch.stack(stacked_frames)
        blink_predictions = self.blink_model.predict(stacked_frames)

        for i, is_blinking in enumerate(blink_predictions):
            self.blink_stats.update_blink_stats(is_blinking)
            frame_info_refs[i].left_eye_pred = is_blinking

        # Reset the buffer by removing batch_size elements
        self.buffer = self.buffer[self.batch_size:]


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

    def predict(self, batch: torch.Tensor) -> list[bool]:
        with torch.no_grad():
            predictions = self.model(batch)
            _, predicted_classes = torch.max(predictions.data, 1)

        return [item == 1 for item in predicted_classes.tolist()]


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
