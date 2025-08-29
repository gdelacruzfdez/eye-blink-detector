"""Command-line interface for running blink prediction on a video file."""

from __future__ import annotations

import argparse
import csv
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the blink predictor CLI."""
    parser = argparse.ArgumentParser(description="Detect blinks in a video file")
    parser.add_argument("--video", required=True, help="Path to the video file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--eye",
        choices=["left", "right"],
        help="Which eye to analyze (single-eye mode)",
    )
    group.add_argument(
        "--two-eyes",
        action="store_true",
        help="Detect and analyze both eyes",
    )
    parser.add_argument("--output", help="Optional path to CSV file for results")
    return parser.parse_args()


def setup_components(args: argparse.Namespace):
    """Create capture, extractor, predictor and list of eyes based on CLI args."""
    # Import heavy modules only when needed
    from video_file_capture import VideoFileCapture
    from eye_extractor import SingleEyeExtractor, DlibEyeExtractor
    from blink_predictor import BlinkPredictor
    from frame_info import Eye

    capture = VideoFileCapture(args.video)

    if args.two_eyes:
        extractor = DlibEyeExtractor()
        eyes: List[Eye] = [Eye.LEFT, Eye.RIGHT]
    else:
        extractor = SingleEyeExtractor(args.eye)
        eyes = [Eye(args.eye)]

    predictor = BlinkPredictor(capture, eyes=eyes)
    predictor.set_export_recording_data(False)
    predictor.start()

    return capture, extractor, predictor, eyes


def process_frames(capture, extractor, predictor):
    """Iterate over video frames and enqueue them for prediction."""
    from frame_info import FrameInfo, Eye, EyeData
    from PIL import Image
    import cv2

    frame_num = 0
    while True:
        frame = capture.get_frame()
        if frame is None:
            break

        left_imgs, right_imgs, frame_with_boxes, eye_boxes = extractor.extract(frame)
        frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        eyes_data: dict[Eye, EyeData] = {}
        if left_imgs:
            eyes_data[Eye.LEFT] = EyeData(img=left_imgs[0])
        if right_imgs:
            eyes_data[Eye.RIGHT] = EyeData(img=right_imgs[0])

        frame_info = FrameInfo(
            frame_num=frame_num,
            frame_img=frame_img,
            frame_with_boxes=frame_with_boxes,
            eye_boxes=eye_boxes,
            eyes=eyes_data,
        )
        predictor.add_frame_to_processing_queue(frame_info)
        frame_num += 1

    predictor.end_recording_session()
    predictor.stop()
    capture.release()

    return predictor.processed_frames


def write_csv(frames: Iterable, path: str, eyes: List):
    """Write processed frame predictions to a CSV file."""
    from frame_info import Eye

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")

        header = ["Frame Number"]
        for eye in eyes:
            label = eye.value.capitalize()
            header.extend(
                [
                    f"{label} Eye Blink Prediction",
                    f"{label} Eye Blink Probability",
                    f"{label} Eye Closed Probability",
                ]
            )
        writer.writerow(header)

        for frame in frames:
            row = [frame.frame_num]
            for eye in eyes:
                eye_data = frame.eyes.get(eye)
                if eye_data:
                    row.extend([eye_data.pred, eye_data.blink_prob, eye_data.closed_prob])
                else:
                    row.extend([None, None, None])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    capture, extractor, predictor, eyes = setup_components(args)
    frames = process_frames(capture, extractor, predictor)
    if args.output:
        write_csv(frames, args.output, eyes)


if __name__ == "__main__":
    main()

