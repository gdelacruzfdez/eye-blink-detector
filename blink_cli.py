"""Command-line interface for running blink prediction on a video file."""

from __future__ import annotations

import argparse
import csv
import os
import glob
from typing import Iterable, List, Dict, Any

from controller import EyeDetectionController
from frame_info import Eye


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the blink predictor CLI."""
    parser = argparse.ArgumentParser(description="Detect blinks in a video file or a directory of video files.")
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument("--video", help="Path to the video file")
    video_group.add_argument("--dir", help="Path to the directory containing video files")

    eye_group = parser.add_mutually_exclusive_group(required=True)
    eye_group.add_argument(
        "--eye",
        choices=["left", "right"],
        help="Which eye to analyze (single-eye mode)",
    )
    eye_group.add_argument(
        "--two-eyes",
        action="store_true",
        help="Detect and analyze both eyes",
    )
    parser.add_argument("--output", help="Optional path to directory for results. Defaults to the current directory.")
    return parser.parse_args()


def setup_components(video_path: str, args: argparse.Namespace):
    """Create capture, extractor, predictor and list of eyes based on CLI args."""
    from video_file_capture import VideoFileCapture
    from eye_extractor import SingleEyeExtractor, DlibEyeExtractor
    from blink_predictor import BlinkPredictor

    capture = VideoFileCapture(video_path)

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


def write_summary_csv(summary_data: List[Dict[str, Any]], output_dir: str):
    """Write summary of blink counts to a CSV file."""
    output_path = os.path.join(output_dir, "summary.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        header = ["Video", "Left Eye Blinks", "Right Eye Blinks"]
        writer.writerow(header)
        for row in summary_data:
            writer.writerow([row["video"], row["left_blinks"], row["right_blinks"]])

def process_video(args: argparse.Namespace, video_path: str = None) -> Dict[str, Any]:
    """Process a single video file."""
    if video_path is None:
        video_path = args.video

    output_dir = args.output or "."
    os.makedirs(output_dir, exist_ok=True)

    video_filename = os.path.basename(video_path)
    csv_filename = os.path.splitext(video_filename)[0] + ".csv"
    output_csv_path = os.path.join(output_dir, csv_filename)

    capture, extractor, predictor, eyes = setup_components(video_path, args)
    frames = process_frames(capture, extractor, predictor)
    write_csv(frames, output_csv_path, eyes)

    frame_rate = capture.get_frame_rate()
    EyeDetectionController.generate_report_from_csv(output_csv_path, frame_rate)

    left_blinks = predictor.left_eye_stats.blink_count
    right_blinks = predictor.right_eye_stats.blink_count

    print(f"Finished processing {video_filename}")
    print(f"  - Left eye blinks: {left_blinks}")
    print(f"  - Right eye blinks: {right_blinks}")

    return {"video": video_filename, "left_blinks": left_blinks, "right_blinks": right_blinks}

def process_directory(args: argparse.Namespace):
    """Process all videos in a directory."""
    video_files = glob.glob(os.path.join(args.dir, "*.mp4")) + \
                  glob.glob(os.path.join(args.dir, "*.avi")) + \
                  glob.glob(os.path.join(args.dir, "*.mov"))

    if not video_files:
        print(f"No video files found in {args.dir}")
        return

    summary_data = []
    for video_file in video_files:
        summary = process_video(args, video_file)
        summary_data.append(summary)

    if summary_data:
        output_dir = args.output or "."
        write_summary_csv(summary_data, output_dir)
        print(f"\nSummary report generated at {os.path.join(output_dir, 'summary.csv')}")

def main() -> None:
    args = parse_args()
    if args.dir:
        process_directory(args)
    else:
        process_video(args)


if __name__ == "__main__":
    main()

