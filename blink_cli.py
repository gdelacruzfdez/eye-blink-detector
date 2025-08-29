"""Command-line interface for running blink prediction on a video file."""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
from PIL import Image
from tqdm import tqdm

from blink_predictor import BlinkPredictor
from controller import EyeDetectionController
from eye_extractor import DlibEyeExtractor, EyeExtractor, SingleEyeExtractor
from frame_info import Eye, EyeData, FrameInfo
from video_file_capture import VideoFileCapture


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the blink predictor CLI."""
    parser = argparse.ArgumentParser(
        description="Detect blinks in a video file or a directory of video files."
    )
    video_group = parser.add_mutually_exclusive_group(required=True)
    video_group.add_argument("--video", help="Path to the video file")
    video_group.add_argument(
        "--dir", help="Path to the directory containing video files"
    )

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
    parser.add_argument(
        "--output",
        help="Optional path to directory for results. Defaults to the current directory.",
    )
    parser.add_argument(
        "--export-frames",
        action="store_true",
        help="Export frames to a directory.",
    )
    args = parser.parse_args()
    logging.info(f"Arguments parsed: {args}")
    return args


def setup_components(
    video_path: str,
    args: argparse.Namespace,
) -> Tuple[VideoFileCapture, EyeExtractor, BlinkPredictor, List[Eye]]:
    """Create capture, extractor, predictor and list of eyes based on CLI args."""
    logging.info(f"Setting up components for video: {video_path}")
    capture = VideoFileCapture(video_path)

    if args.two_eyes:
        extractor: EyeExtractor = DlibEyeExtractor()
        eyes: List[Eye] = [Eye.LEFT, Eye.RIGHT]
        logging.info("Using DlibEyeExtractor for both eyes.")
    else:
        extractor = SingleEyeExtractor(args.eye)
        eyes = [Eye(args.eye)]
        logging.info(f"Using SingleEyeExtractor for {args.eye} eye.")

    predictor = BlinkPredictor(capture, eyes=eyes)
    predictor.set_export_recording_data(args.export_frames)
    predictor.start()
    logging.info("BlinkPredictor started.")

    return capture, extractor, predictor, eyes


def process_frames(
    capture: VideoFileCapture,
    extractor: EyeExtractor,
    predictor: BlinkPredictor,
    export_frames_path: Optional[str] = None,
) -> List[FrameInfo]:
    """Iterate over video frames and enqueue them for prediction."""
    logging.info("Starting frame processing.")
    total_frames = capture.get_frame_count()
    with tqdm(total=total_frames, desc="Processing frames", position=1, leave=False) as pbar:
        frame_num = 0
        while True:
            frame = capture.get_frame()
            if frame is None:
                logging.info("End of video file reached.")
                break

            left_imgs, right_imgs, frame_with_boxes, eye_boxes = extractor.extract(frame)
            frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if export_frames_path:
                frame_img.save(
                    os.path.join(export_frames_path, f"frame_{frame_num:04d}.jpg")
                )

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
            pbar.update(1)

    predictor.end_recording_session()
    predictor.stop()
    capture.release()
    logging.info(f"Finished processing {frame_num} frames.")

    return predictor.processed_frames


def write_csv(frames: Iterable[FrameInfo], path: str, eyes: List[Eye]) -> None:
    """Write processed frame predictions to a CSV file."""
    logging.info(f"Writing results to CSV file: {path}")
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
            row: List[Any] = [frame.frame_num]
            for eye in eyes:
                eye_data = frame.eyes.get(eye)
                if eye_data:
                    row.extend(
                        [eye_data.pred, eye_data.blink_prob, eye_data.closed_prob]
                    )
                else:
                    row.extend([None, None, None])
            writer.writerow(row)
    logging.info("Finished writing CSV file.")


def write_summary_csv(summary_data: List[Dict[str, Any]], output_dir: str) -> None:
    """Write summary of blink counts to a CSV file."""
    output_path = os.path.join(output_dir, "summary.csv")
    logging.info(f"Writing summary to CSV file: {output_path}")
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        header = ["Video", "Left Eye Blinks", "Right Eye Blinks"]
        writer.writerow(header)
        for row in summary_data:
            writer.writerow([row["video"], row["left_blinks"], row["right_blinks"]])
    logging.info("Finished writing summary CSV file.")


def process_video(
    args: argparse.Namespace,
    video_path: str | None = None,
) -> Dict[str, Any] | None:
    """Process a single video file."""
    if video_path is None:
        video_path = args.video

    logging.info(f"Processing video: {video_path}")

    video_filename = os.path.basename(video_path)
    video_filename_without_ext = os.path.splitext(video_filename)[0]

    output_dir = args.output or "."
    video_output_dir = os.path.join(output_dir, video_filename_without_ext)

    if os.path.exists(video_output_dir):
        logging.warning(
            f"Output directory {video_output_dir} already exists, skipping video."
        )
        return None

    os.makedirs(video_output_dir)
    logging.info(f"Output directory for video: {video_output_dir}")

    csv_filename = "blink_data.csv"
    output_csv_path = os.path.join(video_output_dir, csv_filename)

    export_frames_path = None
    if args.export_frames:
        export_frames_path = os.path.join(video_output_dir, "frames")
        os.makedirs(export_frames_path, exist_ok=True)
        logging.info(f"Exporting frames to {export_frames_path}")

    capture, extractor, predictor, eyes = setup_components(video_path, args)
    if args.export_frames:
        predictor.set_session_save_dir(video_output_dir)

    frames = process_frames(capture, extractor, predictor, export_frames_path)
    write_csv(frames, output_csv_path, eyes)

    frame_rate = capture.get_fps()
    EyeDetectionController.generate_report_from_csv(output_csv_path, int(frame_rate))

    left_blinks = predictor.left_eye_stats.blink_count
    right_blinks = predictor.right_eye_stats.blink_count

    logging.info(f"Finished processing {video_filename}")
    logging.info(f"  - Left eye blinks: {left_blinks}")
    logging.info(f"  - Right eye blinks: {right_blinks}")

    return {
        "video": video_filename,
        "left_blinks": left_blinks,
        "right_blinks": right_blinks,
    }


def process_directory(args: argparse.Namespace) -> None:
    """Process all videos in a directory."""
    logging.info(f"Processing directory: {args.dir}")

    if args.output is None:
        args.output = f"{os.path.basename(os.path.abspath(args.dir))}_analysis"

    video_files = (
        glob.glob(os.path.join(args.dir, "*.mp4"))
        + glob.glob(os.path.join(args.dir, "*.avi"))
        + glob.glob(os.path.join(args.dir, "*.mov"))
    )

    if not video_files:
        logging.warning(f"No video files found in {args.dir}")
        return

    logging.info(f"Found {len(video_files)} video files to process.")

    summary_data = []
    for video_file in tqdm(video_files, desc="Processing videos", leave=True, position=0):
        summary = process_video(args, video_file)
        if summary:
            summary_data.append(summary)

    if summary_data:
        output_dir = args.output or "."
        write_summary_csv(summary_data, output_dir)
        logging.info(
            f"\nSummary report generated at {os.path.join(output_dir, 'summary.csv')}"
        )

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    if args.dir:
        process_directory(args)
    else:
        process_video(args)


if __name__ == "__main__":
    main()
