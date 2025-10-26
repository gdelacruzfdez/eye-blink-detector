#!/usr/bin/env python3
"""Convert Fabiana's annotated videos into the flat dataset format used elsewhere.

This script reads every ``.avi`` file inside ``videos_fabiana/<dataset>`` alongside
its ``.annotations.xlsx`` file, extracts eye crops for the annotated frames and
writes both the cropped images and a consolidated CSV describing each crop.

The output layout matches the format described by the user:

```
<output_root>/
    SLIT_LAMP/               # Directory with every cropped eye image
    SLIT_LAMP.csv            # Metadata with one row per eye per frame
    TEARSCOPE/
    TEARSCOPE.csv
```

Example:
    python convert_videos_fabiana.py --input-dir videos_fabiana \
        --output-dir fabiana_datasets --datasets SLIT_LAMP TEARSCOPE
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import pandas as pd

from video_file_capture import VideoFileCapture


REQUIRED_COLUMNS = {"video", "frameId", "eye", "blink", "NV", "blink_id"}


@dataclass
class DatasetRecord:
    videoFrame: str
    frame: str
    blink: int
    blink_id: int
    video: int
    frameId: int
    NV: int
    eye: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Fabiana's raw videos into flat eye datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("videos_fabiana"),
        help="Directory that contains SLIT_LAMP, TEARSCOPE, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fabiana_datasets"),
        help="Directory where converted datasets will be written.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Names of dataset folders inside --input-dir to process. "
        "Defaults to every immediate child directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow deleting previous outputs for the requested datasets.",
    )
    parser.add_argument(
        "--image-format",
        default="jpg",
        choices=["jpg", "png"],
        help="Image format/extension for the cropped eyes (default: jpg).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional safety valve to stop after this many frames per video (useful for quick tests).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (DEBUG, INFO, ...).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def resolve_dataset_dirs(input_dir: Path, requested: Optional[Sequence[str]]) -> List[Path]:
    if requested:
        dirs = []
        for name in requested:
            path = input_dir / name
            if not path.is_dir():
                raise FileNotFoundError(f"Dataset directory {path} does not exist.")
            dirs.append(path)
        return dirs

    dirs = [p for p in sorted(input_dir.iterdir()) if p.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No dataset directories found inside {input_dir}.")
    return dirs


def prepare_output_paths(image_dir: Path, csv_path: Path, overwrite: bool) -> None:
    if image_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory {image_dir} already exists. "
                "Pass --overwrite if you want to regenerate it."
            )
        shutil.rmtree(image_dir)
    if csv_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output file {csv_path} already exists. "
                "Pass --overwrite if you want to replace it."
            )
        csv_path.unlink()
    image_dir.mkdir(parents=True, exist_ok=True)


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    logging.debug("Reading annotations from %s", annotation_path)
    df = pd.read_excel(annotation_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{annotation_path} is missing required columns: {missing}")

    df = df.loc[:, ["video", "frameId", "eye", "blink", "NV", "blink_id"]].copy()
    df["eye"] = df["eye"].astype(str).str.strip().str.upper()
    df["frameId"] = pd.to_numeric(df["frameId"], errors="raise").astype(int)
    for col in ("blink", "NV", "blink_id"):
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .fillna(-1 if col == "blink_id" else 0)
            .astype(int)
        )
    df.sort_values(["frameId", "eye"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def annotations_by_frame(df: pd.DataFrame) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for _, row in df.iterrows():
        grouped.setdefault(int(row["frameId"]), []).append(row.to_dict())
    return grouped


def process_video(
    video_path: Path,
    frame_annotations: Dict[int, List[dict]],
    image_dir: Path,
    starting_index: int,
    video_id: int,
    image_format: str,
    max_frames: Optional[int],
) -> tuple[List[DatasetRecord], int, List[str]]:
    capture = VideoFileCapture(str(video_path))
    max_frame_id = max(frame_annotations.keys()) if frame_annotations else -1
    if max_frame_id < 0:
        logging.warning("No annotations found for %s, skipping video.", video_path.name)
        capture.release()
        return [], starting_index, []

    frame_limit = max_frame_id + 1 if max_frames is None else min(max_frames, max_frame_id + 1)

    records: List[DatasetRecord] = []
    missing_entries: List[str] = []
    processed_frames: set[int] = set()
    frame_idx = 0

    while frame_idx < frame_limit:
        frame = capture.get_frame()
        if frame is None:
            logging.warning(
                "Video %s ended at frame %d before exhausting annotations.",
                video_path.name,
                frame_idx,
            )
            break
        rows = frame_annotations.get(frame_idx)
        if rows:
            processed_frames.add(frame_idx)
            video_frame_name = f"{video_id}_frame{frame_idx}.jpg"
            for row in rows:
                image_name = f"{starting_index}.{image_format}"
                if not cv2.imwrite(str(image_dir / image_name), frame):
                    missing_entries.append(
                        f"Failed to save {image_name} for {video_path.name} frame {frame_idx}"
                    )
                    continue
                records.append(
                    DatasetRecord(
                        videoFrame=video_frame_name,
                        frame=image_name,
                        blink=int(row["blink"]),
                        blink_id=int(row["blink_id"]),
                        video=video_id,
                        frameId=frame_idx,
                        NV=int(row["NV"]),
                        eye=row["eye"],
                    )
                )
                starting_index += 1
        frame_idx += 1

    capture.release()

    remaining = set(frame_annotations) - processed_frames
    if remaining:
        logging.warning(
            "Video %s skipped %d annotated frames (no corresponding frame read). "
            "First few: %s",
            video_path.name,
            len(remaining),
            sorted(list(remaining))[:5],
        )

    return records, starting_index, missing_entries


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    overwrite: bool,
    image_format: str,
    max_frames: Optional[int],
) -> None:
    dataset_name = dataset_dir.name
    image_dir = output_dir / dataset_name
    csv_path = output_dir / f"{dataset_name}.csv"
    prepare_output_paths(image_dir, csv_path, overwrite)

    records: List[DatasetRecord] = []
    missing_entries: List[str] = []
    image_index = 0

    videos = sorted(dataset_dir.glob("*.avi"))
    if not videos:
        logging.warning("No videos found inside %s.", dataset_dir)
        return

    for video_idx, video_path in enumerate(videos, start=1):
        annotation_path = Path(f"{video_path}.annotations.xlsx")
        if not annotation_path.exists():
            logging.warning(
                "Annotation file %s missing for video %s, skipping.",
                annotation_path,
                video_path.name,
            )
            continue
        logging.info("Processing %s (%s)", dataset_name, video_path.name)
        df = load_annotations(annotation_path)
        frame_ann = annotations_by_frame(df)
        video_records, image_index, video_missing = process_video(
            video_path=video_path,
            frame_annotations=frame_ann,
            image_dir=image_dir,
            starting_index=image_index,
            video_id=video_idx,
            image_format=image_format,
            max_frames=max_frames,
        )
        records.extend(video_records)
        missing_entries.extend(video_missing)

    if not records:
        logging.warning("No records generated for dataset %s.", dataset_name)
        return

    df = pd.DataFrame([record.__dict__ for record in records])
    df.sort_values(["video", "frameId", "eye"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(csv_path, index=False)

    logging.info(
        "Finished dataset %s: %d images saved to %s, CSV rows: %d",
        dataset_name,
        len(records),
        image_dir,
        len(df),
    )
    if missing_entries:
        logging.warning(
            "Missing crops for %d annotations. See details below:\n%s",
            len(missing_entries),
            "\n".join(missing_entries[:20]),
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    dataset_dirs = resolve_dataset_dirs(args.input_dir, args.datasets)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_dir in dataset_dirs:
        process_dataset(
            dataset_dir=dataset_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            image_format=args.image_format,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
