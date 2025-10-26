# Eye Blink Detector

Eye Blink Detector is a research toolkit for (1) converting raw ophthalmic videos into per-eye datasets, (2) running blink classifiers over recorded material, and (3) exploring/annotating blinks live through a desktop app. The repository brings together preprocessing utilities, inference pipelines, visualization/annotation tools, and evaluation helpers.

---

## Table of Contents
1. [Features](#features)
2. [Repository Layout](#repository-layout)
3. [Environment Setup](#environment-setup)
4. [Datasets & Conversion Pipeline](#datasets--conversion-pipeline)
5. [Blink CLI Workflows](#blink-cli-workflows)
6. [Real-Time Blink Application](#real-time-blink-application)
7. [Annotation & Evaluation Helpers](#annotation--evaluation-helpers)
8. [Development Tips](#development-tips)

---

## Features
- **Batch video processing** – `blink_cli.py` can scan a directory of `.avi` files, detect both eyes (or a single eye), run the blink predictor model, and export per-frame predictions, thumbnails, and summaries.
- **Interactive recording & inference** – `app.py` launches a Tk/Tkinter UI that streams a webcam feed, runs blinks in real time, counts blinks, and optionally records sessions.
- **Dataset conversion** – `convert_videos_fabiana.py` consumes the `videos_fabiana/<DATASET>/<video>.avi` + `.annotations.xlsx` pairs and emits normalized datasets where every frame lives in one folder alongside a CSV with blink metadata.
- **Annotation tooling** – `blink_cli.py annotate` integrates with `annotator_ui.py` to review or edit blink labels frame by frame.
- **Evaluation/reporting** – `evaluator.py` and `blink_data_exporter.py` turn prediction CSVs into metrics, blink IDs, and Excel reports.

---

## Repository Layout
| Path | Purpose |
| --- | --- |
| `blink_cli.py` | Main CLI for batch processing or annotation. |
| `app.py`, `ui.py`, `controller.py` | Desktop app entry, UI, and orchestration logic. |
| `convert_videos_fabiana.py` | Dataset converter that flattens annotated videos into image+CSV datasets. |
| `eye_detector.py`, `eye_extractor.py`, `frame_processor.py`, `frame_info.py` | Vision utilities for detecting faces/eyes, cropping, and tracking metadata. |
| `blink_predictor.py`, `model/*.ckpt` | Core inference engine and trained weights. |
| `annotator_ui.py`, `evaluator.py`, `blink_data_exporter.py` | Labeling, evaluation, and reporting helpers. |
| `videos_fabiana/` | Raw videos + annotation spreadsheets (not tracked). |
| `fabiana_datasets/`, `<dataset>_analysis/` | Generated datasets and experiment outputs. |

See `AGENTS.md` for contributor-focused guidelines.

---

## Environment Setup
This project relies on dlib, OpenCV, Pillow, pandas, and Tkinter. A pre-built Conda env (`blink_app`) is assumed.

```bash
git clone <repo>
cd eye-blink-detector
source activate blink_app              # loads Python 3.11, dlib, OpenCV, etc.
# optional: install extra deps
pip install -r requirements.txt        # if you maintain a requirements file
```

If you create a new environment manually:
```bash
conda create -n blink_app python=3.11 \
    opencv dlib pillow pandas scipy scikit-learn tk screeninfo tqdm
conda activate blink_app
pip install -r optional-requirements.txt
```

> Large model files (`shape_predictor_68_face_landmarks.dat`, `model/*.ckpt`, etc.) must stay in place for inference to work.

---

## Datasets & Conversion Pipeline
Raw Fabiana datasets contain `.avi` videos and Excel annotations with columns: `video`, `frameId`, `eye`, `blink`, `NV`, `blink_id`. Use the converter to create the flat format used by training code:

```bash
python convert_videos_fabiana.py \
    --input-dir videos_fabiana \
    --output-dir fabiana_datasets \
    --datasets SLIT_LAMP TEARSCOPE \
    --overwrite                        # clobber previous exports
```

Flags:
- `--max-frames N` – dry run on the first N frames per video.
- `--image-format {jpg,png}` – choose output extension.

Output layout:
```
fabiana_datasets/
  SLIT_LAMP/
    0.jpg
    1.jpg
    ...
  SLIT_LAMP.csv    # videoFrame, frame, blink, blink_id, video, frameId, NV, eye
  TEARSCOPE/
  TEARSCOPE.csv
```

Keep `videos_fabiana/` read-only and regenerate datasets when annotations change.

---

## Blink CLI Workflows
`blink_cli.py` is organized as a subcommand-based CLI.

### Batch processing
```bash
python blink_cli.py process \
    --dir videos_fabiana/SLIT_LAMP \
    --two-eyes \
    --export-frames \
    --output runs/slit_lamp
```
- Accepts either `--video file.avi` (single video) or `--dir directory`.
- Choose `--two-eyes` for dual-eye detection or `--eye {left,right}` for single-eye inference.
- `--export-frames` saves RGB frames/thumbnails; outputs are stored under `<output>/<video>/`.
- Produces `blink_data.csv` (per-frame predictions) and `summary.csv` (blink counts).

### Annotation mode
```bash
python blink_cli.py annotate \
    --dir videos_fabiana/TEARSCOPE \
    --two-eyes
```
This launches the annotation UI to adjust or label blinks. Annotations are stored beside each video and can later be merged into datasets.

Logging is configured via the Python logger; add `LOGLEVEL=DEBUG` (or edit the script) for verbose traces.

---

## Real-Time Blink Application
`app.py` starts a Tkinter UI for live webcam capture, visualization, and recording.

```bash
python main.py           # launches EyeDetectionApp with default settings
# or explicitly
python - <<'PY'
from app import EyeDetectionApp
EyeDetectionApp().start()
PY
```

Capabilities:
- Select among detected webcams, display the live feed, and overlay detected eye boxes.
- Show cropped eye views plus a per-eye blink counter.
- Start/stop recording; videos and metadata are written under timestamped folders via `video_recorder.py`.
- Generate Excel reports from existing CSVs through the “Report” menu.

Useful knobs:
- `controller.py` wires together `WebcamCapture`, `EyeDetectionController`, and `BlinkPredictor`.
- Toggle frame export or change collection directories via the Options menu.

---

## Annotation & Evaluation Helpers
- **Annotator UI** (`blink_cli.py annotate` / `annotator_ui.py`) – frame-by-frame editing with blink/NV/blink_id fields.
- **Blink Data Exporter** (`blink_data_exporter.py`) – merges processed frames with ground truth to create Excel reports and ensures blink IDs are sequential.
- **Evaluator** (`evaluator.py`) – computes blink-level metrics from prediction CSVs; ideal for regression testing or model comparisons.

Example evaluation flow:
```bash
python evaluator.py \
    --pred fabiana_datasets/SLIT_LAMP.csv \
    --gt path/to/ground_truth.csv
```
Consult the script for exact argument names; adapt as needed for your experiment harness.

---

## Development Tips
- Follow the contributor guide in `AGENTS.md` for coding conventions, dataset policies, and PR expectations.
- Prefer `source activate blink_app` before running tools; many scripts import dlib/OpenCV at import time.
- Store generated datasets outside of Git (e.g., `fabiana_datasets/`, `runs/`). If you need to share results, provide reproduction commands instead of committing large files.
- When adding new datasets, keep the CSV schema identical (`videoFrame, frame, blink, blink_id, video, frameId, NV, eye`) so downstream tooling remains compatible.
- Manual QA is common: inspect exported frames and CSVs, run the evaluator, and note validation commands in commits/PRs.

Happy blinking!
