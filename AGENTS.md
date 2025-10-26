# Repository Guidelines

## Project Structure & Responsibilities
- **Entry points** live in `blink_cli.py`, `app.py`, and `controller.py`; treat them as the only modules that wire together video capture, inference, and UI flows.  
- **Vision stack** (`eye_detector.py`, `eye_extractor.py`, `frame_processor.py`, `frame_info.py`, `video_file_capture.py`) houses model loading, eye-box math, and frame bookkeeping—keep these files pure and stateless.  
- **Data tooling** (e.g., `convert_videos_fabiana.py`, `blink_data_exporter.py`, `annotator_ui.py`, `evaluator.py`, `*_analysis/`) handles dataset preparation, labeling, and evaluation. Place newly generated artifacts under `fabiana_datasets/` or dataset-specific folders; avoid polluting `videos_fabiana/`.

## Environment & Key Commands
1. **Activate tooling stack**: `source activate blink_app` (ships dlib, OpenCV, Pillow).  
2. **Convert annotated videos**:  
   `python convert_videos_fabiana.py --input-dir videos_fabiana --output-dir fabiana_datasets --datasets SLIT_LAMP TEARSCOPE --overwrite`  
   Use `--max-frames N` for dry runs.  
3. **Run inference**:  
   `python blink_cli.py process --video inputs/clip.avi --two-eyes --export-frames`  
4. **Annotate or review**:  
   `python blink_cli.py annotate --dir videos_fabiana/SLIT_LAMP --two-eyes`

## Coding Standards
- Python 3.11+, 4-space indentation, exhaustive type hints for new functions, and module-level docstrings that explain intent.  
- Prefer pure functions and dependency injection; avoid global state except for model weights.  
- Use the `logging` module with contextual prefixes (`INFO` for milestones, `DEBUG` for per-frame details).  
- Name files and symbols descriptively (`snake_case` for functions/variables, `PascalCase` for classes); mirror existing naming for eyes (`LEFT`, `RIGHT`) and blink labels (`blink`, `NV`, `blink_id`).

## Testing & QA Practices
- No unified test harness yet—validate changes via targeted scripts. Record the exact commands and sample outputs in PRs.  
- For data tooling, run `convert_videos_fabiana.py --max-frames 50` before full exports; inspect a few frames (`open fabiana_datasets/SLIT_LAMP/0.jpg`) and verify CSV headers.  
- For inference changes, capture before/after metrics with `blink_cli.py process` + `evaluator.py` to prove there are no regressions.  
- Document manual verification steps so future agents can replay them.

## Dataset & Artifact Handling
- Raw videos live in `videos_fabiana/<dataset>/<video>.avi`; never edit them in place.  
- Generated images/CSVs belong in sibling directories (`fabiana_datasets/<dataset>/`, `SLIT_LAMP_analysis/<id>/`). Reference paths rather than committing bulky outputs.  
- When scripts mutate existing outputs, state the exact command in commits/PRs so datasets remain reproducible.

## Commit & Pull Request Workflow
- **Commits**: small, atomic, imperative titles (“Add evaluator metrics”). Include context or commands in the body if reproduction steps are non-trivial.  
- **PRs**: provide summary, motivation, validation logs/commands, and explicit notes on dataset, model, or config changes. Link issues, flag breaking changes, and list any manual follow-up required post-merge.  
- **Reviews**: highlight risky areas (model weights, data schemas) and request focused testing when relevant. Clean up debug artifacts before requesting review.

## Security & Configuration Tips
- Keep credentialed files (API keys, license files) out of the repo; use environment variables or `.env` files listed in `.gitignore`.  
- Large models (`model/*.ckpt`, `shape_predictor_68_face_landmarks.dat`) are already tracked—coordinate before replacing them to avoid accidental deletions.  
- When sharing datasets externally, anonymize file paths and remove any frames flagged as NV (not visible) if privacy is a concern.
