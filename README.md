# eye-blink-detector

This project detects eye blinks using a neural network.

## Command Line Interface

A simple CLI is provided to run blink prediction on a video file.

```bash
# Single-eye mode
python blink_cli.py --video path/to/video.mp4 --eye left --output results.csv

# Two-eye mode
python blink_cli.py --video path/to/video.mp4 --two-eyes --output results.csv
```

Arguments:

- `--video PATH` – path to an input video.
- `--eye {left,right}` – analyze a single eye and treat the frame as that eye.
- `--two-eyes` – detect and analyze both eyes in each frame.
- `--output CSV` – optional path to a CSV file where predictions will be written.

The CSV will contain one row per frame with the blink prediction and probabilities
for the selected eye or for both eyes when `--two-eyes` is used.
