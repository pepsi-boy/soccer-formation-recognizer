# Soccer Formation Recognizer

Interactive computer vision web app for estimating soccer team formations from short match clips.

The app uses YOLO player detection/tracking, jersey-color clustering, scoreboard team/kit parsing, coordinate extraction, and rule-based formation matching to return an estimated visible formation for a selected team.

## Features

- Upload a short soccer clip through a Gradio web interface
- Enter a team abbreviation or name, such as `TOT`, `LEE`, `Tottenham`, or `Leeds`
- Detect and track players with Ultralytics YOLO
- Separate teams by kit color and filter officials/outliers into an `Official/Other` group
- Use scoreboard team abbreviations and kit-color swatches when available
- Estimate formations such as `4-3-3`, `4-4-2`, `4-2-3-1`, `3-5-2`, and `5-3-2`
- Display an annotated video frame, tracked-player table, and realistic top-down pitch map
- Speed up analysis with a frame-stride control
- Cancel a running analysis from the UI

## Tech Stack

- Python
- PyTorch-backed Ultralytics YOLO
- OpenCV
- scikit-learn
- Gradio
- Matplotlib
- Tesseract OCR via `pytesseract`

## Setup

Install Tesseract OCR:

```bash
brew install tesseract
```

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open the local Gradio URL shown in the terminal.

## How It Works

1. The user uploads a soccer clip.
2. YOLO tracks people across the clip.
3. The app groups tracked people by jersey color into two teams plus `Official/Other`.
4. OCR and scoreboard kit-color swatches are used when possible to associate team names with detected kit clusters.
5. The selected team's tracked positions are averaged and mapped onto a top-down pitch.
6. The app groups the selected team's players into depth lines and compares those line counts against common formation templates.

## Notes And Limitations

This is a portfolio prototype, not a production scouting system. Broadcast camera angle, missing players, occlusion, similar kit colors, OCR quality, and tracking ID switches can affect the result. The app reports an estimated visible shape rather than a guaranteed tactical formation.
