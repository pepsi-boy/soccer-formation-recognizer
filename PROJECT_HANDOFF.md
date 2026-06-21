# Soccer Formation Recognizer - Full Project Handoff

This document summarizes the full context of the soccer formation recognizer project so it can be transferred to another LLM, coding assistant, collaborator, recruiter prep workflow, or future development session.

## Project Identity

Project name:

```text
Soccer Formation Recognizer
```

GitHub repository:

```text
https://github.com/pepsi-boy/soccer-formation-recognizer
```

Local project folder:

```text
/Users/omarjebril/Documents/soccer-formation-recognizer
```

Main file:

```text
app.py
```

Current app type:

```text
Interactive computer vision web app
```

Best short description:

```text
An interactive AI web app that uses YOLO-based player detection/tracking, jersey-color clustering, scoreboard OCR/kit-color parsing, coordinate analysis, and formation-template matching to estimate a soccer team’s visible formation from a short match clip.
```

## Original Goal

The user wanted to build a portfolio-ready AI project inspired by prior ACCRE/Vanderbilt GPU work:

```text
User uploads a short soccer match clip.
The model detects players.
The app identifies which team is being analyzed.
The app estimates the formation the selected team is using.
The app returns a formation guess, confidence, explanation, annotated frame, coordinate table, and top-down pitch map.
```

Example intended use case:

```text
Upload a 5-10 second Arsenal vs Manchester United clip.
Select or type a team.
The app says something like:
"The selected team appears to be using a 4-3-3 / 4-2-3-1 / 5-3-2 shape."
```

Important correction made early:

```text
The user originally mentioned 4-4-3, but normal soccer formations describe 10 outfield players, so 4-3-3 is the intended formation.
```

## Current Tech Stack

Core tools:

```text
Python
Gradio
Ultralytics YOLO
PyTorch backend through YOLO
OpenCV
NumPy
scikit-learn
Matplotlib
Tesseract OCR
pytesseract
```

The app uses YOLO through Ultralytics. This means PyTorch is being used indirectly because Ultralytics YOLO models run on PyTorch.

## Current Repository Files

Tracked files:

```text
app.py
README.md
requirements.txt
.gitignore
PROJECT_HANDOFF.md
```

Ignored files:

```text
.venv/
__pycache__/
yolov8n.pt
video files
outputs/
runs/
```

The YOLO weights file `yolov8n.pt` is intentionally ignored and should not be committed.

## GitHub Status

Git repo was initialized locally and pushed to GitHub.

Remote:

```text
origin https://github.com/pepsi-boy/soccer-formation-recognizer.git
```

Initial commit:

```text
e144897 Initial soccer formation recognizer
```

Branch:

```text
main
```

Note:

```text
PROJECT_HANDOFF.md was created after the initial GitHub push and may need to be added/committed/pushed if desired.
```

## Setup Commands

Project folder:

```bash
cd ~/Documents/soccer-formation-recognizer
```

Activate environment:

```bash
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install OCR system dependency:

```bash
brew install tesseract
```

Run app:

```bash
python app.py
```

Then open the Gradio local URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

## OCR Installation Status

Tesseract OCR was installed via Homebrew:

```text
tesseract 5.5.2
```

`pytesseract` was installed into both the Anaconda/global Python environment and the project `.venv`.

Verification previously passed:

```text
pytesseract ok in .venv
5.5.2
```

## High-Level App Flow

Current intended pipeline:

```text
User uploads video
↓
User enters team to analyze, e.g. TOT, LEE, Tottenham, Leeds
↓
OCR attempts to read scoreboard team abbreviations
↓
Scoreboard kit-color swatches are sampled left-to-right
↓
YOLO tracks people in the clip
↓
Tracked detections are collapsed into track segments
↓
Jersey colors are clustered into Team 1, Team 2, and Official/Other
↓
Scoreboard colors map the left/right teams to the closest detected jersey clusters
↓
The selected team’s tracked players are used for formation estimation
↓
The app outputs:
  - annotated frame
  - realistic top-down pitch map
  - tracked player coordinate table
  - formation result text
```

## UI State

The current UI should have:

```text
Upload Soccer Video
Team to Analyze
Frame Stride
Team Name Detection
Analyze Video button
Stop Analysis button
Detected Players image
Top-Down Pitch Map image
Tracked Player Coordinates table
Formation Result text box
```

The old UI had:

```text
Optional Matchup Hint
Team dropdown
Scoreboard First Team Detected As
Scoreboard Second Team Detected As
```

Those controls were removed because they confused the workflow.

The current desired interaction:

```text
Upload clip.
Type "TOT", "LEE", "Tottenham", or "Leeds" into Team to Analyze.
Click Analyze Video.
```

## Important UI Error Encountered

At one point Gradio threw:

```text
Value: Tottenham Hotspur is not in the list of choices: ['Team 1', 'Team 2']
```

Meaning:

```text
The browser/server was still using stale Gradio state from the old dropdown version.
```

Fix:

```text
Stop the server with Control + C.
Restart python app.py.
Open a fresh browser tab or hard refresh the Gradio page.
```

## Current Formation Templates

The app currently compares against:

```python
FORMATIONS = {
    "4-3-3": [4, 3, 3],
    "4-4-2": [4, 4, 2],
    "4-2-3-1": [4, 2, 3, 1],
    "3-5-2": [3, 5, 2],
    "5-3-2": [5, 3, 2],
}
```

These templates describe outfield-player line counts.

Example:

```text
[5, 3, 2] means five players in one line, three in another, two in another.
This maps to a likely 5-3-2 shape.
```

## How Formation Estimation Works

The current approach is rule-based, not a trained formation classifier.

Steps:

```text
1. Take selected team’s tracked player positions.
2. Use normalized y-position/depth values.
3. Cluster players into formation lines using KMeans.
4. Compare detected line-count proportions against known formation templates.
5. Pick the closest formation.
6. Return confidence and explanation.
```

The app reports:

```text
Formation Guess
Confidence
Projected Formation Lines
Tracked Player Line Counts
Explanation
```

Important limitation:

```text
The app estimates visible shape from broadcast footage, not guaranteed tactical truth.
```

## Coordinate System

For each detected/tracked person, YOLO gives:

```text
x1, y1, x2, y2
```

The app approximates the player’s standing point with:

```python
center_x = (x1 + x2) / 2
foot_y = y2
```

Then normalizes:

```python
normalized_x = center_x / frame_width
normalized_y = foot_y / frame_height
```

These normalized coordinates are used for the top-down pitch map.

## Top-Down Pitch Map

The pitch map was updated to resemble a real soccer pitch:

```text
goals on left/right
halfway line top-to-bottom
center circle
penalty boxes
six-yard boxes
penalty spots
goal frames
```

This replaced the earlier simplified rectangle where the halfway line ran horizontally.

Current note:

```text
The map is still based on normalized screen coordinates, not true pitch calibration/homography.
```

Future improvement:

```text
Use pitch-line detection and homography to map broadcast camera coordinates to real field coordinates.
```

## Team Detection And Labeling

This evolved through several versions.

### Version 1

Only used:

```text
Team 1
Team 2
```

from jersey-color clustering.

### Version 2

Added:

```text
Official/Other
```

because referees were being incorrectly treated as players.

The app now creates a third color cluster when enough people are visible:

```text
Team 1
Team 2
Official/Other
```

The smallest/outlier cluster tends to become:

```text
Official/Other
```

This helps filter referees and possibly goalkeepers/other outliers from formation logic.

### Version 3

Added OCR/team-name detection:

```text
TOT -> Tottenham Hotspur
LEE -> Leeds United
```

### Version 4

Added scoreboard kit-color swatch parsing.

The scoreboard often shows each team abbreviation with a kit-color block. For example:

```text
TOT has a white swatch
LEE has a black/gray swatch
```

The app now tries to:

```text
read scoreboard colors left-to-right
compare those to detected jersey clusters
map the requested team to the closest cluster
```

## Team Query Behavior

The user should no longer type the full matchup.

Current behavior:

```text
Type "TOT" or "Tottenham" to analyze Tottenham.
Type "LEE" or "Leeds" to analyze Leeds.
```

The app tries to OCR the scoreboard to detect both teams, then uses the requested team string to select one.

If OCR fails:

```text
The app may fall back incorrectly or need future improvements.
```

Potential fallback to add later:

```text
If OCR fails, allow a hidden/advanced manual matchup field again or infer from filename.
```

## Scoreboard Color Logic

Important functions:

```text
extract_scoreboard_kit_colors(frame)
dominant_lab_color(crop)
resolve_team_cluster_map(team_names, scoreboard_kit_colors, team_color_data)
```

General idea:

```text
1. Crop likely top-left scoreboard kit-color regions.
2. Find dominant color in each region.
3. Compare those LAB colors to detected team jersey-cluster LAB colors.
4. If swapped mapping is closer than normal mapping, map first scoreboard team to Team 2 and second to Team 1.
```

Known risk:

```text
The scoreboard crop zones are heuristic and may not work on every broadcaster/layout.
```

For the Tottenham vs Leeds NBC/Premier League clip, the user expected:

```text
TOT = white/light kit
LEE = dark/gray kit
```

## Jersey Color Logic

Initial approach:

```text
Crop torso area from each bounding box.
Average all pixels.
Cluster colors.
```

Problem:

```text
Small/loose bounding boxes included grass, causing white shirts to look green.
```

Fix:

```text
Convert torso crop to HSV.
Mask out likely grass pixels.
Average remaining pixels in LAB color space.
```

Known limitation:

```text
White kits can still be hard if the crop includes background, shadows, or compression artifacts.
```

## Tracking Logic

The app uses:

```python
model.track(...)
```

Instead of only:

```python
model(...)
```

Reason:

```text
We want repeated detections of the same player across frames to use tracking IDs.
```

Each tracked detection has:

```text
track_id
frame_index
bbox
normalized position
confidence
jersey color
team
```

Then:

```text
build_unique_tracked_players()
```

collapses many detections with the same `track_id` into one averaged player/track segment.

Important terminology change:

Originally the app said:

```text
Tracking found 162 unique Team 1 players
```

This was misleading because YOLO can create many track fragments.

The wording was changed to:

```text
Tracking produced 162 Team 1 track segments
```

This is more honest.

## Why Analysis Was Slow

Reason:

```text
YOLO tracking was processing the whole video frame-by-frame.
```

For a 30-second clip, that can mean hundreds of frames.

Work being done:

```text
YOLO detection
tracking ID maintenance
jersey-color extraction
team clustering
formation estimation
visualization
```

## Speed Improvements Added

Added:

```text
Frame Stride slider
```

Meaning:

```text
1 = analyze every frame
5 = analyze every 5th frame
10 = analyze every 10th frame
```

Current default:

```python
DEFAULT_TRACK_FRAME_STRIDE = 5
```

YOLO call now uses:

```python
vid_stride=frame_stride
```

The result text reports:

```text
Speed setting: processed every N frame(s).
```

Recommended:

```text
Frame Stride 5 for normal testing
Frame Stride 8-10 for faster demos
Frame Stride 1 for more detailed but slower analysis
```

## Cancel/Stop Button

Added:

```text
Stop Analysis button
```

Using Gradio event cancellation:

```python
stop_button.click(
    fn=None,
    inputs=None,
    outputs=None,
    cancels=[analyze_event],
)
```

Limitation:

```text
The stop button cancels the Gradio job, but a deep model call may take a moment to release.
```

## Current README

The README explains:

```text
project purpose
features
tech stack
setup
how it works
limitations
```

It includes the recent performance/cancel features:

```text
Frame stride control
Cancel a running analysis from the UI
```

## ACCRE Context

The user previously had access to Vanderbilt ACCRE GPUs through the visualization terminal.

Relevant scratch commands from the original background:

```bash
salloc --time=60:00 --account=es3890_acc --partition=batch_gpu --gres=gpu:nvidia_rtx_a6000:1
module purge
module load python/3.12.4
module load opencv/4.13.0
source soccer_env/bin/activate
```

Advice given:

```text
Build locally first.
Use ACCRE only later for heavy training/fine-tuning/batch processing.
Keep code on laptop and GitHub so ACCRE access loss does not risk losing work.
Do not rely on /nobackup as sole storage.
```

Current project does not require ACCRE.

## Data/Clip Advice

Suggested sources:

```text
Pexels soccer match videos
Pixabay soccer/football videos
SoccerNet for serious research later
```

For testing:

```text
wide camera angle
not a replay
not too zoomed
many players visible
clear kit color differences
horizontal video preferred
```

Avoid:

```text
celebration close-ups
vertical clips
crowd shots
training close-ups
clips with only one or two players
```

Copyright note:

```text
Do not commit copyrighted match clips to GitHub.
Use them privately for testing only.
```

## Major Bugs/Issues Fixed

### Indentation/Heredoc Confusion

Early issue:

```text
The user pasted Python code directly into the terminal heredoc prompt and hit IndentationError.
```

Resolution:

```text
Use TextEdit/VS Code to edit app.py.
Terminal is for commands.
Python code belongs in .py files.
```

### Missing Pitch Dots

Issue:

```text
User did not see yellow pitch-map dots.
```

Resolution:

```text
Scroll/check second Gradio panel.
Restart app if old version is running.
```

### Referee Detected As Player

Issue:

```text
Blue referee was boxed and assigned to a team.
```

Resolution:

```text
Add third color cluster Official/Other.
```

### Pitch Map Too Messy

Issue:

```text
Map showed too many tracking fragments/labels.
```

Resolution:

```text
Pitch map and coordinate table now show only formation_players, not every unique tracked segment.
```

### Misleading Unique Player Count

Issue:

```text
"162 unique players" sounded impossible.
```

Resolution:

```text
Wording changed to "track segments."
```

### Team Name Mapping Confusion

Issue:

```text
The app mapped Tottenham to wrong kit cluster/color.
```

Resolution:

```text
Use scoreboard kit-color swatches as source of truth where possible.
```

### Old Gradio Dropdown State

Issue:

```text
Value: Tottenham Hotspur is not in the list of choices: ['Team 1', 'Team 2']
```

Resolution:

```text
Restart app and open fresh browser tab.
New UI uses text input, not team dropdown.
```

## Current Known Limitations

The app is a prototype and has these limitations:

```text
No true pitch homography/calibration yet.
Formation is based on visible shape, not guaranteed tactical formation.
Tracking IDs can fragment when players disappear/reappear.
OCR can fail depending on scoreboard style/video quality.
Scoreboard color crop zones are tuned heuristically.
Jersey clustering can fail with similar kits, shadows, or small players.
Goalkeepers may be filtered as Official/Other because formations usually describe outfield players.
Broadcast camera angle affects coordinate interpretation.
The selected team may have fewer than 10 usable tracked players.
```

## Suggested Next Improvements

Best next technical upgrades:

```text
1. Add pitch homography / camera calibration.
2. Improve OCR scoreboard cropping and debug visualization.
3. Add a small image preview showing the cropped scoreboard kit-color regions.
4. Add better team-color mapping fallback if OCR fails.
5. Improve tracking by merging nearby fragmented track IDs.
6. Add manual override only as an advanced/collapsed control, not primary UI.
7. Add more formation templates.
8. Add confidence based on player count, track stability, and formation-template distance.
9. Add result export screenshot/GIF for portfolio.
10. Polish UI visually after backend is stable.
```

Best portfolio upgrades:

```text
1. Add screenshots to README.
2. Record a short demo video.
3. Add a "Limitations" section in the app itself.
4. Add cleaner cards/results layout.
5. Add example output in README without committing copyrighted footage.
```

## Suggested Resume Description

Possible resume bullet:

```text
Built an interactive soccer formation recognition web app using Python, Gradio, YOLO, OpenCV, scikit-learn, and OCR to detect/track players, separate teams by kit color, extract spatial coordinates, and estimate visible formations from match footage.
```

More technical version:

```text
Implemented a YOLO-based computer vision pipeline for soccer footage analysis, combining player tracking, jersey-color clustering, scoreboard OCR, coordinate normalization, and rule-based formation-template matching in an interactive Gradio app.
```

## Suggested GitHub Description

Repository description:

```text
Interactive YOLO-powered soccer formation recognizer with player tracking, team-color clustering, OCR-assisted team labeling, and Gradio UI.
```

Topics:

```text
computer-vision
yolo
soccer-analytics
sports-analytics
gradio
opencv
pytorch
ocr
machine-learning
```

## How To Commit This Handoff File

If desired:

```bash
cd ~/Documents/soccer-formation-recognizer
git add PROJECT_HANDOFF.md
git commit -m "Add project handoff notes"
git push
```

## Useful Commands

Run app:

```bash
cd ~/Documents/soccer-formation-recognizer
source .venv/bin/activate
python app.py
```

Check syntax:

```bash
python -c "compile(open('app.py').read(), 'app.py', 'exec')"
```

Check Git:

```bash
git status
git log --oneline -5
git remote -v
```

Push:

```bash
git push
```

## Mental Model For Another LLM

If continuing this project, preserve the core philosophy:

```text
Do not pretend the app perfectly knows tactics.
Frame results as "estimated visible shape."
Keep the UI interactive and portfolio-friendly.
Prioritize functionality first, visual polish second.
Avoid committing videos/model weights/env folders.
Explain tradeoffs clearly to the user.
```

The user is learning as they build. They appreciate step-by-step explanations, especially:

```text
what command/code to enter
where to enter it
why each step exists
what each output means
```

The user prefers concrete action over abstract outlines.

