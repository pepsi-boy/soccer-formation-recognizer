import cv2
import numpy as np
import re

from .config import TEAM_ABBREVIATIONS, TEAM_NAME_ALIASES

def get_video_path(video_input):
    if isinstance(video_input, dict):
        return video_input.get("path")
    return video_input

def normalize_text(value):
    if value is None:
        return ""
    return str(value).replace("_", " ").replace("-", " ").strip()

def extract_team_names_from_text(text):
    normalized = normalize_text(text)
    found_names = []

    for token in re.findall(r"\b[A-Z]{2,4}\b", normalized.upper()):
        team_name = TEAM_ABBREVIATIONS.get(token)
        if team_name and team_name not in found_names:
            found_names.append(team_name)

    lowered = normalized.lower()
    alias_matches = []
    for alias, team_name in TEAM_NAME_ALIASES.items():
        match_index = lowered.find(alias)
        if match_index >= 0:
            alias_matches.append((match_index, team_name))

    for _, team_name in sorted(alias_matches):
        if team_name not in found_names:
            found_names.append(team_name)

    return found_names[:2]

def extract_scoreboard_text(frame):
    try:
        import pytesseract
    except ImportError:
        return ""

    height, width = frame.shape[:2]
    scoreboard_crop = frame[0:int(height * 0.22), 0:int(width * 0.45)]
    gray = cv2.cvtColor(scoreboard_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    try:
        return pytesseract.image_to_string(thresholded, config="--psm 6")
    except Exception:
        return ""

def read_first_frame(video_path):
    capture = cv2.VideoCapture(video_path)
    success, frame = capture.read()
    capture.release()

    if not success:
        return None

    return frame

def resolve_matchup(video_path):
    searchable_text = normalize_text(video_path)
    source = "filename"
    team_names = extract_team_names_from_text(searchable_text)

    if len(team_names) < 2 and video_path:
        source = "OCR"
        first_frame = read_first_frame(video_path)
        if first_frame is not None:
            ocr_text = extract_scoreboard_text(first_frame)
            for team_name in extract_team_names_from_text(ocr_text):
                if team_name not in team_names:
                    team_names.append(team_name)

    if len(team_names) < 2:
        source = "fallback"

    return team_names[:2], source

def resolve_requested_team(team_query, team_names):
    query_matches = extract_team_names_from_text(team_query)

    for query_match in query_matches:
        if query_match in team_names:
            return query_match

    normalized_query = normalize_text(team_query).lower()
    for team_name in team_names:
        if normalized_query and normalized_query in team_name.lower():
            return team_name

    if team_names:
        return team_names[0]

    return None

def describe_lab_color(lab_color):
    lab_array = np.uint8([[lab_color]])
    bgr = cv2.cvtColor(lab_array, cv2.COLOR_LAB2BGR)[0][0]
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    hue, saturation, value = [int(channel) for channel in hsv]

    if value < 55:
        return "dark kit"
    if saturation < 35 and value > 190:
        return "white/light kit"
    if saturation < 45:
        return "gray kit"
    if hue < 10 or hue >= 170:
        return "red kit"
    if hue < 25:
        return "orange kit"
    if hue < 35:
        return "yellow kit"
    if hue < 85:
        return "green kit"
    if hue < 100:
        return "cyan kit"
    if hue < 130:
        return "blue kit"
    if hue < 160:
        return "purple kit"
    return "pink kit"

def average_lab_to_bgr(lab_color):
    lab_array = np.uint8([[np.clip(lab_color, 0, 255)]])
    return cv2.cvtColor(lab_array, cv2.COLOR_LAB2BGR)[0][0]

def bgr_to_lab_color(bgr_color):
    bgr_array = np.uint8([[np.clip(bgr_color, 0, 255)]])
    return cv2.cvtColor(bgr_array, cv2.COLOR_BGR2LAB)[0][0].astype(float)

def describe_bgr_color(bgr_color):
    return describe_lab_color(bgr_to_lab_color(bgr_color))
