import cv2
import numpy as np
import re
import os

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
    for alias, team_name in sorted(
        TEAM_NAME_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        match_index = lowered.find(alias)
        if match_index >= 0:
            alias_matches.append((match_index, team_name))

    for _, team_name in sorted(alias_matches):
        if team_name not in found_names:
            found_names.append(team_name)

    return found_names[:2]

def ocr_scorebug_strip(frame):
    """
    OCR a wide horizontal strip across the top of the frame where the
    scorebug lives on any broadcaster layout. Returns the raw OCR text.
    Works for NBC Sports, Sky Sports, BT Sport, ESPN, beIN, etc. because
    we read the entire strip rather than guessing which x-position each
    team name sits at.
    """
    try:
        import pytesseract
    except ImportError:
        return ""

    height, width = frame.shape[:2]

    # Wide strip: full width, top 22% of frame — captures any scorebug position
    strip = frame[
        int(height * 0.04): int(height * 0.22),
        int(width * 0.00): int(width * 1.00),
    ]

    if strip.size == 0:
        return ""

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    # Upscale 3x so small text becomes legible
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    best_text = ""
    for variant in [gray, cv2.bitwise_not(gray)]:
        _, thresh = cv2.threshold(variant, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            # psm 11 = sparse text, finds words anywhere in the image
            text = pytesseract.image_to_string(
                thresh,
                config="--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ&.- "
            ).strip()
        except Exception:
            text = ""
        if sum(ch.isalpha() for ch in text) > sum(ch.isalpha() for ch in best_text):
            best_text = text

    return best_text

def extract_scorebug_abbreviations(frame):
    """
    Read the full scorebug strip and return the first two team abbreviations
    found in it as (left_abbr, right_abbr). Order is best-effort based on
    which known abbreviation appears first in the OCR text.
    """
    text = ocr_scorebug_strip(frame)
    print("[OCR scorebug strip]", repr(text))

    tokens = re.findall(r"\b[A-Z]{2,4}\b", text.upper())
    found = []
    for token in tokens:
        if token in TEAM_ABBREVIATIONS and token not in found:
            found.append(token)
        if len(found) >= 2:
            break

    left  = found[0] if len(found) > 0 else ""
    right = found[1] if len(found) > 1 else ""
    return left, right

def abbreviations_to_teams(left_abbr, right_abbr):
    left_team  = TEAM_ABBREVIATIONS.get(left_abbr)
    right_team = TEAM_ABBREVIATIONS.get(right_abbr)
    return left_team, right_team

def extract_team_names_from_scoreboard_text(text):
    normalized = normalize_text(text).upper()
    found_names = []

    for token in re.findall(r"\b[A-Z]{2,4}\b", normalized):
        team_name = TEAM_ABBREVIATIONS.get(token)
        if team_name and team_name not in found_names:
            found_names.append(team_name)

    return found_names[:2]

def extract_scoreboard_text(frame):
    return ocr_scorebug_strip(frame)

def read_first_frame(video_path):
    capture = cv2.VideoCapture(video_path)
    success, frame = capture.read()
    capture.release()

    if not success:
        return None

    return frame

def resolve_matchup(video_path):
    basename = os.path.basename(video_path) if video_path else ""
    searchable_text = normalize_text(basename)
    team_names = extract_team_names_from_text(searchable_text)
    source = "filename"

    if len(team_names) < 2 and video_path:
        capture = cv2.VideoCapture(video_path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample early frames — scorebug is almost always visible from the start
        sample_frames = [0, frame_count // 10, frame_count // 5, frame_count // 3]
        sample_frames = sorted(set(max(0, idx) for idx in sample_frames))

        for frame_index in sample_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue

            left_abbr, right_abbr = extract_scorebug_abbreviations(frame)
            print("LEFT ABBR:", repr(left_abbr))
            print("RIGHT ABBR:", repr(right_abbr))

            left_team, right_team = abbreviations_to_teams(left_abbr, right_abbr)

            for team_name in [left_team, right_team]:
                if team_name and team_name not in team_names:
                    team_names.append(team_name)

            if len(team_names) >= 2:
                source = "OCR"
                break

        capture.release()

    if len(team_names) < 2:
        source = "fallback"

    return team_names[:2], source

def resolve_requested_team(team_query, team_names):
    if not team_query or not team_names:
        return None

    query_matches = extract_team_names_from_text(team_query)

    for query_match in query_matches:
        if query_match in team_names:
            return query_match

    normalized_query = normalize_text(team_query).lower()

    for team_name in team_names:
        if normalized_query and normalized_query in team_name.lower():
            return team_name

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