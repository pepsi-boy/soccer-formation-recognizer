import os
import re

import cv2
import numpy as np

from .config import TEAM_ABBREVIATIONS, TEAM_NAME_ALIASES, TEAM_TYPE


# ============================================================
# Color helpers
# ============================================================

def bgr_to_lab_color(bgr):
    """Convert a single BGR color (3,) to LAB via OpenCV."""
    pixel = np.uint8([[bgr]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return lab[0, 0].astype(np.float32)


def describe_lab_color(lab):
    """Return a rough human-readable color name from a LAB value."""
    l, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    if l < 35:
        return "Black/Dark"
    if l > 200:
        return "White/Light"
    if a > 150:
        return "Red"
    if a < 110:
        return "Green"
    if b > 150:
        return "Yellow"
    if b < 110:
        return "Blue"
    return "Mid-tone"


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    """Basic text normalization for OCR output."""
    text = text.strip()
    text = re.sub(r"[^\w\s\-&.']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# Abbreviation disambiguation
# ============================================================

def _pick_from_candidates(candidates, preferred_type=None):
    """Pick a single team from a list of candidates sharing an abbreviation."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if preferred_type:
        typed = [c for c in candidates if TEAM_TYPE.get(c) == preferred_type]
        if typed:
            return typed[0]
    # Default: first registered wins (club, since clubs are registered first)
    return candidates[0]


def _infer_context(left_candidates, right_candidates):
    """
    If either side has an unambiguous type, use that as context for both.
    If both sides have international teams available, prefer international.
    """
    left_types = {TEAM_TYPE.get(c) for c in left_candidates} if left_candidates else set()
    right_types = {TEAM_TYPE.get(c) for c in right_candidates} if right_candidates else set()

    # If one side is unambiguously international, treat both as international
    if left_candidates and len(left_candidates) == 1 and "international" in left_types:
        return "international"
    if right_candidates and len(right_candidates) == 1 and "international" in right_types:
        return "international"

    # If one side is unambiguously club, treat both as club
    if left_candidates and len(left_candidates) == 1 and "club" in left_types:
        return "club"
    if right_candidates and len(right_candidates) == 1 and "club" in right_types:
        return "club"

    # If both sides have international candidates, prefer international
    if "international" in left_types and "international" in right_types:
        return "international"

    # If both sides have club candidates, prefer club
    if "club" in left_types and "club" in right_types:
        return "club"

    return None


def abbreviations_to_teams(left_abbr, right_abbr):
    """
    Resolve two abbreviations to canonical team names,
    disambiguating shared codes using match context.
    """
    left_candidates = TEAM_ABBREVIATIONS.get(left_abbr, [])
    right_candidates = TEAM_ABBREVIATIONS.get(right_abbr, [])

    context = _infer_context(left_candidates, right_candidates)

    left_team = _pick_from_candidates(left_candidates, preferred_type=context)
    right_team = _pick_from_candidates(right_candidates, preferred_type=context)

    return left_team, right_team


# ============================================================
# OCR helpers
# ============================================================

def ocr_scorebug_strip(frame):
    """
    OCR a wide horizontal strip across the top of the frame where the
    scorebug lives on any broadcaster layout.
    Returns the raw OCR text.
    """
    try:
        import pytesseract
    except ImportError:
        return ""

    height, width = frame.shape[:2]
    strip_height = int(height * 0.12)
    strip = frame[0:strip_height, 0:width]
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(thresh, config="--psm 7")
    except Exception:
        text = ""
    return text


def extract_scoreboard_abbreviations(text):
    """
    Extract two 2-4 letter abbreviations from OCR text (left team, right team).
    """
    normalized = normalize_text(text).upper()
    found = re.findall(r"\b[A-Z]{2,4}\b", normalized)
    # Filter to only those that exist in our abbreviation registry
    valid = [f for f in found if f in TEAM_ABBREVIATIONS]
    left = valid[0] if len(valid) > 0 else ""
    right = valid[1] if len(valid) > 1 else ""
    return left, right


def extract_scoreboard_text(frame):
    """OCR the scorebug and return raw text."""
    return ocr_scorebug_strip(frame)


# ============================================================
# Team name extraction from text
# ============================================================

def extract_team_names_from_text(text):
    """
    Extract up to 2 team names from arbitrary text using abbreviations and aliases.
    Returns list of canonical team names.
    """
    if not text:
        return []

    normalized = normalize_text(text)
    found_names = []

    # Check abbreviations (now list-based, take first candidate as fallback)
    for token in re.findall(r"\b[A-Z]{2,4}\b", normalized.upper()):
        candidates = TEAM_ABBREVIATIONS.get(token, [])
        if candidates:
            team_name = candidates[0]
            if team_name not in found_names:
                found_names.append(team_name)

    # Check aliases (longest match first)
    lowered = normalized.lower()
    alias_matches = []
    for alias, team_name in sorted(
        TEAM_NAME_ALIASES.items(), key=lambda item: len(item[0]), reverse=True,
    ):
        match_index = lowered.find(alias)
        if match_index >= 0:
            alias_matches.append((match_index, team_name))

    for _, team_name in sorted(alias_matches):
        if team_name not in found_names:
            found_names.append(team_name)

    return found_names[:2]


def extract_team_names_from_scoreboard_text(text):
    """
    Extract team names from scoreboard OCR text specifically.
    Uses context-aware disambiguation for shared abbreviations.
    """
    if not text:
        return []

    normalized = normalize_text(text).upper()
    found_abbrs = re.findall(r"\b[A-Z]{2,4}\b", normalized)
    valid_abbrs = [f for f in found_abbrs if f in TEAM_ABBREVIATIONS]

    # If we found two abbreviations, use context-aware resolution
    if len(valid_abbrs) >= 2:
        left_team, right_team = abbreviations_to_teams(valid_abbrs[0], valid_abbrs[1])
        results = []
        if left_team:
            results.append(left_team)
        if right_team and right_team != left_team:
            results.append(right_team)
        if len(results) >= 2:
            return results[:2]

    # Fall back to alias matching
    found_names = []
    lowered = normalize_text(text).lower()
    alias_matches = []
    for alias, team_name in sorted(
        TEAM_NAME_ALIASES.items(), key=lambda item: len(item[0]), reverse=True,
    ):
        match_index = lowered.find(alias)
        if match_index >= 0:
            alias_matches.append((match_index, team_name))

    for _, team_name in sorted(alias_matches):
        if team_name not in found_names:
            found_names.append(team_name)

    return found_names[:2]


# ============================================================
# Video path / frame helpers
# ============================================================

def get_video_path(video_input):
    """Get the file path from Gradio video input (handles str or dict)."""
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        return video_input.get("video", video_input.get("name", ""))
    return str(video_input)


def read_first_frame(video_path):
    """Read and return the first frame of a video file."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


# ============================================================
# Matchup resolution
# ============================================================

def resolve_matchup(video_path):
    """
    Attempt to detect the two teams in the match from scoreboard OCR.
    Falls back to filename parsing.
    Returns (list_of_team_names, source_string).
    """
    team_names = []
    source = "unknown"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _fallback_from_filename(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample a few frames from the beginning
    sample_indices = [int(total_frames * f) for f in [0.0, 0.02, 0.05, 0.1, 0.15]]

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        text = ocr_scorebug_strip(frame)
        if not text.strip():
            continue

        left_abbr, right_abbr = extract_scoreboard_abbreviations(text)
        if not left_abbr or not right_abbr:
            continue

        left_team, right_team = abbreviations_to_teams(left_abbr, right_abbr)
        for team_name in [left_team, right_team]:
            if team_name and team_name not in team_names:
                team_names.append(team_name)
        if len(team_names) >= 2:
            source = "OCR"
            break

    cap.release()

    if len(team_names) < 2:
        fallback_names, fallback_source = _fallback_from_filename(video_path)
        for name in fallback_names:
            if name not in team_names:
                team_names.append(name)
        if len(team_names) >= 2 and source == "unknown":
            source = fallback_source

    if len(team_names) < 2:
        source = "fallback"

    return team_names[:2], source


def _fallback_from_filename(video_path):
    """Try to extract team names from the video filename."""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    names = extract_team_names_from_text(basename)
    if len(names) >= 2:
        return names[:2], "filename"
    return names, "filename"


# ============================================================
# User team query resolution
# ============================================================

def resolve_requested_team(team_query, team_names):
    """
    Resolve user's team query against detected team names.
    Returns the canonical name that matches, or the first detected team as fallback.
    """
    if not team_query or not team_names:
        return None

    # Try extracting team names from the query text
    query_matches = extract_team_names_from_text(team_query)
    if query_matches:
        for match in query_matches:
            if match in team_names:
                return match

    # Try abbreviation lookup (list-based)
    abbr_key = team_query.strip().upper()
    candidates = TEAM_ABBREVIATIONS.get(abbr_key, [])
    for candidate in candidates:
        if candidate in team_names:
            return candidate

    # Try alias lookup
    alias_match = TEAM_NAME_ALIASES.get(team_query.strip().lower())
    if alias_match and alias_match in team_names:
        return alias_match

    # Fuzzy: check if query is substring of any detected name
    for name in team_names:
        if team_query.strip().lower() in name.lower():
            return name

    # Last resort: return first detected team
    return team_names[0] if team_names else None