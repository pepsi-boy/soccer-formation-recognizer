import os
import re
import time

import cv2
import numpy as np

from .config import TEAM_ABBREVIATIONS, TEAM_NAME_ALIASES, TEAM_TYPE


# ============================================================
# Ambiguity / context filtering
# ============================================================

AMBIGUOUS_ABBREVIATIONS = {
    "NEW", "FOR", "THE", "CAN", "COM", "WIN", "TOP",
    "USE", "GET", "HAM", "ALL", "MAD", "MAN", "RUN",
    "OUR", "VAN", "NOR", "MON",
}

INTERNATIONAL_CONTEXT_KEYWORDS = [
    "world cup", "fifa", "euro ", "euros", "euro2",
    "copa america", "afcon", "nations league",
    "qualifier", "international",
    "group a", "group b", "group c", "group d",
    "group e", "group f", "group g", "group h",
]

CLUB_CONTEXT_KEYWORDS = [
    "premier league", "la liga", "serie a", "bundesliga",
    "ligue 1", "champions league", "europa league",
    "conference league", "fa cup", "carabao", "efl", "championship",
]


def detect_competition_context(text):
    if not text:
        return None
    lowered = text.lower()
    for keyword in INTERNATIONAL_CONTEXT_KEYWORDS:
        if keyword in lowered:
            return "international"
    for keyword in CLUB_CONTEXT_KEYWORDS:
        if keyword in lowered:
            return "club"
    return None


def is_valid_abbreviation_pair(abbr1, abbr2, competition_context=None):
    candidates1 = TEAM_ABBREVIATIONS.get(abbr1, [])
    candidates2 = TEAM_ABBREVIATIONS.get(abbr2, [])

    if not candidates1 or not candidates2:
        return False

    if isinstance(candidates1, str):
        candidates1 = [candidates1]
    if isinstance(candidates2, str):
        candidates2 = [candidates2]

    types1 = {TEAM_TYPE.get(c) for c in candidates1} - {None}
    types2 = {TEAM_TYPE.get(c) for c in candidates2} - {None}

    if competition_context:
        has_type1 = any(TEAM_TYPE.get(c) == competition_context for c in candidates1)
        has_type2 = any(TEAM_TYPE.get(c) == competition_context for c in candidates2)
        if not has_type1 or not has_type2:
            return False
        return True

    if types1 and types2:
        if types1 == {"international"} and types2 == {"club"}:
            return False
        if types1 == {"club"} and types2 == {"international"}:
            return False

    return True


def abbreviation_fits_context(abbr, competition_context):
    if not competition_context:
        return True
    candidates = TEAM_ABBREVIATIONS.get(abbr, [])
    if isinstance(candidates, str):
        candidates = [candidates]
    return any(TEAM_TYPE.get(c) == competition_context for c in candidates)


# ============================================================
# Color helpers
# ============================================================

def bgr_to_lab_color(bgr):
    pixel = np.uint8([[bgr]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return lab[0, 0].astype(np.float32)


def describe_lab_color(lab):
    """
    Describe a LAB color in human-friendly terms.
    OpenCV LAB: L=0-255, a=0-255 (128=neutral), b=0-255 (128=neutral).
    """
    l, a, b = float(lab[0]), float(lab[1]), float(lab[2])

    a_shift = a - 128  # positive = red, negative = green
    b_shift = b - 128  # positive = yellow, negative = blue

    if l < 45:
        return "Darker"
    if l > 190 and abs(a_shift) < 10 and abs(b_shift) < 10:
        return "Lighter"
    if l > 210:
        return "Lighter"

    if a_shift > 15:
        if b_shift > 20:
            return "Orange"
        return "Red"
    if a_shift < -15:
        return "Green"
    if b_shift > 20:
        return "Yellow"
    if b_shift < -15:
        return "Blue"

    if a_shift > 8:
        return "Red"
    if a_shift < -8:
        return "Green"
    if b_shift > 12:
        return "Yellow"
    if b_shift < -8:
        return "Blue"

    if l > 150:
        return "Lighter"
    if l < 80:
        return "Darker"

    return "Darker" if l < 128 else "Lighter"


def describe_team_color_comparative(team1_lab, team2_lab):
    """
    Compare two team colors and return descriptive labels for each.
    More reliable than labeling each in isolation because it handles
    grass contamination by looking at relative differences.
    
    Returns (team1_label, team2_label).
    """
    if team1_lab is None or team2_lab is None:
        return describe_lab_color(team1_lab) if team1_lab is not None else "Unknown", \
               describe_lab_color(team2_lab) if team2_lab is not None else "Unknown"

    l1, a1, b1 = float(team1_lab[0]), float(team1_lab[1]), float(team1_lab[2])
    l2, a2, b2 = float(team2_lab[0]), float(team2_lab[1]), float(team2_lab[2])

    l_diff = l1 - l2       # positive = team1 lighter
    a_diff = a1 - a2       # positive = team1 more red
    b_diff = b1 - b2       # positive = team1 more yellow

    # Check if one team is clearly more red than the other
    a1_shift = a1 - 128
    a2_shift = a2 - 128

    # If one team has strong red channel AND the other doesn't
    if a1_shift > 8 and a2_shift < 8 and a_diff > 10:
        return "Red", "Lighter" if l2 > 140 else "Darker"
    if a2_shift > 8 and a1_shift < 8 and a_diff < -10:
        return "Lighter" if l1 > 140 else "Darker", "Red"

    # If one team has strong blue channel
    b1_shift = b1 - 128
    b2_shift = b2 - 128

    if b1_shift < -10 and b2_shift > -5 and b_diff < -10:
        return "Blue", "Lighter" if l2 > 140 else "Darker"
    if b2_shift < -10 and b1_shift > -5 and b_diff > 10:
        return "Lighter" if l1 > 140 else "Darker", "Blue"

    # If significant lightness difference, just use Lighter/Darker
    if abs(l_diff) > 25:
        if l_diff > 0:
            return "Lighter", "Darker"
        else:
            return "Darker", "Lighter"

    # Fall back to individual labeling
    return describe_lab_color(team1_lab), describe_lab_color(team2_lab)

def normalize_text(text):
    text = text.strip()
    text = re.sub(r"[^\w\s\-&.']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ============================================================
# Video path / frame helpers
# ============================================================

def get_video_path(video_input):
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input if video_input else None
    if isinstance(video_input, dict):
        path = video_input.get("video", video_input.get("name", ""))
        return path if path else None
    path = str(video_input)
    return path if path and path != "None" else None


def read_first_frame(video_path):
    if not video_path:
        return None
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ============================================================
# Abbreviation resolution helpers
# ============================================================

def pick_from_candidates(candidates, preferred_type=None):
    if not candidates:
        return None
    if isinstance(candidates, str):
        return candidates
    if len(candidates) == 1:
        return candidates[0]
    if preferred_type:
        typed = [c for c in candidates if TEAM_TYPE.get(c) == preferred_type]
        if typed:
            return typed[0]
    return candidates[0]


def infer_context(left_candidates, right_candidates):
    if isinstance(left_candidates, str):
        left_candidates = [left_candidates]
    if isinstance(right_candidates, str):
        right_candidates = [right_candidates]

    left_types = {TEAM_TYPE.get(c) for c in left_candidates} - {None} if left_candidates else set()
    right_types = {TEAM_TYPE.get(c) for c in right_candidates} - {None} if right_candidates else set()

    if left_candidates and len(left_candidates) == 1:
        lt = TEAM_TYPE.get(left_candidates[0])
        if lt:
            return lt
    if right_candidates and len(right_candidates) == 1:
        rt = TEAM_TYPE.get(right_candidates[0])
        if rt:
            return rt

    if "international" in left_types and "international" in right_types:
        return "international"
    if "club" in left_types and "club" in right_types:
        return "club"
    return None


def abbreviations_to_teams(left_abbr, right_abbr):
    left_candidates = TEAM_ABBREVIATIONS.get(left_abbr, [])
    right_candidates = TEAM_ABBREVIATIONS.get(right_abbr, [])

    if isinstance(left_candidates, str):
        left_candidates = [left_candidates]
    if isinstance(right_candidates, str):
        right_candidates = [right_candidates]

    context = infer_context(left_candidates, right_candidates)

    left_team = pick_from_candidates(left_candidates, preferred_type=context)
    right_team = pick_from_candidates(right_candidates, preferred_type=context)

    return left_team, right_team


def _name_to_abbreviation(team_name):
    """Reverse lookup: canonical team name -> its first abbreviation."""
    for abbr, candidates in TEAM_ABBREVIATIONS.items():
        if isinstance(candidates, str):
            if candidates == team_name:
                return abbr
        elif team_name in candidates:
            return abbr
    return ""


# ============================================================
# OCR — Scorebug detection
# ============================================================

def extract_scoreboard_text(frame):
    try:
        import pytesseract
    except ImportError:
        return ""

    height, width = frame.shape[:2]
    y1 = int(height * 0.02)
    y2 = int(height * 0.20)
    strip = frame[y1:y2, 0:int(width * 0.50)]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    try:
        text = pytesseract.image_to_string(gray, config="--psm 6")
    except Exception:
        text = ""

    return text


def _ocr_single_frame(frame, timeout=8):
    """
    Try to find two valid team abbreviations from a single frame.
    Returns (abbr1, abbr2, competition_context) or ("", "", None).
    """
    try:
        import pytesseract
    except ImportError:
        return "", "", None

    start_time = time.time()
    height, width = frame.shape[:2]

    # --- PHASE 1: Context OCR (narrow left side, scoreboard area) ---
    competition_context = None
    ctx_text = ""
    ctx_crop = frame[int(height * 0.02):int(height * 0.19), 0:int(width * 0.50)]

    if ctx_crop.size > 0:
        try:
            ctx_gray = cv2.cvtColor(ctx_crop, cv2.COLOR_BGR2GRAY)
            ctx_gray = cv2.resize(ctx_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, ctx_thresh = cv2.threshold(ctx_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ctx_text = pytesseract.image_to_string(ctx_thresh, config="--psm 6 --oem 3")
            print(f"[OCR context] text: {repr(ctx_text[:120])}")
            competition_context = detect_competition_context(ctx_text)
            if competition_context:
                print(f"[OCR context] Competition: {competition_context}")
        except Exception:
            pass

    # --- PHASE 1b: Extract full team names from context text ---
    if ctx_text:
        full_names = extract_team_names_from_text(ctx_text)
        if len(full_names) >= 2:
            abbr1 = _name_to_abbreviation(full_names[0])
            abbr2 = _name_to_abbreviation(full_names[1])
            if abbr1 and abbr2:
                print(f"[OCR context] ✓ Found full names: {full_names} -> [{abbr1}, {abbr2}]")
                return abbr1, abbr2, competition_context

    # --- PHASE 2: Scan scorebug regions ---
    scorebug_regions = [
        # Very top scorebug (FOX/FIFA style, y=0-5%)
        (0.03, 0.40, 0.00, 0.05),
        (0.03, 0.40, 0.01, 0.05),
        (0.03, 0.40, 0.00, 0.06),
        (0.05, 0.35, 0.00, 0.05),
        (0.05, 0.35, 0.01, 0.06),
        # Wide PL style (y≈2-9%)
        (0.05, 0.65, 0.02, 0.07),
        (0.05, 0.65, 0.03, 0.08),
        (0.05, 0.65, 0.04, 0.09),
        (0.05, 0.60, 0.03, 0.07),
        (0.05, 0.60, 0.04, 0.08),
        (0.05, 0.60, 0.05, 0.09),
        # FIFA top-left style (y≈12-19%)
        (0.02, 0.45, 0.12, 0.17),
        (0.02, 0.45, 0.13, 0.18),
        (0.02, 0.45, 0.14, 0.19),
        (0.02, 0.40, 0.12, 0.16),
        (0.02, 0.40, 0.13, 0.17),
        (0.02, 0.40, 0.14, 0.18),
        # ESPN / general (y≈5-13%)
        (0.00, 0.45, 0.05, 0.10),
        (0.00, 0.45, 0.06, 0.11),
        (0.00, 0.45, 0.07, 0.12),
        (0.00, 0.45, 0.08, 0.13),
        # Upper-left (y≈8-16%)
        (0.00, 0.45, 0.08, 0.13),
        (0.00, 0.45, 0.10, 0.15),
        (0.00, 0.45, 0.11, 0.16),
        (0.00, 0.45, 0.12, 0.17),
        # Centered scorebugs
        (0.20, 0.80, 0.02, 0.07),
        (0.20, 0.80, 0.03, 0.08),
        (0.25, 0.75, 0.03, 0.08),
        # Narrower fallbacks
        (0.00, 0.30, 0.03, 0.08),
        (0.00, 0.30, 0.05, 0.10),
        (0.00, 0.35, 0.10, 0.15),
        (0.00, 0.35, 0.12, 0.17),
    ]

    best_single = ""
    fallback_pair = None

    for x_start_pct, x_end_pct, y_start_pct, y_end_pct in scorebug_regions:
        if time.time() - start_time > timeout:
            break

        x1 = int(width * x_start_pct)
        x2 = int(width * x_end_pct)
        y1 = int(height * y_start_pct)
        y2 = int(height * y_end_pct)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop_height = y2 - y1
        scale = max(2, int(100 / max(crop_height, 1)))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        images_to_try = []

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        images_to_try.append(otsu)
        _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        images_to_try.append(otsu_inv)

        for t in [200, 210, 220, 230]:
            _, white_text = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            images_to_try.append(white_text)

        for t in [120, 140, 160, 180]:
            _, fixed = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            images_to_try.append(fixed)
            _, fixed_inv = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
            images_to_try.append(fixed_inv)

        _, white_base = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(white_base, cv2.MORPH_OPEN, kernel)
        images_to_try.append(cleaned)

        for img in images_to_try:
            if time.time() - start_time > timeout:
                break

            for psm in ["--psm 7 --oem 3", "--psm 6 --oem 3"]:
                try:
                    text = pytesseract.image_to_string(img, config=psm)
                except Exception:
                    continue

                text = text.strip()
                if not text or len(text) > 100:
                    continue

                tokens = re.findall(r"\b([A-Z]{3})\b", text.upper())
                found = []
                for token in tokens:
                    if token in TEAM_ABBREVIATIONS and token not in found:
                        found.append(token)
                    if len(found) >= 2:
                        break

                if len(found) >= 2:
                    if is_valid_abbreviation_pair(found[0], found[1], competition_context):
                        has_digit = bool(re.search(r"\d", text))
                        if has_digit:
                            print(f"[OCR scorebug] ✓ Valid pair (with score): {found}")
                            return found[0], found[1], competition_context
                        elif fallback_pair is None:
                            fallback_pair = (found[0], found[1])
                            print(f"[OCR scorebug] ~ Valid pair (no score): {found}")
                    else:
                        print(f"[OCR scorebug] ✗ Rejected: {found} (context={competition_context})")
                        for f in found:
                            if not best_single and abbreviation_fits_context(f, competition_context):
                                best_single = f
                elif len(found) == 1:
                    if not best_single and abbreviation_fits_context(found[0], competition_context):
                        best_single = found[0]

    if fallback_pair:
        print(f"[OCR scorebug] Using fallback pair: {list(fallback_pair)}")
        return fallback_pair[0], fallback_pair[1], competition_context

    return best_single, "", competition_context


def extract_scorebug_abbreviations(frame, timeout=25):
    """Public wrapper. Returns (abbr1, abbr2)."""
    a1, a2, _ = _ocr_single_frame(frame, timeout=timeout)
    if a1 and not a2:
        print(f"[OCR scorebug] Partial: only found {a1}")
    elif not a1:
        print("[OCR scorebug] No valid abbreviations found")
    return a1, a2


# ============================================================
# Kit color extraction from scorebug (no OCR — fast pixel sampling)
# ============================================================

def extract_scoreboard_kit_colors(frame):
    """
    Extract kit colors from the scoreboard by sampling the left and right
    halves of the scorebug region.
    Returns list of 2 LAB colors [left_color, right_color] or empty list.
    """
    height, width = frame.shape[:2]

    candidate_strips = [
        (0.05, 0.60, 0.02, 0.07),
        (0.05, 0.60, 0.03, 0.08),
        (0.05, 0.60, 0.04, 0.09),
        (0.02, 0.40, 0.12, 0.17),
        (0.02, 0.40, 0.13, 0.18),
        (0.02, 0.40, 0.14, 0.19),
        (0.00, 0.45, 0.05, 0.10),
        (0.00, 0.45, 0.06, 0.11),
        (0.00, 0.45, 0.08, 0.13),
        (0.00, 0.45, 0.10, 0.15),
    ]

    for x_start_pct, x_end_pct, y_start_pct, y_end_pct in candidate_strips:
        x1 = int(width * x_start_pct)
        x2 = int(width * x_end_pct)
        y1 = int(height * y_start_pct)
        y2 = int(height * y_end_pct)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        if crop_w < 60 or crop_h < 10:
            continue

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        std_dev = np.std(gray_crop)
        if std_dev < 20:
            continue

        left_end = int(crop_w * 0.30)
        right_start = int(crop_w * 0.70)

        left_section = crop[:, :left_end]
        right_section = crop[:, right_start:]

        colors = []
        for section in [left_section, right_section]:
            if section.size == 0:
                break

            lab_section = cv2.cvtColor(section, cv2.COLOR_BGR2LAB)
            pixels = lab_section.reshape(-1, 3).astype(np.float32)

            if len(pixels) < 10:
                break

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            try:
                _, labels, centers = cv2.kmeans(
                    pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
                )
            except cv2.error:
                break

            counts = [np.sum(labels == i) for i in range(2)]
            bg_idx = np.argmax(counts)
            dominant = centers[bg_idx].astype(np.float32)
            colors.append(dominant)

        if len(colors) == 2:
            color_diff = np.linalg.norm(colors[0] - colors[1])
            if color_diff > 10:
                print(f"[Scoreboard colors] Left: LAB={colors[0].astype(int)} ({describe_lab_color(colors[0])})")
                print(f"[Scoreboard colors] Right: LAB={colors[1].astype(int)} ({describe_lab_color(colors[1])})")
                return colors

    print("[Scoreboard colors] Could not extract kit colors")
    return []


# ============================================================
# Team name extraction from text
# ============================================================

def extract_team_names_from_text(text):
    if not text:
        return []

    normalized = normalize_text(text)
    found_names = []
    matched_spans = []

    lowered = normalized.lower()

    for alias, team_name in sorted(
        TEAM_NAME_ALIASES.items(), key=lambda item: len(item[0]), reverse=True,
    ):
        if len(alias) < 3:
            continue
        match_index = lowered.find(alias)
        if match_index >= 0:
            span_end = match_index + len(alias)
            overlaps = False
            for s, e in matched_spans:
                if match_index < e and span_end > s:
                    overlaps = True
                    break
            if not overlaps and team_name not in found_names:
                found_names.append(team_name)
                matched_spans.append((match_index, span_end))
        if len(found_names) >= 2:
            return found_names[:2]

    tokens = re.finditer(r"\b([A-Z]{3})\b", normalized.upper())
    for match in tokens:
        token = match.group(1)
        start = match.start()
        end = match.end()

        inside = False
        for s, e in matched_spans:
            if start >= s and end <= e:
                inside = True
                break
        if inside:
            continue

        if token in TEAM_ABBREVIATIONS:
            candidates = TEAM_ABBREVIATIONS[token]
            if isinstance(candidates, str):
                candidates = [candidates]
            for candidate in candidates:
                if candidate not in found_names:
                    found_names.append(candidate)
                    matched_spans.append((start, end))
                    break
        if len(found_names) >= 2:
            break

    return found_names[:2]


def extract_team_names_from_scoreboard_text(text):
    if not text:
        return []

    normalized = normalize_text(text).upper()
    found_abbrs = re.findall(r"\b[A-Z]{2,4}\b", normalized)
    valid_abbrs = [f for f in found_abbrs if f in TEAM_ABBREVIATIONS]

    if len(valid_abbrs) >= 2:
        competition_context = detect_competition_context(text)
        if is_valid_abbreviation_pair(valid_abbrs[0], valid_abbrs[1], competition_context):
            left_team, right_team = abbreviations_to_teams(valid_abbrs[0], valid_abbrs[1])
            results = []
            if left_team:
                results.append(left_team)
            if right_team and right_team != left_team:
                results.append(right_team)
            if len(results) >= 2:
                return results[:2]

    return extract_team_names_from_text(text)


# ============================================================
# Matchup resolution (scoreboard — tries multiple frames)
# ============================================================

def resolve_matchup(video_path):
    """
    Detect the two teams from the scoreboard.
    Tries multiple frames. Total time capped at ~30 seconds.
    """
    if not video_path:
        return [], "no video"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], "no video"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    overall_start = time.time()

    sample_positions = [0.25, 0.50, 0.10, 0.75]
    best_single = ""

    for pos in sample_positions:
        if time.time() - overall_start > 28:
            print("[resolve_matchup] Overall timeout")
            break

        target_frame = int(total_frames * pos)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        a1, a2, ctx = _ocr_single_frame(frame, timeout=7)

        if a1 and a2:
            left_team, right_team = abbreviations_to_teams(a1, a2)
            team_names = [t for t in [left_team, right_team] if t]
            if len(team_names) >= 2:
                cap.release()
                return team_names[:2], "OCR (abbreviation)"

        if a1 and not best_single:
            best_single = a1

    cap.release()

    if best_single:
        candidates = TEAM_ABBREVIATIONS.get(best_single, [])
        if isinstance(candidates, str):
            candidates = [candidates]
        if candidates:
            return [candidates[0]], "OCR (partial)"

    # Last resort: full-text OCR
    frame = read_first_frame(video_path)
    if frame is not None:
        ocr_text = extract_scoreboard_text(frame)
        if ocr_text:
            ocr_names = extract_team_names_from_text(ocr_text)
            if len(ocr_names) >= 2:
                return ocr_names[:2], "OCR (text)"

    return [], "scoreboard not readable"


# ============================================================
# Filename fallback (kept for compatibility)
# ============================================================

def fallback_from_filename(video_path):
    if not video_path:
        return [], "no video"
    basename = os.path.splitext(os.path.basename(video_path))[0]
    names = extract_team_names_from_text(basename)
    if len(names) >= 2:
        return names[:2], "filename"
    return names, "filename"


# ============================================================
# User team query resolution
# ============================================================

def resolve_requested_team(team_query, team_names):
    if not team_query or not team_names:
        return team_names[0] if team_names else None

    query_lower = team_query.strip().lower()
    for name in team_names:
        if query_lower == name.lower():
            return name

    abbr_key = team_query.strip().upper()
    candidates = TEAM_ABBREVIATIONS.get(abbr_key, [])
    if isinstance(candidates, str):
        candidates = [candidates]
    for candidate in candidates:
        if candidate in team_names:
            return candidate

    alias_match = TEAM_NAME_ALIASES.get(query_lower)
    if alias_match and alias_match in team_names:
        return alias_match

    for name in team_names:
        if query_lower in name.lower():
            return name

    return team_names[0] if team_names else None