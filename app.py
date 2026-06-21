import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.cluster import KMeans
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

FORMATIONS = {
    "4-3-3": [4, 3, 3],
    "4-4-2": [4, 4, 2],
    "4-2-3-1": [4, 2, 3, 1],
    "3-5-2": [3, 5, 2],
    "5-3-2": [5, 3, 2],
}

SAMPLE_FRAME_COUNT = 10
EXPECTED_OUTFIELD_PLAYERS = 10
MIN_TRACK_OBSERVATIONS = 2
DEFAULT_TRACK_FRAME_STRIDE = 5

TEAM_ABBREVIATIONS = {
    "ARS": "Arsenal",
    "AVL": "Aston Villa",
    "BOU": "Bournemouth",
    "BRE": "Brentford",
    "BHA": "Brighton & Hove Albion",
    "BRI": "Brighton & Hove Albion",
    "BUR": "Burnley",
    "CHE": "Chelsea",
    "CRY": "Crystal Palace",
    "EVE": "Everton",
    "FUL": "Fulham",
    "IPS": "Ipswich Town",
    "LEE": "Leeds United",
    "LEI": "Leicester City",
    "LIV": "Liverpool",
    "LUT": "Luton Town",
    "MCI": "Manchester City",
    "MUN": "Manchester United",
    "NEW": "Newcastle United",
    "NFO": "Nottingham Forest",
    "NOT": "Nottingham Forest",
    "SHU": "Sheffield United",
    "SOU": "Southampton",
    "SUN": "Sunderland",
    "TOT": "Tottenham Hotspur",
    "WBA": "West Bromwich Albion",
    "WHU": "West Ham United",
    "WOL": "Wolverhampton Wanderers",
    "ALA": "Alaves",
    "ATH": "Athletic Club",
    "ATM": "Atletico Madrid",
    "ATL": "Atletico Madrid",
    "BAR": "Barcelona",
    "BET": "Real Betis",
    "CAD": "Cadiz",
    "CEL": "Celta Vigo",
    "ELC": "Elche",
    "ESP": "Espanyol",
    "FCB": "Barcelona",
    "GET": "Getafe",
    "GIR": "Girona",
    "GRA": "Granada",
    "LPA": "Las Palmas",
    "LAS": "Las Palmas",
    "LEG": "Leganes",
    "MLL": "Mallorca",
    "OSA": "Osasuna",
    "RAY": "Rayo Vallecano",
    "RMA": "Real Madrid",
    "RMD": "Real Madrid",
    "RSO": "Real Sociedad",
    "SEV": "Sevilla",
    "VAL": "Valencia",
    "VIL": "Villarreal",
    "VLL": "Real Valladolid",
}

TEAM_NAME_ALIASES = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton & Hove Albion",
    "burnley": "Burnley",
    "chelsea": "Chelsea",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "ipswich": "Ipswich Town",
    "leeds": "Leeds United",
    "leeds united": "Leeds United",
    "leicester": "Leicester City",
    "liverpool": "Liverpool",
    "manchester city": "Manchester City",
    "man city": "Manchester City",
    "manchester united": "Manchester United",
    "man united": "Manchester United",
    "newcastle": "Newcastle United",
    "nottingham forest": "Nottingham Forest",
    "southampton": "Southampton",
    "sunderland": "Sunderland",
    "tottenham": "Tottenham Hotspur",
    "tottenham hotspur": "Tottenham Hotspur",
    "west ham": "West Ham United",
    "wolves": "Wolverhampton Wanderers",
    "wolverhampton": "Wolverhampton Wanderers",
    "athletic club": "Athletic Club",
    "atletico madrid": "Atletico Madrid",
    "barcelona": "Barcelona",
    "real madrid": "Real Madrid",
    "real sociedad": "Real Sociedad",
    "sevilla": "Sevilla",
    "valencia": "Valencia",
    "villarreal": "Villarreal",
}


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


def dominant_lab_color(crop):
    if crop.size == 0:
        return None

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = crop.reshape(-1, 3)
    hsv_pixels = hsv_crop.reshape(-1, 3)

    hue = hsv_pixels[:, 0]
    saturation = hsv_pixels[:, 1]
    value = hsv_pixels[:, 2]
    grass_mask = (hue >= 35) & (hue <= 90) & (saturation > 45) & (value > 45)
    usable_pixels = pixels[~grass_mask]

    if len(usable_pixels) < 20:
        usable_pixels = pixels

    cluster_count = min(4, len(usable_pixels))
    if cluster_count < 1:
        return None

    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(usable_pixels)
    cluster_sizes = np.bincount(labels)
    dominant_cluster = int(np.argmax(cluster_sizes))
    dominant_bgr = kmeans.cluster_centers_[dominant_cluster]

    return bgr_to_lab_color(dominant_bgr)


def extract_scoreboard_kit_colors(frame):
    height, width = frame.shape[:2]

    # These zones target the left and right team-color panels in common TV scorebugs.
    zones = [
        (0.03, 0.13, 0.07, 0.16),
        (0.19, 0.32, 0.07, 0.16),
    ]

    kit_colors = []
    for x_start, x_end, y_start, y_end in zones:
        x1 = int(width * x_start)
        x2 = int(width * x_end)
        y1 = int(height * y_start)
        y2 = int(height * y_end)
        crop = frame[y1:y2, x1:x2]
        kit_colors.append(dominant_lab_color(crop))

    if any(color is None for color in kit_colors):
        return []

    return kit_colors


def get_team_color_data(players):
    color_data = {}

    for team in ["Team 1", "Team 2"]:
        team_players = [player for player in players if player["team"] == team]
        if not team_players:
            color_data[team] = {
                "lab": None,
                "label": "unknown kit",
            }
            continue

        average_lab = np.mean(
            [player["jersey_color"] for player in team_players],
            axis=0,
        )
        color_data[team] = {
            "lab": average_lab,
            "label": describe_lab_color(average_lab),
        }

    return color_data


def resolve_team_cluster_map(team_names, scoreboard_kit_colors, team_color_data):
    if len(team_names) < 2:
        return {}

    if len(scoreboard_kit_colors) >= 2:
        team_1_lab = team_color_data.get("Team 1", {}).get("lab")
        team_2_lab = team_color_data.get("Team 2", {}).get("lab")

        if team_1_lab is not None and team_2_lab is not None:
            left_to_team_1 = np.linalg.norm(scoreboard_kit_colors[0] - team_1_lab)
            right_to_team_2 = np.linalg.norm(scoreboard_kit_colors[1] - team_2_lab)
            left_to_team_2 = np.linalg.norm(scoreboard_kit_colors[0] - team_2_lab)
            right_to_team_1 = np.linalg.norm(scoreboard_kit_colors[1] - team_1_lab)

            normal_score = left_to_team_1 + right_to_team_2
            swapped_score = left_to_team_2 + right_to_team_1

            if swapped_score < normal_score:
                return {
                    team_names[0]: "Team 2",
                    team_names[1]: "Team 1",
                }

    return {
        team_names[0]: "Team 1",
        team_names[1]: "Team 2",
    }


def build_team_display_map(team_names, color_data, team_cluster_map=None, scoreboard_kit_colors=None):
    team_cluster_map = team_cluster_map or {}
    scoreboard_kit_colors = scoreboard_kit_colors or []
    display_map = {}

    team_name_by_cluster = {
        cluster: team_name
        for team_name, cluster in team_cluster_map.items()
    }
    scoreboard_label_by_team = {
        team_name: describe_lab_color(scoreboard_kit_colors[index])
        for index, team_name in enumerate(team_names[:len(scoreboard_kit_colors)])
    }

    for index, internal_team in enumerate(["Team 1", "Team 2"]):
        team_name = team_name_by_cluster.get(internal_team)
        color_label = color_data.get(internal_team, {}).get("label", "unknown kit")
        if team_name in scoreboard_label_by_team:
            color_label = scoreboard_label_by_team[team_name]

        if internal_team in team_name_by_cluster:
            display_map[internal_team] = f"{team_name_by_cluster[internal_team]} ({color_label})"
        elif index < len(team_names):
            display_map[internal_team] = f"{team_names[index]}? ({internal_team}, {color_label})"
        else:
            display_map[internal_team] = f"{internal_team} ({color_label})"

    display_map["Official/Other"] = "Official/Other"

    return display_map


def selected_display_to_internal(selected_team, team_names, team_cluster_map=None):
    if selected_team in ["Team 1", "Team 2"]:
        return selected_team

    team_cluster_map = team_cluster_map or {}
    if selected_team in team_cluster_map:
        return team_cluster_map[selected_team]

    return "Team 1"


def update_team_status(video_input):
    video_path = get_video_path(video_input)
    team_names, source = resolve_matchup(video_path)

    if len(team_names) >= 2:
        status = f"Detected matchup: {team_names[0]} vs {team_names[1]} ({source})"
        return status

    return "Could not detect both team names from the scoreboard yet."


def get_jersey_color(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)

    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)

    torso_x1 = x1 + int(box_width * 0.2)
    torso_x2 = x1 + int(box_width * 0.8)
    torso_y1 = y1 + int(box_height * 0.15)
    torso_y2 = y1 + int(box_height * 0.6)

    torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

    if torso_crop.size == 0:
        return [0, 0, 0]

    hsv_crop = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    flattened_hsv = hsv_crop.reshape(-1, 3)
    hue = flattened_hsv[:, 0]
    saturation = flattened_hsv[:, 1]
    value = flattened_hsv[:, 2]

    grass_mask = (hue >= 35) & (hue <= 90) & (saturation > 45) & (value > 45)
    usable_pixels = torso_crop.reshape(-1, 3)[~grass_mask]

    if len(usable_pixels) < 20:
        usable_pixels = torso_crop.reshape(-1, 3)

    lab_pixels = cv2.cvtColor(usable_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    average_color = np.mean(lab_pixels.reshape(-1, 3), axis=0)

    return average_color


def extract_players_from_frame(frame):
    height, width = frame.shape[:2]

    results = model(frame, classes=[0], conf=0.35, verbose=False)

    players = []

    for i, box in enumerate(results[0].boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])

        center_x = (x1 + x2) / 2
        foot_y = y2

        normalized_x = center_x / width
        normalized_y = foot_y / height

        bbox = [float(x1), float(y1), float(x2), float(y2)]

        players.append(
            {
                "id": i,
                "x": float(center_x),
                "y": float(foot_y),
                "normalized_x": float(normalized_x),
                "normalized_y": float(normalized_y),
                "confidence": confidence,
                "bbox": bbox,
                "jersey_color": get_jersey_color(frame, bbox),
                "team": "Unknown",
            }
        )

    return players


def extract_players_from_tracking_result(result, frame_index):
    frame = result.orig_img
    height, width = frame.shape[:2]

    if result.boxes is None:
        return []

    players = []
    track_ids = result.boxes.id

    for i, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])

        if track_ids is not None:
            track_id = int(track_ids[i - 1].cpu().item())
        else:
            track_id = int((frame_index * 1000) + i)

        center_x = (x1 + x2) / 2
        foot_y = y2

        normalized_x = center_x / width
        normalized_y = foot_y / height

        bbox = [float(x1), float(y1), float(x2), float(y2)]

        players.append(
            {
                "id": track_id,
                "track_id": track_id,
                "frame_index": frame_index,
                "x": float(center_x),
                "y": float(foot_y),
                "normalized_x": float(normalized_x),
                "normalized_y": float(normalized_y),
                "confidence": confidence,
                "bbox": bbox,
                "jersey_color": get_jersey_color(frame, bbox),
                "team": "Unknown",
            }
        )

    return players


def assign_teams_by_color(players):
    if len(players) < 2:
        return players

    jersey_colors = np.array([player["jersey_color"] for player in players])

    cluster_count = 3 if len(players) >= 6 else 2
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    team_labels = kmeans.fit_predict(jersey_colors)
    cluster_sizes = {
        cluster_id: int(np.sum(team_labels == cluster_id))
        for cluster_id in range(cluster_count)
    }
    clusters_by_size = sorted(
        cluster_sizes,
        key=lambda cluster_id: cluster_sizes[cluster_id],
        reverse=True,
    )

    cluster_names = {}
    for rank, cluster_id in enumerate(clusters_by_size):
        if rank == 0:
            cluster_names[cluster_id] = "Team 1"
        elif rank == 1:
            cluster_names[cluster_id] = "Team 2"
        else:
            cluster_names[cluster_id] = "Official/Other"

    for player, label in zip(players, team_labels):
        player["team"] = cluster_names[label]

    return players


def build_unique_tracked_players(tracked_detections):
    tracks = {}

    for detection in tracked_detections:
        track_id = detection["track_id"]
        tracks.setdefault(track_id, []).append(detection)

    unique_players = []

    for track_id, detections in tracks.items():
        best_detection = max(detections, key=lambda detection: detection["confidence"])

        unique_players.append(
            {
                "id": track_id,
                "track_id": track_id,
                "team": "Unknown",
                "x": float(np.mean([detection["x"] for detection in detections])),
                "y": float(np.mean([detection["y"] for detection in detections])),
                "normalized_x": float(
                    np.mean([detection["normalized_x"] for detection in detections])
                ),
                "normalized_y": float(
                    np.mean([detection["normalized_y"] for detection in detections])
                ),
                "confidence": float(
                    np.mean([detection["confidence"] for detection in detections])
                ),
                "bbox": best_detection["bbox"],
                "jersey_color": np.mean(
                    [detection["jersey_color"] for detection in detections], axis=0
                ),
                "observations": len(detections),
            }
        )

    return unique_players


def propagate_track_teams(detections, unique_players):
    team_by_track_id = {
        player["track_id"]: player["team"]
        for player in unique_players
    }

    for detection in detections:
        detection["team"] = team_by_track_id.get(detection["track_id"], "Unknown")

    return detections


def choose_formation_players(unique_players, selected_team):
    selected_players = [
        player
        for player in unique_players
        if player["team"] == selected_team
        and player.get("observations", 0) >= MIN_TRACK_OBSERVATIONS
    ]

    if len(selected_players) < 6:
        selected_players = [
            player for player in unique_players if player["team"] == selected_team
        ]

    selected_players = sorted(
        selected_players,
        key=lambda player: (player.get("observations", 0), player["confidence"]),
        reverse=True,
    )

    return selected_players[:EXPECTED_OUTFIELD_PLAYERS]


def scale_line_counts(raw_counts, expected_total=EXPECTED_OUTFIELD_PLAYERS):
    if not raw_counts:
        return []

    total = sum(raw_counts)
    if total == 0:
        return [0 for _ in raw_counts]

    scaled_values = [(count / total) * expected_total for count in raw_counts]
    scaled_counts = [int(value) for value in scaled_values]
    remaining = expected_total - sum(scaled_counts)
    remainders = [value - int(value) for value in scaled_values]

    for index in np.argsort(remainders)[::-1][:remaining]:
        scaled_counts[index] += 1

    return scaled_counts


def estimate_formation(selected_players):
    if len(selected_players) < 6:
        return {
            "formation": "Unknown",
            "confidence": 0.2,
            "line_counts": [],
            "raw_line_counts": [],
            "explanation": "Not enough selected-team players are visible to make a useful formation estimate.",
        }

    y_positions = np.array([[player["normalized_y"]] for player in selected_players])

    best_formation = "Unknown"
    best_score = float("inf")
    best_line_counts = []
    best_raw_line_counts = []

    for formation_name, template_counts in FORMATIONS.items():
        line_count = len(template_counts)

        if len(selected_players) < line_count:
            continue

        kmeans = KMeans(n_clusters=line_count, random_state=42, n_init=10)
        line_labels = kmeans.fit_predict(y_positions)

        detected_counts = []
        for line_id in range(line_count):
            detected_counts.append(int(np.sum(line_labels == line_id)))

        detected_counts = sorted(detected_counts, reverse=True)
        expected_counts = sorted(template_counts, reverse=True)

        detection_total = sum(detected_counts)
        expected_total = sum(expected_counts)
        score = sum(
            abs((detected_count / detection_total) - (expected_count / expected_total))
            for detected_count, expected_count in zip(detected_counts, expected_counts)
        )

        if score < best_score:
            best_score = score
            best_formation = formation_name
            best_raw_line_counts = detected_counts
            best_line_counts = scale_line_counts(detected_counts)

    confidence = max(0.25, min(0.9, 1 - best_score))

    explanation = (
        f"The selected team's tracked players were grouped into depth lines across the clip. "
        f"The projected line counts {best_line_counts} were closest to the {best_formation} template. "
        "This is an estimate based on visible player positions, not a guaranteed full-match tactical formation."
    )

    return {
        "formation": best_formation,
        "confidence": round(confidence, 2),
        "line_counts": best_line_counts,
        "raw_line_counts": best_raw_line_counts,
        "explanation": explanation,
    }


def team_box_color(team):
    if team == "Team 1":
        return (255, 0, 255)
    if team == "Team 2":
        return (255, 255, 0)
    return (160, 160, 160)


def draw_boxes(frame, players, selected_team, display_map=None):
    annotated = frame.copy()
    display_map = display_map or {}

    for player in players:
        x1, y1, x2, y2 = player["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        center_x = int(player["x"])
        foot_y = int(player["y"])

        color = team_box_color(player["team"])
        thickness = 3 if player["team"] == selected_team else 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(annotated, (center_x, foot_y), 5, (0, 0, 255), -1)

        team_label = display_map.get(player["team"], player["team"])
        cv2.putText(
            annotated,
            f"T{player.get('track_id', player['id'])} {team_label}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def create_pitch_map(players, selected_team, display_map=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    display_map = display_map or {}

    ax.set_facecolor("#2f8f46")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    line_color = "white"
    line_width = 2

    # Real pitch orientation: goals left/right, halfway line top-to-bottom.
    pitch_x = 0.05
    pitch_y = 0.08
    pitch_width = 0.9
    pitch_height = 0.84
    center_x = pitch_x + pitch_width / 2
    center_y = pitch_y + pitch_height / 2

    penalty_depth = pitch_width * 0.16
    penalty_height = pitch_height * 0.58
    penalty_y = center_y - penalty_height / 2
    six_yard_depth = pitch_width * 0.055
    six_yard_height = pitch_height * 0.28
    six_yard_y = center_y - six_yard_height / 2
    goal_height = pitch_height * 0.14
    goal_y = center_y - goal_height / 2

    ax.add_patch(
        plt.Rectangle(
            (pitch_x, pitch_y),
            pitch_width,
            pitch_height,
            fill=False,
            color=line_color,
            linewidth=line_width,
        )
    )
    ax.plot([center_x, center_x], [pitch_y, pitch_y + pitch_height], color=line_color, linewidth=1.5)
    ax.add_patch(plt.Circle((center_x, center_y), 0.085, fill=False, color=line_color, linewidth=1.5))
    ax.scatter(center_x, center_y, s=12, color=line_color)

    # Left penalty area, six-yard box, goal, and penalty spot.
    ax.add_patch(
        plt.Rectangle(
            (pitch_x, penalty_y),
            penalty_depth,
            penalty_height,
            fill=False,
            color=line_color,
            linewidth=1.5,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (pitch_x, six_yard_y),
            six_yard_depth,
            six_yard_height,
            fill=False,
            color=line_color,
            linewidth=1.5,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (pitch_x - 0.018, goal_y),
            0.018,
            goal_height,
            fill=False,
            color=line_color,
            linewidth=1.5,
        )
    )
    ax.scatter(pitch_x + penalty_depth * 0.68, center_y, s=10, color=line_color)

    # Right penalty area, six-yard box, goal, and penalty spot.
    ax.add_patch(
        plt.Rectangle(
            (pitch_x + pitch_width - penalty_depth, penalty_y),
            penalty_depth,
            penalty_height,
            fill=False,
            color=line_color,
            linewidth=1.5,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (pitch_x + pitch_width - six_yard_depth, six_yard_y),
            six_yard_depth,
            six_yard_height,
            fill=False,
            color=line_color,
            linewidth=1.5,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (pitch_x + pitch_width, goal_y),
            0.018,
            goal_height,
            fill=False,
            color=line_color,
            linewidth=1.5,
        )
    )
    ax.scatter(pitch_x + pitch_width - penalty_depth * 0.68, center_y, s=10, color=line_color)

    for player in players:
        x = player["normalized_x"]
        y = player["normalized_y"]

        if player["team"] == selected_team:
            dot_color = "yellow"
            dot_size = 130
            alpha = 1.0
        else:
            dot_color = "lightgray"
            dot_size = 70
            alpha = 0.35

        ax.scatter(x, y, s=dot_size, color=dot_color, edgecolor="black", linewidth=1.5, alpha=alpha)
        ax.text(x, y - 0.03, f"T{player.get('track_id', player['id'])}", ha="center", color="black", fontsize=8)

    ax.set_title(f"Tracked Shape Used For Formation - {display_map.get(selected_team, selected_team)}")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.canvas.draw()

    pitch_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)

    return pitch_image


def make_coordinate_table(players, display_map=None):
    table = []
    display_map = display_map or {}

    for player in players:
        table.append(
            [
                player.get("track_id", player["id"]),
                display_map.get(player["team"], player["team"]),
                round(player["x"], 1),
                round(player["y"], 1),
                round(player["normalized_x"], 3),
                round(player["normalized_y"], 3),
                round(player["confidence"], 2),
                player.get("observations", 1),
            ]
        )

    return table


def get_sample_frame_indices(frame_count, sample_count=SAMPLE_FRAME_COUNT):
    if frame_count <= 0:
        return []

    if frame_count <= sample_count:
        return list(range(frame_count))

    start_frame = int(frame_count * 0.1)
    end_frame = max(start_frame + 1, int(frame_count * 0.9))
    frame_indices = np.linspace(start_frame, end_frame, sample_count, dtype=int)

    return sorted(set(int(frame_index) for frame_index in frame_indices))


def score_frame(players, selected_team):
    selected_players = [player for player in players if player["team"] == selected_team]
    average_confidence = 0

    if players:
        average_confidence = float(np.mean([player["confidence"] for player in players]))

    return (len(selected_players), len(players), average_confidence)


def assign_teams_across_candidates(candidates):
    all_players = []

    for candidate in candidates:
        all_players.extend(candidate["players"])

    if len(all_players) < 2:
        return candidates

    jersey_colors = np.array([player["jersey_color"] for player in all_players])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    team_labels = kmeans.fit_predict(jersey_colors)

    for player, label in zip(all_players, team_labels):
        player["team"] = f"Team {label + 1}"

    return candidates


def collect_sampled_frame_candidates(cap, frame_count):
    sample_indices = get_sample_frame_indices(frame_count)
    candidates = []

    for frame_index in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            continue

        players = extract_players_from_frame(frame)
        candidates.append(
            {
                "frame": frame,
                "frame_index": frame_index,
                "players": players,
            }
        )

    candidates = assign_teams_across_candidates(candidates)

    return candidates, len(sample_indices)


def find_best_visual_candidate(candidates, selected_team):
    best_candidate = None

    for candidate in candidates:
        players = candidate["players"]
        candidate["selected_players"] = [
            player for player in players if player["team"] == selected_team
        ]
        candidate["score"] = score_frame(players, selected_team)

        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    return best_candidate


def get_aggregate_selected_players(candidates, selected_team):
    selected_players = []

    for candidate in candidates:
        for player in candidate["players"]:
            if player["team"] == selected_team:
                selected_players.append(player)

    return selected_players


def analyze_video_with_tracking(video_path, frame_count, frame_stride):
    frame_stride = max(1, int(frame_stride))
    processed_frame_count = max(1, int(np.ceil(frame_count / frame_stride)))
    sample_indices = set(get_sample_frame_indices(processed_frame_count))
    detections = []
    candidates = []

    results = model.track(
        video_path,
        stream=True,
        persist=True,
        classes=[0],
        conf=0.35,
        vid_stride=frame_stride,
        verbose=False,
    )

    for processed_index, result in enumerate(results):
        frame_index = processed_index * frame_stride
        players = extract_players_from_tracking_result(result, frame_index)
        detections.extend(players)

        if processed_index in sample_indices or (not sample_indices and processed_index == 0):
            candidates.append(
                {
                    "frame": result.orig_img.copy(),
                    "frame_index": frame_index,
                    "players": players,
                }
            )

    unique_players = build_unique_tracked_players(detections)
    unique_players = assign_teams_by_color(unique_players)
    propagate_track_teams(detections, unique_players)

    return candidates, unique_players, len(sample_indices)


def analyze_video(
    video_input,
    team_query,
    frame_stride,
):
    video_path = get_video_path(video_input)

    if video_path is None:
        return None, None, [], "Please upload a video first."

    team_names, matchup_source = resolve_matchup(video_path)
    requested_team_name = resolve_requested_team(team_query, team_names)
    first_frame = read_first_frame(video_path)
    scoreboard_kit_colors = []
    if first_frame is not None:
        scoreboard_kit_colors = extract_scoreboard_kit_colors(first_frame)

    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()

    candidates, unique_players, sampled_frame_count = analyze_video_with_tracking(
        video_path,
        frame_count,
        frame_stride,
    )

    for candidate in candidates:
        propagate_track_teams(candidate["players"], unique_players)

    color_data = get_team_color_data(unique_players)
    team_cluster_map = resolve_team_cluster_map(
        team_names,
        scoreboard_kit_colors,
        color_data,
    )
    selected_team = selected_display_to_internal(
        requested_team_name,
        team_names,
        team_cluster_map,
    )

    best_candidate = find_best_visual_candidate(candidates, selected_team)

    if best_candidate is None:
        return None, None, [], "Could not find a readable frame in the video."

    frame = best_candidate["frame"]
    frame_index = best_candidate["frame_index"]
    players = best_candidate["players"]
    selected_players = best_candidate["selected_players"]
    formation_players = choose_formation_players(unique_players, selected_team)
    tracked_selected_players = [
        player for player in unique_players if player["team"] == selected_team
    ]
    display_map = build_team_display_map(
        team_names,
        color_data,
        team_cluster_map,
        scoreboard_kit_colors,
    )
    selected_team_label = display_map.get(selected_team, selected_team)
    formation_result = estimate_formation(formation_players)
    scoreboard_color_summary = "not detected"
    if len(scoreboard_kit_colors) >= 2:
        scoreboard_color_summary = (
            f"{describe_lab_color(scoreboard_kit_colors[0])} / "
            f"{describe_lab_color(scoreboard_kit_colors[1])}"
        )

    annotated_frame = draw_boxes(frame, players, selected_team, display_map)
    pitch_map = create_pitch_map(formation_players, selected_team, display_map)
    coordinate_table = make_coordinate_table(formation_players, display_map)

    timestamp = 0
    if fps:
        timestamp = frame_index / fps

    summary = (
        f"Sampled {sampled_frame_count} frames and selected frame {frame_index} "
        f"at about {timestamp:.1f} seconds for the visual preview.\n"
        f"Speed setting: processed every {int(frame_stride)} frame(s).\n"
        f"Detected {len(players)} total players in the selected frame.\n"
        f"{selected_team_label} contains {len(selected_players)} detected players in that frame.\n\n"
        f"Requested team: {team_query or 'not provided'}.\n"
        f"Team names source: {matchup_source}.\n"
        f"Scoreboard kit colors read left-to-right: {scoreboard_color_summary}.\n"
        f"Detected cluster labels: {display_map['Team 1']} / {display_map['Team 2']}.\n"
        f"Tracking produced {len(tracked_selected_players)} {selected_team_label} track segments across the clip.\n"
        f"Formation analysis used the top {len(formation_players)} tracked {selected_team_label} players.\n"
        "The pitch map and table now show only those formation players, not every tracking fragment.\n"
        f"Formation Guess: {formation_result['formation']}\n"
        f"Confidence: {formation_result['confidence']}\n"
        f"Projected Formation Lines: {formation_result['line_counts']}\n"
        f"Tracked Player Line Counts: {formation_result['raw_line_counts']}\n\n"
        f"Explanation: {formation_result['explanation']}\n\n"
        "Player identity is based on YOLO tracking IDs. "
        "Team separation uses jersey-color clustering with a third Official/Other group for referees or outliers. "
        "If the teams have similar colors or the image is blurry, the grouping may be imperfect."
    )

    return annotated_frame, pitch_map, coordinate_table, summary


with gr.Blocks() as demo:
    gr.Markdown("# Soccer Formation Recognizer")
    gr.Markdown("Upload a short soccer clip. This version tracks players, labels teams when possible, separates teams by jersey color, and estimates the selected team's visible formation.")

    video_input = gr.Video(label="Upload Soccer Video")
    team_query = gr.Textbox(
        label="Team to Analyze",
        placeholder="Example: TOT, LEE, Tottenham, Leeds",
    )
    frame_stride = gr.Slider(
        minimum=1,
        maximum=10,
        value=DEFAULT_TRACK_FRAME_STRIDE,
        step=1,
        label="Frame Stride",
        info="Higher is faster. 1 analyzes every frame; 5 analyzes every 5th frame.",
    )
    matchup_status = gr.Textbox(label="Team Name Detection", interactive=False)
    with gr.Row():
        analyze_button = gr.Button("Analyze Video")
        stop_button = gr.Button("Stop Analysis", variant="stop")

    with gr.Row():
        output_image = gr.Image(label="Detected Players")
        pitch_image = gr.Image(label="Top-Down Pitch Map")

    coordinate_table = gr.Dataframe(
        headers=["Track ID", "Team", "X Pixel", "Y Pixel", "X Normalized", "Y Normalized", "Confidence", "Observations"],
        label="Tracked Player Coordinates",
    )

    output_text = gr.Textbox(label="Formation Result", lines=10)

    video_input.change(
        update_team_status,
        inputs=video_input,
        outputs=matchup_status,
    )
    analyze_event = analyze_button.click(
        analyze_video,
        inputs=[
            video_input,
            team_query,
            frame_stride,
        ],
        outputs=[output_image, pitch_image, coordinate_table, output_text],
    )
    stop_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[analyze_event],
    )


demo.launch()
