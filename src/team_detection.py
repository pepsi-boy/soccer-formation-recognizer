import cv2
import numpy as np
from sklearn.cluster import KMeans

from .utils import (
    bgr_to_lab_color,
    describe_lab_color,
    describe_team_color_comparative,
    get_video_path,
    normalize_text,
    read_first_frame,
    resolve_matchup,
    extract_scoreboard_text,
    extract_team_names_from_text,
)

from .config import EXPECTED_OUTFIELD_PLAYERS, MIN_TRACK_OBSERVATIONS


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

    # Get LAB colors for both teams
    team1_lab = color_data.get("Team 1", {}).get("lab")
    team2_lab = color_data.get("Team 2", {}).get("lab")

    # Use comparative labeling (handles grass contamination better)
    label1, label2 = describe_team_color_comparative(team1_lab, team2_lab)

    color_labels = {
        "Team 1": label1,
        "Team 2": label2,
    }

    for internal_team in ["Team 1", "Team 2"]:
        team_name = team_name_by_cluster.get(internal_team)
        color_label = color_labels[internal_team]

        if team_name:
            display_map[internal_team] = f"{team_name} ({color_label})"
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

    if team_names:
        if selected_team == team_names[0]:
            return "Team 1"

        if len(team_names) > 1 and selected_team == team_names[1]:
            return "Team 2"

    return None


def update_team_status(video_input):
    video_path = get_video_path(video_input)
    team_names, source = resolve_matchup(video_path)

    if len(team_names) >= 2:
        return f"{team_names[0]} and {team_names[1]}"

    if len(team_names) == 1:
        return f"{team_names[0]} (other team not detected)"

    return "Could not detect teams from the scoreboard."


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

def override_team_cluster_by_color_hint(color_hint, team_names, team_cluster_map, color_data):
    """
    If user specified a jersey color hint for their team (team_names[0] or the queried team),
    override the cluster mapping to match.
    
    color_hint: one of "Lighter/White", "Darker/Black", "Red", "Blue", "Yellow", "Green", "Orange"
    Returns updated team_cluster_map.
    """
    if not color_hint or color_hint == "Auto-detect" or len(team_names) < 2:
        return team_cluster_map

    team1_lab = color_data.get("Team 1", {}).get("lab")
    team2_lab = color_data.get("Team 2", {}).get("lab")

    if team1_lab is None or team2_lab is None:
        return team_cluster_map

    l1, a1, b1 = float(team1_lab[0]), float(team1_lab[1]), float(team1_lab[2])
    l2, a2, b2 = float(team2_lab[0]), float(team2_lab[1]), float(team2_lab[2])

    # Score each cluster for how well it matches the hint
    def score_cluster(l, a, b, hint):
        a_shift = a - 128
        b_shift = b - 128
        if hint in ("Lighter/White",):
            return l  # higher L = better match
        elif hint in ("Darker/Black",):
            return 255 - l  # lower L = better match
        elif hint == "Red":
            return a_shift  # higher a = more red
        elif hint == "Blue":
            return -b_shift  # lower b = more blue
        elif hint == "Yellow":
            return b_shift  # higher b = more yellow
        elif hint == "Green":
            return -a_shift  # lower a = more green
        elif hint == "Orange":
            return a_shift + b_shift  # high a and high b
        return 0

    score1 = score_cluster(l1, a1, b1, color_hint)
    score2 = score_cluster(l2, a2, b2, color_hint)

    # The cluster with the higher score matches the user's team
    # team_names[0] is assumed to be the user's selected team
    # (or we can pass in the specific team — for now, use first team in team_names)
    if score1 >= score2:
        # Team 1 cluster matches the hint → first team name maps to Team 1
        return {
            team_names[0]: "Team 1",
            team_names[1]: "Team 2",
        }
    else:
        # Team 2 cluster matches the hint → first team name maps to Team 2
        return {
            team_names[0]: "Team 2",
            team_names[1]: "Team 1",
        }


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