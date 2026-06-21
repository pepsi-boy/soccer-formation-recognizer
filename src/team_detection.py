import cv2
import numpy as np
from sklearn.cluster import KMeans

from .utils import (
    bgr_to_lab_color,
    describe_lab_color,
    get_video_path,
    normalize_text,
    read_first_frame,
    resolve_matchup,
    extract_scoreboard_text,
    extract_team_names_from_text,
)

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

    return None

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
