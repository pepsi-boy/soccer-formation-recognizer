import numpy as np
from sklearn.cluster import KMeans

from .config import EXPECTED_OUTFIELD_PLAYERS, FORMATIONS, MIN_TRACK_OBSERVATIONS

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
