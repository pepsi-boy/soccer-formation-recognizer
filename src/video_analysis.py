import cv2
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO

from .config import (
    DEFAULT_TRACK_FRAME_STRIDE,
    EXPECTED_OUTFIELD_PLAYERS,
    FORMATIONS,
    MIN_TRACK_OBSERVATIONS,
    SAMPLE_FRAME_COUNT,
)
from .formation import choose_formation_players, estimate_formation
from .team_detection import (
    assign_teams_by_color,
    build_team_display_map,
    extract_scoreboard_kit_colors,
    get_jersey_color,
    get_team_color_data,
    resolve_team_cluster_map,
    selected_display_to_internal,
)
from .utils import (
    describe_lab_color,
    get_video_path,
    read_first_frame,
    resolve_matchup,
    resolve_requested_team,
    extract_scoreboard_text,
    extract_team_names_from_text,
)
from .visualization import create_pitch_map, draw_boxes, make_coordinate_table

model = YOLO("yolov8n.pt")

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
    return None, None, [], (
    f"Input team '{team_query}' could not be recognized or matched "
    f"to the detected teams in this video. "
    f"Detected teams: {', '.join(team_names) if team_names else 'none'}."
)
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
