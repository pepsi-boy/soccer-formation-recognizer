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
    override_team_cluster_by_color_hint,
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


# ============================================================
# YOLO model
# ============================================================

model = YOLO("yolov8n.pt")


# ============================================================
# Tracking helpers
# ============================================================

def get_sample_frame_indices(processed_frame_count):
    """Return indices of frames to use as visual candidates."""
    if processed_frame_count <= SAMPLE_FRAME_COUNT:
        return list(range(processed_frame_count))
    step = processed_frame_count / SAMPLE_FRAME_COUNT
    return [int(i * step) for i in range(SAMPLE_FRAME_COUNT)]


def extract_players_from_frame(frame):
    """Single-frame detection (no tracking)."""
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
        jersey_color = get_jersey_color(frame, bbox)

        players.append({
            "id": i,
            "track_id": None,
            "frame_index": 0,
            "x": float(center_x),
            "y": float(foot_y),
            "normalized_x": float(normalized_x),
            "normalized_y": float(normalized_y),
            "confidence": confidence,
            "bbox": bbox,
            "jersey_color": jersey_color,
            "team": "Unknown",
        })

    return players


def extract_players_from_tracking_result(result, frame_index):
    """Extract player dicts from a single YOLO tracking result."""
    frame = result.orig_img
    height, width = frame.shape[:2]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    players = []
    boxes = result.boxes
    track_ids = boxes.id

    for i in range(len(boxes)):
        bbox = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = bbox
        confidence = float(boxes.conf[i].cpu().numpy())

        track_id = None
        if track_ids is not None:
            track_id = int(track_ids[i].cpu().numpy())

        center_x = (x1 + x2) / 2
        foot_y = y2
        normalized_x = center_x / width
        normalized_y = foot_y / height

        bbox_list = [float(x1), float(y1), float(x2), float(y2)]
        jersey_color = get_jersey_color(frame, bbox_list)

        players.append({
            "id": track_id or i,
            "track_id": track_id,
            "frame_index": frame_index,
            "x": float(center_x),
            "y": float(foot_y),
            "normalized_x": float(normalized_x),
            "normalized_y": float(normalized_y),
            "confidence": confidence,
            "bbox": bbox_list,
            "jersey_color": jersey_color,
            "team": "Unknown",
        })

    return players


def build_unique_tracked_players(tracked_detections):
    """Aggregate detections by track_id into unique players with averaged positions."""
    tracks = {}
    for detection in tracked_detections:
        track_id = detection.get("track_id")
        if track_id is None:
            continue
        tracks.setdefault(track_id, []).append(detection)

    unique_players = []
    for track_id, detections in tracks.items():
        best_detection = max(detections, key=lambda d: d["confidence"])

        unique_players.append({
            "id": track_id,
            "track_id": track_id,
            "team": "Unknown",
            "x": float(np.mean([d["x"] for d in detections])),
            "y": float(np.mean([d["y"] for d in detections])),
            "normalized_x": float(np.mean([d["normalized_x"] for d in detections])),
            "normalized_y": float(np.mean([d["normalized_y"] for d in detections])),
            "confidence": float(np.mean([d["confidence"] for d in detections])),
            "bbox": best_detection["bbox"],
            "jersey_color": np.mean(
                [d["jersey_color"] for d in detections], axis=0
            ),
            "observations": len(detections),
        })

    return unique_players


def propagate_track_teams(detections, unique_players):
    """Copy team labels from unique_players back to per-frame detections."""
    team_by_track = {p["track_id"]: p.get("team") for p in unique_players}
    for detection in detections:
        track_id = detection.get("track_id")
        if track_id and track_id in team_by_track:
            detection["team"] = team_by_track[track_id]


def analyze_video_with_tracking(video_path, frame_count, frame_stride):
    """Run YOLO tracking across the video and return candidates + unique players."""
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

        if processed_index in sample_indices:
            candidates.append({
                "frame": result.orig_img.copy(),
                "frame_index": frame_index,
                "players": players,
            })

    unique_players = build_unique_tracked_players(detections)
    unique_players = assign_teams_by_color(unique_players)
    propagate_track_teams(detections, unique_players)

    return candidates, unique_players, len(sample_indices)


# ============================================================
# Main analysis function
# ============================================================

def find_best_visual_candidate(candidates, selected_team):
    """Pick the candidate frame with the most players from the selected team."""
    best = None
    best_score = -1

    for candidate in candidates:
        players = candidate["players"]
        team_players = [p for p in players if p.get("team") == selected_team]
        candidate["selected_players"] = team_players

        avg_conf = (
            np.mean([p.get("confidence", 0) for p in team_players])
            if team_players
            else 0
        )
        score = len(team_players) * 1000 + avg_conf

        if score > best_score:
            best_score = score
            best = candidate

    return best


def analyze_video(video_input, team_query, frame_stride, jersey_color_hint="Auto-detect"):
    """Main analysis pipeline. Returns 5 outputs for the Gradio UI."""
    video_path = get_video_path(video_input)
    if not video_path:
        empty_msg = "*Please upload a video first.*"
        return None, None, [], empty_msg, ""

    # ------------------------------------------------------------------
    # 1. Resolve matchup
    # ------------------------------------------------------------------
    team_names, matchup_source = resolve_matchup(video_path)

    # ------------------------------------------------------------------
    # 1b. Validate user's team query against detected teams
    # ------------------------------------------------------------------
    if team_query and team_query.strip() and len(team_names) >= 2:
        resolved = resolve_requested_team(team_query, team_names)
        if resolved is None:
            return (
                None,
                None,
                [],
                f"⚠️ **'{team_query}' is not one of the detected teams.**\n\n"
                f"Detected teams: **{team_names[0]}** and **{team_names[1]}**.\n\n"
                f"Please enter one of those team names (or their abbreviation), "
                f"or leave the field blank to analyze the first team.",
                "",
            )

    # ------------------------------------------------------------------
    # 2. Track players across the clip
    # ------------------------------------------------------------------
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()

    candidates, unique_players, sampled_frame_count = analyze_video_with_tracking(
        video_path,
        frame_count,
        frame_stride,
    )

    if not candidates:
        empty_msg = "**⚠️ No players detected.** Try a different clip or lower the frame stride."
        return None, None, [], empty_msg, ""

    # Propagate team labels to candidate frames
    for candidate in candidates:
        propagate_track_teams(candidate["players"], unique_players)

    # ------------------------------------------------------------------
    # 3. Detect scoreboard kit colors and resolve cluster-to-team mapping
    # ------------------------------------------------------------------
    first_frame_cap = cv2.VideoCapture(video_path)
    ret, first_frame = first_frame_cap.read()
    first_frame_cap.release()

    scoreboard_kit_colors = []
    if ret:
        scoreboard_kit_colors = extract_scoreboard_kit_colors(first_frame)

    color_data = get_team_color_data(unique_players)
    team_cluster_map = resolve_team_cluster_map(
        team_names, scoreboard_kit_colors, color_data
    )

    # ------------------------------------------------------------------
    # 3b. Override mapping if user specified jersey color hint
    # ------------------------------------------------------------------
    if jersey_color_hint and jersey_color_hint != "Auto-detect" and len(team_names) >= 2:
        requested_name = resolve_requested_team(team_query, team_names)
        if requested_name:
            ordered_names = [requested_name, [n for n in team_names if n != requested_name][0]]
        else:
            ordered_names = team_names
        team_cluster_map = override_team_cluster_by_color_hint(
            jersey_color_hint, ordered_names, team_cluster_map, color_data
        )

    # ------------------------------------------------------------------
    # 4. Resolve user's requested team
    # ------------------------------------------------------------------
    requested_team_name = resolve_requested_team(team_query, team_names)
    selected_team = selected_display_to_internal(
        requested_team_name, team_names, team_cluster_map
    )

    if not selected_team:
        selected_team = "Team 1"

    # ------------------------------------------------------------------
    # 5. Find best visual candidate frame
    # ------------------------------------------------------------------
    best_candidate = find_best_visual_candidate(candidates, selected_team)

    if best_candidate is None:
        empty_msg = "**⚠️ Could not find a readable frame.** Try a different clip."
        return None, None, [], empty_msg, ""

    frame = best_candidate["frame"]
    frame_index = best_candidate["frame_index"]
    players = best_candidate["players"]
    selected_players = best_candidate.get("selected_players", [])

    # ------------------------------------------------------------------
    # 6. Formation analysis
    # ------------------------------------------------------------------
    formation_players = choose_formation_players(unique_players, selected_team)
    tracked_selected_players = [
        player for player in unique_players if player.get("team") == selected_team
    ]
    formation_result = estimate_formation(formation_players)

    # ------------------------------------------------------------------
    # 7. Build display map
    # ------------------------------------------------------------------
    display_map = build_team_display_map(
        team_names, color_data, team_cluster_map, scoreboard_kit_colors
    )
    selected_team_label = display_map.get(selected_team, selected_team)

    # ------------------------------------------------------------------
    # 8. Timestamp
    # ------------------------------------------------------------------
    timestamp = frame_index / fps if fps > 0 else 0.0

    # ------------------------------------------------------------------
    # 9. Draw visuals
    # ------------------------------------------------------------------
    annotated_frame = draw_boxes(frame, players, selected_team, display_map)
    pitch_map = create_pitch_map(formation_players, selected_team, display_map)

    # ------------------------------------------------------------------
    # 10. Build coordinate table
    # ------------------------------------------------------------------
    coordinate_table = make_coordinate_table(formation_players, display_map)

    # ------------------------------------------------------------------
    # 11. Build user-friendly formation summary (Markdown)
    # ------------------------------------------------------------------
    confidence_val = formation_result.get("confidence", 0)
    confidence_pct = int(confidence_val * 100)

          # Confidence bar
    filled = confidence_pct // 10
    empty_blocks = 10 - filled
    bar = "█" * filled + "░" * empty_blocks

    # Confidence label
    if confidence_val >= 0.85:
        conf_label = "High"
        conf_emoji = "✅"
    elif confidence_val >= 0.65:
        conf_label = "Medium"
        conf_emoji = "●"
    else:
        conf_label = "Low"
        conf_emoji = "⚠️"

    line_counts = formation_result.get("line_counts", [])
    raw_line_counts = formation_result.get("raw_line_counts", line_counts)

    formation_summary = f"""
## {selected_team_label}

### Estimated Formation: **{formation_result.get('formation', 'Unknown')}**

{conf_emoji} **Confidence: {conf_label} ({confidence_pct}%)**

{bar}

---

**Line Structure:** {' - '.join(str(x) for x in line_counts)}

**Players Used:** {len(formation_players)} outfield players tracked across the clip

---

? **What does this mean?**

The model detected {len(formation_players)} players for {selected_team_label} and grouped them into
defensive, midfield, and attacking lines based on their average positions across multiple frames.
The shape **{formation_result.get('formation', 'Unknown')}** (lines of {', '.join(str(x) for x in line_counts)} players)
was the closest match from the template library.

{"⚠️ *Note: Confidence is below 70%. The camera angle, missing players, or similar kit colors may have affected accuracy.*" if confidence_val < 0.7 else ""}
"""

    # ------------------------------------------------------------------
    # 12. Build technical details (Markdown)
    # ------------------------------------------------------------------
    scoreboard_color_summary = "not detected"
    if len(scoreboard_kit_colors) >= 2:
        scoreboard_color_summary = (
            f"{describe_lab_color(scoreboard_kit_colors[0])} / "
            f"{describe_lab_color(scoreboard_kit_colors[1])}"
        )

    technical_output = f"""
#### Video & Sampling
| Metric | Value |
|--------|-------|
| Total frames | {frame_count} |
| Frames sampled | {sampled_frame_count} |
| Frame stride | {int(frame_stride)} |
| Preview frame | #{frame_index} (~{timestamp:.1f}s) |
| FPS | {fps:.1f} |

#### Detection & Tracking
| Metric | Value |
|--------|-------|
| Players in preview frame | {len(players)} |
| {selected_team_label} in preview frame | {len(selected_players)} |
| Total {selected_team_label} track segments | {len(tracked_selected_players)} |
| Players used for formation | {len(formation_players)} |
| Min observations threshold | {MIN_TRACK_OBSERVATIONS} |

#### Team Identification
| Metric | Value |
|--------|-------|
| Requested team input | `{team_query or 'not provided'}` |
| Resolved team | {selected_team_label} |
| Internal cluster | {selected_team} |
| Team names source | {matchup_source} |
| Scoreboard kit colors | {scoreboard_color_summary} |
| Cluster labels | {display_map.get('Team 1', 'N/A')} / {display_map.get('Team 2', 'N/A')} |

#### Formation Internals
| Metric | Value |
|--------|-------|
| Formation guess | {formation_result.get('formation', 'Unknown')} |
| Raw confidence score | {confidence_val:.4f} |
| Projected line counts | {line_counts} |
| Raw line counts | {raw_line_counts} |

#### How It Works
1. **YOLO Detection** — YOLOv8 detects all people in each sampled frame
2. **Multi-frame Tracking** — Players are tracked across frames using YOLO's built-in tracker
3. **Kit Color Clustering** — K-Means (k=3) on jersey LAB colors separates Team 1, Team 2, and Officials
4. **Scoreboard OCR** — Abbreviations and kit-color swatches from the broadcast graphic help label clusters
5. **Position Averaging** — Each player's position is averaged across all frames they appear in
6. **Line Grouping** — Players are grouped into depth lines (defense, midfield, attack) by Y-coordinate clustering
7. **Template Matching** — Line counts are compared to formation templates (4-3-3, 4-4-2, etc.) to find the best fit
"""

    return annotated_frame, pitch_map, coordinate_table, formation_summary, technical_output