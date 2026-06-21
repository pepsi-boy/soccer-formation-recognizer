import cv2
import matplotlib.pyplot as plt
import numpy as np

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
