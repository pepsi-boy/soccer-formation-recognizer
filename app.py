import gradio as gr

from src.config import DEFAULT_TRACK_FRAME_STRIDE
from src.team_detection import update_team_status
from src.video_analysis import analyze_video

with gr.Blocks(title="Soccer Formation Recognizer") as demo:
    gr.Markdown("# ⚽ Soccer Formation Recognizer")
    gr.Markdown(
        "Upload a short soccer clip. Enter a team name or abbreviation. "
        "The app will detect players, separate teams by kit color, and estimate the visible formation."
    )

    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Upload Soccer Video")
            team_query = gr.Textbox(
                label="Team to Analyze",
                placeholder="Example: TOT, LEE, Tottenham, Leeds, Argentina, FRA",
            )
            jersey_color_hint = gr.Dropdown(
                label="Your Team's Jersey Color (optional)",
                choices=["Auto-detect", "Lighter/White", "Darker/Black", "Red", "Blue", "Yellow", "Green", "Orange"],
                value="Auto-detect",
                info="If the model swaps colors, manually select your team's jersey color here.",
            )
            frame_stride = gr.Slider(
                minimum=1,
                maximum=10,
                value=DEFAULT_TRACK_FRAME_STRIDE,
                step=1,
                label="Frame Stride",
                info="Higher = faster. 1 analyzes every frame; 5 analyzes every 5th frame.",
            )
            matchup_status = gr.Textbox(label="Teams Detected", interactive=False)
            with gr.Row():
                analyze_button = gr.Button("⚡ Analyze Video", variant="primary")
                stop_button = gr.Button("⏹ Stop Analysis", variant="stop")

        with gr.Column(scale=3):
            gr.Markdown("### 🏆 Formation Result")
            formation_summary = gr.Markdown(
                value="*Upload a video and click Analyze to see results.*"
            )

            with gr.Row():
                output_image = gr.Image(label="Detected Players")
                pitch_image = gr.Image(label="Top-Down Pitch Map")

            with gr.Accordion("🔍 Technical Details", open=False):
                gr.Markdown(
                    "*Detailed stats for those interested in the model's internals.*"
                )
                technical_output = gr.Markdown(value="")
                coordinate_table = gr.Dataframe(
                    headers=[
                        "Track ID",
                        "Team",
                        "X Pixel",
                        "Y Pixel",
                        "X Normalized",
                        "Y Normalized",
                        "Confidence",
                        "Observations",
                    ],
                    label="Tracked Player Coordinates",
                )

    team_detect_event = video_input.change(
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
            jersey_color_hint,
        ],
        outputs=[
            output_image,
            pitch_image,
            coordinate_table,
            formation_summary,
            technical_output,
        ],
    )

    stop_button.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[analyze_event, team_detect_event],
    )

demo.launch()