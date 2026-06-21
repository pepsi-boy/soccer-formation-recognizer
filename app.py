import gradio as gr

from src.config import DEFAULT_TRACK_FRAME_STRIDE
from src.team_detection import update_team_status
from src.video_analysis import analyze_video

with gr.Blocks() as demo:
    gr.Markdown("# Soccer Formation Recognizer")
    gr.Markdown(
        "Upload a short soccer clip. This version tracks players, labels teams when possible, "
        "separates teams by jersey color, and estimates the selected team's visible formation."
    )

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
