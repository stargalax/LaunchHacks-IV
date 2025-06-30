import gradio as gr
import cv2
import numpy as np
import time
from model import process_video_stream  # assuming model.py is in same directory

def create_sector_ui(sector_idx):
    with gr.Column():
        gr.Markdown(f"### Sector {sector_idx}")
        video_in = gr.Video(label=f"Upload Video for Sector {sector_idx}", interactive=True)
        start_btn = gr.Button("Start Processing")
        video_out = gr.Image(label=f"Processed Feed Sector {sector_idx}")
        alert_out = gr.Textbox(label="Alert Status", interactive=False)

        def start_processing(video_path):
            for frame, alert_triggered in process_video_stream(video_path, f"Sector {sector_idx}"):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                alert_msg = "⚠️ Alert!" if alert_triggered else "✅ Safe"
                yield frame_rgb, alert_msg
                time.sleep(0.03)


        start_btn.click(
            fn=start_processing,
            inputs=video_in,
            outputs=[video_out, alert_out],
            show_progress=True,
            #stream=True
        )

        return video_in, video_out, alert_out

with gr.Blocks() as demo:
    gr.Markdown("## Sector-wise Real-time Video Analysis Demo")

    with gr.Row():
        for sector_idx in range(1, 7):
            create_sector_ui(sector_idx)
    


demo.queue()
demo.launch()
