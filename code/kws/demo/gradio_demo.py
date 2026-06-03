import gradio as gr
import pandas as pd

"""
gradio_demo.py

Interactive Gradio demo for the project.

The demo visualizes examples where speech enhancement
improved keyword classification performance.

Displayed information:
- clean audio
- noisy audio
- enhanced audio
- model predictions
- SNR
- noise type
"""
# =====================================================
# LOAD DATA
# =====================================================

CSV_PATH = (
    "/home/dsi/skopavi/Project/kws_project/"
    "outputs/new_runs/2026-05-31_20-09-53/"
    "interesting_examples.csv"
)

df = pd.read_csv(CSV_PATH)

# show just the "fixed" examples where enhancement improved performance
df = df[df["category"] == "fixed"].reset_index(drop=True)

# =====================================================
# BUILD DROPDOWN LABELS
# =====================================================

example_labels = []
label_to_filename = {}

for _, row in df.iterrows():

    label = (
        f"{row['filename']}\n"
        f"{row['true_label']} | "
        f"noisy→{row['pred_noisy']} | "
        f"enh→{row['pred_enh']} | "
        f"{row['snr']} dB | "
        f"{row['noise']}"
)

    example_labels.append(label)

    label_to_filename[label] = row["filename"]

# =====================================================
# CALLBACK
# =====================================================

def load_example(selected_label):

    filename = label_to_filename[selected_label]

    row = df[df["filename"] == filename].iloc[0]

    info = f"""
True Label:
{row['true_label']}

Clean Prediction:
{row['pred_clean']}
(correct={row['correct_clean']})

Noisy Prediction:
{row['pred_noisy']}
(correct={row['correct_noisy']})

Enhanced Prediction:
{row['pred_enh']}
(correct={row['correct_enh']})

Noise:
{row['noise']}

SNR:
{row['snr']} dB

Filename:
{row['filename']}
"""

    return (
        info,
        row["clean_path"],
        row["noisy_path"],
        row["enh_path"]
    )

# =====================================================
# UI
# =====================================================

with gr.Blocks(title="KWS Enhancement Demo") as demo:

    gr.Markdown("# KWS Enhancement Demo")

    example_dd = gr.Dropdown(
        choices=example_labels,
        value=example_labels[0],
        label="Example"
    )

    info_box = gr.Textbox(
        label="Prediction Information",
        lines=15
    )

    with gr.Row():

        clean_audio = gr.Audio(
            label="Clean Audio"
        )

        noisy_audio = gr.Audio(
            label="Noisy Audio"
        )

        enh_audio = gr.Audio(
            label="Enhanced Audio"
        )

    example_dd.change(
        fn=load_example,
        inputs=example_dd,
        outputs=[
            info_box,
            clean_audio,
            noisy_audio,
            enh_audio
        ]
    )

    # טload the first example on startup
    demo.load(
        fn=load_example,
        inputs=example_dd,
        outputs=[
            info_box,
            clean_audio,
            noisy_audio,
            enh_audio
        ]
    )

# =====================================================
# RUN
# =====================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    show_api=False
)