import argparse
from pathlib import Path
import os
import pandas as pd
import gradio as gr

from demo_utils import DEFAULT_PATH_MAP, parse_path_map, remap_audio_paths, AUDIO_PATH_COLUMNS


# =====================================================
# ARGUMENTS
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Browse and select interesting KWS examples.")
    parser.add_argument("--csv", type=str, required=True, help="Path to an interesting_examples.csv file.",)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                        help="Extra audio path prefix mapping (repeatable). Takes precedence over the built-in "
                             "historical-data -> GannotLab-archive mapping.")
    parser.add_argument("--no-path-remap", action="store_true",
                        help="Use the audio paths exactly as stored in the CSV.")

    return parser.parse_args()


args = parse_args()

CSV_PATH = Path(args.csv).expanduser().resolve()
SAVE_PATH = Path(__file__).resolve().parent / "selected_examples.csv"

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(CSV_PATH)

# Historical CSVs hold absolute paths to the original data location; remap them
# in memory to the archive (the CSV file itself is not modified).
if not args.no_path_remap:
    df = remap_audio_paths(df, parse_path_map(args.path_map) + DEFAULT_PATH_MAP)

_missing = sum(not (isinstance(p, str) and os.path.exists(p))
               for col in AUDIO_PATH_COLUMNS if col in df.columns for p in df[col])
if _missing:
    print(f"WARNING: {_missing} audio paths in {CSV_PATH} do not exist (use --path-map OLD=NEW).")

df = (
    df[df["category"] == "fixed"]
    .reset_index(drop=True)
)

N = len(df)

# =====================================================
# BUILD HEADER
# =====================================================

def build_header(idx):

    row = df.iloc[idx]

    return f"""
# Example {idx + 1} / {N}

**Filename:** `{row['filename']}`

**True Label:** `{row['true_label']}`

**Noise:** `{row['noise']}`

**SNR:** `{row['snr']} dB`

---

### ❌ Noisy Prediction

**{row['pred_noisy']}**

---

### ✅ Enhanced Prediction

**{row['pred_enh']}**
"""


# =====================================================
# LOAD EXAMPLE
# =====================================================

def load_example(idx):

    idx = max(0, min(int(idx), N - 1))

    row = df.iloc[idx]

    return (
        idx,
        build_header(idx),
        row["clean_path"],
        row["noisy_path"],
        row["enh_path"],
        ""
    )


# =====================================================
# NAVIGATION
# =====================================================

def next_example(idx):

    idx = min(int(idx) + 1, N - 1)

    return load_example(idx)


def previous_example(idx):

    idx = max(int(idx) - 1, 0)

    return load_example(idx)


# =====================================================
# SAVE EXAMPLE
# =====================================================

def save_example(idx):

    idx = int(idx)

    row = df.iloc[idx]

    cols = [
        "filename",
        "true_label",
        "pred_clean",
        "pred_noisy",
        "pred_enh",
        "noise",
        "snr"
    ]

    selected = row[cols].to_frame().T

    if os.path.exists(SAVE_PATH):

        saved = pd.read_csv(SAVE_PATH)

        exists = (
            (saved["filename"] == row["filename"]) &
            (saved["noise"] == row["noise"]) &
            (saved["snr"] == row["snr"])
        ).any()

        if exists:
            return "✅ Already saved."

        saved = pd.concat(
            [saved, selected],
            ignore_index=True
        )

    else:

        saved = selected

    saved.to_csv(
        SAVE_PATH,
        index=False
    )

    return "⭐ Example saved."


# =====================================================
# UI
# =====================================================

with gr.Blocks(title="KWS Enhancement Demo") as demo:

    gr.Markdown("# KWS Enhancement Demo")

    current_idx = gr.State(0)

    header = gr.Markdown()

    with gr.Row():

        prev_btn = gr.Button("⬅ Previous")

        save_btn = gr.Button("⭐ Save Example")

        next_btn = gr.Button("Next ➡")

    status = gr.Markdown()

    with gr.Row():

        clean_audio = gr.Audio(
            label="Clean Audio"
        )

        noisy_audio = gr.Audio(
            label="Noisy Audio"
        )

        enhanced_audio = gr.Audio(
            label="Enhanced Audio"
        )

    demo.load(
        fn=load_example,
        inputs=current_idx,
        outputs=[
            current_idx,
            header,
            clean_audio,
            noisy_audio,
            enhanced_audio,
            status
        ]
    )

    next_btn.click(
        fn=next_example,
        inputs=current_idx,
        outputs=[
            current_idx,
            header,
            clean_audio,
            noisy_audio,
            enhanced_audio,
            status
        ]
    )

    prev_btn.click(
        fn=previous_example,
        inputs=current_idx,
        outputs=[
            current_idx,
            header,
            clean_audio,
            noisy_audio,
            enhanced_audio,
            status
        ]
    )

    save_btn.click(
        fn=save_example,
        inputs=current_idx,
        outputs=status
    )

# =====================================================
# RUN
# =====================================================

demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    show_api=False
)