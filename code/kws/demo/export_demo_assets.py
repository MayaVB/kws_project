# export_demo_assets.py
import os
import json
import shutil

import librosa
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram
import pandas as pd

import argparse
from pathlib import Path

# =====================================================
# CONFIG
# =====================================================

DEMO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = DEMO_ROOT / "demo_assets"

SAMPLING_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description="Export audio and visualization assets for the KWS demo.")
    parser.add_argument("--metadata", type=str, required=True, help="Path to the enhanced dataset metadata CSV.",)
    parser.add_argument("--clean_root", type=str, required=True,  help="Root directory of the clean speech dataset.",)
    parser.add_argument("--noisy_root", type=str, required=True, help="Root directory of the noisy test dataset.",)
    parser.add_argument("--enhanced_root", type=str, required=True, help="Root directory of the enhanced test dataset.",)
    return parser.parse_args()

# =====================================================
# EXAMPLES
# =====================================================

DEMO_EXAMPLES = [

    {
        "label": "four",
        "filename": "e0315cf6_nohash_3.wav",
        "true_word": "four",
        "pred_noisy": "no",
        "pred_enh": "four",
        "noise": "Cat Meowing",
        "snr": 2.5,

    },

    {
        "label": "off",
        "filename": "8e05039f_nohash_4.wav",
        "true_word": "off",                                    
        "pred_noisy": "on",
        "pred_enh": "off",
        "noise": "Exercise Bike",
        "snr": 7.5,
    },

    {
        "label": "tree",
        "filename": "b72e58c9_nohash_1.wav",
        "true_word": "tree",
        "pred_noisy": "two",
        "pred_enh": "tree",
        "noise": "Dishwashing",
        "snr": 2.5,
    },

    {
        "label": "go",  
        "filename": "f17be97f_nohash_1.wav",
        "true_word": "go",
        "pred_noisy": "dog",
        "pred_enh": "go",
        "noise": "Doing the Dishes",
        "snr": 2.5,

    },

    {
        "label": "bird",
        "filename": "c1d39ce8_nohash_0.wav",
        "true_word": "bird",
        "pred_noisy": "bed",
        "pred_enh": "bird",
        "noise": "Exercise Bike",
        "snr": 7.5,
    },

    # {
        # "label": "off",
        # "filename": "f4504600_nohash_0.wav",
        # "true_word": "off",
        # "pred_noisy": "on",
        # "pred_enh": "off",
        # "noise": "Dishwashing",
        # "snr": 2.5,
    # },

    # {
        # "label": "go",
        # "filename": "5e033479_nohash_0.wav",
        # "true_word": "go",
        # "pred_noisy": "stop",
        # "pred_enh": "go",
        # "noise": "Cat Meowing",
        # "snr": 2.5,
    # },
    
]

# =====================================================
# SPECTROGRAM FUNCTION
# =====================================================

def save_spectrogram(
    signal,
    fs,
    save_path,
    title,
    show_info=False
):

    D = librosa.stft(
        signal,
        n_fft=512,
        hop_length=128
    )

    S_db = librosa.amplitude_to_db(
        np.abs(D),
        ref=np.max
    )

    times = librosa.times_like(
        S_db,
        sr=fs,
        hop_length=128
    )

    freqs = librosa.fft_frequencies(
        sr=fs,
        n_fft=512
    ) / 1000.0

    plt.figure(figsize=(8,4))

    img = plt.pcolormesh(
        times,
        freqs,
        S_db,
        shading="auto",
        cmap="inferno"
    )

    cbar = plt.colorbar(img)
    cbar.set_label(
        "Magnitude (dB)"
    )

    if show_info:
        plt.title(
                title,
                fontsize=11,
                pad=10
            )
        
    else:
        plt.title("")

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# WAVEFORM
# =====================================================

def save_waveform(
    signal,
    fs,
    save_path,
    title,
    color,
    show_info=True  
):

    t = np.arange(len(signal)) / fs

    plt.figure(figsize=(8, 3))

    plt.plot(t, signal, linewidth=1, color=color)

    if show_info:
        plt.title(
            title,
            fontsize=11,
            pad=10
        )       
    else:
        plt.title("")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.xlim(0, len(signal) / fs)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# MFCC
# =====================================================

def save_mfcc(
    signal,
    fs,
    save_path,
    title,
    show_info=False
):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=fs,
        n_mfcc=13
    )

    plt.figure(figsize=(8,4))

    img = plt.imshow(
        mfcc,
        aspect="auto",
        origin="lower",
        cmap="viridis"
    )

    plt.colorbar(img)

    if show_info:
        plt.title(
            title,
            fontsize=11,
            pad=10
        )       
    else:
        plt.title("")

    plt.xlabel("Frame")
    plt.ylabel("MFCC Coefficient")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# HTML
# =====================================================

def generate_example_html(idx, ex):

    return f"""
<section>

<h2>Example {idx}</h2>

<div class="example-meta">

True Word: {ex["true_word"]} | Noisy Prediction: {ex["pred_noisy"]} | Enhanced Prediction: {ex["pred_enh"]} |

Noise: {ex["noise"]} | SNR: {ex["snr"]} dB

</div>

<div class="result-line">

Noisy Prediction:

<span class="bad">
{ex["pred_noisy"]}
</span>

&nbsp;&nbsp;&nbsp;&nbsp;

Enhanced Prediction:

<span class="good">
{ex["pred_enh"]}
</span>

</div>

<div class="example-grid">

<div class="card">

<h3>Noisy Speech</h3>

<audio controls>
<source src="demo_assets/example_{idx:02d}/noisy.wav" type="audio/wav">
</audio>

<img src="demo_assets/example_{idx:02d}/noisy_spectrogram.png">

</div>

<div class="card">

<h3>Enhanced Speech</h3>

<audio controls>
<source src="demo_assets/example_{idx:02d}/enhanced.wav" type="audio/wav">
</audio>

<img src="demo_assets/example_{idx:02d}/enhanced_spectrogram.png">

</div>

</div>

</section>
"""

def generate_html():

    examples_html = ""

    for idx, ex in enumerate(DEMO_EXAMPLES, start=1):

        examples_html += generate_example_html(idx, ex)

    html = f"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<title>
KWS Enhancement Demo
</title>

<style>

body{{
    font-family:Arial,sans-serif;
    max-width:1400px;
    margin:auto;
    padding:40px;
    line-height:1.6;
}}

h1{{
    text-align:center;
    color:#003366;
}}

h2{{
    text-align:center;
    color:#003366;
    margin-top:60px;
}}

.intro{{
    max-width:800px;
    margin:auto;
    text-align:justify;
    text-justify:inter-word;
    font-size:18px;
}}

.example-grid{{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:25px;
    margin-top:25px;
}}

.card{{
    border:1px solid #d9d9d9;
    border-radius:12px;
    padding:15px;
    background:#fafafa;
}}

.card h3{{
    margin-top:0;
    margin-bottom:10px;
    color:#003366;
    font-size:16px;
    font-weight:600;
    text-align:left;
}}

audio{{
    width:100%;
    margin-bottom:15px;
}}

img{{
    width:100%;
    border:1px solid #cccccc;
}}

.bad{{
    color:red;
    font-weight:bold;
}}

.good{{
    color:green;
    font-weight:bold;
}}

.example-meta{{
    text-align:center;
    font-size:16px;
    color:#666;
    margin-top:-5px;
    margin-bottom:8px;
}}

.result-line{{
    text-align:left;
    font-size:20px;
    font-weight:600;
    margin-bottom:18px;

    width:fit-content;
    margin-left:auto;
    margin-right:auto;
}}

footer{{
    margin-top:60px;
    text-align:center;
    color:#666;
}}

</style>

</head>

<body>

<h1>
Speech Enhancement for Robust Keyword Spotting
</h1>

<p class="intro">

This demo presents examples from our speech enhancement
pipeline for robust keyword spotting (KWS).

Each example contains a spoken keyword mixed with
real-world background noise at different signal-to-noise
ratios (SNRs).

In the noisy condition, the keyword recognition model
produced an incorrect prediction.

After enhancement using an SGMSE model trained on our project dataset,
the correct keyword was successfully recovered.

These examples illustrate how speech enhancement improves
noise robustness and enables more reliable keyword
recognition.

</p>

{examples_html}

<footer>

KWS Enhancement Demo<br>
DS-CNN + Speech Enhancement (SGMSE)

</footer>

</body>
</html>
"""

    with open(
        DEMO_ROOT / "index.html",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(html)

# =====================================================
# MAIN
# =====================================================

def main():
    args = parse_args()

    metadata_csv = Path(args.metadata).expanduser().resolve()
    clean_root = Path(args.clean_root).expanduser().resolve()
    noisy_root = Path(args.noisy_root).expanduser().resolve()
    enhanced_root = Path(args.enhanced_root).expanduser().resolve()

    df_meta = pd.read_csv(metadata_csv)
    print(df_meta[df_meta["filename"] == "e0315cf6_nohash_3.wav"])

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)
    
    for idx, ex in enumerate(
        DEMO_EXAMPLES,
        start=1
    ):
    
        label = ex["label"]
        filename = ex["filename"]
    
        print(
            f"\nProcessing {label}/{filename}"
        )
    
        example_dir = os.path.join(
            OUTPUT_DIR,
            f"example_{idx:02d}"
        )
    
        os.makedirs(
            example_dir,
            exist_ok=True
        )
    
        # -------------------------------------------------
        # PATHS
        # -------------------------------------------------
    
        clean_path = clean_root / label / filename
        noisy_path = noisy_root / label / filename
        enh_path = enhanced_root / label / filename
    
        # -------------------------------------------------
        # COPY WAVS
        # -------------------------------------------------
    
        shutil.copy2(
            clean_path,
            os.path.join(
                example_dir,
                "clean.wav"
            )
        )
    
        shutil.copy2(
            noisy_path,
            os.path.join(
                example_dir,
                "noisy.wav"
            )
        )
    
        shutil.copy2(
            enh_path,
            os.path.join(
                example_dir,
                "enhanced.wav"
            )
        )
    
        # -------------------------------------------------
        # LOAD AUDIO
        # -------------------------------------------------
    
        clean_sig, _ = librosa.load(
            clean_path,
            sr=SAMPLING_RATE
        )
    
        noisy_sig, _ = librosa.load(
            noisy_path,
            sr=SAMPLING_RATE
        )
    
        enh_sig, _ = librosa.load(
            enh_path,
            sr=SAMPLING_RATE
        )
    
        # -------------------------------------------------
        # SPECTROGRAMS
        # -------------------------------------------------
    
        save_spectrogram(
            clean_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "clean_spectrogram.png"
            ),
            f"""Clean Speech | Keyword: {ex["true_word"]}""",
            show_info=False
        )
    
        save_spectrogram(
            noisy_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "noisy_spectrogram.png"
            ),
             f"""Noisy Speech | Keyword: {ex["true_word"]} | {ex["noise"]} | SNR = {ex["snr"]} dB""",
            show_info=False
        )
    
        save_spectrogram(
            enh_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "enhanced_spectrogram.png"
            ),
            f"""Enhanced Speech | Keyword: {ex["true_word"]} | {ex["noise"]} | SNR = {ex["snr"]} dB""",
            show_info=False
        )
    
        # -------------------------------------------------
        # WAVEFORMS
        # -------------------------------------------------
    
        save_waveform(
            clean_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "clean_waveform.png"
            ),
            f"""Clean Speech | Keyword: {ex["true_word"]}""",
            "forestgreen",
            show_info=False
        )
    
        save_waveform(
            noisy_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "noisy_waveform.png"
            ),
            f"""Noisy Speech | Keyword: {ex["true_word"]} | {ex["noise"]} | SNR = {ex["snr"]} dB""",
            "darkorange",
            show_info=False
        )
    
        save_waveform(
            enh_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "enhanced_waveform.png"
            ),
            f"""Enhanced Speech | Keyword: {ex["true_word"]} | {ex["noise"]} | SNR = {ex["snr"]} dB""",
            "royalblue",
            show_info=False
        )
    
        # -------------------------------------------------
        # MFCC
        # -------------------------------------------------
    
        save_mfcc(
            clean_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "clean_mfcc.png"
            ),
            f"""Clean Speech | Keyword: {ex["true_word"]}""",
            show_info=False
        )
    
        save_mfcc(
            noisy_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "noisy_mfcc.png"
            ),
            f"""Noisy Speech | Keyword: {ex["true_word"]}""",
            show_info=False
        )
    
        save_mfcc(
            enh_sig,
            SAMPLING_RATE,
            os.path.join(
                example_dir,
                "enhanced_mfcc.png"
            ),
            f"""Enhanced Speech | Keyword: {ex["true_word"]}""",
            show_info=False
        )
    
        # -------------------------------------------------
        # METADATA
        # -------------------------------------------------
    
        noise = ex["noise"]
        snr = ex["snr"]
            
        metadata = {
    
            "filename": filename,
    
            "true_word": ex["true_word"],
    
            "pred_noisy": ex["pred_noisy"],
            "pred_enh": ex["pred_enh"],
    
            "noise": noise,
            "snr": snr,
    
            "label": label,
    
            "noisy_correct":
                ex["pred_noisy"] == ex["true_word"],
    
            "enh_correct":
                ex["pred_enh"] == ex["true_word"]
        }
    
        with open(
            os.path.join(
                example_dir,
                "metadata.json"
            ),
            "w"
        ) as f:
    
            json.dump(
                metadata,
                f,
                indent=4
            )
    
    generate_html()
    
    print("\nHTML generated.")
    print("Done.")


if __name__ == "__main__":
    main()

# Run from the repository root:
#
# python code/kws/demo/export_demo_assets.py \
#     --metadata data/enhanced_trained_ep176_new_metadata.csv \
#     --clean_root data/raw/data_new \
#     --noisy_root data/noisy_new/test \
#     --enhanced_root data/enhanced_new/trained_ep176
#
# Then update the GitHub Pages demo:
# cd ../kws-demo
# ./update_demo.sh