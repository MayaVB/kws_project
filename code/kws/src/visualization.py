# visualization.py
import os
from os import path
import random
import librosa
import numpy as np

from noise_onthefly_dataset import NoisyTestDataset, mix_with_noise_at_snr
from kws.denoiser.stft_mask.denoise import denoise_signal
from metrics import plot_signal_comparison

def plot_example_signals(
    df_test_audio,
    sampling_rate,
    plots_dir,
    noisy_root,
    enhanced_versions,  # dict of {mode_name: enhanced_root}
    idx_vis=0,
):
    """
    Plot example: clean vs noisy vs denoised vs enhanced
    Uses FIXED test dataset (no randomness)
    """

    # BASIC INFO
    filename = df_test_audio["filename"].iloc[idx_vis]
    label = df_test_audio["label"].iloc[idx_vis]

    print(f"[EXAMPLE] label={label}, file={filename}")

    # CLEAN
    clean_sig = df_test_audio["audio_data"].iloc[idx_vis]

    # NOISY 
    noisy_path = os.path.join(noisy_root, "test", label, filename)

    if not os.path.exists(noisy_path):
        print(f"❌ Missing noisy file: {noisy_path}")
        return

    noisy_sig, _ = librosa.load(noisy_path, sr=sampling_rate)

    # DENOISED
    denoised_sig = denoise_signal(
        noisy_signal=noisy_sig,
        fs=sampling_rate
    )

    # ENHANCED
    enhanced_signals = {}
    for method_name, root in enhanced_versions.items():
        path = os.path.join(root, label, filename)
        if os.path.exists(path):
            sig, _ = librosa.load(path, sr=sampling_rate)
            enhanced_signals[method_name] = sig
        else:
            print(f"⚠️ Missing enhanced: {path}")

    # PLOT
    plot_signal_comparison(
        clean=clean_sig,
        noisy=noisy_sig,
        denoised=denoised_sig,
        enhanced_signals=enhanced_signals,
        fs=sampling_rate,
        title=f"{label}/{filename}",
        save_path=plots_dir / f"Signal_Comparison_Example_{filename}.png"
    )

# TODO: add option to plot random examples from the test set (not just fixed idx)
# TODO: add option to plot examples from the enhanced dataset (not just the noisy dataset)
# TODO: CHOOSE EXAMPLE WITH SPECIFIC SNR
