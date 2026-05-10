# visualization.py
import os
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
    enh_pretrained,
    enh_trained=None, 
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

    # ENHANCED SGMSE
    enhanced_sgmse_path = os.path.join(
        enh_pretrained,
        label,
        filename
    )

    enhanced_sgmse_sig = None
    if os.path.exists(enhanced_sgmse_path):
        enhanced_sgmse_sig, _ = librosa.load(enhanced_sgmse_path, sr=sampling_rate)
    else:
        print(f"⚠️ Missing enhanced SGMSE: {enhanced_sgmse_path}")

    # ENHANCED TRAINED EP100
    enhanced_trained_sig = None
    
    if enh_trained is not None:
        path = os.path.join(enh_trained, label, filename)
        if os.path.exists(path):
            enhanced_trained_sig, _ = librosa.load(path, sr=sampling_rate)
        else:
            print(f"⚠️ Missing enhanced trained: {path}")
    

    # PLOT
    plot_signal_comparison(
        clean=clean_sig,
        noisy=noisy_sig,
        denoised=denoised_sig,
        enhanced_sgmse=enhanced_sgmse_sig,
        enhanced_trained=enhanced_trained_sig,
        fs=sampling_rate,
        title=f"{label}/{filename}",
        save_path=plots_dir / f"example_{filename}.png"
    )

# TODO: add option to plot random examples from the test set (not just fixed idx)
# TODO: add option to plot examples from the enhanced dataset (not just the noisy dataset)
# TODO: CHOOSE EXAMPLE WITH SPECIFIC SNR
