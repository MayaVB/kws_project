import os
import random
import librosa
import numpy as np

from noise_dataset import NoisyTestDataset, mix_with_noise_at_snr
from denoiser.stft_mask.denoise import denoise_signal
from metrics import plot_signal_comparison


def plot_example_signals(
    df_test_audio,
    label_encoder,
    scaler,
    sampling_rate,
    n_mfcc,
    max_len,
    cfg,
    plots_dir,
    device,
    idx_vis=0,
):
    """
    Plot example: clean vs noisy vs denoised vs enhanced
    Independent from main pipeline
    """

    # 🔥 create small noisy dataset JUST for noise access
    example_ds = NoisyTestDataset(
        audio_list=df_test_audio["audio_data"].values,
        labels_list=df_test_audio["label"].values,
        filenames_list=df_test_audio["filename"].values,
        label_encoder=label_encoder,
        scaler=scaler,
        sampling_rate=sampling_rate,
        n_mfcc=n_mfcc,
        max_len=max_len,
        bg_noise_dir=str(cfg["bg_noise_dir"]),
        noise_ops=list(cfg["noise_ops"]),
        snr_choices=list(cfg["snr_choices"]),
        seed=123,
        mode="noisy",
        device=device,
    )

    # =========================
    # CLEAN
    # =========================
    clean_sig = df_test_audio["audio_data"].iloc[idx_vis]
    filename = df_test_audio["filename"].iloc[idx_vis]
    label = df_test_audio["label"].iloc[idx_vis]

    # =========================
    # NOISY
    # =========================
    noise_name = example_ds._choose_noise_name()
    noise_arr = random.choice(example_ds.noise_bank[noise_name])

    snr_db = -10

    noisy_sig = mix_with_noise_at_snr(
        clean=clean_sig,
        noise=noise_arr,
        snr_db=snr_db
    )

    # =========================
    # DENOISED
    # =========================
    denoised_sig = denoise_signal(
        noisy_signal=noisy_sig,
        fs=sampling_rate,
        device=device
    )

    # =========================
    # ENHANCED (from disk)
    # =========================
    enhanced_path = os.path.join(
        "/home/dsi/skopavi/Project/kws_project/generated_enhanced",
        label,
        filename
    )

    enhanced_sig = None
    if os.path.exists(enhanced_path):
        enhanced_sig, _ = librosa.load(enhanced_path, sr=sampling_rate)
    else:
        print(f"⚠️ Enhanced file not found: {enhanced_path}")

    # =========================
    # PLOT
    # =========================
    plot_signal_comparison(
        clean=clean_sig,
        noisy=noisy_sig,
        denoised=denoised_sig,
        enhanced=enhanced_sig,
        fs=sampling_rate,
        title=f"file={filename} | noise={noise_name} | SNR={snr_db} dB",
        save_path=plots_dir / f"example_{filename}_snr{snr_db}.png"
    )