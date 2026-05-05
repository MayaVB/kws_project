# run_pipeline.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

from stft_func import load_wav_mono, compute_stft_db, reconstruct_from_stft
from spp_mask import compute_time_freq_spp


def save_waveform_png(x, fs, out_png, title):
    """Save a time-domain waveform plot."""
    x = np.asarray(x)
    t = np.arange(len(x)) / float(fs)

    plt.figure()
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_spectrogram_png(f, t, S_db, out_png, title, fs=None):
    """Save a dB spectrogram plot."""
    plt.figure()
    plt.pcolormesh(t, f, S_db, shading="auto")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.colorbar(label="Magnitude [dB]")
    if fs is not None:
        plt.ylim(0, fs / 2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_mask_png(f, t, mask, out_png, title, fs=None):
    """Save a time-frequency mask plot (0/1)."""
    plt.figure()
    plt.pcolormesh(t, f, mask, shading="auto")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.colorbar(label="Mask")
    if fs is not None:
        plt.ylim(0, fs / 2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def align_tf(A, B):
    """
    Align two STFT-like arrays by cropping to the minimum (F,T).
    Returns cropped_A, cropped_B, F_min, T_min
    """
    F = min(A.shape[0], B.shape[0])
    T = min(A.shape[1], B.shape[1])
    return A[:F, :T], B[:F, :T], F, T


def main():
    clean_wav_path = r"C:\Users\idobe\Ido_code\Data\archive\eight\0a2b400e_nohash_0.wav"
    noisy_wav_path = r"C:\Users\idobe\Ido_code\temp_git\project_spp\generated_noisy\noisy_eight_N1_skip0_doing_the_dishes_snr-20dB\eight\0a2b400e_nohash_0__noisy__doing_the_dishes__snr-20.0dB.wav"
    # Output folder: next to the noisy file (easy to find)
    noisy_path = Path(noisy_wav_path)
    out_dir = noisy_path.parent / "pipeline_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # STFT params (must match between STFT and iSTFT)
    n_fft = 1024
    hop = 256
    win = "hamming"

    # -------------------------
    # 1) Load signals
    # -------------------------
    x_clean, fs = load_wav_mono(clean_wav_path)
    x_noisy, fs_noisy = load_wav_mono(noisy_wav_path)
    if fs != fs_noisy:
        raise ValueError(f"Sample rate mismatch: clean fs={fs}, noisy fs={fs_noisy}")

    # -------------------------
    # 2) Save clean waveform (Figure 1)
    # -------------------------
    save_waveform_png(
        x_clean, fs,
        out_dir / "1_clean_waveform.png",
        title="Clean waveform"
    )

    # -------------------------
    # 3) STFT of clean + save spectrogram (Figure 2)
    # -------------------------
    f_c, t_c, Z_clean, S_clean_mag, S_clean_db = compute_stft_db(
        x_clean, fs, n_fft=n_fft, hop=hop, win=win
    )
    save_spectrogram_png(
        f_c, t_c, S_clean_db,
        out_dir / "2_clean_stft_db.png",
        title="Clean STFT (dB)",
        fs=fs
    )

    # -------------------------
    # 4) Compute mask (from clean magnitude) + save mask (Figure 3)
    #   mask function currently expects (F,T,1), so we wrap it.
    # -------------------------
    S_clean_mag_3d = S_clean_mag[:, :, np.newaxis]  # (F,T,1)
    vad_mask_3d = compute_time_freq_spp(S_clean_mag_3d, threshold=0.6, low_freq_emphasis=True)
    mask = vad_mask_3d[:, :, 0].astype(np.float32)  # (F,T)

    save_mask_png(
        f_c, t_c, mask,
        out_dir / "3_mask.png",
        title="Time-Frequency Mask (from clean)",
        fs=fs
    )

    # -------------------------
    # 5) Save noisy waveform (Figure 4)
    # -------------------------
    save_waveform_png(
        x_noisy, fs,
        out_dir / "4_noisy_waveform.png",
        title="Noisy waveform"
    )

    # -------------------------
    # 6) STFT of noisy + save spectrogram (Figure 5)
    # -------------------------
    f_n, t_n, Z_noisy, S_noisy_mag, S_noisy_db = compute_stft_db(
        x_noisy, fs, n_fft=n_fft, hop=hop, win=win
    )

    save_spectrogram_png(
        f_n, t_n, S_noisy_db,
        out_dir / "5_noisy_stft_db.png",
        title="Noisy STFT (dB)",
        fs=fs
    )

    # -------------------------
    # 7) Align mask to noisy STFT if needed (crop to min F,T)
    # -------------------------
    Z_noisy_aligned, mask_aligned, Fmin, Tmin = align_tf(Z_noisy, mask)
    f = f_n[:Fmin]
    t = t_n[:Tmin]

    # -------------------------
    # 8) Apply mask element-wise on COMPLEX noisy STFT
    # -------------------------
    Z_masked = Z_noisy_aligned * mask_aligned  # element-wise

    # Save spectrogram after masking (Figure 6)
    S_masked_db = 20.0 * np.log10(np.abs(Z_masked) + 1e-12)
    save_spectrogram_png(
        f, t, S_masked_db,
        out_dir / "6_masked_noisy_stft_db.png",
        title="Masked Noisy STFT (dB)",
        fs=fs
    )

    # -------------------------
    # 9) Reconstruct time-domain signal from masked STFT (iSTFT)
    # -------------------------
    x_rec = reconstruct_from_stft(Z_masked, fs, n_fft=n_fft, hop=hop, win=win)

    # Normalize to prevent clipping (optional but safe)
    peak = np.max(np.abs(x_rec)) + 1e-12
    if peak > 1.0:
        x_rec = x_rec / peak

    # Save reconstructed waveform (Figure 7)
    save_waveform_png(
        x_rec, fs,
        out_dir / "7_reconstructed_waveform.png",
        title="Reconstructed waveform (after masking)"
    )

    # -------------------------
    # 10) Save reconstructed WAV
    # -------------------------
    out_wav = noisy_path.with_name(noisy_path.stem + "__constructed.wav")
    sf.write(out_wav, x_rec.astype(np.float32), fs)

    print("Saved outputs to:", out_dir)
    print("Saved reconstructed WAV:", out_wav)


if __name__ == "__main__":
    main()
