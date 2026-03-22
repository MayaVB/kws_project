import numpy as np
import soundfile as sf
from scipy.signal import stft, istft
import matplotlib.pyplot as plt

def load_wav_mono(wav_path: str):
    x, fs = sf.read(wav_path)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x, fs

def compute_stft_db(
    x: np.ndarray,
    fs: int,
    n_fft: int = 1024,
    hop: int = 256,
    win: str = "hamming",
    eps: float = 1e-12,
):
    """
    Returns:
      f (Hz), t (sec), Zxx (complex), S_mag (abs), S_db (dB)
    """
    f, t, Zxx = stft(
        x, fs=fs,
        window=win,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
        boundary=None,
        padded=False
    )
    S_mag = np.abs(Zxx)
    S_db = 20 * np.log10(S_mag + eps)
    return f, t, Zxx, S_mag, S_db

def plot_spectrogram_db(f, t, S_db, fs=None, title="STFT Spectrogram (dB)"):
    plt.figure()
    plt.pcolormesh(t, f, S_db, shading="auto")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.title(title)
    plt.colorbar(label="Magnitude [dB]")
    if fs is not None:
        plt.ylim(0, fs / 2)
    plt.tight_layout()
    plt.show()

def reconstruct_from_stft(
    Zxx,
    fs,
    n_fft=1024,
    hop=256,
    win="hamming"
):
    """
    Reconstruct a time-domain signal from a complex STFT.

    Parameters:
        Zxx : ndarray (F, T)
            Complex STFT (after masking or processing).
        fs : int
            Sampling frequency [Hz].
        n_fft : int
            FFT size used in STFT.
        hop : int
            Hop size used in STFT.
        win : str or array
            Window type (must match STFT).

    Returns:
        x_rec : ndarray
            Reconstructed time-domain signal.
    """
    noverlap = n_fft - hop

    _, x_rec = istft(
        Zxx,
        fs=fs,
        window=win,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        input_onesided=True,
        boundary=False
    )

    return x_rec

def apply_mask(Z_noisy_complex, mask):
    """Element-wise mask on complex STFT."""
    F = min(Z_noisy_complex.shape[0], mask.shape[0])
    T = min(Z_noisy_complex.shape[1], mask.shape[1])
    return Z_noisy_complex[:F, :T] * mask[:F, :T]
