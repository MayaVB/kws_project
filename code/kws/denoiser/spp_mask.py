import numpy as np
from scipy.ndimage import gaussian_filter

def compute_time_freq_spp(stft, threshold=0.5, low_freq_emphasis=True):
    """
    Input:
        stft : 2D magnitude spectrogram, shape (F, T)

    Output:
        vad_mask : boolean mask, shape (F, T)
    """
    # Energy
    energy = np.abs(stft) ** 2   # (F, T)

    # Optional low-frequency emphasis
    """
    if low_freq_emphasis:
        F = energy.shape[0]
        bin_idx = np.arange(F)
        low_freq_weight = np.exp(-bin_idx / 30.0)[:, None]  # (F,1)
        energy = energy * low_freq_weight
    """

    # Mean energy per frequency bin
    # mean_energy = np.mean(energy, axis=1, keepdims=True)  # (F,1)
    mean_energy = np.median(energy, axis=1, keepdims=True)

    # Time-frequency VAD mask
    # vad_mask = energy > (threshold * mean_energy)

    # soft mask version instead of binary mask (not used currently)
    ratio = energy / (mean_energy + 1e-12)
    # mask = ratio / (ratio + threshold)

    mask = energy / (energy + threshold * mean_energy)
    mask = np.clip(mask, 0, 1)

    # Optional smoothing (e.g. Gaussian filter)
    # vad_mask = gaussian_filter(vad_mask.astype(float), sigma=1)
    vad_mask = gaussian_filter( mask.astype(float),sigma=(0.5, 1.0))

    return vad_mask.astype(np.float32)
