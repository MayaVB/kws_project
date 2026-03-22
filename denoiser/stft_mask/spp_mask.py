import numpy as np

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
    if low_freq_emphasis:
        F = energy.shape[0]
        bin_idx = np.arange(F)
        low_freq_weight = np.exp(-bin_idx / 50.0)[:, None]  # (F,1)
        energy = energy * low_freq_weight

    # Mean energy per frequency bin
    mean_energy = np.mean(energy, axis=1, keepdims=True)  # (F,1)

    # Time-frequency VAD mask
    vad_mask = energy > (threshold * mean_energy)

    return vad_mask.astype(np.float32)
