# add_noise.py
import numpy as np

def rms(x, eps=1e-12):
    return np.sqrt(np.mean(x.astype(np.float64)**2) + eps)

def mix_at_snr(speech, noise, snr_db):
    s_rms = rms(speech)
    n_rms = rms(noise)

    desired_noise_rms = s_rms / (10**(snr_db/20))
    scale = desired_noise_rms / (n_rms + 1e-12)

    return speech + noise * scale

def add_noise(speech, noise_full, snr_db, rng):
    L = len(speech)

    start = rng.integers(0, max(1, len(noise_full) - L))
    noise = noise_full[start:start+L]

    if len(noise) < L:
        noise = np.pad(noise, (0, L-len(noise)))

    return mix_at_snr(speech, noise, snr_db)