# noise_utils.py
import numpy as np

def mix_with_noise_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    random_start: bool = True,
    noise_if_short: str = "loop",
) -> np.ndarray:
    """
    Mix clean + noise at a desired SNR (dB).
    clean/noise are 1D float arrays.
    Returns noisy signal length == len(clean).
    """
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)

    L = len(clean)
    if len(noise) >= L:
        if random_start:
            start = np.random.randint(0, len(noise) - L + 1)
        else:
            start = 0
        n = noise[start:start+L]
    else:
        # noise shorter than clean
        if noise_if_short == "loop":
            reps = int(np.ceil(L / len(noise)))
            n = np.tile(noise, reps)[:L]
        elif noise_if_short == "pad_zeros":
            n = np.zeros(L, dtype=np.float32)
            n[:len(noise)] = noise
        else:
            raise ValueError(f"Unknown noise_if_short: {noise_if_short}")

    Ps = np.mean(clean**2) + 1e-12
    Pn = np.mean(n**2) + 1e-12

    # scale noise to achieve SNR
    # snr_db = 10*log10(Ps / Pn_scaled)
    # => Pn_scaled = Ps / 10^(snr/10)
    desired_Pn = Ps / (10 ** (snr_db / 10.0))
    scale = np.sqrt(desired_Pn / Pn)
    n_scaled = n * scale

    y = clean + n_scaled
    # optional clip (keeps things safe)
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32)
