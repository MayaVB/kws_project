import torch
import numpy as np
from .stft_func import compute_stft_db, reconstruct_from_stft


def denoise_signal_unet(
    noisy_signal,
    fs,
    model,
    device="cpu",
    n_fft=1024,
    hop=256,
    win="hamming",
):

    f, t, Z_noisy, S_mag, _ = compute_stft_db(
        noisy_signal,
        fs,
        n_fft=n_fft,
        hop=hop,
        win=win
    )

    # remove last freq bin to make shape divisible by 2 (513->512)
    S_mag = S_mag[:-1, :]
    Z_noisy = Z_noisy[:-1, :]
    
    spec = torch.tensor(S_mag, dtype=torch.float32)

    spec = spec.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = model(spec)

    mask = mask.squeeze().cpu().numpy()

    Z_masked = Z_noisy * mask

    x_rec = reconstruct_from_stft(
        Z_masked,
        fs,
        n_fft=n_fft,
        hop=hop,
        win=win
    )

    return x_rec.astype(np.float32)