import numpy as np
from .stft_func import compute_stft_db, reconstruct_from_stft, apply_mask
from .spp_mask import compute_time_freq_spp
from .unet_denoise import denoise_signal_unet

def denoise_signal(
    noisy_signal,
    fs,
    n_fft=1024,
    hop=256,
    win="hamming",
    threshold=0.6,
    device="cpu",
    use_unet=False,
    unet_model=None,
):
    """
    noisy_signal -> denoised_signal
    """
    # UNET method 
    if use_unet:

        return denoise_signal_unet(
            noisy_signal,
            fs,
            model=unet_model,
            device=device,
            n_fft=n_fft,
            hop=hop,
            win=win
        )
    
    # SPP method
    f, t, Z_noisy, S_mag, _ = compute_stft_db(
        noisy_signal,
        fs,
        n_fft=n_fft,
        hop=hop,
        win=win
    )

    mask = compute_time_freq_spp(S_mag, threshold=threshold)

    Z_masked = apply_mask(Z_noisy, mask)

    x_rec = reconstruct_from_stft(
        Z_masked,
        fs,
        n_fft=n_fft,
        hop=hop,
        win=win
    )

    return x_rec.astype(np.float32)