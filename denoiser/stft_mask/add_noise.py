from pathlib import Path
import numpy as np
import soundfile as sf

# =========================
# USER CONTROLS
# =========================
dataset_root = Path(r"C:\Users\idobe\Ido_code\Data\archive")
specific_file_name = r"C:\Users\idobe\Ido_code\Data\archive\eight\0a2b400e_nohash_0.wav"

word = "eight"
noise_file = "doing_the_dishes.wav"
snr_db = -20.0
target_duration_sec = 1.0
noise_offset_samples = 100
# Selection controls (NO sorting, uses filesystem order)
N = 1          # how many files to take
J = 0           # skip first J files, then take next N

# Output folder
run_name = f"noisy_{word}_N{N}_skip{J}_{Path(noise_file).stem}_snr{snr_db:g}dB"
out_root = Path.cwd() / "generated_noisy" / run_name
# =========================

def rms(x, eps=1e-12):
    return np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)

def ensure_len(x: np.ndarray, L: int):
    """Crop or zero-pad to length L."""
    x = x.astype(np.float32)
    if len(x) >= L:
        return x[:L]
    y = np.zeros(L, dtype=np.float32)
    y[:len(x)] = x
    return y

def get_noise_segment(noise_full: np.ndarray, L: int, start: int):
    """
    Take exactly L samples from noise starting at 'start'.
    If segment reaches the end, wrap around (circular).
    """
    n = noise_full.astype(np.float32)
    if len(n) == 0:
        return np.zeros(L, dtype=np.float32)

    start = start % len(n)
    if start + L <= len(n):
        return n[start:start + L]

    first = n[start:]
    remain = L - len(first)
    reps = int(np.ceil(remain / len(n)))
    tail = np.tile(n, reps)[:remain]
    return np.concatenate([first, tail]).astype(np.float32)

def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db_val: float):
    """Scale noise to achieve desired SNR (dB) w.r.t. speech RMS."""
    s = speech.astype(np.float32)
    n = noise.astype(np.float32)

    s_rms = rms(s)
    n_rms = rms(n)

    desired_n_rms = s_rms / (10 ** (snr_db_val / 20.0))
    scale = desired_n_rms / (n_rms + 1e-12)

    y = s + n * scale

    """peak = np.max(np.abs(y)) + 1e-12
    if peak > 0.99:
        y = y / peak * 0.99"""
    return y

def main():
    word_dir = dataset_root / word
    noise_path = dataset_root / "_background_noise_" / noise_file

    """if not word_dir.exists():
        raise RuntimeError(f"Word folder not found: {word_dir}")"""
    if not noise_path.exists():
        raise RuntimeError(f"Noise file not found: {noise_path}")

    # IMPORTANT: no sorting here (order is filesystem-dependent)
    all_files = [p for p in word_dir.rglob("*.wav") if p.is_file()]

    if len(all_files) == 0:
        raise RuntimeError(f"No wav files found in: {word_dir}")

    start_idx = max(0, J)
    end_idx = min(len(all_files), start_idx + max(0, N))
    #chosen = all_files[start_idx:end_idx]
    chosen = [Path(specific_file_name)]  

    if len(chosen) == 0:
        raise RuntimeError(f"Selection is empty. J={J}, N={N}, total={len(all_files)}")

    # Load noise once
    noise_full, sr_n = sf.read(noise_path, dtype="float32")
    if noise_full.ndim > 1:
        noise_full = noise_full.mean(axis=1)

    target_len = int(sr_n * target_duration_sec)

    # Create output folder
    out_word_dir = out_root / word
    out_word_dir.mkdir(parents=True, exist_ok=True)

    for i, sp_path in enumerate(chosen):
        speech, sr_s = sf.read(sp_path, dtype="float32")
        if speech.ndim > 1:
            speech = speech.mean(axis=1)

        if sr_s != sr_n:
            raise RuntimeError(f"Sample-rate mismatch: speech {sr_s}, noise {sr_n}")

        # --- choose length based on speech length ---
        L = len(speech)  # exact length of the wav (after mono)
        speech_L = speech.astype(np.float32)

        # Noise chunk selection: start + optional offset
        noise_start = noise_offset_samples
        noise_L = get_noise_segment(noise_full, L, start=noise_start)

        mixed = mix_at_snr(speech_L, noise_L, snr_db)

        out_name = f"{sp_path.stem}__noisy__{Path(noise_file).stem}__snr{snr_db:.1f}dB.wav"
        sf.write(out_word_dir / out_name, mixed, sr_s)
        print("Wrote:", out_name)


    print("\nDone.")
    print("Saved under:", out_word_dir)
    print(f"Selected files: {len(chosen)} (from index {start_idx} to {end_idx-1})")

if __name__ == "__main__":
    main()
