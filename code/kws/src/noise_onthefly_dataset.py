# noise_dataset.py
import sys
import os
sys.path.append("/home/dsi/skopavi/Project/kws_project/sgmse_repo/sgmse")
sys.path.append("/home/dsi/skopavi/Project/kws_project/code")

import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from typing import List, Sequence, Dict
import random
from kws.denoiser.stft_mask.denoise import denoise_signal
from kws.denoiser.stft_mask.unet_model import UNetDenoiser

def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(x**2) + eps))

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


class NoisyTestDataset(Dataset):
    """
    Builds MFCC on-the-fly from CLEAN test audio, optionally adds random noise (and later denoise).
    Output is shaped like training: (1, T, F) with padding to max_len.
    Returns: X, y, filename
    """
    def __init__(
        self,
        audio_list: Sequence[np.ndarray],
        labels_list: Sequence[str],
        filenames_list: Sequence[str],
        label_encoder: LabelEncoder,
        scaler,
        sampling_rate: int,
        n_mfcc: int,
        max_len: int,
        bg_noise_dir: str,
        noise_ops: Sequence[str],
        snr_choices: Sequence[float],
        random_noise_start: bool = True,
        noise_if_short: str = "loop",
        seed: int = 123,
        mode: str = "noisy",   # "noisy" / "denoised" 
        return_audio: bool = True,
        device: str = "cpu",
    ):
        self.audio = list(audio_list)
        self.labels = list(labels_list)
        self.filenames = list(filenames_list)

        self.enc = label_encoder
        self.scaler = scaler
        self.sr = int(sampling_rate)
        self.n_mfcc = int(n_mfcc)
        self.max_len = int(max_len)

        self.bg_noise_dir = bg_noise_dir
        self.noise_ops = list(noise_ops)
        self.snr_choices = list(snr_choices)
        self.random_noise_start = bool(random_noise_start)
        self.noise_if_short = str(noise_if_short)
        self.mode = str(mode).lower()

        # reproducible randomness
        self.seed = int(seed)

        # load all noise wavs once (cache)
        self.noise_bank: Dict[str, List[np.ndarray]] = {}
        self._build_noise_bank()

        # pre-encode labels 
        self.y_enc = self.enc.transform(np.array(self.labels))

        self.return_audio = bool(return_audio)
        self.device = device

        import pandas as pd

        self.meta_dict = {}

        meta_path = "/home/dsi/skopavi/Project/kws_project/data/generated_noisy_metadata.csv"

        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path)
            self.meta_dict = {
                (row["label"], row["filename"]): row
                for _, row in df.iterrows()
            }

    def _build_noise_bank(self):
        # load all wav files in bg_noise_dir
        wavs = [f for f in os.listdir(self.bg_noise_dir) if f.lower().endswith(".wav")]
        if len(wavs) == 0:
            raise ValueError(f"No .wav found in bg_noise_dir: {self.bg_noise_dir}")

        for w in wavs:
            name = os.path.splitext(w)[0]  # without .wav
            path = os.path.join(self.bg_noise_dir, w)
            y, _ = librosa.load(path, sr=self.sr)
            y = y.astype(np.float32)

            if name not in self.noise_bank:
                self.noise_bank[name] = []
            self.noise_bank[name].append(y)

    def _choose_noise_name(self) -> str:
        ops = self.noise_ops
        if "all" in ops:
            # choose any available noise file name from bank
            keys = sorted(self.noise_bank.keys())
            return random.choice(keys)
        # else: choose among requested ops, but only those existing
        valid = [n for n in ops if n in self.noise_bank]
        if len(valid) == 0:
            raise ValueError(
                f"None of noise_ops exist in bg_noise_dir. "
                f"noise_ops={ops}, available={sorted(self.noise_bank.keys())}"
            )
        return random.choice(valid)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx: int):
        snr_db = None
        noise_name = None
        # make per-sample reproducible randomness (so reruns give same noisy test)
        np.random.seed(self.seed + idx)
        random.seed(self.seed + idx)

        clean = self.audio[idx]
        fname = self.filenames[idx]
        y = self.y_enc[idx]

        if self.mode == "clean":
            sig = clean.astype(np.float32)

        elif self.mode == "enhanced":
            label = self.labels[idx]

            enhanced_path = os.path.join(
                "/home/dsi/skopavi/Project/kws_project/data/enhanced_sgmse/generated_enhanced",
                label,
                fname
            )

            sig, _ = librosa.load(enhanced_path, sr=self.sr)
            sig = librosa.util.fix_length(sig, size=self.sr)
            sig = sig.astype(np.float32)

            key = (label, fname)
            if key in self.meta_dict:
                snr_db = self.meta_dict[key]["snr"]
                noise_name = self.meta_dict[key]["noise"]

            if key not in self.meta_dict:
                print(f"Missing metadata for {fname}")
                
        else:
            # pick noise + snr
            noise_name = self._choose_noise_name()
            noise_arr = random.choice(self.noise_bank[noise_name])
            snr_db = float(random.choice(self.snr_choices))

            sig = mix_with_noise_at_snr(
                clean=clean,
                noise=noise_arr,
                snr_db=snr_db,
                random_start=self.random_noise_start,
                noise_if_short=self.noise_if_short,
            )

            if self.mode == "denoised":
                sig = denoise_signal(
                noisy_signal=sig,
                fs=self.sr,
                n_fft=1024,
                hop=256,
                win="hamming",
                threshold=0.6,
                device=self.device,
            )
                
            # fix signal length
            target_len = self.sr   # 16000
            sig = librosa.util.fix_length(sig, size=self.sr)   
            sig = sig.astype(np.float32)   

            if len(sig) != len(clean):
                print("problem file:", fname, "len sig:", len(sig), "len clean:", len(clean))

        # MFCC 
        mfcc = librosa.feature.mfcc(y=sig, sr=self.sr, n_mfcc=self.n_mfcc).T  # (T, F)

        # scale with TRAIN scaler
        mfcc_sc = self.scaler.transform(mfcc).astype(np.float32)

        # pad to max_len
        T = mfcc_sc.shape[0]
        F = mfcc_sc.shape[1]
        Xp = np.zeros((self.max_len, F), dtype=np.float32)
        Xp[:min(T, self.max_len), :] = mfcc_sc[:self.max_len, :]

        X_tensor = torch.from_numpy(Xp).unsqueeze(0)  # (1, T, F)
        y_tensor = torch.tensor(y).long()

        if self.return_audio:
                sig_tensor = torch.from_numpy(sig.astype(np.float32))
                return X_tensor, y_tensor, fname, sig_tensor, snr_db, noise_name
        return X_tensor, y_tensor, fname
