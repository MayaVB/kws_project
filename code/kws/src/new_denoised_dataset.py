# new_denoised_dataset.py

import librosa
import numpy as np
import torch
import os
from torch.utils.data import Dataset

from kws.denoiser.stft_mask.denoise import denoise_signal

class DenoisedDataset(Dataset):
    def __init__(self, base_dataset, sampling_rate, n_mfcc, scaler, max_len, root):
        self.base = base_dataset
        self.sr = sampling_rate
        self.n_mfcc = n_mfcc
        self.scaler = scaler
        self.max_len = max_len
        self.root = root  

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, fname, snr, noise = self.base[idx]

        label = self.base.labels[idx]

        path = os.path.join(self.root, label, fname)

        audio, _ = librosa.load(path, sr=self.sr)

        audio = librosa.util.fix_length(audio, size=self.sr)

        # denoise
        denoised = denoise_signal(audio, self.sr)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=denoised,
            sr=self.sr,
            n_mfcc=self.n_mfcc
        ).T

        mfcc = self.scaler.transform(mfcc)

        # pad
        if mfcc.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:self.max_len]

        return (
            torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0),
            y,
            fname,
            snr,
            noise
        )