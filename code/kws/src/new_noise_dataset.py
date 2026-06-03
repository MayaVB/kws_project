"""
new_noise_dataset.py

Dataset wrapper for evaluating keyword classification
on pre-generated noisy speech samples.

The dataset loads noisy audio files, extracts MFCC
features, applies normalization and returns optional
SNR/noise metadata for analysis.
"""
import os
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FixedNoisyDataset(Dataset):
    """
    Dataset for evaluating keyword classification
    on fixed noisy audio files.

    Each sample returns:

    - MFCC tensor
    - encoded label
    - filename
    - SNR value
    - noise type
    """
    def __init__(
        self,
        root,                 
        labels_list,
        filenames_list,
        label_encoder,
        scaler,
        sampling_rate,
        n_mfcc,
        max_len,
        meta_csv=None,
        split="test",       
    ):
        self.root = root
        self.labels = list(labels_list)
        self.filenames = list(filenames_list)
        self.le = label_encoder
        self.scaler = scaler
        self.sr = sampling_rate
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.split = split

        # BUILD FAST META LOOKUP
        self.meta_dict = {}

        if meta_csv is not None:
            meta_df = pd.read_csv(meta_csv)

            meta_df = meta_df[meta_df["split"] == self.split]

            for _, row in meta_df.iterrows():
                key = (row["filename"], row["label"])
                self.meta_dict[key] = (
                    float(row["snr"]),
                    str(row["noise"])
                )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Load a noisy speech sample and prepare it
        for DS-CNN inference.

        Returns
        -------
        (
            mfcc_tensor,
            encoded_label,
            filename,
            snr_db,
            noise_name
        )
        """
        fname = self.filenames[idx]
        label = self.labels[idx]

        path = os.path.join(self.root, label, fname)

        # Load and normalize audio length
        audio, _ = librosa.load(path, sr=self.sr)
        audio = librosa.util.fix_length(audio, size=self.sr)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc
        ).T

        mfcc = self.scaler.transform(mfcc)

        # Force all MFCC matrices to the same temporal length
        if mfcc.shape[0] < self.max_len:
            pad = np.zeros((self.max_len - mfcc.shape[0], mfcc.shape[1]))
            mfcc = np.vstack([mfcc, pad])
        else:
            mfcc = mfcc[:self.max_len]

        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
        y = self.le.transform([label])[0]

        # Retrieve SNR and noise information if available
        key = (fname, label)

        if key in self.meta_dict:
            snr, noise = self.meta_dict[key]
        else:
            snr, noise = None, None

        return x, y, fname, snr, noise