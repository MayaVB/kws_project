# enhanced_dataset.py
import os
import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
import pandas as pd

class EnhancedTestDataset(Dataset):
    def __init__(
        self,
        labels_list,
        filenames_list,
        label_encoder,
        scaler,
        sampling_rate,
        n_mfcc,
        max_len,
        enhanced_root,
        meta_csv=None,
    ):
        self.labels = list(labels_list)
        self.filenames = list(filenames_list)

        self.enc = label_encoder
        self.scaler = scaler
        self.sr = sampling_rate
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.enhanced_root = enhanced_root

        self.y_enc = self.enc.transform(self.labels)

        # optional metadata
        self.meta_dict = {}
        if meta_csv and os.path.exists(meta_csv):
            df = pd.read_csv(meta_csv)
            self.meta_dict = {
                (row["label"], row["filename"]): row
                for _, row in df.iterrows()
            }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        fname = self.filenames[idx]

        path = os.path.join(self.enhanced_root, label, fname)

        sig, _ = librosa.load(path, sr=self.sr)
        sig = librosa.util.fix_length(sig, size=self.sr)

        # MFCC
        mfcc = librosa.feature.mfcc(y=sig, sr=self.sr, n_mfcc=self.n_mfcc).T
        mfcc_sc = self.scaler.transform(mfcc).astype(np.float32)

        # pad
        T, F = mfcc_sc.shape
        Xp = np.zeros((self.max_len, F), dtype=np.float32)
        Xp[:min(T, self.max_len)] = mfcc_sc[:self.max_len]

        X_tensor = torch.from_numpy(Xp).unsqueeze(0)
        y_tensor = torch.tensor(self.y_enc[idx]).long()

        # metadata
        key = (label, fname)
        # print("example meta key:", list(self.meta_dict.keys())[0])
        
        snr_db = None
        noise_name = None

        if key in self.meta_dict:
            snr_db = self.meta_dict[key]["snr"]
            noise_name = self.meta_dict[key]["noise"]

        sig_tensor = torch.from_numpy(sig.astype(np.float32))

        return X_tensor, y_tensor, fname, snr_db, noise_name, sig_tensor