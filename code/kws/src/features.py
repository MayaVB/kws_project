# features.py
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

def extract_mfcc(audio_data: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Return MFCC as (T, n_mfcc) to match the notebook convention.
    """
    return librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc).T

def add_mfcc_column(df, sr: int = 16000, n_mfcc: int = 13, use_parallel: bool = True):
    audio_list = df["audio_data"].tolist()

    if use_parallel:
        with ThreadPool(os.cpu_count()) as pool:
            fn = partial(extract_mfcc, sr=sr, n_mfcc=n_mfcc)  
            mfcc_list = list(tqdm(pool.imap(fn, audio_list),
                                  total=len(audio_list),
                                  desc="Extracting MFCCs",
                                  leave=True))
    else:
        mfcc_list = [
            extract_mfcc(a, sr, n_mfcc)
            for a in tqdm(audio_list, desc="Extracting MFCCs")
        ]

    df["mfcc"] = mfcc_list
    return df


def fit_scaler(mfcc_list):
    """
    Fit StandardScaler on the concatenation of all frames from all samples:
      concat over time -> big matrix of shape (sum_T, F)
    """
    mfcc_flat = np.concatenate(mfcc_list, axis=0)
    scaler = StandardScaler().fit(mfcc_flat)
    return scaler


def apply_scaler(mfcc_list, scaler):
    """
    Apply the scaler per sample (frame-wise).
    Returns list of scaled MFCC arrays (T, F) float32.
    """
    scaled = []
    for m in mfcc_list:
        scaled.append(scaler.transform(m).astype(np.float32))
    # return np.array(scaled, dtype=object)
    return scaled

