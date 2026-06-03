"""
features.py

This module contains:
- MFCC extraction
- parallel MFCC processing
- feature normalization using StandardScaler
"""
import os
import numpy as np
import librosa

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

def extract_mfcc(audio_data: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features from an audio signal.

    Parameters
    ----------
    audio_data : np.ndarray
        Input audio waveform.
    sr : int
        Sampling rate.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    np.ndarray
        MFCC matrix with shape (T, n_mfcc).
    """
    return librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc).T

def add_mfcc_column(df, sr: int = 16000, n_mfcc: int = 13, use_parallel: bool = True):
    """
    Extract MFCC features from an audio signal.

    Parameters
    ----------
    audio_data : np.ndarray
        Input audio waveform.
    sr : int
        Sampling rate.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    np.ndarray
        MFCC matrix with shape (T, n_mfcc).
    """
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
    Fit a StandardScaler using all MFCC frames from the training set.

    All MFCC matrices are concatenated along the time axis
    before fitting the scaler.
    """
    # Stack all frames from all samples to estimate
    # global feature mean and variance
    mfcc_flat = np.concatenate(mfcc_list, axis=0)
    scaler = StandardScaler().fit(mfcc_flat)
    return scaler


def apply_scaler(mfcc_list, scaler):
    """
    Apply a fitted StandardScaler to each MFCC sample.

    Parameters
    ----------
    mfcc_list : list[np.ndarray]
        List of MFCC matrices.
    scaler : StandardScaler
        Previously fitted scaler.

    Returns
    -------
    list[np.ndarray]
        Scaled MFCC matrices.
    """
    scaled = []
    for m in mfcc_list:
        scaled.append(scaler.transform(m).astype(np.float32))
    # return np.array(scaled, dtype=object)
    return scaled

