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

def plot_audio_and_features(
    audio_df,
    label_name: str,
    sampling_rate: int = 16000,
    n_mfcc: int = 13,
    mfcc_col: str = "mfcc",
    scaled_mfcc_col: str = "scaled_mfcc",
    random_example: bool = False,
):
    """
    Plot waveform, log spectrogram, MFCC, and chromagram for a single example.
    Optionally, also plot scaled (normalized) MFCC if available in the dataframe.

    Parameters
    ----------
    audio_df : pd.DataFrame
        Must contain columns: 'label', 'audio_data'.
        Optionally contains columns: mfcc_col, scaled_mfcc_col.
    label_name : str
        Class label (folder name) to select an example from.
    sampling_rate : int
        Sampling rate used to load audio.
    n_mfcc : int
        Number of MFCC coefficients (for re-computing MFCC if needed).
    mfcc_col : str
        Column name for MFCC features stored as (T, n_mfcc) or (n_mfcc, T).
    scaled_mfcc_col : str
        Column name for scaled MFCC features stored as (T, n_mfcc) or (n_mfcc, T).
    random_example : bool
        If True, pick a random example from the label. Otherwise pick the first.
    """
    # Select one sample
    df_label = audio_df[audio_df["label"] == label_name]
    if len(df_label) == 0:
        raise ValueError(f"No samples found for label='{label_name}'")

    if random_example:
        row = df_label.sample(1).iloc[0]
    else:
        row = df_label.iloc[0]

    audio_sample = row["audio_data"]

    # Compute spectrogram (in dB)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_sample)), ref=np.max)

    # MFCC (raw) 
    if mfcc_col in row and row[mfcc_col] is not None:
        mfcc_raw = row[mfcc_col]
        mfcc_raw = _to_mfcc_FxT(mfcc_raw)  # (n_mfcc, T)
    else:
        mfcc_raw = librosa.feature.mfcc(y=audio_sample, sr=sampling_rate, n_mfcc=n_mfcc)

    # MFCC (scaled / normalized) 
    mfcc_scaled = row[scaled_mfcc_col]
    mfcc_scaled = _to_mfcc_FxT(mfcc_scaled)  # (n_mfcc, T)

    # Chromagram 
    chroma = librosa.feature.chroma_stft(y=audio_sample, sr=sampling_rate)

    # Plot layout 
    fig, axes = plt.subplots(2, 2, figsize=(18, 7))

    # Waveform
    librosa.display.waveshow(audio_sample, sr=sampling_rate, ax=axes[0, 0])
    axes[0, 0].set_title("Waveform")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")

    # Spectrogram
    librosa.display.specshow(D, sr=sampling_rate, x_axis="time", y_axis="log", ax=axes[0, 1], cmap="viridis")
    axes[0, 1].set_title("Spectrogram (log freq)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")

    # MFCC (raw)
    img1 = librosa.display.specshow(mfcc_raw, x_axis="time", sr=sampling_rate, ax=axes[1, 0])
    axes[1, 0].set_title("MFCC (raw)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("MFCC index")
    fig.colorbar(img1, ax=axes[1, 0])

    # MFCC (_scaled)
    if mfcc_scaled is not None:
        img2 = librosa.display.specshow(mfcc_scaled, x_axis="time", sr=sampling_rate, ax=axes[1, 1])
        axes[1, 1].set_title("MFCC (scaled / normalized)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("MFCC index")
        fig.colorbar(img2, ax=axes[1, 1])
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


def _to_mfcc_FxT(mfcc):
    """
    Ensure MFCC is shaped as (n_mfcc, T) for librosa.display.specshow.
    Accepts input shaped (T, n_mfcc) or (n_mfcc, T).
    """
    mfcc = np.asarray(mfcc)
    if mfcc.ndim != 2:
        raise ValueError("MFCC must be a 2D array")

    # If it's (T, n_mfcc), transpose to (n_mfcc, T)
    if mfcc.shape[0] > mfcc.shape[1]:
        # Often T is larger than n_mfcc, so this is likely (T, n_mfcc)
        return mfcc.T
    return mfcc

