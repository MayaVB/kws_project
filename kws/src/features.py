import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
import multiprocessing.dummy as mp
import os
from sklearn.preprocessing import StandardScaler

def extract_mfcc(audio_data: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Return MFCC as (T, n_mfcc) to match the notebook convention.
    """
    return librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc).T

def add_mfcc_column(df, sr: int = 16000, n_mfcc: int = 13, use_parallel: bool = True):
    audio_list = df["audio_data"].tolist()

    if use_parallel:
        tasks = [(a, sr, n_mfcc) for a in audio_list]
        with mp.Pool(os.cpu_count()) as pool:
            mfcc_list = list(tqdm(
                pool.starmap(extract_mfcc, tasks),
                total=len(tasks),
                desc="Extracting MFCCs"
            ))
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
    return np.array(scaled, dtype=object)

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

    # MFCC (scaled / normalized) if exists 
    mfcc_scaled = None
    if scaled_mfcc_col in row and row[scaled_mfcc_col] is not None:
        mfcc_scaled = row[scaled_mfcc_col]
        mfcc_scaled = _to_mfcc_FxT(mfcc_scaled)  # (n_mfcc, T)

    # Chromagram 
    chroma = librosa.feature.chroma_stft(y=audio_sample, sr=sampling_rate)

    # Plot layout 
    # If we have scaled MFCC, make 2 rows x 3 cols; otherwise 2x2 as before
    if mfcc_scaled is None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))

        # 1) Waveform
        librosa.display.waveshow(audio_sample, sr=sampling_rate, ax=axes[0, 0])
        axes[0, 0].set_title("Waveform")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")

        # 2) Spectrogram
        librosa.display.specshow(D, sr=sampling_rate, x_axis="time", y_axis="log", ax=axes[0, 1], cmap="viridis")
        axes[0, 1].set_title("Spectrogram (log freq)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Frequency (Hz)")

        # 3) MFCC (raw)
        img = librosa.display.specshow(mfcc_raw, x_axis="time", sr=sampling_rate, ax=axes[1, 0])
        axes[1, 0].set_title("MFCC (raw)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("MFCC index")
        fig.colorbar(img, ax=axes[1, 0])

        # 4) Chromagram
        img2 = librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", sr=sampling_rate, ax=axes[1, 1])
        axes[1, 1].set_title("Chromagram")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Pitch class")
        fig.colorbar(img2, ax=axes[1, 1])

        plt.tight_layout()
        plt.show()

    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 7))

        # 1) Waveform
        librosa.display.waveshow(audio_sample, sr=sampling_rate, ax=axes[0, 0])
        axes[0, 0].set_title("Waveform")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")

        # 2) Spectrogram
        librosa.display.specshow(D, sr=sampling_rate, x_axis="time", y_axis="log", ax=axes[0, 1], cmap="viridis")
        axes[0, 1].set_title("Spectrogram (log freq)")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Frequency (Hz)")

        # 3) Chromagram
        img_ch = librosa.display.specshow(chroma, y_axis="chroma", x_axis="time", sr=sampling_rate, ax=axes[0, 2])
        axes[0, 2].set_title("Chromagram")
        axes[0, 2].set_xlabel("Time (s)")
        axes[0, 2].set_ylabel("Pitch class")
        fig.colorbar(img_ch, ax=axes[0, 2])

        # 4) MFCC (raw)
        img1 = librosa.display.specshow(mfcc_raw, x_axis="time", sr=sampling_rate, ax=axes[1, 0])
        axes[1, 0].set_title("MFCC (raw)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("MFCC index")
        fig.colorbar(img1, ax=axes[1, 0])

        # 5) MFCC (scaled)
        img2 = librosa.display.specshow(mfcc_scaled, x_axis="time", sr=sampling_rate, ax=axes[1, 1])
        axes[1, 1].set_title("MFCC (scaled / normalized)")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("MFCC index")
        fig.colorbar(img2, ax=axes[1, 1])

        # 6) Difference (optional view)
        diff = mfcc_scaled - mfcc_raw
        img3 = librosa.display.specshow(diff, x_axis="time", sr=sampling_rate, ax=axes[1, 2])
        axes[1, 2].set_title("Scaled - Raw (difference)")
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].set_ylabel("MFCC index")
        fig.colorbar(img3, ax=axes[1, 2])

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

