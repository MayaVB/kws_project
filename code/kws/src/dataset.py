# dataset.py
import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import multiprocessing.dummy as mp 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader


def list_folders(main_dir: str):
    """Return directory names under main_dir."""
    folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]
    folders = [f for f in folders if f != "_background_noise_"]
    folders = sorted(folders)  
    return folders

def collect_wav_paths(main_dir: str, subset_folders):
    """Collect full paths to .wav files from the selected folders."""
    file_paths = []
    for folder in subset_folders:
        folder_dir = os.path.join(main_dir, folder)
        for fname in os.listdir(folder_dir):
            if fname.lower().endswith(".wav"):
                file_paths.append(os.path.join(folder_dir, fname))
    return file_paths


def load_audio_file(file_path: str, sampling_rate: int = 16000):
    """
    Load one WAV file and return:
    - filename (basename)
    - label (parent folder name)
    - audio array (y)
    # - full path
    """
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    audio = fix_audio_length(audio, sampling_rate)  # ensure 1 second length
    label = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)
    return filename, label, audio

def fix_audio_length(audio, target_len=16000):
    """
    Ensure audio has length exactly target_len (e.g. 1 second at 16kHz).
    If too long, truncate. If too short, pad with zeros at the end.
    """
    if len(audio) > target_len:
        audio = audio[:target_len]

    elif len(audio) < target_len:
        pad = target_len - len(audio)
        audio = np.pad(audio, (0, pad))

    return audio

def build_audio_dataframe(file_paths, sampling_rate: int = 16000, use_parallel: bool = True):
    """
    Build a DataFrame with columns:
    filename, label, audio_data, full_path

    If use_parallel=True, loads audio using a thread pool for speed.
    """
    if use_parallel:
        with mp.Pool(os.cpu_count()) as pool:
            rows = list(
                tqdm(
                    pool.imap(lambda p: load_audio_file(p, sampling_rate), file_paths),
                    total=len(file_paths),
                    desc="Loading WAV files"
                )
            )
    else:
        rows = [load_audio_file(p, sampling_rate) for p in tqdm(file_paths, desc="Loading WAV files")]

    df = pd.DataFrame(rows, columns=["filename", "label", "audio_data"])
    return df


def split_train_val_test(X, y, filenames,
                         train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                         random_state=42):
    """
    Split X,y,filenames into train/val/test using given ratios.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1. Got {total}")
    # split train vs temp
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp, fn_train, fn_temp = train_test_split(
        X, y, filenames,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y
    )

    # split temp into val/test
    # val is a fraction of temp
    val_fraction_of_temp = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test, fn_val, fn_test = train_test_split(
        X_temp, y_temp, fn_temp,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=random_state,
        stratify=y_temp
    )

    return (X_train, y_train, fn_train), (X_val, y_val, fn_val), (X_test, y_test, fn_test)


def pad_mfcc_list(mfcc_list, max_len=None):
    """
    Pad a list of (T_i, F) arrays into a single (N, T_max, F) float32 array.
    """
    if max_len is None:
        max_len = max(m.shape[0] for m in mfcc_list)
    # If no length provided, use the longest sequence
    n_samples = len(mfcc_list) 
    n_features = mfcc_list[0].shape[1]
    X_padded = np.zeros((n_samples, max_len, n_features), dtype=np.float32)
    # Copy each MFCC matrix into padded container
    for i, m in enumerate(mfcc_list):
        length = m.shape[0]
        X_padded[i, :length, :] = m

    return X_padded


class MFCCDataset(Dataset):
    """
    Holds padded MFCC tensors of shape:
      X: (N, 1, T, F)
      y: (N,)
    Also keeps filenames so we can find misclassified files later.
    """
    def __init__(self, X_padded: np.ndarray, y_encoded: np.ndarray, filenames):
        self.X = torch.from_numpy(X_padded).unsqueeze(1)  # (N, 1, T, F)
        self.y = torch.from_numpy(y_encoded).long()
        self.filenames = list(filenames)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.filenames[idx]


def make_label_encoder(y_train_labels, y_val_labels, y_test_labels):
    """
    Fit encoder on train labels, transform val/test with same mapping.
    """
    enc = LabelEncoder()
    y_train_enc = enc.fit_transform(y_train_labels)
    y_val_enc = enc.transform(y_val_labels)
    y_test_enc = enc.transform(y_test_labels)
    return enc, y_train_enc, y_val_enc, y_test_enc


def make_loaders(
    X_train, y_train, fn_train,
    X_val, y_val, fn_val,
    X_test, y_test, fn_test,
    batch_size=64,
    num_workers=0,
    shuffle_train=True
):
    """
    Create DataLoaders.

    Parameters
    ----------
    batch_size : int
        Number of samples per batch.
    num_workers : int
        How many subprocesses to use for data loading.
    shuffle_train : bool
        Whether to shuffle training data.
    Notes
    -----
    Validation and test loaders are never shuffled
    to keep filename order deterministic.
    """
    train_ds = MFCCDataset(X_train, y_train, fn_train)
    val_ds   = MFCCDataset(X_val,   y_val,   fn_val)
    test_ds  = MFCCDataset(X_test,  y_test,  fn_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

def make_test_loader(
    X_test,
    y_test,
    fn_test,
    batch_size=64,
    num_workers=0
):
    """
    Create ONLY test DataLoader.
    Same behavior as make_loaders().
    """

    test_ds = MFCCDataset(
        X_test,
        y_test,
        fn_test
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader