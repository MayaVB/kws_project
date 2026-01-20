import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import random

from features import plot_audio_and_features


def plot_history(history, title_prefix=""):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], "bo-", label="Training Loss")
    plt.plot(epochs_range, history["val_loss"],   "r*-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], "bo-", label="Training Accuracy")
    plt.plot(epochs_range, history["val_acc"],   "r*-", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def confusion_and_report(model, loader, class_names, device, model_name=""):
    """
    Build confusion matrix + classification report from a loader that returns (X,y,filename).
    Returns: cm, report_text, (all_true, all_pred, all_filenames)
    """
    model.eval()
    all_true, all_pred, all_files = [], [], []

    with torch.no_grad():
        for X_batch, y_batch, fn_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            all_true.extend(y_batch.numpy())
            all_pred.extend(preds.cpu().numpy())
            all_files.extend(list(fn_batch))

    cm = confusion_matrix(all_true, all_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.show()

    report = classification_report(all_true, all_pred, target_names=class_names)
    print("Classification Report:")
    print(report)

    return cm, report, (np.array(all_true), np.array(all_pred), np.array(all_files))

def plot_two_confusion_matrices(
    cm_left,
    cm_right,
    class_names,
    title_left="VAL",
    title_right="TEST",
    suptitle="Confusion Matrices Comparison"
):
    """
    Plot two confusion matrices side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm_left,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title(title_left)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_right,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title(title_right)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


# TODO
# add random choise of classes to misclassified

def pick_random_class_pair(class_names, seed=123):
    rng = random.Random(seed)
    a, b = rng.sample(list(class_names), 2)
    return a, b

def misclassified_between_two_classes(t, p, f, class_names, class_a, class_b, max_rows=20):
    """
    Show misclassifications A->B and B->A only.
    t,p are integer class indices, f filenames.
    """
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    ia = name_to_idx[class_a]
    ib = name_to_idx[class_b]

    t = np.asarray(t)
    p = np.asarray(p)
    f = np.asarray(f)

    mask_ab = (t == ia) & (p == ib)   # A predicted as B
    mask_ba = (t == ib) & (p == ia)   # B predicted as A

    rows = []
    for fn in f[mask_ab][:max_rows]:
        rows.append({"true": class_a, "pred": class_b, "filename": fn})
    for fn in f[mask_ba][:max_rows]:
        rows.append({"true": class_b, "pred": class_a, "filename": fn})

    return pd.DataFrame(rows)

def find_misclassified_files(all_true, all_pred, all_files,
                             class_names,
                             true_label_name: str,
                             pred_label_name: str):
    """
    Return a DataFrame listing filenames where:
      true == true_label_name AND pred == pred_label_name
    """
    true_idx = np.where(class_names == true_label_name)[0][0]
    pred_idx = np.where(class_names == pred_label_name)[0][0]

    mask = (all_true == true_idx) & (all_pred == pred_idx)
    mis_files = all_files[mask]

    df = pd.DataFrame({
        "filename": mis_files,
        "true_label": [true_label_name] * len(mis_files),
        "pred_label": [pred_label_name] * len(mis_files),
    })
    return df


# -----------------------------
# Helpers: single consistent split by indices
# -----------------------------
def make_split_indices(labels: np.ndarray,
                       train_ratio: float,
                       val_ratio: float,
                       test_ratio: float,
                       random_state: int):
    """
    Returns idx_train, idx_val, idx_test (numpy arrays) with stratified split.
    """
    from sklearn.model_selection import train_test_split

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1. Got {total}")

    all_idx = np.arange(len(labels))
    temp_ratio = val_ratio + test_ratio

    idx_train, idx_temp = train_test_split(
        all_idx,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=labels
    )

    val_fraction_of_temp = val_ratio / temp_ratio
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=random_state,
        stratify=labels[idx_temp]
    )

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)


def subset_by_idx(arr, idx):
    return np.array(arr, dtype=object)[idx]


def plot_one_from_clean_df_row(row, scaler, sampling_rate, n_mfcc, title_prefix=""):
    """
    row is a single-row DataFrame or a Series from clean_df/df_train/df_test
    Must include: audio_data, label, filename, mfcc
    """
    mfcc_raw = row["mfcc"]
    mfcc_scaled = scaler.transform(mfcc_raw).astype(np.float32)

    plot_df = row.to_frame().T.copy()  # single-row DF
    plot_df["scaled_mfcc"] = [mfcc_scaled]

    print(f"\nPLOT {title_prefix}: label={row['label']} filename={row['filename']} mfcc_raw={mfcc_raw.shape}")
    plot_audio_and_features(
        audio_df=plot_df,
        label_name=str(row["label"]),
        sampling_rate=sampling_rate,
        n_mfcc=n_mfcc,
        random_example=False
    )


import numpy as np
import pandas as pd

def plot_one_noisy_item(noisy_ds, idx, sampling_rate, n_mfcc, title_prefix=""):
    """
    Supports both:
      - NoisyTestDataset(return_audio=True): returns (X, y, fname, sig)
      - NoisyTestDataset(return_audio=False): returns (X, y, fname)
    Plots waveform/spectrogram from sig if available, and scaled MFCC from X.
    """

    out = noisy_ds[idx]

    if len(out) == 4:
        X_tensor, y_tensor, fname, sig_tensor = out
        sig = sig_tensor.numpy().astype(np.float32)
    elif len(out) == 3:
        X_tensor, y_tensor, fname = out
        # fallback waveform (so plot_audio_and_features won't crash)
        sig = np.zeros(int(sampling_rate), dtype=np.float32)
    else:
        raise ValueError(f"Unexpected noisy_ds[idx] output length={len(out)} (expected 3 or 4).")

    mfcc_sc_padded = X_tensor.squeeze(0).numpy().astype(np.float32)  # (Tmax, F)

    label_for_plot = "NOISY_SAMPLE"
    plot_df = pd.DataFrame([{
        "filename": fname,
        "label": label_for_plot,
        "audio_data": sig,               # real noisy audio if exists, else zeros
        "mfcc": None,
        "scaled_mfcc": mfcc_sc_padded,   # already scaled+padded
    }])

    print(
        f"\n{title_prefix} | idx={idx} | file={fname} | "
        f"audio_len={len(sig)} | mfcc_padded={mfcc_sc_padded.shape}"
    )

    plot_audio_and_features(
        audio_df=plot_df,
        label_name=label_for_plot,
        sampling_rate=sampling_rate,
        n_mfcc=n_mfcc,
        random_example=False
    )
