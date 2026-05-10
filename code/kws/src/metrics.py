# metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import random
from features import plot_audio_and_features
from scipy.signal import stft
import os
import shutil
import subprocess
import soundfile as sf
import sys



def plot_history(history, save_path=None):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], "bo-", label="Training Loss")
    plt.plot(epochs_range, history["val_loss"],   "r*-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], "bo-", label="Training Accuracy")
    plt.plot(epochs_range, history["val_acc"],   "r*-", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def confusion_and_report(model, loader, class_names, device, model_name="",
                        return_meta=False, save_path=None, save_txt_path=None):
    """
    Build confusion matrix + classification report from a loader that returns (X,y,filename).
    Returns: cm, report_text, (all_true, all_pred, all_filenames)
    """
    model.eval()
    all_true, all_pred, all_files = [], [], []

    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0]
            y_batch = batch[1]
            fn_batch = batch[2]
            
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    report = classification_report(all_true, all_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
    if save_txt_path:
        with open(save_txt_path, "a") as f:
            f.write(f"\n{model_name.upper()}\n")
            f.write(report)
            f.write("\n")

    return cm, report, (np.array(all_true), np.array(all_pred), np.array(all_files))

def plot_two_confusion_matrices(
    cm_left,
    cm_right,
    class_names,
    title_left="VAL",
    title_right="TEST",
    suptitle="Confusion Matrices Comparison",
    save_path=None
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# TODO
# add random choise of classes to misclassified

def pick_random_class_pair(class_names, seed=123):
    rng = random.Random(seed)
    a, b = rng.sample(list(class_names), 2)
    return a, b

def misclassified_between_two_classes(t, p, f, class_names, class_a, class_b, snr=None, noise=None, max_rows=20):
    """
    Show misclassifications A->B.
    t,p are integer class indices, f filenames.
    """
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    ia = name_to_idx[class_a]
    ib = name_to_idx[class_b]

    t = np.asarray(t)
    p = np.asarray(p)
    f = np.asarray(f)
    if snr is not None:
        snr = np.array(snr)
    if noise is not None:
        noise = np.array(noise)

    mask_ab = (t == ia) & (p == ib)   # A predicted as B

    rows = []

    idx_ab = np.where(mask_ab)[0][:max_rows]

    for i in idx_ab:

        row = {
            "true": class_a,
            "pred": class_b,
            "filename": f[i],
        }

        if snr is not None and len(snr) > i:
            row["snr_db"] = snr[i]

        if noise is not None and len(noise) > i:
            row["noise"] = noise[i]

        rows.append(row)

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


# Helpers: single consistent split by indices
def make_split_indices(labels: np.ndarray,
                       train_ratio: float,
                       val_ratio: float,
                       test_ratio: float,
                       random_state: int):
    """
    Returns idx_train, idx_val, idx_test (numpy arrays) with stratified split.
    """

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


def plot_one_from_clean_df_row(row, scaler, sampling_rate, n_mfcc, title_prefix="", save_path=None):
    """
    row is a single-row DataFrame or a Series from clean_df/df_train/df_test
    Must include: audio_data, label, filename, mfcc
    """
    mfcc_raw = row["mfcc"]
    mfcc_scaled = scaler.transform(mfcc_raw).astype(np.float32)

    plot_df = row.to_frame().T.copy()  # single-row DF
    plot_df["scaled_mfcc"] = [mfcc_scaled]

    print(
        f"\nPLOT {title_prefix}: label={row['label']} | filename={row['filename']} | "
        f"mfcc={mfcc_raw.shape}"
    )
    
    plot_audio_and_features(
    audio_df=plot_df,
    label_name=str(row["label"]),
    sampling_rate=sampling_rate,
    n_mfcc=n_mfcc,
    random_example=False,
    title_extra=f"file={row['filename']} | label={row['label']}"
    )

def plot_one_noisy_item(noisy_ds, idx, sampling_rate, n_mfcc, title):
    """
    Supports both:
      - NoisyTestDataset(return_audio=True): returns (X, y, fname, sig)
      - NoisyTestDataset(return_audio=False): returns (X, y, fname)
    Plots waveform/spectrogram from sig if available, and scaled MFCC from X.
    """

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
    """
    X_tensor, y_tensor, fname, sig_tensor, snr_db, noise_name = noisy_ds[idx]

    sig = sig_tensor.numpy().astype(np.float32)
    mfcc_sc_padded = X_tensor.squeeze(0).numpy()

    plot_df = pd.DataFrame([{
        "filename": fname,
        "label": "NOISY_SAMPLE",
        "audio_data": sig,
        "scaled_mfcc": mfcc_sc_padded
    }])

    print(
    f"\nPLOT {title} : idx={idx} | filename={fname} | "
    f"noise={noise_name} | SNR={snr_db} dB | "
    f"mfcc={mfcc_sc_padded.shape}"
)

    plot_audio_and_features(
    audio_df=plot_df,
    label_name="NOISY_SAMPLE",
    sampling_rate=sampling_rate,
    n_mfcc=n_mfcc,
    random_example=False,
    title_extra=f"file={fname} | noise={noise_name} | SNR={snr_db} dB"
    ) 


def plot_accuracy_vs_snr(true_labels, pred_labels, snr_values, title, save_path=None):
    """
    Plot classification accuracy as a function of SNR.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    snr_values = np.asarray(snr_values)

    snr_unique = np.sort(np.unique(snr_values))

    acc_per_snr = []
    count = []

    for snr in snr_unique:
        mask = snr_values == snr
        acc = np.mean(true_labels[mask] == pred_labels[mask])
        acc_per_snr.append(acc)
        count.append(np.sum(mask))

    plt.figure(figsize=(7,5))
    plt.plot(snr_unique, acc_per_snr, marker="o", linewidth=2)
    
    for x,y,c in zip(snr_unique, acc_per_snr, count):
        plt.text(x, y, f"{c}", ha="center", va="bottom", fontsize=9)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs SNR")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 


def plot_confusion_per_snr(true_labels, pred_labels, snr_values, class_names, title, save_path=None):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    snr_values = np.asarray(snr_values)

    snrs = np.sort(np.unique(snr_values))

    fig, axes = plt.subplots(2, 3, figsize=(12,10))
    axes = axes.flatten()

    for i,s in enumerate(snrs):
        mask = snr_values == s
        cm = confusion_matrix(
            true_labels[mask],
            pred_labels[mask],
            labels=np.arange(len(class_names))
        )

        sns.heatmap(
            cm,
            ax=axes[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        axes[i].set_title(f"SNR={s} dB")

    for i in range(len(snrs), len(axes)):
        axes[i].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_per_noise(true_labels, pred_labels, noise_names, class_names, title, save_path=None):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    noise_names = np.asarray(noise_names)
    noises = np.unique(noise_names)
    
    fig, axes = plt.subplots(2, 3, figsize=(12,10))
    axes = axes.flatten()

    for i, n in enumerate(noises):
        mask = noise_names == n
        cm = confusion_matrix(
            true_labels[mask],
            pred_labels[mask],
            labels=np.arange(len(class_names))
        )

        sns.heatmap(
            cm,
            ax=axes[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        axes[i].set_title(f"noise={n}")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def _compute_spectrogram(x, fs):
    f, t, Zxx = stft(x, fs=fs, nperseg=512, noverlap=256)
    S = np.abs(Zxx)
    return f, t, 20 * np.log10(S + 1e-10)


def plot_signal_comparison(clean, noisy, denoised, enhanced_sgmse=None,
                            enhanced_trained=None,
                            fs=16000, title="Signal Comparison", save_path=None):
    """
    Plot waveform + spectrogram for clean / noisy / denoised signals
    """

    signals = [
        ("Clean", clean),
        ("Noisy", noisy),
        ("Denoised", denoised),
        ("Enhanced (SGMSE)", enhanced_sgmse),
        ("Enhanced (Trained EP100)", enhanced_trained),
        ]
    

    fig, axes = plt.subplots(5, 2, figsize=(12, 10))

    for i, (name, sig) in enumerate(signals):

        # ===== waveform =====
        axes[i, 0].plot(sig)
        axes[i, 0].set_title(f"{name} - Waveform")
        axes[i, 0].set_xlim([0, len(sig)])

        # ===== spectrogram =====
        f, t, S_db = _compute_spectrogram(sig, fs)

        im = axes[i, 1].pcolormesh(t, f, S_db, shading='auto')
        axes[i, 1].set_title(f"{name} - Spectrogram")
        axes[i, 1].set_ylabel("Hz")
        axes[i, 1].set_xlabel("Time")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

"""
def run_calc_metrics(df_test_audio, sampling_rate, run_dir, enhanced_root, tag):

    TMP_ROOT = "/home/dsi/skopavi/Project/kws_project/tmp/tmp_metrics"

    test_dir = os.path.join(TMP_ROOT, "test")
    clean_dir = os.path.join(test_dir, "clean")
    noisy_dir = os.path.join(test_dir, "noisy")
    enhanced_tmp_dir = os.path.join(TMP_ROOT, "enhanced")

    print("\n[METRICS] Reset TMP...")
    if os.path.exists(TMP_ROOT):
        shutil.rmtree(TMP_ROOT)

    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(enhanced_tmp_dir, exist_ok=True)

    print("[METRICS] Saving TEST files...")

    noisy_root = "/home/dsi/skopavi/Project/kws_project/data/noisy_new/test"
    enhanced_root = enhanced_root

    count = 0
    print("\n[DEBUG] Checking file consistency...\n")
    missing_clean = []
    missing_noisy = []
    missing_enhanced = []
    ok_files = []

    for _, row in df_test_audio.iterrows():
        fname = row["filename"]
        label = row["label"]

        # CLEAN
        clean_path = os.path.join(clean_dir, fname)
        sf.write(clean_path, row["audio_data"], sampling_rate)

        # NOISY (copy)
        noisy_src = os.path.join(noisy_root, label, fname)
        noisy_dst = os.path.join(noisy_dir, fname)

        if os.path.exists(noisy_src):
            shutil.copy2(noisy_src, noisy_dst)

        # ENHANCED (copy + flatten)
        enhanced_src = os.path.join(enhanced_root, label, fname)
        enhanced_dst = os.path.join(enhanced_tmp_dir, fname)

        has_noisy = os.path.exists(noisy_src)
        has_enhanced = os.path.exists(enhanced_src)
        if not has_noisy:
            missing_noisy.append((label, fname))
        if not has_enhanced:
            missing_enhanced.append((label, fname))
        if has_noisy and has_enhanced:
            ok_files.append((label, fname))

        if os.path.exists(noisy_src) and os.path.exists(enhanced_src):
            count += 1
            shutil.copy2(enhanced_src, enhanced_dst)

    print("DEBUG: FILES USED FOR METRICS:", count)
    print(f"\nTOTAL df_test_audio: {len(df_test_audio)}")
    print("DEBUG: MISSING CLEAN FILES:", len(missing_clean))
    print("DEBUG: MISSING NOISY FILES:", len(missing_noisy))
    print("DEBUG: MISSING ENHANCED FILES:", len(missing_enhanced))
    print("DEBUG: OK FILES:", len(ok_files))

    MAX_PRINT = 10
    if missing_noisy:
        print("\n[DEBUG] Missing in NOISY:")
        for label, fname in missing_noisy[:MAX_PRINT]:
            print(f"NOISY MISSING → {label}/{fname}")
    if missing_enhanced:
        print("\n[DEBUG] Missing in ENHANCED:")
        for label, fname in missing_enhanced[:MAX_PRINT]:
            print(f"ENHANCED MISSING → {label}/{fname}")
    only_noisy = []
    only_enhanced = []
    for _, row in df_test_audio.iterrows():
        fname = row["filename"]
        label = row["label"]
        noisy_src = os.path.join(noisy_root, label, fname)
        enhanced_src = os.path.join(enhanced_root, label, fname)
        if os.path.exists(noisy_src) and not os.path.exists(enhanced_src):
            only_noisy.append((label, fname))
        if os.path.exists(enhanced_src) and not os.path.exists(noisy_src):
            only_enhanced.append((label, fname))
    if only_noisy:
        print("\n[DEBUG] Exists ONLY in NOISY:")
        for label, fname in only_noisy[:MAX_PRINT]:
            print(f"ONLY NOISY → {label}/{fname}")
    if only_enhanced:
        print("\n[DEBUG] Exists ONLY in ENHANCED:")
        for label, fname in only_enhanced[:MAX_PRINT]:
            print(f"ONLY ENHANCED → {label}/{fname}")

    print("[METRICS] Running calc_metrics...")

    metrics_out_path = run_dir / "metrics.txt"

    print(clean_dir)
    print(noisy_dir)
    print(enhanced_tmp_dir)
    print("DEBUG:")
    print("df_test_audio:", len(df_test_audio))
    print("noisy_tmp:", len(os.listdir(noisy_dir)))
    print("enhanced_tmp:", len(os.listdir(enhanced_tmp_dir)))

    # cmd = f
    # python /home/dsi/skopavi/Project/kws_project/code/sgmse/calc_metrics.py \
        # --clean_dir {clean_dir} \
        # --noisy_dir {noisy_dir} \
        # --enhanced_dir {enhanced_tmp_dir}
    

    with open(metrics_out_path, "a") as f:
        f.write(f"\n\n{tag}\n")
        f.flush()
        subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
        f.write("\n") 

    print("[METRICS] Done! Saved to:", metrics_out_path)

"""

def run_calc_metrics(df_test_audio, sampling_rate, run_dir, enhanced_root, tag):

    import os, shutil, subprocess
    import soundfile as sf

    TMP_ROOT = "/home/dsi/skopavi/Project/kws_project/tmp/tmp_metrics"

    test_dir = os.path.join(TMP_ROOT, "test")
    clean_dir = os.path.join(test_dir, "clean")
    noisy_dir = os.path.join(test_dir, "noisy")
    enhanced_tmp_dir = os.path.join(TMP_ROOT, "enhanced")

    noisy_root = "/home/dsi/skopavi/Project/kws_project/data/noisy_new/test"

    # =========================
    # RESET TMP
    # =========================
    print("\n[METRICS] Reset TMP...")
    if os.path.exists(TMP_ROOT):
        shutil.rmtree(TMP_ROOT)

    os.makedirs(clean_dir)
    os.makedirs(noisy_dir)
    os.makedirs(enhanced_tmp_dir)

    # =========================
    # DEBUG STORAGE
    # =========================
    missing_noisy = []
    missing_enhanced = []
    used_files = []

    # =========================
    # MAIN LOOP
    # =========================
    print("[METRICS] Building dataset for metrics...\n")

    for _, row in df_test_audio.iterrows():

        fname = row["filename"]
        label = row["label"]

        noisy_src = os.path.join(noisy_root, label, fname)
        enhanced_src = os.path.join(enhanced_root, label, fname)

        has_noisy = os.path.exists(noisy_src)
        has_enhanced = os.path.exists(enhanced_src)

        # =========================
        # DEBUG TRACKING
        # =========================
        if not has_noisy:
            missing_noisy.append((label, fname))
            continue

        if not has_enhanced:
            missing_enhanced.append((label, fname))
            continue

        # =========================
        # COPY ONLY VALID FILES
        # =========================
        safe_name = f"{label}__{fname}"

        clean_path = os.path.join(clean_dir, safe_name)
        noisy_dst = os.path.join(noisy_dir, safe_name)
        enhanced_dst = os.path.join(enhanced_tmp_dir, safe_name)
        # clean_path = os.path.join(clean_dir, fname)
        # noisy_dst = os.path.join(noisy_dir, fname)
        # enhanced_dst = os.path.join(enhanced_tmp_dir, fname)

        sf.write(clean_path, row["audio_data"], sampling_rate)
        shutil.copy2(noisy_src, noisy_dst)
        shutil.copy2(enhanced_src, enhanced_dst)

        used_files.append((label, fname))

    # =========================
    # DEBUG PRINTS
    # =========================
    print("====================================")
    print("[DEBUG SUMMARY]")
    print("====================================")

    print(f"Total df_test_audio: {len(df_test_audio)}")
    print(f"Used for metrics (intersection): {len(used_files)}")
    print(f"Missing NOISY: {len(missing_noisy)}")
    print(f"Missing ENHANCED: {len(missing_enhanced)}")

    MAX_PRINT = 10

    if missing_noisy:
        print("\n[DEBUG] Missing in NOISY:")
        for label, fname in missing_noisy[:MAX_PRINT]:
            print(f"  NOISY MISSING → {label}/{fname}")

    if missing_enhanced:
        print("\n[DEBUG] Missing in ENHANCED:")
        for label, fname in missing_enhanced[:MAX_PRINT]:
            print(f"  ENHANCED MISSING → {label}/{fname}")

    # =========================
    # FINAL COUNTS (CRITICAL)
    # =========================
    print("\n[FINAL COUNTS]")
    print("clean:", len(os.listdir(clean_dir)))
    print("noisy:", len(os.listdir(noisy_dir)))
    print("enhanced:", len(os.listdir(enhanced_tmp_dir)))

    # sanity check
    assert len(os.listdir(clean_dir)) == len(os.listdir(noisy_dir)) == len(os.listdir(enhanced_tmp_dir)), \
        "❌ Mismatch between clean/noisy/enhanced!"

    # =========================
    # RUN METRICS
    # =========================
    print("\n[METRICS] Running calc_metrics...")

    metrics_out_path = run_dir / "metrics.txt"

    cmd = f"""
    python /home/dsi/skopavi/Project/kws_project/code/sgmse/calc_metrics.py \
        --clean_dir {clean_dir} \
        --noisy_dir {noisy_dir} \
        --enhanced_dir {enhanced_tmp_dir}
    """

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    print("\n[METRICS OUTPUT]")
    print(result.stdout)

    with open(metrics_out_path, "a") as f:
        f.write(f"\n\n{tag}\n")
        f.write(result.stdout)
        f.write("\n")

    print("[METRICS] Done! Saved to:", metrics_out_path)