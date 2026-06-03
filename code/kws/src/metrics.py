# metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import random

import os
import shutil
import subprocess
import soundfile as sf
import sys
import time


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

    # ONLY TRUE LABELS INSIDE CURRENT SUBSET
    labels_subset = sorted(np.unique(all_true))

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=labels_subset)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

    # SUBSET ACCURACY/REPORT - ONLY SELECTED FOLDERS
    report = classification_report(
        all_true,
        all_pred,
        labels=labels_subset,
        target_names=class_names,
        zero_division=0)
    
    print("Classification Report:")
    print(report)
    if save_txt_path:
        with open(save_txt_path, "a") as f:
            f.write(f"\n{model_name.upper()}\n")
            f.write(report)
            f.write("\n")

    return cm, report, (np.array(all_true), np.array(all_pred), np.array(all_files))


def pick_random_class_pair(class_names, seed=123):
    rng = random.Random(seed)
    a, b = rng.sample(list(class_names), 2)
    return a, b

def misclassified_between_two_classes(t, p, f, class_names, class_a, class_b, snr=None, noise=None, max_rows=50):
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


def run_calc_metrics(df_test_audio, sampling_rate, run_dir, enhanced_root, tag):

    TMP_ROOT = os.path.join(
    "/home/dsi/skopavi/Project/kws_project/tmp",
    f"tmp_metrics_{tag}_{os.getpid()}_{int(time.time())}"
)

    test_dir = os.path.join(TMP_ROOT, "test")
    clean_dir = os.path.join(test_dir, "clean")
    noisy_dir = os.path.join(test_dir, "noisy")
    enhanced_tmp_dir = os.path.join(TMP_ROOT, "enhanced")

    noisy_root = "/home/dsi/skopavi/Project/kws_project/data/noisy_new/test"

    # RESET TMP
    # print("[METRICS] Reset TMP...")
    # if os.path.exists(TMP_ROOT):
        # shutil.rmtree(TMP_ROOT)

    os.makedirs(clean_dir)
    os.makedirs(noisy_dir)
    os.makedirs(enhanced_tmp_dir)

    # DEBUG STORAGE
    missing_noisy = []
    missing_enhanced = []
    used_files = []

    # MAIN LOOP
    print("[METRICS] Building dataset for metrics...")

    for _, row in df_test_audio.iterrows():

        fname = row["filename"]
        label = row["label"]

        noisy_src = os.path.join(noisy_root, label, fname)
        enhanced_src = os.path.join(enhanced_root, label, fname)

        has_noisy = os.path.exists(noisy_src)
        has_enhanced = os.path.exists(enhanced_src)

        # DEBUG TRACKING
        if not has_noisy:
            missing_noisy.append((label, fname))
            continue

        if not has_enhanced:
            missing_enhanced.append((label, fname))
            continue

        # COPY ONLY VALID FILES
        safe_name = f"{label}__{fname}"

        clean_path = os.path.join(clean_dir, safe_name)
        noisy_dst = os.path.join(noisy_dir, safe_name)
        enhanced_dst = os.path.join(enhanced_tmp_dir, safe_name)

        sf.write(clean_path, row["audio_data"], sampling_rate)
        shutil.copy2(noisy_src, noisy_dst)
        shutil.copy2(enhanced_src, enhanced_dst)

        used_files.append((label, fname))

    """
    # DEBUG PRINTS
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

    """

    # RUN METRICS
    print("[METRICS] Running calc_metrics...")

    metrics_out_path = run_dir / "metrics.txt"

    cmd = f"""
    python /home/dsi/skopavi/Project/kws_project/code/sgmse/calc_metrics.py \
        --clean_dir {clean_dir} \
        --noisy_dir {noisy_dir} \
        --enhanced_dir {enhanced_tmp_dir}
    """

    metrics_start = time.time()

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    metrics_time = time.time() - metrics_start

    print(
        f"\nMETRICS TIME: "
        f"{metrics_time/60:.2f} minutes"
    )

    print("\n[METRICS OUTPUT]")
    print(result.stdout)

    metrics = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("PESQ:"):
            metrics["pesq"] = (line.replace("PESQ:", "").split("(N=")[0].strip())
        elif line.startswith("ESTOI:"):
            metrics["estoi"] = (line.replace("ESTOI:", "").split("(N=")[0].strip())
        elif line.startswith("SI-SDR:"):
            metrics["si_sdr"] = (line.replace("SI-SDR:", "").split("(N=")[0].strip())
        elif line.startswith("SI-SIR:"):
            metrics["si_sir"] = (line.replace("SI-SIR:", "").split("(N=")[0].strip())
        elif line.startswith("SI-SAR:"):
            metrics["si_sar"] = (line.replace("SI-SAR:", "").split("(N=")[0].strip())

    with open(metrics_out_path, "a") as f:
        f.write(f"\n\n{tag}\n")
        f.write(result.stdout)
        f.write("\n")

    print("[METRICS] Done! Saved to:", metrics_out_path)

    return metrics