from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from denoiser.stft_mask.denoise import denoise_signal
import torch
import matplotlib
matplotlib.use("Agg")
from datetime import datetime
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tabulate import tabulate
from config import load_config
from torch.utils.data import DataLoader
from noise_dataset import NoisyTestDataset, mix_with_noise_at_snr
from sklearn.model_selection import train_test_split
from models import DSCNN
from train import get_device, train_model, load_model, evaluate_loader
from denoiser.stft_mask.unet_model import UNetDenoiser
from metrics import (
    plot_history, 
    confusion_and_report,
    plot_signal_comparison, 
    plot_two_confusion_matrices,
    pick_random_class_pair,
    misclassified_between_two_classes,
    plot_one_from_clean_df_row,
    plot_one_noisy_item,
    plot_accuracy_vs_snr,
    plot_confusion_per_snr,
    plot_confusion_per_noise,
)

from dataset import (
    list_folders,
    collect_wav_paths,
    build_audio_dataframe,
    pad_mfcc_list,
    make_label_encoder,
    make_loaders,
)

from features import (
    add_mfcc_column,
    fit_scaler,
    apply_scaler,
    plot_audio_and_features,
)


def run(cfg: dict):
    # 1. SETUP
    device = get_device(cfg["device"])
    print("Using device:", device)

    sampling_rate = int(cfg["sampling_rate"])
    n_mfcc = int(cfg["n_mfcc"])

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(cfg["output_dir"]) / f"{run_stamp}"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print("Run dir:", run_dir)
    print("Models dir:", models_dir)
    print("Plots dir:", plots_dir)

    # 2. Load CLEAN audio df
    clean_root = str(cfg["clean_dir"])
    folder_start = int(cfg["folder_start"])
    folder_end = int(cfg["folder_end"])

    use_parallel = bool(cfg["use_parallel"])
    random_state = int(cfg["random_state"])

    all_folders = list_folders(clean_root)
    selected_folders = all_folders[folder_start:folder_end]
    print("\nCLEAN selected folders:", selected_folders)

    clean_paths = collect_wav_paths(clean_root, selected_folders)
    print("CLEAN total wav files:", len(clean_paths))

    clean_df = build_audio_dataframe(
        clean_paths,
        sampling_rate=sampling_rate,
        use_parallel=use_parallel
    )
    print("CLEAN df shape (audio loaded):", clean_df.shape)
    print(tabulate(clean_df.head(3), headers="keys", tablefmt="psql", showindex=False))

    # Keep arrays for later
    all_audio = clean_df["audio_data"].values
    all_labels = clean_df["label"].values
    all_filenames = clean_df["filename"].values

    # ONE consistent split (indices)
    n_total = len(clean_df)
    all_indices = np.arange(n_total)

    train_ratio = float(cfg["train_ratio"])
    val_ratio = float(cfg["val_ratio"])
    test_ratio = float(cfg["test_ratio"])
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    # First split: train vs temp (val+test)
    temp_ratio = val_ratio + test_ratio
    idx_train, idx_temp = train_test_split(
        all_indices,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=all_labels
    )

    # Second split: val vs test (within temp)
    val_fraction_of_temp = val_ratio / temp_ratio  # 0.1/(0.1+0.1)=0.5
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=random_state,
        stratify=all_labels[idx_temp]
    )

    # Build split dataframes (audio + label + filename are guaranteed consistent)
    df_train_audio = clean_df.iloc[idx_train].reset_index(drop=True)
    df_val_audio = clean_df.iloc[idx_val].reset_index(drop=True)
    df_test_audio = clean_df.iloc[idx_test].reset_index(drop=True)

    print("\nSPLIT (AUDIO) sizes:",
          f"train={len(df_train_audio)}, val={len(df_val_audio)}, test={len(df_test_audio)}")

    # 3. FEATURES (MFCC)
    # Compute MFCC for ALL clean df (then slice by SAME indices)
    clean_df = add_mfcc_column(
        clean_df,
        sr=sampling_rate,
        n_mfcc=n_mfcc,
        use_parallel=use_parallel
    )
    print("\nCLEAN after MFCC computed for ALL:")

    df_train = clean_df.iloc[idx_train].reset_index(drop=True)
    df_val = clean_df.iloc[idx_val].reset_index(drop=True)
    df_test = clean_df.iloc[idx_test].reset_index(drop=True)

    print("\nSPLIT (MFCC) sizes:",
          f"train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    # print("Example MFCC shape (raw) from train[0]:", df_train["mfcc"].iloc[0].shape)

    # Fit scaler on CLEAN TRAIN only
    X_train_mfcc_list = df_train["mfcc"].values
    scaler = fit_scaler(X_train_mfcc_list)

    # Scale + pad MFCC for clean train/val/test
    X_train_sc = apply_scaler(df_train["mfcc"].values, scaler)
    X_val_sc = apply_scaler(df_val["mfcc"].values, scaler)
    X_test_sc = apply_scaler(df_test["mfcc"].values, scaler)

    max_len = max(m.shape[0] for m in X_train_sc)
    X_train_padded = pad_mfcc_list(X_train_sc, max_len=max_len)
    X_val_padded = pad_mfcc_list(X_val_sc, max_len=max_len)
    X_test_padded = pad_mfcc_list(X_test_sc, max_len=max_len)

    print("\nPADDING:")
    print("  max_len from CLEAN TRAIN:", max_len)
    print("  X_train_padded:", X_train_padded.shape)
    print("  X_val_padded  :", X_val_padded.shape)
    print("  X_test_padded :", X_test_padded.shape)

    # Label encoder (fit on CLEAN TRAIN labels only)
    y_train_labels = df_train["label"].values
    y_val_labels = df_val["label"].values
    y_test_labels = df_test["label"].values

    label_encoder, y_train_enc, y_val_enc, y_test_enc = make_label_encoder(
        y_train_labels, y_val_labels, y_test_labels
    )

    class_names = label_encoder.classes_
    num_classes = len(class_names)
    print("\nCLASSES:", list(class_names))

    fn_train = df_train["filename"].values
    fn_val = df_val["filename"].values
    fn_test = df_test["filename"].values

    # Build loaders
    # Train/Val ALWAYS clean
    # Test depends on cfg["test_mode"]
    train_loader, val_loader, test_loader_clean = make_loaders(
        X_train_padded, y_train_enc, fn_train,
        X_val_padded, y_val_enc, fn_val,
        X_test_padded, y_test_enc, fn_test,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle_train=bool(cfg["shuffle_train"])
    )

    print("Loader batches:",
              f"train={len(train_loader)}, val={len(val_loader)}, test_clean={len(test_loader_clean)}")

    # 4. TRAIN (CLEAN) + EVAL (CLEAN)
    print("\n=== TRAIN CLEAN ===")

    model = DSCNN(num_classes)
    hist = train_model(
        model, train_loader, val_loader,
        num_epochs=int(cfg["epochs"]),
        patience=int(cfg["patience"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        model_name="dscnn",
        device=device,
        best_dir=models_dir
    )

    plot_history(hist, save_path=plots_dir / "training_history.png")

    best_model = load_model(DSCNN, num_classes, hist["best_path"], device=device)

    tr_loss, tr_acc = evaluate_loader(best_model, train_loader, device=device)
    va_loss, va_acc = evaluate_loader(best_model, val_loader, device=device)

    # 5. TEST (CLEAN)
    test_modes_cfg = cfg.get("test_mode", "all")

    if test_modes_cfg == "all":
        test_modes = ["clean", "noisy", "denoised", "enhanced"]
    elif isinstance(test_modes_cfg, list):
        test_modes = [m for m in test_modes_cfg if m != "all"]
    else:
        test_modes = [test_modes_cfg]

    # 6. TEST - NOISY + DENOISED (ON-THE-FLY)
    results = {}

    for mode in test_modes:
        if mode not in ["clean", "noisy", "denoised", "enhanced"]:
             continue  # skip invalid modes
        print(f"\n=== TEST MODE: {mode.upper()} ===")
        if mode == "clean":
             plot_one_from_clean_df_row(
                row=df_test.iloc[2], scaler=scaler,
                sampling_rate=sampling_rate, n_mfcc=n_mfcc,
                title_prefix=f"TEST CLEAN example",
                save_path=plots_dir / "example_clean.png")
             loss_c, acc_c, t_c, p_c, f_c, _, _ = evaluate_loader(
                    best_model, test_loader_clean, device=device, return_meta=True)
                    
             cm, rep, (t, p, f) = confusion_and_report(best_model, test_loader_clean, class_names, 
                                                    device, model_name=f"{mode.upper()}")
             
             print("\n=== RESULTS (CLEAN) ===")
             print(f"train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
             print(f"val: loss={va_loss:.4f}, acc={va_acc:.4f}")
             print(f"test: loss={loss_c:.4f}, acc={acc_c:.4f}")
             continue
        
        elif mode in ["noisy", "denoised", "enhanced"]:   
            ds = NoisyTestDataset(
                audio_list=df_test_audio["audio_data"].values,
                labels_list=df_test_audio["label"].values,
                filenames_list=df_test_audio["filename"].values,
                label_encoder=label_encoder,
                scaler=scaler,
                sampling_rate=sampling_rate,
                n_mfcc=n_mfcc,
                max_len=max_len,
                bg_noise_dir=str(cfg["bg_noise_dir"]),
                noise_ops=list(cfg["noise_ops"]),
                snr_choices=list(cfg["snr_choices"]),
                seed=int(cfg.get("noisy_test_seed", 123)),
                mode=mode,
                device=device,)

            loader = DataLoader(ds, batch_size=int(cfg["batch_size"]),
                                shuffle=False, num_workers=int(cfg["num_workers"]))
            
            """
            # show example
            if bool(cfg["show_plots"]) and mode != "clean":
                plot_one_noisy_item(noisy_ds=ds, idx=1, sampling_rate=sampling_rate,
                    n_mfcc=n_mfcc,title_prefix=f"TEST {mode.upper()} example")
            """

            loss, acc, t, p, f, snr, noise = evaluate_loader(
                best_model, loader, device=device, return_meta=True
            )

            print(f"test ({mode}): loss={loss:.4f}, acc={acc:.4f}")

            cm, rep, (t, p, f) = confusion_and_report(best_model, loader, class_names, 
                                                    device, model_name=f"{mode.upper()}")

            results[mode] = dict(loss=loss, acc=acc, t=t, p=p, f=f, snr=snr, noise=noise, cm=cm, rep=rep)
            
    
    # SHOW EXAMPLES TEST ITEMS
    idx_vis = 0
    clean_sig = df_test_audio["audio_data"].iloc[idx_vis]
    filename = df_test_audio["filename"].iloc[idx_vis]
    noise_name = ds._choose_noise_name()
    noise_arr = ds.noise_bank[noise_name][0]
    snr_db = -10
    noisy_sig = mix_with_noise_at_snr(clean=clean_sig, noise=noise_arr, snr_db=snr_db)
    denoised_sig = denoise_signal(noisy_signal=noisy_sig, fs=sampling_rate, device=device)
    # === ENHANCED SIGNAL ===
    enhanced_path = os.path.join(
        "/home/dsi/skopavi/Project/kws_project/generated_enhanced",
        df_test_audio["label"].iloc[idx_vis],filename)

    if os.path.exists(enhanced_path):
        enhanced_sig, _ = librosa.load(enhanced_path, sr=sampling_rate)
    else:
        enhanced_sig = None

    plot_signal_comparison(
        clean=clean_sig,
        noisy=noisy_sig,
        denoised=denoised_sig,
        enhanced=enhanced_sig,  
        fs=sampling_rate,
        title=f"file={filename} | noise={noise_name} | SNR={snr_db} dB",
        save_path=plots_dir / f"example_{filename}_snr{snr_db}.png")

    # 7. CONFUSION MATRICES (3 SUBPLOTS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    titles = ["Clean Test", "Noisy Test", "Denoised Test", "Enhanced Test"]  
    for i, (name, t, p) in enumerate([
        ("Clean", t_c, p_c),
        ("Noisy", results["noisy"]["t"], results["noisy"]["p"]),
        ("Denoised", results["denoised"]["t"], results["denoised"]["p"]),
        ("Enhanced", results["enhanced"]["t"], results["enhanced"]["p"]),
    ]):
        cm = confusion_matrix(t, p)
        sns.heatmap(cm, ax=axes[i], annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(titles[i], fontsize=14)
        axes[i].set_xlabel("Predicted", fontsize=12)
        axes[i].set_ylabel("True", fontsize=12) 
    plt.suptitle("Confusion Matrices - Clean vs Noisy vs Denoised vs Enhanced Test", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if plots_dir:
        plt.savefig(plots_dir/ "confusion_matrices.png")
        plt.close()
    else:
        plt.show()

    # 8. ANALYSIS - NOISY
    print("\n=== ANALYSIS NOISY ===")

    plot_confusion_per_snr(
        results["noisy"]["t"],
        results["noisy"]["p"],
        results["noisy"]["snr"],
        class_names,
        title = "Confusion per SNR - NOISY TEST",
        save_path=plots_dir / "confusion_per_snr_noisy.png"
    )

    plot_confusion_per_noise(
        results["noisy"]["t"],
        results["noisy"]["p"],
        results["noisy"]["noise"],
        class_names,
        title = "Confusion per Noise - NOISY TEST",
        save_path=plots_dir / "confusion_per_noise_noisy.png"
    )

    # MISCLASSIFIED ANALYSIS (RANDOM PAIR)
    a, b = pick_random_class_pair(class_names, seed=int(cfg.get("mis_seed", 123)))
    print(f"\nRandom class pair for misclassified: A='{a}' vs B='{b}'")

    print("\n=== MISCLASSIFIED NOISY ===")
    mis_noisy = misclassified_between_two_classes(
        results["noisy"]["t"],
        results["noisy"]["p"],
        results["noisy"]["f"],
        class_names,
        a, b,
        results["noisy"]["snr"],
        results["noisy"]["noise"],
        max_rows=int(cfg.get("mis_max_rows", 20)),
    )
    if len(mis_noisy) == 0:
            print("No misclassifications found between these two classes.")
    else:
        print("\nMisclassified files between the pair:")
        print(tabulate(mis_noisy, headers="keys", tablefmt="psql", showindex=True))
        print("\nErrors by SNR:")
        print(mis_noisy.groupby("snr_db").size())
        print("\nErrors by noise:")
        print(mis_noisy.groupby("noise").size())

    # 9. ANALYSIS - DENOISED
    print("\n=== ANALYSIS DENOISED ===")

    plot_confusion_per_snr(
        results["denoised"]["t"],
        results["denoised"]["p"],
        results["denoised"]["snr"],
        class_names,
        title = "Confusion per SNR - DENOISED TEST",
        save_path=plots_dir / "confusion_per_snr_denoised.png"
    )

    plot_confusion_per_noise(
        results["denoised"]["t"],
        results["denoised"]["p"],
        results["denoised"]["noise"],
        class_names,
        title = "Confusion per Noise - DENOISED TEST",
        save_path=plots_dir / "confusion_per_noise_denoised.png"
    )

    # MISCLASSIFIED ANALYSIS (RANDOM PAIR)
    print("\n=== MISCLASSIFIED DENOISED ===")
    mis_denoised = misclassified_between_two_classes(
        results["denoised"]["t"],
        results["denoised"]["p"],
        results["denoised"]["f"],
        class_names,
        a, b,
        results["denoised"]["snr"],
        results["denoised"]["noise"],
        
    )
    if len(mis_denoised) == 0:
            print("No misclassifications found between these two classes.")
    else:
        print("\nMisclassified files between the pair:")
        print(tabulate(mis_denoised, headers="keys", tablefmt="psql", showindex=True))
        print("\nErrors by SNR:")
        print(mis_denoised.groupby("snr_db").size())
        print("\nErrors by noise:")
        print(mis_denoised.groupby("noise").size())

    # 11. ANALYSIS - ENHANCED
    print("\n=== ANALYSIS ENHANCED ===")

    plot_confusion_per_snr(
        results["enhanced"]["t"],
        results["enhanced"]["p"],
        results["enhanced"]["snr"],
        class_names,
        title="Confusion per SNR - ENHANCED TEST",
        save_path=plots_dir / "confusion_per_snr_enhanced.png"
    )

    plot_confusion_per_noise(
        results["enhanced"]["t"],
        results["enhanced"]["p"],
        results["enhanced"]["noise"],
        class_names,
        title="Confusion per Noise - ENHANCED TEST",
        save_path=plots_dir / "confusion_per_noise_enhanced.png"
    )

    print("\n=== MISCLASSIFIED ENHANCED ===")

    mis_enhanced = misclassified_between_two_classes(
        results["enhanced"]["t"],
        results["enhanced"]["p"],
        results["enhanced"]["f"],
        class_names,
        a, b,
        results["enhanced"]["snr"],
        results["enhanced"]["noise"],
    )

    if len(mis_enhanced) == 0:
        print("No misclassifications found between these two classes.")
    else:
        print("\nMisclassified files between the pair:")
        print(tabulate(mis_enhanced, headers="keys", tablefmt="psql", showindex=True))

        print("\nErrors by SNR:")
        print(mis_enhanced.groupby("snr_db").size())

        print("\nErrors by noise:")
        print(mis_enhanced.groupby("noise").size())

if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    run(cfg)