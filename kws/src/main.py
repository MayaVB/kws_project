from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import torch
from datetime import datetime
import numpy as np
from pathlib import Path
from tabulate import tabulate
from config import load_config
from torch.utils.data import DataLoader
from noise_dataset import NoisyTestDataset
from sklearn.model_selection import train_test_split
from models import DSCNN
from train import get_device, train_model, load_model, evaluate_loader
from denoiser.stft_mask.unet_model import UNetDenoiser
from metrics import (
    plot_history, 
    confusion_and_report, 
    plot_two_confusion_matrices,
    pick_random_class_pair,
    misclassified_between_two_classes,
    plot_one_from_clean_df_row,
    plot_one_noisy_item,
    plot_accuracy_vs_snr,
    plot_accuracy_per_noise,
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
    # Device + output dirs
    device = get_device(cfg["device"])
    print("Using device:", device)

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{run_stamp}"

    run_dir = Path(cfg["output_dir"]) / run_id
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Run dir:", run_dir)
    print("Models dir:", models_dir)

    # UNet model directory (shared between runs)
    unet_dir = Path(cfg["output_dir"]) / "unet_denoiser"
    unet_dir.mkdir(parents=True, exist_ok=True)
    # path for the best UNet model
    unet_best_path = unet_dir / "best_model.pth"
    print("UNet dir:", unet_dir)
    print("UNet best model path:", unet_best_path)

    sampling_rate = int(cfg["sampling_rate"])
    n_mfcc = int(cfg["n_mfcc"])
    use_parallel = bool(cfg["use_parallel"])
    random_state = int(cfg["random_state"])

    train_mode = str(cfg.get("train_mode", "clean")).lower()
    val_mode = str(cfg.get("val_mode", "clean")).lower()
    test_mode = str(cfg.get("test_mode", "clean")).lower()

    print("\n=== DATA MODES ===")
    print(f"Train mode: {train_mode}")
    print(f"Val mode  : {val_mode}")
    print(f"Test mode : {test_mode}")
    print("\n=== MODEL PATHS ===")
    print("DSCNN best model will be saved in:", models_dir)
    print("UNet best model:", unet_best_path)

    # Load CLEAN audio df
    clean_root = str(cfg["clean_dir"])
    folder_start = int(cfg["folder_start"])
    folder_end = int(cfg["folder_end"])

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

    # Compute MFCC for ALL clean df (then slice by SAME indices)
    clean_df = add_mfcc_column(
        clean_df,
        sr=sampling_rate,
        n_mfcc=n_mfcc,
        use_parallel=use_parallel
    )
    print("\nCLEAN after MFCC computed for ALL:")
    # print(tabulate(clean_df.head(1), headers="keys", tablefmt="psql", showindex=False))

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
    train_loader, val_loader_clean, test_loader_clean = make_loaders(
        X_train_padded, y_train_enc, fn_train,
        X_val_padded, y_val_enc, fn_val,
        X_test_padded, y_test_enc, fn_test,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle_train=bool(cfg["shuffle_train"])
    )

    print("Loader batches:",
              f"train={len(train_loader)}, val={len(val_loader_clean)}, test_clean={len(test_loader_clean)}")

    if val_mode == "clean":
        val_loader = val_loader_clean  # fixed clean validation
    else:
        pass
    
    if test_mode == "clean":
        test_loader = test_loader_clean
        
    else:
        # Build on-the-fly noisy test from CLEAN TEST AUDIO of THE SAME SPLIT
        test_audio_list = df_test_audio["audio_data"].values
        test_label_list = df_test_audio["label"].values
        test_filename_list = df_test_audio["filename"].values

        unet_model = None
        use_unet = bool(cfg.get("use_unet_denoiser", False))
        if use_unet:
            print("Loading UNet denoiser...")
            unet_model = UNetDenoiser().to(device)
            if os.path.exists(cfg["unet_model_path"]):
                ckpt = torch.load(cfg["unet_model_path"], map_location=device)
                unet_model.load_state_dict(ckpt)
            unet_model.eval()

        # Dataset for EVAL/TEST
        noisy_ds = NoisyTestDataset(
            audio_list=test_audio_list,
            labels_list=test_label_list,
            filenames_list=test_filename_list,
            label_encoder=label_encoder,
            scaler=scaler,
            sampling_rate=sampling_rate,
            n_mfcc=n_mfcc,
            max_len=max_len,
            bg_noise_dir=str(cfg["bg_noise_dir"]),
            noise_ops=list(cfg["noise_ops"]),
            snr_choices=list(cfg["snr_choices"]),
            random_noise_start=bool(cfg.get("random_noise_start", True)),
            noise_if_short=str(cfg.get("noise_if_short", "loop")),
            seed=int(cfg.get("noisy_test_seed", 123)),
            mode=test_mode,
            return_audio=True,   
            device=device,
            use_unet = use_unet,
            unet_model = unet_model if use_unet else None,
        )
        
        test_loader = DataLoader(
            noisy_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["num_workers"]),
        )

        print(f"TEST mode: {test_mode} (on-the-fly). size={len(noisy_ds)}, batches={len(test_loader)}")

        # show examples (TRAIN clean + TEST noisy)
        if bool(cfg["show_plots"]):
            # TRAIN clean example (from df_train) 
            row_train = df_train.iloc[2]
            plot_one_from_clean_df_row(
                row=row_train,
                scaler=scaler,
                sampling_rate=sampling_rate,
                n_mfcc=n_mfcc,
                title_prefix=f"TRAIN {train_mode} example"
            )

            plot_one_noisy_item(
                noisy_ds=noisy_ds,
                idx=1,
                sampling_rate=sampling_rate,
                n_mfcc=n_mfcc,
                title_prefix=f"TEST {test_mode} example"
            )


    # Train + Eval
    print("\n=== TRAIN DSCNN Model ===")
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

    best_model = load_model(DSCNN, num_classes, hist["best_path"], device=device)

    tr_loss, tr_acc = evaluate_loader(best_model, train_loader, device=device)
    va_loss, va_acc = evaluate_loader(best_model, val_loader, device=device)

    if test_mode != "clean":
        te_loss, te_acc, t_test, p_test, f_test, snr_test, noise_test = evaluate_loader(
        best_model, test_loader, device=device, return_meta=True)
    else:
        te_loss, te_acc = evaluate_loader(best_model, test_loader, device=device)
    
    print(f"\ntrain: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
    print(f"val: loss={va_loss:.4f}, acc={va_acc:.4f}")
    print(f"test: loss={te_loss:.4f}, acc={te_acc:.4f}")

    if bool(cfg["show_plots"]):
        plot_history(hist)

        if test_mode != "clean":
            print(f"\n=== NOISE ANALYSIS (mode={test_mode}) ===")
            plot_accuracy_vs_snr(t_test, p_test, snr_test)
            # plot_accuracy_per_noise(t_test, p_test, noise_test)
            plot_confusion_per_snr(t_test, p_test, snr_test, class_names)
            plot_confusion_per_noise(t_test, p_test, noise_test, class_names)

    cm_single  = bool(cfg.get("cm_single", True))
    cm_compare = bool(cfg.get("cm_compare", False))

    cm_val = cm_test = None
    t_val = p_val = f_val = None
    t_test = p_test = f_test = None

    if bool(cfg["make_confusion"]):
        cm_val, rep_val, (t_val, p_val, f_val) = confusion_and_report(
            best_model, val_loader, class_names, device,
            model_name=f"VAL {val_mode}" if cm_single else ""
        )

        cm_test, rep_test, (t_test, p_test, f_test) = confusion_and_report(
            best_model, test_loader, class_names, device,
            model_name=f"TEST {test_mode}" if cm_single else ""
        )

        if cm_compare:
            plot_two_confusion_matrices(
                cm_left=cm_val,
                cm_right=cm_test,
                class_names=class_names,
                title_left=f"VAL {val_mode}",
                title_right=f"TEST {test_mode}",
                suptitle="Confusion Matrices - VAL vs TEST"
            )

    # Misclassified list (based on VAL by default)
    if bool(cfg.get("mis_enabled", False)):
        if bool(cfg.get("mis_random_pair", True)):
            a, b = pick_random_class_pair(class_names, seed=int(cfg.get("mis_seed", 123)))
            print(f"\nRandom class pair for misclassified: A='{a}' vs B='{b}'")
        else:
            a = str(cfg["mis_true_label"])
            b = str(cfg["mis_pred_label"])

        test_mode = str(cfg.get("test_mode", "clean")).strip().lower()

        mis_df = misclassified_between_two_classes(
            t=t_test,
            p=p_test,
            f=f_test,
            class_names=class_names,
            class_a=a,
            class_b=b,
            snr=snr_test if test_mode != "clean" else None,
            noise=noise_test if test_mode != "clean" else None,
            max_rows=int(cfg.get("mis_max_rows", 20)),
        )

        if len(mis_df) == 0:
            print("No misclassifications found between these two classes.")
        else:
            print("\nMisclassified files between the pair:")
            print(tabulate(mis_df, headers="keys", tablefmt="psql", showindex=True))
            print("\nErrors by SNR:")
            print(mis_df.groupby("snr_db").size())
            print("\nErrors by noise:")
            print(mis_df.groupby("noise").size())


if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    run(cfg)
