# new_main.py
from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import json
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
from torch.utils.data import DataLoader

from new_noise_dataset import FixedNoisyDataset
from new_denoised_dataset import DenoisedDataset
from enhanced_dataset import EnhancedTestDataset

from dataset import build_audio_dataframe
from metrics import  confusion_and_report, pick_random_class_pair, plot_history
from models import DSCNN
from visualization import plot_example_signals
from config import load_config
from train import get_device, load_model, train_model, evaluate_loader
from dataset import list_folders, collect_wav_paths, make_label_encoder, make_loaders, pad_mfcc_list
from features import add_mfcc_column, apply_scaler, fit_scaler
from analysis_utils import analyze_mode

def run(cfg: dict):
        
    # CONFIG
    device = get_device(cfg["device"])
    sampling_rate = int(cfg["sampling_rate"])
    n_mfcc = int(cfg["n_mfcc"])
    output_dir = Path(cfg["output_dir"])
    clean_root = str(cfg["clean_dir"])
    folder_start = int(cfg["folder_start"])
    folder_end = int(cfg["folder_end"])

    meta_path = str(cfg["meta"])   # ✅ FIX
    noisy_root = str(cfg["noisy_dir"])
    enh_pretrained = str(cfg["enh_pretrained"])
    enh_trained = str(cfg["enh_trained"])

    batch_size = int(cfg["batch_size"])
    use_parallel = bool(cfg["use_parallel"])
    random_state = int(cfg["random_state"])

    # DIRECTORIES
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"{run_stamp}"
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"

    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Run dir:", run_dir)

    # LOAD METADATA
    meta_df = pd.read_csv(meta_path)   # ✅ FIX

    # SELECT FOLDERS
    all_folders = sorted(meta_df["label"].unique())
    selected_folders = all_folders[folder_start:folder_end]
    print("Selected folders:", selected_folders)

    df_train_meta = meta_df[meta_df["split"] == "train"].reset_index(drop=True)
    df_val_meta   = meta_df[meta_df["split"] == "val"].reset_index(drop=True)
    df_test_meta  = meta_df[meta_df["split"] == "test"].reset_index(drop=True)

    # FILTER TO SELECTED FOLDERS
    df_train_meta = df_train_meta[df_train_meta["label"].isin(selected_folders)].reset_index(drop=True)
    df_val_meta   = df_val_meta[df_val_meta["label"].isin(selected_folders)].reset_index(drop=True)
    df_test_meta  = df_test_meta[df_test_meta["label"].isin(selected_folders)].reset_index(drop=True)

    print(f"\nSPLIT: train={len(df_train_meta)}, val={len(df_val_meta)}, test={len(df_test_meta)}")

    # BUILD CLEAN AUDIO
    def build_paths(df):
        return [
            os.path.join(clean_root, row["label"], row["filename"])
            for _, row in df.iterrows()
        ]

    train_paths = build_paths(df_train_meta)
    val_paths   = build_paths(df_val_meta)
    test_paths  = build_paths(df_test_meta)

    df_train_audio = build_audio_dataframe(train_paths, sampling_rate, use_parallel)
    df_val_audio   = build_audio_dataframe(val_paths, sampling_rate, use_parallel)
    df_test_audio  = build_audio_dataframe(test_paths, sampling_rate, use_parallel)

    # MFCC
    df_train = add_mfcc_column(df_train_audio, sr=sampling_rate, n_mfcc=n_mfcc, use_parallel=use_parallel)
    df_val   = add_mfcc_column(df_val_audio,   sr=sampling_rate, n_mfcc=n_mfcc, use_parallel=use_parallel)
    df_test  = add_mfcc_column(df_test_audio,  sr=sampling_rate, n_mfcc=n_mfcc, use_parallel=use_parallel)

    scaler = fit_scaler(df_train["mfcc"].values)

    X_train = pad_mfcc_list(apply_scaler(df_train["mfcc"].values, scaler))
    X_val   = pad_mfcc_list(apply_scaler(df_val["mfcc"].values, scaler))
    X_test  = pad_mfcc_list(apply_scaler(df_test["mfcc"].values, scaler))

    y_train = df_train["label"].values
    y_val   = df_val["label"].values
    y_test  = df_test["label"].values

    label_encoder, y_train_enc, y_val_enc, y_test_enc = make_label_encoder(y_train, y_val, y_test)

    class_names = label_encoder.classes_

    fn_train = df_train["filename"].values
    fn_val   = df_val["filename"].values
    fn_test  = df_test["filename"].values

    train_loader, val_loader, test_loader_clean = make_loaders(
        X_train, y_train_enc, fn_train,
        X_val,   y_val_enc,   fn_val,
        X_test,  y_test_enc,  fn_test,
        batch_size=batch_size,
        num_workers=int(cfg["num_workers"]),
        shuffle_train=bool(cfg["shuffle_train"])
    )

    # TRAIN
    model = DSCNN(len(class_names))

    hist = train_model(
        model, train_loader, val_loader,
        num_epochs=int(cfg["epochs"]),
        patience=int(cfg["patience"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        device=device,
        best_dir=models_dir
    )

    best_model = load_model(DSCNN, len(class_names), hist["best_path"], device=device)
    plot_history(hist, save_path=plots_dir / "training_history.png")

    # TEST MODES
    # test_modes = ["clean", "noisy", "denoised", "enhanced_sgmse"]
    test_modes = ["clean", "noisy", "denoised", "enhanced_sgmse", "enhanced_trained_ep100"]

    # PICK RANDOM CLASS PAIR FOR MISCLASSIFICATION ANALYSIS
    a, b = pick_random_class_pair(class_names, seed=int(cfg.get("mis_seed", 123)))
    print(f"Random pair for misclassification analysis: {a} vs {b}")

    results = {}

    for mode in test_modes:

        print(f"\n=== {mode.upper()} ===")

        # CLEAN
        if mode == "clean":
            loader = test_loader_clean

            loss, acc, t, p, f, *_ = evaluate_loader(
                best_model, loader, device=device, return_meta=True
            )
            snr = None
            noise = None
            t_c, p_c = t, p  

        # NOISY
        elif mode == "noisy":
            loader = DataLoader(
                FixedNoisyDataset(
                    root=os.path.join(noisy_root, "test"),
                    labels_list=df_test["label"].values,
                    filenames_list=df_test["filename"].values,
                    label_encoder=label_encoder,
                    scaler=scaler,
                    sampling_rate=sampling_rate,
                    n_mfcc=n_mfcc,
                    max_len=X_train.shape[1],
                    meta_csv=meta_path,
                    split="test"   
                ),
                batch_size=batch_size,
                shuffle=False
            )

        # DENOISED
        elif mode == "denoised":
            base = FixedNoisyDataset(
                root=os.path.join(noisy_root, "test"),
                labels_list=df_test["label"].values,
                filenames_list=df_test["filename"].values,
                label_encoder=label_encoder,
                scaler=scaler,
                sampling_rate=sampling_rate,
                n_mfcc=n_mfcc,
                max_len=X_train.shape[1],
                meta_csv=meta_path,
                split="test"
            )

            loader = DataLoader(
                DenoisedDataset(
                    base,
                    sampling_rate,
                    n_mfcc,
                    scaler,
                    X_train.shape[1],
                    root=os.path.join(noisy_root, "test")   
                ),
                batch_size=batch_size,
                shuffle=False
)

        # ENHANCED PRETRAINED
        elif mode == "enhanced_sgmse":
            loader = DataLoader(
                EnhancedTestDataset(
                    labels_list=df_test["label"].values,
                    filenames_list=df_test["filename"].values,
                    label_encoder=label_encoder,
                    scaler=scaler,
                    sampling_rate=sampling_rate,
                    n_mfcc=n_mfcc,
                    max_len=X_train.shape[1],
                    enhanced_root=enh_pretrained,
                    meta_csv=meta_path
                ),
                batch_size=batch_size,
                shuffle=False
            )

        # ENHANCED TRAINED
        elif mode == "enhanced_trained_ep100":
            loader = DataLoader(
                EnhancedTestDataset(
                    labels_list=df_test["label"].values,
                    filenames_list=df_test["filename"].values,
                    label_encoder=label_encoder,
                    scaler=scaler,
                    sampling_rate=sampling_rate,
                    n_mfcc=n_mfcc,
                    max_len=X_train.shape[1],
                    enhanced_root=enh_trained,
                    meta_csv=meta_path
                ),
                batch_size=batch_size,
                shuffle=False
            )

        # EVALUATION
        loss, acc, t, p, f, snr, noise = evaluate_loader(
            best_model, loader, device=device, return_meta=True
        )

        if mode == "clean":
            snr = None
            noise = None

        results[mode] = {
            "acc": acc,
            "loss": loss,
            "t": t,
            "p": p,
            "f": f,
            "snr": snr,
            "noise": noise
        }

        print(f"\ntest ({mode}): loss={loss:.4f}, acc={acc:.4f}\n")
        
        cm, rep, _ = confusion_and_report(
        best_model,
        loader,
        class_names,
        device=device,
        model_name=mode,
        save_txt_path=run_dir / "reports.txt")

        analyze_mode(
            mode=mode,
            results=results[mode],
            class_names=class_names,
            plots_dir=plots_dir,
            run_dir=run_dir,
            a=a,
            b=b,
            df_test_audio=df_test_audio,
            sampling_rate=sampling_rate,
            enhanced_root=(enh_pretrained if mode == "enhanced_sgmse" else
            enh_trained if mode == "enhanced_trained_ep100" else None)
        )

    plot_example_signals(
    df_test_audio=df_test_audio,
    sampling_rate=sampling_rate,
    plots_dir=plots_dir,
    noisy_root=noisy_root,
    enh_pretrained=enh_pretrained,
    enh_trained=enh_trained,
    idx_vis=8,)

    # CONFUSION MATRICES
    print("\n=== CONFUSION MATRICES COMPARISON ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    # fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    titles = [
        "Clean Test",
        "Noisy Test",
        "Denoised Test",
        "Enhanced SGMSE Test",
        "Enhanced Trained Ep100 Test"
    ]
    pairs = [
        ("Clean", t_c, p_c),
        ("Noisy", results["noisy"]["t"], results["noisy"]["p"]),
        ("Denoised", results["denoised"]["t"], results["denoised"]["p"]),
        ("Enhanced SGMSE", results["enhanced_sgmse"]["t"], results["enhanced_sgmse"]["p"]),
        ("Enhanced Trained Ep100", results["enhanced_trained_ep100"]["t"], results["enhanced_trained_ep100"]["p"]),
    ]

    print("DEBUG: starting confusion matrices block")

    for i, (name, t, p) in enumerate(pairs):
        print(f"DEBUG: {name} len={len(t)}")
        cm = confusion_matrix(t, p)
        sns.heatmap(
            cm,
            ax=axes[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    print("DEBUG: finished loop")

    # hide empty subplot
    # axes[-1].axis("off")

    plt.suptitle("Confusion Matrices Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plots_dir / "confusion_matrices.png")
    plt.close()


if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parents[1] / "new_config.yaml"
    cfg = load_config(cfg_path)
    run(cfg)