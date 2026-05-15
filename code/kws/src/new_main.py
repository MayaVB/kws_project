# new_main.py
from __future__ import annotations
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import json
import librosa
import math
import numpy as np
import pandas as pd
from tabulate import tabulate
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
    enh_new_dir = str(cfg["enh_new_dir"])

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
    # test_modes = ["clean", "noisy", "denoised", "enhanced_sgmse", "enhanced_trained_ep100"]

    modes_config = {

        "clean": {
            "type": "clean"
        },

        "noisy": {
            "type": "noisy",
            "root": os.path.join(noisy_root, "test")
        },

        "denoised": {
            "type": "denoised",
            "root": os.path.join(noisy_root, "test")
        },
    }

    enhanced_parent = Path(enh_new_dir)
    for folder in sorted(enhanced_parent.iterdir()):

        if not folder.is_dir():
            continue

        name = folder.name.lower()
        mode_name = f"enh_{name}"
        modes_config[mode_name] = {
            "type": "enhanced",
            "root": str(folder)
        }

    # PICK RANDOM CLASS PAIR FOR MISCLASSIFICATION ANALYSIS
    a, b = pick_random_class_pair(class_names, seed=int(cfg.get("mis_seed", 123)))
    print(f"Random pair for misclassification analysis: {a} vs {b}")

    results = {}
    all_metrics = []
    
    for mode_name, mode_cfg in modes_config.items():
        print(f"\n=== {mode_name.upper()} ===")
        mode_type = mode_cfg["type"]

        # CLEAN
        if mode_type == "clean":
            loader = test_loader_clean
            loss, acc, t, p, f, *_ = evaluate_loader(
                best_model,
                loader,
                device=device,
                return_meta=True
            )
            snr = None
            noise = None
            t_c, p_c = t, p  

        # NOISY
        elif mode_type == "noisy":
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
        elif mode_type == "denoised":
            base = FixedNoisyDataset(
                root=mode_cfg["root"],
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
                    root=mode_cfg["root"]   
                ),
                batch_size=batch_size,
                shuffle=False
)

        # ENHANCED 
        elif mode_type == "enhanced":
            loader = DataLoader(
                EnhancedTestDataset(
                    labels_list=df_test["label"].values,
                    filenames_list=df_test["filename"].values,
                    label_encoder=label_encoder,
                    scaler=scaler,
                    sampling_rate=sampling_rate,
                    n_mfcc=n_mfcc,
                    max_len=X_train.shape[1],
                    enhanced_root=mode_cfg["root"],
                    meta_csv=meta_path
                ),
                batch_size=batch_size,
                shuffle=False
            )

        # EVALUATION
        loss, acc, t, p, f, snr, noise = evaluate_loader(
            best_model, loader, device=device, return_meta=True
        )

        if mode_type == "clean":
            snr = None
            noise = None

        results[mode_name] = {
            "acc": acc,
            "loss": loss,
            "t": t,
            "p": p,
            "f": f,
            "snr": snr,
            "noise": noise
        }

        print(f"\ntest ({mode_name}): loss={loss:.4f}, acc={acc:.4f}\n")
        
        cm, rep, _ = confusion_and_report(
        best_model,
        loader,
        class_names,
        device=device,
        model_name=mode_name,
        save_txt_path=run_dir / "reports.txt")

        metrics_dict = analyze_mode(
            acc,
            mode=mode_name,
            results=results[mode_name],
            class_names=class_names,
            plots_dir=plots_dir,
            run_dir=run_dir,
            a=a,
            b=b,
            df_test_audio=df_test_audio,
            sampling_rate=sampling_rate,
            enhanced_root=(mode_cfg["root"] if mode_type == "enhanced" else None)
        )
        if metrics_dict is not None:
            all_metrics.append(metrics_dict)

    enhanced_versions = {}
    for mode_name, mode_cfg in modes_config.items():
        if mode_cfg["type"] == "enhanced":
            enhanced_versions[mode_name] = mode_cfg["root"]
    
    plot_example_signals(
    df_test_audio=df_test_audio,
    sampling_rate=sampling_rate,
    plots_dir=plots_dir,
    noisy_root=noisy_root,
    enhanced_versions=enhanced_versions,
    idx_vis=8,
    )

    # CONFUSION MATRICES
    print("\n=== CONFUSION MATRICES COMPARISON ===")
    
    pairs = []

    for mode_name, mode_res in results.items():
        pairs.append((mode_name, mode_res["t"], mode_res["p"]))

    n_modes = len(pairs)
    ncols = 3
    nrows = math.ceil(n_modes / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5*ncols, 4*nrows)
    )

    axes = np.array(axes).reshape(-1)

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
        axes[i].set_title(name)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

        single_fig, single_ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            ax=single_ax,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        single_ax.set_title(name)
        plt.tight_layout()
        plt.savefig(
            plots_dir / f"confusion_{name}.png"
        )
        plt.close(single_fig)

    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

        

    print("DEBUG: finished loop")

    # hide empty subplot
    # axes[-1].axis("off")

    plt.suptitle("Confusion Matrices Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plots_dir / "confusion_matrices.png")
    plt.close()

    if len(all_metrics) > 0:
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics = df_metrics.fillna("-")
        # df_metrics = df_metrics.sort_values("pesq", ascending=False)
        print("\n=== METRICS TABLE ===")
        print(df_metrics)
        df_metrics.to_csv(
            run_dir / "metrics_comparison.csv",
            index=False
        )

        table_str = tabulate(
        df_metrics,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )

    print(table_str)

    with open(run_dir / "metrics_comparison.txt", "w") as f:
        f.write(table_str)


if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parents[1] / "new_config.yaml"
    cfg = load_config(cfg_path)
    run(cfg)