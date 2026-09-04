"""
run_training.py

Main training and evaluation pipeline for the
Keyword Spotting project.

Pipeline overview
-----------------
1. Load configuration
2. Load metadata
3. Build train/validation/test datasets
4. Extract MFCC features
5. Fit feature scaler
6. Train DS-CNN model
7. Evaluate multiple modes:
   - clean
   - noisy
   - enhanced baseline
   - enhanced trained
8. Generate reports and visualizations
9. Save demo artifacts

Outputs
-------
- trained model
- scaler
- label encoder
- confusion matrices
- classification reports
- speech enhancement metrics
- demo CSV files
"""
from __future__ import annotations
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import time
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from new_noise_dataset import FixedNoisyDataset
from enhanced_dataset import EnhancedTestDataset
from dataset import build_audio_dataframe, build_paths
from metrics import  confusion_and_report, pick_random_class_pair
from models import DSCNN
from visualization import plot_example_signals, plot_history, plot_confusion_comparison
from config import load_config
from train import get_device, load_model, train_model, evaluate_loader
from dataset import make_label_encoder, make_loaders, pad_mfcc_list
from features import add_mfcc_column, apply_scaler, fit_scaler
from analysis_utils import analyze_mode, save_prediction_reports, save_metrics_summary
from kws.demo.demo_utils import compare_prediction_modes

def run(cfg: dict):
    total_start = time.time()
    # =====================================
    # LOAD CONFIGURATION
    # =====================================
    device = get_device(cfg["device"])
    sampling_rate = int(cfg["sampling_rate"])
    n_mfcc = int(cfg["n_mfcc"])
    output_dir = Path(cfg["output_dir"])
    clean_root = str(cfg["clean_dir"])
    folder_start = int(cfg["folder_start"])
    folder_end = int(cfg["folder_end"])

    meta_path = str(cfg["meta"])  
    noisy_root = str(cfg["noisy_dir"])
    enh_new_dir = str(cfg["enh_new_dir"])
    enh_trained = str(cfg["enh_trained"])
    enh_baseline = str(cfg["enh_baseline"])

    batch_size = int(cfg["batch_size"])
    use_parallel = bool(cfg["use_parallel"])
    random_state = int(cfg["random_state"])

    # =====================================
    # CREATE OUTPUT DIRECTORIES
    # =====================================
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / f"{run_stamp}"
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"

    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Run dir:", run_dir)

    # =====================================
    # LOAD DATASET METADATA
    # =====================================
    meta_df = pd.read_csv(meta_path) 

    # SELECT FOLDERS
    all_folders = sorted(meta_df["label"].unique())
    if "selected_folders" in cfg and cfg["selected_folders"]:
        selected_folders = cfg["selected_folders"]
    else:
        selected_folders = all_folders[
            int(cfg["folder_start"]):
            int(cfg["folder_end"])
        ]
    print("Selected folders:", selected_folders)

    df_train_meta = meta_df[meta_df["split"] == "train"].reset_index(drop=True)
    df_val_meta   = meta_df[meta_df["split"] == "val"].reset_index(drop=True)
    df_test_meta  = meta_df[meta_df["split"] == "test"].reset_index(drop=True)

    # FILTER TO SELECTED FOLDERS
    df_train_meta = df_train_meta[df_train_meta["label"].isin(selected_folders)].reset_index(drop=True)
    df_val_meta   = df_val_meta[df_val_meta["label"].isin(selected_folders)].reset_index(drop=True)
    df_test_meta  = df_test_meta[df_test_meta["label"].isin(selected_folders)].reset_index(drop=True)

    print(f"\nSPLIT: train={len(df_train_meta)}, val={len(df_val_meta)}, test={len(df_test_meta)}")

    # =====================================
    # BUILD CLEAN DATASETS
    # =====================================
    train_paths = build_paths(df_train_meta, clean_root)
    val_paths   = build_paths(df_val_meta, clean_root)
    test_paths  = build_paths(df_test_meta, clean_root)

    df_train_audio = build_audio_dataframe(train_paths, sampling_rate, use_parallel)
    df_val_audio   = build_audio_dataframe(val_paths, sampling_rate, use_parallel)
    df_test_audio  = build_audio_dataframe(test_paths, sampling_rate, use_parallel)

    # =====================================
    # MFCC EXTRACTION AND NORMALIZATION
    # =====================================
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

    # =====================================
    # CREATE DATALOADERS
    # =====================================
    train_loader, val_loader, test_loader_clean = make_loaders(
        X_train, y_train_enc, fn_train,
        X_val,   y_val_enc,   fn_val,
        X_test,  y_test_enc,  fn_test,
        batch_size=batch_size,
        num_workers=int(cfg["num_workers"]),
        shuffle_train=bool(cfg["shuffle_train"])
    )   

    # =====================================
    # SAVE PREPROCESSING ARTIFACTS
    # =====================================
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    print("\nSaved scaler + label encoder")

    
    # =====================================
    # TRAIN MODEL
    # =====================================
    model = DSCNN(len(class_names))

    train_start = time.time()

    hist = train_model(
        model, train_loader, val_loader,
        num_epochs=int(cfg["epochs"]),
        patience=int(cfg["patience"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        device=device,
        best_dir=models_dir
    )

    train_time = time.time() - train_start
    print(f"\nTRAINING TIME: {train_time/60:.2f} minutes")

    best_model = load_model(DSCNN, len(class_names), hist["best_path"], device=device)
    plot_history(hist, save_path=plots_dir / "training_history.png")
    

    # LOAD BEST MODEL (TRAIN ALL)
    # best_model_path = cfg["best_model_path"]
    # best_model = load_model(DSCNN, len(class_names), best_model_path, device=device)

    # =====================================
    # DEFINE EVALUATION MODES
    # =====================================
    modes_config = {

        "clean": {
            "type": "clean"
        },

        "noisy": {
            "type": "noisy",
            "root": os.path.join(noisy_root, "test")
        },
        
        "enh_baseline": {
            "type": "enhanced",
            "root": os.path.join(enh_baseline)
        },
        "enh_trained": {
            "type": "enhanced",
            "root": os.path.join(enh_trained)
        } 
    }

    """
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
    """
    # =====================================
    # PREPARE ANALYSIS
    # =====================================
    # PICK RANDOM CLASS PAIR FOR MISCLASSIFICATION ANALYSIS
    a, b = pick_random_class_pair(class_names, seed=int(cfg.get("mis_seed", 123)))
    print(f"Random pair for misclassification analysis: {a} vs {b}")

    results = {}
    all_metrics = []

    # =====================================
    # EVALUATE ALL MODES
    # =====================================
    for mode_name, mode_cfg in modes_config.items():
        print(f"\n=== {mode_name.upper()} ===")
        mode_type = mode_cfg["type"]

        eval_start = time.time()

        # CLEAN
        if mode_type == "clean":
            loader = test_loader_clean

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

        
        # SAVE MISCLASSIFICATIONS FOR GRADIO
        pred_df, mis_df = save_prediction_reports(
            t=t,
            p=p,
            f=f,
            mode_name=mode_name,
            label_encoder=label_encoder,
            run_dir=run_dir,
            snr=snr,
            noise=noise
        )
        

        eval_time = time.time() - eval_start

        print(
            f"\nEVALUATION TIME ({mode_name}): "
            f"{eval_time:.2f} sec")

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
        
        cm, _, _ = confusion_and_report(
        best_model,
        loader,
        class_names,
        device=device,
        model_name=mode_name,
        save_txt_path=run_dir / "reports.txt")

        # POST-PROCESSING ANALYSIS
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
            noisy_root=os.path.join(noisy_root, "test"),
            enhanced_root=(mode_cfg["root"] if mode_type == "enhanced" else None)
        )
        if metrics_dict is not None:
            all_metrics.append(metrics_dict)

    # =====================================
    # COMPARE MODES
    # =====================================

    compare_prediction_modes(
        clean_csv=run_dir /
        "clean_all_predictions.csv",

        noisy_csv=run_dir /
        "noisy_all_predictions.csv",

        enh_csv=run_dir /
        "enh_trained_all_predictions.csv",

        output_dir=run_dir
    )

    enhanced_versions = {}
    for mode_name, mode_cfg in modes_config.items():
        if mode_cfg["type"] == "enhanced":
            enhanced_versions[mode_name] = mode_cfg["root"]
    
    # =====================================
    # GENERATE EXAMPLE VISUALIZATIONS
    # =====================================
    plot_example_signals(
    df_test_audio=df_test_audio,
    sampling_rate=sampling_rate,
    plots_dir=plots_dir,
    noisy_root=noisy_root,
    enhanced_versions=enhanced_versions,
    idx_vis=8,
    run_dir=run_dir
    )

    # =====================================
    # CONFUSION MATRIX COMPARISON
    # =====================================
    print("\n=== CONFUSION MATRICES COMPARISON ===")
    plot_confusion_comparison(
        results=results,
        class_names=class_names,
        plots_dir=plots_dir,
        confusion_modes=[
            "clean",
            "noisy",
            "enh_trained"
        ]
    )

    # =====================================
    # SAVE FINAL METRICS SUMMARY
    # =====================================
    save_metrics_summary(
        all_metrics,
        run_dir
    )

    # =====================================
    # FINAL RUNTIME REPORT
    # =====================================
    total_time = time.time() - total_start
    print("\n===================================")
    print(f"TOTAL PROGRAM TIME: {total_time/60:.2f} minutes")
    print("===================================\n")

    with open(run_dir / "runtime_report.txt", "a") as f:
        f.write(f"\nTOTAL PROGRAM TIME: {total_time/60:.2f} minutes\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the DS-CNN keyword spotting model.")
    parser.add_argument("--config", type=str, default=None,
        help=(
            "Path to the YAML configuration file. "
            "Defaults to new_config.yaml in the KWS directory."
        ),)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config is None:
        cfg_path = Path(__file__).resolve().parents[1] / "new_config.yaml"
    else:
        cfg_path = Path(args.config).expanduser().resolve()

    print(f"Using configuration: {cfg_path}")

    cfg = load_config(cfg_path)
    run(cfg)
