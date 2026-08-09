# =========================================================
# new_main_inference.py
# INFERENCE ONLY PIPELINE
# uses:
#   - pretrained 30-class model
#   - saved scaler
#   - saved label encoder
#
# supports:
#   - evaluating only selected folders
#   - clean / noisy / enhanced modes
#   - confusion matrices
#   - metrics comparison
#   - runtime measurements
# =========================================================

from __future__ import annotations
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
)

import time
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from dataset import build_paths

# PROJECT IMPORTS
from config import load_config
from models import DSCNN
from dataset import build_audio_dataframe, make_test_loader, pad_mfcc_list
from features import add_mfcc_column, apply_scaler
from train import get_device, load_model, evaluate_loader
from metrics import confusion_and_report, pick_random_class_pair
from visualization import plot_example_signals, plot_confusion_comparison
from analysis_utils import analyze_mode, save_metrics_summary
from new_noise_dataset import FixedNoisyDataset
from enhanced_dataset import EnhancedTestDataset

# MAIN
def run(cfg: dict):

    total_start = time.time()
    # =====================================
    # LOAD CONFIGURATION
    # =====================================
    device = get_device(cfg["device"])
    sampling_rate = int(cfg["sampling_rate"])
    n_mfcc = int(cfg["n_mfcc"])
    batch_size = int(cfg["batch_size"])
    use_parallel = bool(cfg["use_parallel"])
    # folder_start = int(cfg["folder_start"])
    # folder_end = int(cfg["folder_end"])

    # =====================================
    # LOAD PATHS
    # =====================================
    output_dir = Path(cfg["output_dir"])
    meta_path = str(cfg["meta"])
    clean_root = str(cfg["clean_dir"])
    noisy_root = str(cfg["noisy_dir"])
    enh_baseline = str(cfg["enh_baseline"])
    enh_trained = str(cfg["enh_trained"])
    best_model_path = str(cfg["best_model_path"])
    scaler_path = str(cfg["scaler_path"])
    label_encoder_path = str(cfg["label_encoder_path"])

    # =====================================
    # CREATE OUTPUT DIRECTORIES
    # =====================================
    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = output_dir / run_stamp
    plots_dir = run_dir / "plots"
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Using device:", device)
    print("Run dir:", run_dir)

    # =====================================
    # LOAD SAVED ARTIFACTS
    # =====================================
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    all_class_names = label_encoder.classes_
    print("\nALL MODEL CLASSES:")
    print(all_class_names)

    # =====================================
    # LOAD DATASET METADATA
    # =====================================
    meta_df = pd.read_csv(meta_path)

    # SELECTED FOLDERS
    all_folders = sorted(meta_df["label"].unique())
    # selected_folders = all_folders[folder_start:folder_end]
    print("\nSELECTED FOLDERS:")
    print(all_folders)

    # TEST META ONLY
    df_test_meta = meta_df[meta_df["split"] == "test"].reset_index(drop=True)

    # FILTER TO SELECTED FOLDERS
    df_test_meta = df_test_meta[df_test_meta["label"].isin(all_folders)].reset_index(drop=True)
    print(f"\nTEST FILES: {len(df_test_meta)}")

    # =====================================
    # BUILD TEST DATASET
    # =====================================
    test_paths = build_paths(
        df_test_meta,
        clean_root
    )

    # BUILD AUDIO
    df_test_audio = build_audio_dataframe(
        test_paths,
        sampling_rate,
        use_parallel
    )

    # =====================================
    # MFCC EXTRACTION AND NORMALIZATION
    # =====================================
    df_test = add_mfcc_column(
        df_test_audio,
        sr=sampling_rate,
        n_mfcc=n_mfcc,
        use_parallel=use_parallel
    )

    # APPLY SAVED SCALER
    X_test = pad_mfcc_list(
        apply_scaler(
            df_test["mfcc"].values,
            scaler
        )
    )

    max_len = X_test.shape[1]

    # LABELS
    y_test = df_test["label"].values
    y_test_enc = label_encoder.transform(y_test)

    # FILENAMES
    fn_test = df_test["filename"].values

    # =====================================
    # CREATE TEST DATALOADER
    # =====================================
    test_loader_clean = make_test_loader(
        X_test,
        y_test_enc,
        fn_test,
        batch_size=batch_size,
        num_workers=int(cfg["num_workers"])
    )

    # =====================================
    # LOAD PRETRAINED MODEL
    # =====================================
    best_model = load_model(
        DSCNN,
        len(all_class_names),   
        best_model_path,
        device=device
    )

    print("\nMODEL LOADED SUCCESSFULLY")

    # PARAMS
    total_params = sum(p.numel() for p in best_model.parameters())
    print("\n===================================")
    print(f"Total params: {total_params:,}")
    print("===================================\n")

    # =====================================
    # DEFINE EVALUATION MODES
    # =====================================
    modes_config = {

        "clean": {
            "type": "clean"
        },

        "noisy": {
            "type": "noisy",
            "root": os.path.join(
                noisy_root,
                "test"
            )
        },

        "enh_baseline": {
            "type": "enhanced",
            "root": enh_baseline
        },

        "enh_trained": {
            "type": "enhanced",
            "root": enh_trained
        }
    }

    # =====================================
    # PREPARE ANALYSIS
    # =====================================
    # RANDOM CLASS PAIR
    a, b = pick_random_class_pair(
        all_folders,
        seed=int(cfg.get("mis_seed", 123))
    )

    print(
        f"\nRandom pair for analysis: "
        f"{a} vs {b}"
    )

    # RESULTS
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
                    root=mode_cfg["root"],
                    labels_list=df_test["label"].values,
                    filenames_list=df_test["filename"].values,
                    label_encoder=label_encoder,
                    scaler=scaler,
                    sampling_rate=sampling_rate,
                    n_mfcc=n_mfcc,
                    max_len=max_len,
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
                    max_len=max_len,
                    enhanced_root=mode_cfg["root"],
                    meta_csv=meta_path
                ),
                batch_size=batch_size,
                shuffle=False
            )

        # GLABAL ACC (because evaluae_model run on best_model = all folders-> 30-class classifier, 
        # so there is 30 possible outputs)
        # EVALUATION
        loss, acc, t, p, f, snr, noise = evaluate_loader(
            best_model,
            loader,
            device=device,
            return_meta=True
        )
        
        eval_time = time.time() - eval_start
        print(
            f"\nEVALUATION TIME ({mode_name}): "
            f"{eval_time:.2f} sec"
        )

        if mode_type == "clean":
            snr = None
            noise = None

        # SAVE RESULTS
        results[mode_name] = {
            "acc": acc,
            "loss": loss,
            "t": t,
            "p": p,
            "f": f,
            "snr": snr,
            "noise": noise
        }

        print(
            f"\nTEST ({mode_name}): "
            f"loss={loss:.4f}, "
            f"acc={acc:.4f}\n"
        )

        # CONFUSION MATRIX
        confusion_and_report(
            best_model,
            loader,
            all_folders,
            device=device,
            model_name=mode_name,
            save_txt_path=run_dir / "reports.txt"
        )

        # ANALYSIS
        metrics_dict = analyze_mode(
            acc,
            mode=mode_name,
            results=results[mode_name],
            class_names=all_folders,
            plots_dir=plots_dir,
            run_dir=run_dir,
            a=a,
            b=b,
            df_test_audio=df_test_audio,
            sampling_rate=sampling_rate,
            enhanced_root=(
                mode_cfg["root"]
                if mode_type == "enhanced"
                else None
            )
        )

        if metrics_dict is not None:
            all_metrics.append(metrics_dict)

    # =====================================
    # GENERATE EXAMPLE VISUALIZATIONS
    # =====================================
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
        idx_vis=0,
        run_dir=run_dir
    )

    # =====================================
    # CONFUSION MATRIX COMPARISON
    # =====================================
    print("\n=== CONFUSION MATRICES COMPARISON ===")
    plot_confusion_comparison(
        results=results,
        class_names=all_class_names,
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
    print(
        f"TOTAL PROGRAM TIME: "
        f"{total_time / 60:.2f} minutes")
    print("===================================\n")


# ENTRY
if __name__ == "__main__":
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "new_config.yaml"
    )

    cfg = load_config(cfg_path)
    run(cfg)