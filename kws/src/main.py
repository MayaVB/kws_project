# main.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from tabulate import tabulate

from config import load_config

from dataset import (
    list_folders,
    collect_wav_paths,
    build_audio_dataframe,
    split_train_val_test,
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

from models import BasicDSCNN, ImprovedDSCNN
from train import get_device, train_model, load_model, evaluate_loader
# from train import set_seed
from metrics import plot_history, confusion_and_report, find_misclassified_files


def run(cfg: dict):
    # seed + device 
    #set_seed(int(cfg["seed"]))

    device = get_device(cfg["device"])
    print("Using device:", device)

    # folders + wav paths
    folders = list_folders(cfg["main_dir"])
    subset_folders = folders[int(cfg["folder_start"]): int(cfg["folder_end"])]
    print("Selected folders:", subset_folders)

    file_paths = collect_wav_paths(cfg["main_dir"], subset_folders)
    print("Total wav files:", len(file_paths))

    # load audio dataframe 
    audio_df = build_audio_dataframe(
        file_paths,
        sampling_rate=int(cfg["sampling_rate"]),
        use_parallel=bool(cfg["use_parallel"]),
    )
    print("Dataframe shape:", audio_df.shape)
    print(tabulate(audio_df.head(), headers="keys", tablefmt="psql", showindex=False))

    # MFCC 
    audio_df = add_mfcc_column(
        audio_df,
        sr=int(cfg["sampling_rate"]),
        n_mfcc=int(cfg["n_mfcc"]),
        use_parallel=bool(cfg["use_parallel"]),
    )
    print(tabulate(audio_df.head(), headers="keys", tablefmt="psql", showindex=False))

    # scaling 
    scaler = fit_scaler(audio_df["mfcc"].values)
    audio_df["scaled_mfcc"] = apply_scaler(audio_df["mfcc"].values, scaler)

    # optional plots
    if bool(cfg["show_plots"]):
        plot_audio_and_features(
            audio_df=audio_df,
            label_name=subset_folders[0],
            sampling_rate=int(cfg["sampling_rate"]),
            n_mfcc=int(cfg["n_mfcc"]),
            random_example=True,
        )

    # split 
    X_all = audio_df["scaled_mfcc"].values
    y_all = audio_df["label"].values
    fn_all = audio_df["filename"].values

    (X_train, y_train, fn_train), (X_val, y_val, fn_val), (X_test, y_test, fn_test) = split_train_val_test(
        X_all, y_all, fn_all,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        random_state=int(cfg["random_state"]),
    )

    # pad 
    global_max_len = max(m.shape[0] for m in X_all)
    X_train_p = pad_mfcc_list(X_train, max_len=global_max_len)
    X_val_p   = pad_mfcc_list(X_val,   max_len=global_max_len)
    X_test_p  = pad_mfcc_list(X_test,  max_len=global_max_len)
    print("Train padded shape:", X_train_p.shape)

    # label encoding 
    label_encoder, y_train_enc, y_val_enc, y_test_enc = make_label_encoder(y_train, y_val, y_test)
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    print("Classes:", list(class_names))

    # loaders 
    train_loader, val_loader, test_loader = make_loaders(
        X_train_p, y_train_enc, fn_train,
        X_val_p,   y_val_enc,   fn_val,
        X_test_p,  y_test_enc,  fn_test,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle_train=bool(cfg["shuffle_train"]),
    )

    # train/eval basic 
    if bool(cfg["train_basic"]):
        basic = BasicDSCNN(num_classes)
        hist_basic = train_model(
            basic, train_loader, val_loader,
            num_epochs=int(cfg["epochs"]),
            patience=int(cfg["patience"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            model_name="basic_dscnn",
            device=device,
        )
        if bool(cfg["show_plots"]):
            plot_history(hist_basic, title_prefix="BasicDSCNN")

        best_basic = load_model(BasicDSCNN, num_classes, hist_basic["best_path"], device=device)
        tr_loss, tr_acc = evaluate_loader(best_basic, train_loader, device=device)
        va_loss, va_acc = evaluate_loader(best_basic, val_loader, device=device)
        te_loss, te_acc = evaluate_loader(best_basic, test_loader, device=device)
        print(f"[BasicDSCNN] train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"[BasicDSCNN]   val: loss={va_loss:.4f}, acc={va_acc:.4f}")
        print(f"[BasicDSCNN]  test: loss={te_loss:.4f}, acc={te_acc:.4f}")

        if bool(cfg["make_confusion"]):
            confusion_and_report(best_basic, val_loader, class_names, device, model_name="BasicDSCNN (Validation)")

    # train/eval improved 
    if bool(cfg["train_improved"]):
        improved = ImprovedDSCNN(num_classes)
        hist_improved = train_model(
            improved, train_loader, val_loader,
            num_epochs=int(cfg["epochs"]),
            patience=int(cfg["patience"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
            model_name="improved_dscnn",
            device=device,
        )
        if bool(cfg["show_plots"]):
            plot_history(hist_improved, title_prefix="ImprovedDSCNN")

        best_improved = load_model(ImprovedDSCNN, num_classes, hist_improved["best_path"], device=device)
        tr_loss, tr_acc = evaluate_loader(best_improved, train_loader, device=device)
        va_loss, va_acc = evaluate_loader(best_improved, val_loader, device=device)
        te_loss, te_acc = evaluate_loader(best_improved, test_loader, device=device)
        print(f"[ImprovedDSCNN] train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"[ImprovedDSCNN]   val: loss={va_loss:.4f}, acc={va_acc:.4f}")
        print(f"[ImprovedDSCNN]  test: loss={te_loss:.4f}, acc={te_acc:.4f}")

        if bool(cfg["make_confusion"]):
            cm, rep, (t_imp, p_imp, f_imp) = confusion_and_report(
                best_improved, val_loader, class_names, device, model_name="ImprovedDSCNN (Validation)"
            )

            # Misclassified list
            if bool(cfg["mis_enabled"]) and (cfg["mis_true_label"] in class_names) and (cfg["mis_pred_label"] in class_names):
                mis_df = find_misclassified_files(
                    t_imp, p_imp, f_imp,
                    class_names,
                    true_label_name=str(cfg["mis_true_label"]),
                    pred_label_name=str(cfg["mis_pred_label"]),
                )
                print("\nMisclassified files:")
                print(mis_df.head(int(cfg["mis_max_rows"])).to_string(index=False))


if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    run(cfg)
