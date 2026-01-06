from __future__ import annotations
import numpy as np
from pathlib import Path
from tabulate import tabulate
from config import load_config
from torch.utils.data import DataLoader
from models import BasicDSCNN, ImprovedDSCNN
from train import get_device, train_model, load_model, evaluate_loader
from metrics import plot_history, confusion_and_report, find_misclassified_files

from dataset import (
    list_folders,
    collect_wav_paths,
    build_audio_dataframe,
    split_train_val_test,
    pad_mfcc_list,
    make_label_encoder,
    make_loaders,
    MFCCDataset
)

from features import (
    add_mfcc_column,
    fit_scaler,
    apply_scaler,
    plot_audio_and_features,
)

def run(cfg: dict):
    # device
    device = get_device(cfg["device"])
    print("Using device:", device)

    # paths for this run 
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    print("Run dir:", run_dir)
    
    # parametes
    sampling_rate = int(cfg["sampling_rate"])
    n_mfcc = int(cfg["n_mfcc"])
    use_parallel = bool(cfg["use_parallel"])

    # load clean data
    clean_root = cfg["data_sources"]["clean_dir"]
    clean_fs, clean_fe = cfg["folders"]["clean"]

    clean_folders = list_folders(clean_root)
    clean_subset_folders = clean_folders[int(clean_fs):int(clean_fe)]
    print("CLEAN selected folders:", clean_subset_folders)

    clean_paths = collect_wav_paths(clean_root, clean_subset_folders)
    print("CLEAN total wav files:", len(clean_paths))

    clean_df = build_audio_dataframe(
        clean_paths,
        sampling_rate=sampling_rate,
        use_parallel=use_parallel
    )
    print("CLEAN df shape:", clean_df.shape)
    print(tabulate(clean_df.head(3), headers="keys", tablefmt="psql", showindex=False))

    clean_df = add_mfcc_column(
        clean_df,
        sr=sampling_rate,
        n_mfcc=n_mfcc,
        use_parallel=use_parallel
    )
    print("CLEAN after MFCC:")
    print(tabulate(clean_df.head(1), headers="keys", tablefmt="psql", showindex=False))

    # split clean into train/val/test
    X_all = clean_df["mfcc"].values
    y_all = clean_df["label"].values
    fn_all = clean_df["filename"].values

    (X_tr, y_tr, fn_tr), (X_va_clean, y_va_clean, fn_va_clean), (X_te_clean, y_te_clean, fn_te_clean) = split_train_val_test(
        X_all, y_all, fn_all,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        random_state=int(cfg["random_state"])
    )
    print(f"CLEAN split sizes: train={len(X_tr)}, val={len(X_va_clean)}, test={len(X_te_clean)}")

    # fit scaler on clean train
    scaler = fit_scaler(X_tr)

    # plot 
    if bool(cfg["show_plots"]):
        # Pick a sample from CLEAN df (raw mfcc already exists)
        # build a tiny df for plotting that contains both mfcc + scaled_mfcc
        label_for_plot = str(clean_df["label"].iloc[0])  
        df_label = clean_df[clean_df["label"] == label_for_plot]

        if len(df_label) == 0:
            # fallback: just take first row
            row = clean_df.iloc[0]
        else:
            row = df_label.sample(1).iloc[0] if True else df_label.iloc[0]  

        mfcc_raw = row["mfcc"]  
        mfcc_scaled = scaler.transform(mfcc_raw).astype(np.float32)  

        plot_df = clean_df.loc[[row.name]].copy()
        plot_df["scaled_mfcc"] = [mfcc_scaled]

        plot_audio_and_features(
            audio_df=plot_df,
            label_name=str(row["label"]),
            sampling_rate=sampling_rate,
            n_mfcc=n_mfcc,
            random_example=False,  
        )

    X_tr_sc = apply_scaler(X_tr, scaler)
    X_va_clean_sc = apply_scaler(X_va_clean, scaler)
    X_te_clean_sc = apply_scaler(X_te_clean, scaler)

    # padding length based on clean train max length
    max_len = max(m.shape[0] for m in X_tr_sc)
    X_tr_p = pad_mfcc_list(X_tr_sc, max_len=max_len)
    X_va_clean_p = pad_mfcc_list(X_va_clean_sc, max_len=max_len)
    X_te_clean_p = pad_mfcc_list(X_te_clean_sc, max_len=max_len)
    print("CLEAN train padded shape:", X_tr_p.shape)

    # label encoder on clean train 
    label_encoder, y_tr_enc, y_va_clean_enc, y_te_clean_enc = make_label_encoder(
        y_tr, y_va_clean, y_te_clean
    )
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    print("Classes:", list(class_names))

    # build clean loaders (train/val/test)
    train_loader, clean_val_loader, clean_test_loader = make_loaders(
        X_tr_p, y_tr_enc, fn_tr,
        X_va_clean_p, y_va_clean_enc, fn_va_clean,
        X_te_clean_p, y_te_clean_enc, fn_te_clean,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle_train=bool(cfg["shuffle_train"])
    )

    # choose val source from config (clean/noisy/denoised)
    val_source = str(cfg["val_source"]).lower()
    if val_source == "clean":
        val_loader = clean_val_loader
        print("VAL source: CLEAN (uses clean_val_loader)")

    else:
        # Load external validation source (noisy/denoised)
        ext_root = cfg["data_sources"][f"{val_source}_dir"]
        ext_fs, ext_fe = cfg["folders"][val_source]

        ext_folders = list_folders(ext_root)
        ext_subset = ext_folders[int(ext_fs):int(ext_fe)]
        print(f"VAL source: {val_source.upper()} folders:", ext_subset)

        ext_paths = collect_wav_paths(ext_root, ext_subset)
        print(f"VAL source: {val_source.upper()} total wav files:", len(ext_paths))

        ext_df = build_audio_dataframe(
            ext_paths,
            sampling_rate=sampling_rate,
            use_parallel=use_parallel
        )
        ext_df = add_mfcc_column(
            ext_df,
            sr=sampling_rate,
            n_mfcc=n_mfcc,
            use_parallel=use_parallel
        )
        # apply same scaler fitted on clean train
        X_ext_sc = apply_scaler(ext_df["mfcc"].values, scaler)
        X_ext_p = pad_mfcc_list(X_ext_sc, max_len=max_len)

        # Encode labels using same encoder mapping
        y_ext_enc = label_encoder.transform(ext_df["label"].values)
        fn_ext = ext_df["filename"].values

        # build val loader (no shuffle)
        ext_ds = MFCCDataset(X_ext_p, y_ext_enc, fn_ext)
        val_loader = DataLoader(
            ext_ds,
            batch_size=int(cfg["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["num_workers"])
        )
        print(f"VAL source: {val_source.upper()} loader ready. size={len(ext_ds)}")

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
            best_dir=models_dir,
        )
        if bool(cfg["show_plots"]):
            plot_history(hist_basic, title_prefix="BasicDSCNN")

        best_basic = load_model(BasicDSCNN, num_classes, hist_basic["best_path"], device=device)
        tr_loss, tr_acc = evaluate_loader(best_basic, train_loader, device=device)
        va_loss, va_acc = evaluate_loader(best_basic, val_loader, device=device)
        te_loss, te_acc = evaluate_loader(best_basic, clean_test_loader, device=device)
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
            best_dir=models_dir,  
        )
        if bool(cfg["show_plots"]):
            plot_history(hist_improved, title_prefix="ImprovedDSCNN")

        best_improved = load_model(ImprovedDSCNN, num_classes, hist_improved["best_path"], device=device)
        tr_loss, tr_acc = evaluate_loader(best_improved, train_loader, device=device)
        va_loss, va_acc = evaluate_loader(best_improved, val_loader, device=device)
        te_loss, te_acc = evaluate_loader(best_improved, clean_test_loader, device=device)
        print(f"[ImprovedDSCNN] train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
        print(f"[ImprovedDSCNN]   val: loss={va_loss:.4f}, acc={va_acc:.4f}")
        print(f"[ImprovedDSCNN]  test: loss={te_loss:.4f}, acc={te_acc:.4f}")

        if bool(cfg["make_confusion"]):
            cm, rep, (t_imp, p_imp, f_imp) = confusion_and_report(
                best_improved, val_loader, class_names, device, model_name="ImprovedDSCNN (Validation)"
            )

            # misclassified list
            if bool(cfg["mis_enabled"]) and (cfg["mis_true_label"] in class_names) and (cfg["mis_pred_label"] in class_names):
                mis_df = find_misclassified_files(
                    t_imp, p_imp, f_imp,
                    class_names,
                    true_label_name=str(cfg["mis_true_label"]),
                    pred_label_name=str(cfg["mis_pred_label"]),
                )
                print("\nMisclassified files:")
                print("\nMisclassified files:")
                print(tabulate(mis_df.head(int(cfg["mis_max_rows"])), headers="keys", tablefmt="psql", showindex=False))


if __name__ == "__main__":
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = load_config(cfg_path)
    run(cfg)
