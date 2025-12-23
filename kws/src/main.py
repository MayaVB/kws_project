import os
import numpy as np
from tabulate import tabulate

from dataset import (
    list_folders, collect_wav_paths, build_audio_dataframe,
    split_train_val_test, pad_mfcc_list,
    make_label_encoder, make_loaders
)
from features import fit_scaler, apply_scaler, add_mfcc_column, plot_audio_and_features
from models import BasicDSCNN, ImprovedDSCNN
from train import get_device, train_model, load_model, evaluate_loader
from metrics import plot_history, confusion_and_report, find_misclassified_files


def run(main_dir: str, folder_slice=(13, 20),
                 sampling_rate=16000, n_mfcc=13,
                 batch_size=64, epochs=150, patience=10,
                 lr=1e-3, weight_decay=1e-3):
    device = get_device()
    print("Using device:", device)

    folders = list_folders(main_dir)
    subset_folders = folders[folder_slice[0]:folder_slice[1]]
    print("Selected folders:", subset_folders)

    file_paths = collect_wav_paths(main_dir, subset_folders)
    print("Total wav files:", len(file_paths))

    audio_df = build_audio_dataframe(file_paths, sampling_rate=sampling_rate, use_parallel=True)
    print("Dataframe shape:", audio_df.shape)
    print(tabulate(audio_df.head(), headers="keys", tablefmt="psql", showindex=False))

    # MFCC extraction 
    audio_df = add_mfcc_column(audio_df, sr=sampling_rate, n_mfcc=n_mfcc, use_parallel=True)
    print(tabulate(audio_df.head(), headers="keys", tablefmt="psql", showindex=False))

    # Scaling 
    scaler = fit_scaler(audio_df["mfcc"].values)
    scaled_mfcc = apply_scaler(audio_df["mfcc"].values, scaler)
    audio_df["scaled_mfcc"] = scaled_mfcc
    print(audio_df.head(5).to_string(index=False))

    plot_audio_and_features(
    audio_df=audio_df,
    label_name="house",   
    sampling_rate=sampling_rate,
    n_mfcc=13,
    random_example=True
)

    # Split 
    X_all = audio_df["scaled_mfcc"].values
    y_all = audio_df["label"].values
    fn_all = audio_df["filename"].values 

    (X_train, y_train, fn_train), (X_val, y_val, fn_val), (X_test, y_test, fn_test) = split_train_val_test(
        X_all, y_all, fn_all,
        test_size=0.2, # 20% total
        val_size=0.5,  # 50% of temp -> 10% total
        random_state=42
    )

    #  Pad using global max length 
    global_max_len = max(m.shape[0] for m in X_all)
    X_train_p = pad_mfcc_list(X_train, max_len=global_max_len)
    X_val_p   = pad_mfcc_list(X_val,   max_len=global_max_len)
    X_test_p  = pad_mfcc_list(X_test,  max_len=global_max_len)
    print("Train padded shape:", X_train_p.shape)

    # Label encoding (word -> number)
    label_encoder, y_train_enc, y_val_enc, y_test_enc = make_label_encoder(y_train, y_val, y_test)
    class_names = label_encoder.classes_
    num_classes = len(class_names)
    print("Classes:", list(class_names))

    # Loaders 
    train_loader, val_loader, test_loader = make_loaders(
        X_train_p, y_train_enc, fn_train,
        X_val_p,   y_val_enc,   fn_val,
        X_test_p,  y_test_enc,  fn_test,
        batch_size=batch_size
    )

    # Basic model
    basic = BasicDSCNN(num_classes)
    hist_basic = train_model(
        basic, train_loader, val_loader,
        num_epochs=epochs, patience=patience,
        lr=lr, weight_decay=weight_decay,
        model_name="basic_dscnn",
        device=device
    )
    plot_history(hist_basic, title_prefix="BasicDSCNN")

    best_basic = load_model(BasicDSCNN, num_classes, hist_basic["best_path"], device=device)
    tr_loss, tr_acc = evaluate_loader(best_basic, train_loader, device=device)
    va_loss, va_acc = evaluate_loader(best_basic, val_loader, device=device)
    te_loss, te_acc = evaluate_loader(best_basic, test_loader, device=device)
    print(f"[BasicDSCNN] train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
    print(f"[BasicDSCNN]   val: loss={va_loss:.4f}, acc={va_acc:.4f}")
    print(f"[BasicDSCNN]  test: loss={te_loss:.4f}, acc={te_acc:.4f}")

    _, _, (t_basic, p_basic, f_basic) = confusion_and_report(
        best_basic, val_loader, class_names, device, model_name="BasicDSCNN (Validation)")

    # Improved model
    improved = ImprovedDSCNN(num_classes)
    hist_improved = train_model(
        improved, train_loader, val_loader,
        num_epochs=epochs, patience=patience,
        lr=lr, weight_decay=weight_decay,
        model_name="improved_dscnn",
        device=device
    )
    plot_history(hist_improved, title_prefix="ImprovedDSCNN")

    best_improved = load_model(ImprovedDSCNN, num_classes, hist_improved["best_path"], device=device)
    tr_loss, tr_acc = evaluate_loader(best_improved, train_loader, device=device)
    va_loss, va_acc = evaluate_loader(best_improved, val_loader, device=device)
    te_loss, te_acc = evaluate_loader(best_improved, test_loader, device=device)
    print(f"[ImprovedDSCNN] train: loss={tr_loss:.4f}, acc={tr_acc:.4f}")
    print(f"[ImprovedDSCNN]   val: loss={va_loss:.4f}, acc={va_acc:.4f}")
    print(f"[ImprovedDSCNN]  test: loss={te_loss:.4f}, acc={te_acc:.4f}")

    # t_imp - true labels
    # p_imp - predict
    # f_imp - filenames
    
    _, _, (t_imp, p_imp, f_imp) = confusion_and_report(
        best_improved, val_loader, class_names, device, model_name="ImprovedDSCNN (Validation)")

    # find mistakes 
    if "learn" in class_names and "house" in class_names:
        mis_df = find_misclassified_files(
            t_imp, p_imp, f_imp,
            class_names,
            true_label_name="learn",
            pred_label_name="house"
        )
        print("\nMisclassified files (true=learn, predicted=house):")
        print(mis_df.head(50).to_string(index=False))


if __name__ == "__main__":
    MAIN_DIR = r"C:\\temp\\KWS\data\speech-commands-v2"
    run(MAIN_DIR, folder_slice=(13, 20))
