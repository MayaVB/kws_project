import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch


def plot_history(history, title_prefix=""):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], "bo-", label="Training Loss")
    plt.plot(epochs_range, history["val_loss"],   "r*-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], "bo-", label="Training Accuracy")
    plt.plot(epochs_range, history["val_acc"],   "r*-", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def confusion_and_report(model, loader, class_names, device, model_name=""):
    """
    Build confusion matrix + classification report from a loader that returns (X,y,filename).
    Returns: cm, report_text, (all_true, all_pred, all_filenames)
    """
    model.eval()
    all_true, all_pred, all_files = [], [], []

    with torch.no_grad():
        for X_batch, y_batch, fn_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            all_true.extend(y_batch.numpy())
            all_pred.extend(preds.cpu().numpy())
            all_files.extend(list(fn_batch))

    cm = confusion_matrix(all_true, all_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.show()

    report = classification_report(all_true, all_pred, target_names=class_names)
    print("Classification Report:")
    print(report)

    return cm, report, (np.array(all_true), np.array(all_pred), np.array(all_files))


def find_misclassified_files(all_true, all_pred, all_files,
                             class_names,
                             true_label_name: str,
                             pred_label_name: str):
    """
    Return a DataFrame listing filenames where:
      true == true_label_name AND pred == pred_label_name
    """
    true_idx = np.where(class_names == true_label_name)[0][0]
    pred_idx = np.where(class_names == pred_label_name)[0][0]

    mask = (all_true == true_idx) & (all_pred == pred_idx)
    mis_files = all_files[mask]

    df = pd.DataFrame({
        "filename": mis_files,
        "true_label": [true_label_name] * len(mis_files),
        "pred_label": [pred_label_name] * len(mis_files),
    })
    return df
