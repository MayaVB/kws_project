"""
visualization.py

This module contains:
- training history plots
- confusion matrix visualizations
- SNR and noise analysis plots
- waveform and spectrogram comparisons
- enhanced speech visualizations
"""
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.signal import stft

def plot_example_signals(
    df_test_audio,
    sampling_rate,
    plots_dir,
    noisy_root,
    enhanced_versions,  # dict of {mode_name: enhanced_root}
    idx_vis=0,
):
    """
    Plot a representative test example.

    The figure compares:
    - clean speech
    - noisy speech
    - one or more enhanced versions

    The example is selected from the fixed test dataset.
    """
    # BASIC INFO
    filename = df_test_audio["filename"].iloc[idx_vis]
    label = df_test_audio["label"].iloc[idx_vis]

    print(f"[EXAMPLE] label={label}, file={filename}")

    # CLEAN
    clean_sig = df_test_audio["audio_data"].iloc[idx_vis]

    # NOISY 
    noisy_path = os.path.join(noisy_root, "test", label, filename)

    if not os.path.exists(noisy_path):
        print(f"❌ Missing noisy file: {noisy_path}")
        return

    noisy_sig, _ = librosa.load(noisy_path, sr=sampling_rate)

    # ENHANCED
    enhanced_signals = {}
    for method_name, root in enhanced_versions.items():
        path = os.path.join(root, label, filename)
        if os.path.exists(path):
            sig, _ = librosa.load(path, sr=sampling_rate)
            enhanced_signals[method_name] = sig
        else:
            print(f"⚠️ Missing enhanced: {path}")

    # PLOT
    plot_signal_comparison(
        clean=clean_sig,
        noisy=noisy_sig,
        enhanced_signals=enhanced_signals,
        fs=sampling_rate,
        title=f"{label}/{filename}",
        save_path=plots_dir / f"Signal_Comparison_Example_{filename}.png"
    )

def plot_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : dict
        Training history returned by train_model().
    save_path : str or Path, optional
        Output path. If None, display interactively.
    """
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], "bo-", label="Training Loss")
    plt.plot(epochs_range, history["val_loss"],   "r*-", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], "bo-", label="Training Accuracy")
    plt.plot(epochs_range, history["val_acc"],   "r*-", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_per_snr(true_labels, pred_labels, snr_values, class_names, title, save_path=None):
    """
    Plot confusion matrices separately for each SNR level.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    snr_values = np.asarray(snr_values)

    snrs = np.sort(np.unique(snr_values))

    fig, axes = plt.subplots(2, 3, figsize=(12,10))
    axes = axes.flatten()

    for i,s in enumerate(snrs):
        mask = snr_values == s
        cm = confusion_matrix(
            true_labels[mask],
            pred_labels[mask],
            labels=np.arange(len(class_names))
        )

        sns.heatmap(
            cm,
            ax=axes[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        axes[i].set_title(f"SNR={s} dB")

    for i in range(len(snrs), len(axes)):
        axes[i].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_per_noise(true_labels, pred_labels, noise_names, class_names, title, save_path=None):
    """
    Plot confusion matrices separately for each noise type.
    """
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    noise_names = np.asarray(noise_names)
    noises = np.unique(noise_names)
    
    fig, axes = plt.subplots(2, 3, figsize=(12,10))
    axes = axes.flatten()

    for i, n in enumerate(noises):
        mask = noise_names == n
        cm = confusion_matrix(
            true_labels[mask],
            pred_labels[mask],
            labels=np.arange(len(class_names))
        )

        sns.heatmap(
            cm,
            ax=axes[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        axes[i].set_title(f"noise={n}")

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def _compute_spectrogram(x, fs):
    """
    Compute a log-magnitude STFT spectrogram.
    """
    f, t, Zxx = stft(x, fs=fs, nperseg=512, noverlap=256)
    S = np.abs(Zxx)
    return f, t, 20 * np.log10(S + 1e-10)


def plot_signal_comparison(clean, noisy,
                            enhanced_signals=None,
                            fs=16000, title="Signal Comparison", save_path=None):
    """
    Plot waveform and spectrogram comparisons for:

    - clean speech
    - noisy speech
    - one or more enhanced versions
    """
    signals = [
    ("Clean", clean),
    ("Noisy", noisy),
    ]

    if enhanced_signals is not None:
        for method_name, sig in enhanced_signals.items():
            signals.append(
                (
                    f"Enhanced ({method_name})",
                    sig
                )
            )
    else:
        print("No enhanced signals provided for comparison.")   

    n_signals = len(signals) 
    fig, axes = plt.subplots(
        n_signals,
        2,
        figsize=(12, 2*n_signals)
    )

    for i, (name, sig) in enumerate(signals):

        # ===== waveform =====
        axes[i, 0].plot(sig)
        axes[i, 0].set_title(f"{name} - Waveform")
        axes[i, 0].set_xlim([0, len(sig)])

        # ===== spectrogram =====
        f, t, S_db = _compute_spectrogram(sig, fs)

        im = axes[i, 1].pcolormesh(t, f, S_db, shading='auto')
        axes[i, 1].set_title(f"{name} - Spectrogram")
        axes[i, 1].set_ylabel("Hz")
        axes[i, 1].set_xlabel("Time")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_comparison(
    results,
    class_names,
    plots_dir,
    confusion_modes=None
):
    """
    Generate side-by-side confusion matrices for multiple modes.

    Examples:
    - clean
    - noisy
    - enhanced baseline
    - enhanced trained

    Individual confusion matrices are also saved separately.
    """
    if confusion_modes is None:
        confusion_modes = [
            "clean",
            "noisy",
            "enh_trained"
        ]

    display_names = {
        "clean": "Clean",
        "noisy": "Noisy",
        "enh_baseline": "Enhanced Baseline",
        "enh_trained": "Enhanced Trained"
    }

    pairs = []

    for mode_name in confusion_modes:

        if mode_name not in results:
            print(f"WARNING: {mode_name} not found in results")
            continue

        mode_res = results[mode_name]

        pairs.append(
            (
                mode_name,
                mode_res["t"],
                mode_res["p"]
            )
        )

    n_modes = len(pairs)

    if n_modes == 0:
        return

    ncols = len(pairs)
    nrows = 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 6)
    )

    axes = np.array(axes).reshape(-1)

    for i, (name, t, p) in enumerate(pairs):

        cm = confusion_matrix(t, p)

        display_name = display_names.get(
            name,
            name
        )

        sns.heatmap(
            cm,
            ax=axes[i],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )

        axes[i].set_title(
            display_name,
            fontsize=14,
            fontweight="bold"
        )

        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

        # save single figure

        single_fig, single_ax = plt.subplots(
            figsize=(8, 6)
        )

        sns.heatmap(
            cm,
            ax=single_ax,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )

        single_ax.set_title(
            display_name,
            fontsize=14,
            fontweight="bold"
        )

        single_ax.set_xlabel("Predicted")
        single_ax.set_ylabel("True")

        plt.tight_layout()

        plt.savefig(
            plots_dir / f"confusion_{name}.png"
        )

        plt.close(single_fig)

    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Confusion Matrices Comparison",
        fontsize=16
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    plt.savefig(
        plots_dir /
        "confusion_matrices_comparison.png"
    )

    plt.close()

# TODO: add option to plot random examples
# TODO: add option to plot examples from enhanced dataset
# TODO: choose example with specific SNR