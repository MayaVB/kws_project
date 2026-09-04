"""
analysis_utils.py

This module contains:
- mode-specific analysis
- prediction report generation
- metrics aggregation and comparison
"""
from tabulate import tabulate
import pandas as pd

from metrics import misclassified_between_two_classes, run_calc_metrics
from visualization import plot_confusion_per_snr, plot_confusion_per_noise

def analyze_mode(
    acc,
    mode,
    results,
    class_names,
    plots_dir,
    run_dir,
    a,
    b,
    df_test_audio,
    sampling_rate,
    noisy_root,
    enhanced_root=None,
):
    """
    Run post-processing analysis for a single evaluation mode.

    The analysis includes:

    - confusion matrices by SNR
    - confusion matrices by noise type
    - misclassification reports
    - optional speech enhancement metrics

    Returns
    -------
    dict
        Summary metrics for the current mode.
    """
    t = results["t"]
    p = results["p"]
    f = results["f"]
    snr = results["snr"]
    noise = results["noise"]

    has_noise_info = (snr is not None) and (noise is not None) and (len(snr)>0) and (len(noise)>0)

    print(f"\n=== ANALYSIS {mode.upper()} ===")

    # CONFUSION PER SNR
    if has_noise_info:
        plot_confusion_per_snr(
            t, p, snr, class_names,
            title=f"Confusion per SNR - {mode}",
            save_path=plots_dir / f"confusion_per_snr_{mode}.png"
        )

        plot_confusion_per_noise(
            t, p, noise, class_names,
            title=f"Confusion per Noise - {mode}",
            save_path=plots_dir / f"confusion_per_noise_{mode}.png"
        )
    else:
        print(f"DEBUG: {mode} has no SNR/noise info, skipping confusion per SNR/Noise plots")

    # MISCLASSIFIED
    mis = misclassified_between_two_classes(
        t, p, f, class_names,
        a, b,
        snr, noise
    )

    if len(mis) == 0:
        print(f"No misclassified for {mode}")

    # PRINT
    print(tabulate(mis, headers="keys", tablefmt="psql"))

    # SAVE TXT (pretty table)
    out_txt = run_dir / "misclassified_report.txt"

    with open(out_txt, "a") as ftxt:
        ftxt.write(f"\n{mode.upper()}\n")
        ftxt.write(tabulate(mis, headers="keys", tablefmt="psql"))
        ftxt.write("\n")

    # GROUP STATS
    if len(mis) > 0:

        if "snr_db" in mis.columns:
            print("\nErrors by SNR:")
            print(mis.groupby("snr_db").size())

        if "noise" in mis.columns:
            print("\nErrors by noise:")
            print(mis.groupby("noise").size())

    metrics_dict = {
        "mode": mode,
        "acc": f"{acc:.2f}"
    }
    
    if enhanced_root is not None:
        print(f"\n=== METRICS {mode.upper()} ===")

        enh_metrics = run_calc_metrics(
            df_test_audio=df_test_audio,
            sampling_rate=sampling_rate,
            run_dir=run_dir,
            noisy_root=noisy_root,
            enhanced_root=enhanced_root,
            tag=mode.upper()
        )
        metrics_dict.update(enh_metrics)
        return metrics_dict
    
    return metrics_dict

def save_prediction_reports(
    t,
    p,
    f,
    mode_name,
    label_encoder,
    run_dir,
    snr=None,
    noise=None
):
    """
    Save prediction results and misclassification reports.

    Two CSV files are generated:

    - all predictions
    - misclassified samples only

    Returns
    -------
    pred_df : DataFrame
    mis_df : DataFrame
    """
    pred_df = pd.DataFrame({
        "filename": f,
        "true_idx": t,
        "pred_idx": p
    })

    pred_df["true_label"] = label_encoder.inverse_transform(
        pred_df["true_idx"]
    )

    pred_df["pred_label"] = label_encoder.inverse_transform(
        pred_df["pred_idx"]
    )

    pred_df["correct"] = (
        pred_df["true_label"]
        ==
        pred_df["pred_label"]
    )

    pred_df["mode"] = mode_name

    if snr is not None and len(snr) == len(pred_df):
        pred_df["snr"] = snr

    if noise is not None and len(noise) == len(pred_df):
        pred_df["noise"] = noise

    mis_df = pred_df[
        pred_df["correct"] == False
    ].copy()

    mis_df = mis_df.sort_values(
        ["true_label", "pred_label", "filename"]
    )

    mis_df.to_csv(
        run_dir / f"{mode_name}_misclassified.csv",
        index=False
    )

    pred_df.to_csv(
        run_dir / f"{mode_name}_all_predictions.csv",
        index=False
    )

    print(
        f"Saved {len(mis_df)} misclassified samples "
        f"for {mode_name}"
    )

    return pred_df, mis_df


def save_metrics_summary(
    all_metrics,
    run_dir
):
    """
    Create and save a summary table comparing all evaluation modes.

    The summary is written to:
    metrics_comparison.txt
    """
    if len(all_metrics) == 0:
        return

    df_metrics = pd.DataFrame(all_metrics)
    df_metrics = df_metrics.fillna("-")

    print("\n=== METRICS TABLE ===")
    print(df_metrics)

    table_str = tabulate(
        df_metrics,
        headers="keys",
        tablefmt="fancy_grid",
        showindex=False
    )

    print(table_str)

    with open(
        run_dir / "metrics_comparison.txt",
        "w"
    ) as f:
        f.write(table_str)