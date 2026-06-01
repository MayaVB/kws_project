# analysis_utils.py
from tabulate import tabulate
import pandas as pd
import os

from metrics import plot_confusion_per_snr, plot_confusion_per_noise, misclassified_between_two_classes, run_calc_metrics

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
    enhanced_root=None,
):
    """
    Run full analysis for one mode
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
    "acc": round(acc, 4)
    }

    if enhanced_root is not None:
        print(f"\n=== METRICS {mode.upper()} ===")

        enh_metrics = run_calc_metrics(
            df_test_audio=df_test_audio,
            sampling_rate=sampling_rate,
            run_dir=run_dir,
            enhanced_root=enhanced_root,
            tag=mode.upper()
        )
        metrics_dict.update(enh_metrics)
        return metrics_dict
    
    return metrics_dict


def compare_prediction_modes(
    clean_csv,
    noisy_csv,
    enh_csv,
    output_dir
):
    """
    Compare clean/noisy/enhanced predictions and
    save useful analysis files for Gradio.
    """

    print("\n[COMPARE MODES] Loading CSVs...")

    clean = pd.read_csv(clean_csv)
    noisy = pd.read_csv(noisy_csv)
    enh = pd.read_csv(enh_csv)

    # ---------------------------------
    # keep useful columns
    # ---------------------------------

    clean = clean[
        [
            "filename",
            "true_label",
            "pred_label",
            "correct"
        ]
    ].rename(
        columns={
            "pred_label": "pred_clean",
            "correct": "correct_clean"
        }
    )

    noisy = noisy[
        [
            "filename",
            "true_label",
            "pred_label",
            "correct",
            "snr",
            "noise"
        ]
    ].rename(
        columns={
            "pred_label": "pred_noisy",
            "correct": "correct_noisy"
        }
    )

    enh = enh[
        [
            "filename",
            "true_label",
            "pred_label",
            "correct"
        ]
    ].rename(
        columns={
            "pred_label": "pred_enh",
            "correct": "correct_enh"
        }
    )

    # ---------------------------------
    # merge
    # ---------------------------------

    merged = clean.merge(
        noisy,
        on=["filename", "true_label"]
    )

    merged = merged.merge(
        enh,
        on=["filename", "true_label"]
    )

    # ---------------------------------
    # categories
    # ---------------------------------

    fixed = merged[
        (~merged["correct_noisy"])
        &
        (merged["correct_enh"])
    ]

    still_wrong = merged[
        (~merged["correct_noisy"])
        &
        (~merged["correct_enh"])
    ]

    degraded = merged[
        (merged["correct_noisy"])
        &
        (~merged["correct_enh"])
    ]

    # ---------------------------------
    # save csv
    # ---------------------------------

    fixed.to_csv(
        os.path.join(
            output_dir,
            "fixed_by_enhancement.csv"
        ),
        index=False
    )

    still_wrong.to_csv(
        os.path.join(
            output_dir,
            "still_wrong.csv"
        ),
        index=False
    )

    degraded.to_csv(
        os.path.join(
            output_dir,
            "degraded_by_enhancement.csv"
        ),
        index=False
    )

    fixed_sorted = fixed.sort_values(
        ["snr", "noise"]
    )

    fixed_sorted.to_csv(
        os.path.join(
            output_dir,
            "demo_examples.csv"
        ),
        index=False
    )

    interesting = pd.concat([
        fixed.assign(category="fixed"),
        still_wrong.assign(category="still_wrong"),
        degraded.assign(category="degraded")
    ])

    clean_paths = []
    noisy_paths = []
    enh_paths = []

    for _, row in interesting.iterrows():

        c, n, e = build_audio_paths(
            row["filename"],
            row["true_label"]
        )

        clean_paths.append(c)
        noisy_paths.append(n)
        enh_paths.append(e)

    interesting["clean_path"] = clean_paths
    interesting["noisy_path"] = noisy_paths
    interesting["enh_path"] = enh_paths

    interesting.to_csv(
        os.path.join(
            output_dir,
            "interesting_examples.csv"
        ),
        index=False
    )


    # ---------------------------------
    # summary txt
    # ---------------------------------

    summary_path = os.path.join(
        output_dir,
        "enhancement_analysis.txt"
    )

    with open(summary_path, "w") as f:

        f.write(
            "Enhancement Analysis\n"
        )

        f.write(
            "====================\n\n"
        )

        f.write(
            f"Fixed by enhancement : {len(fixed)}\n"
        )

        f.write(
            f"Still wrong          : {len(still_wrong)}\n"
        )

        f.write(
            f"Degraded             : {len(degraded)}\n"
        )

    print(
        "\n[COMPARE MODES]"
    )

    print(
        f"Fixed      : {len(fixed)}"
    )

    print(
        f"StillWrong : {len(still_wrong)}"
    )

    print(
        f"Degraded   : {len(degraded)}"
    )

def build_audio_paths(
    filename,
    label
):
    clean_path = (
        f"/home/dsi/skopavi/Project/kws_project/"
        f"data/raw/data_new/{label}/{filename}"
    )

    noisy_path = (
        f"/home/dsi/skopavi/Project/kws_project/"
        f"data/noisy_new/test/{label}/{filename}"
    )

    enhanced_path = (
        f"/home/dsi/skopavi/Project/kws_project/"
        f"data/enhanced_new/trained_ep176/"
        f"{label}/{filename}"
    )

    return clean_path, noisy_path, enhanced_path