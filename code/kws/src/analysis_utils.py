from tabulate import tabulate
import pandas as pd

from metrics import plot_confusion_per_snr, plot_confusion_per_noise, misclassified_between_two_classes, run_calc_metrics

def analyze_mode(
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

    # print(f"DEBUG: mode {mode} snr: {snr[:5] if snr is not None else None} noise: {noise[:5] if noise is not None else None }")

    has_noise_info = (snr is not None) and (noise is not None) and (len(snr)>0) and (len(noise)>0)
    # print(f"\nDEBUG: mode {mode} has_noise_info: {has_noise_info}")

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
        return

    # PRINT
    print(tabulate(mis, headers="keys", tablefmt="psql"))

    # SAVE CSV 
    # out_csv = run_dir / f"misclassified_{mode}.csv"
    # mis.to_csv(out_csv, index=False)

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

    if enhanced_root is not None:
        print(f"\n=== METRICS {mode.upper()} ===")

        run_calc_metrics(
            df_test_audio=df_test_audio,
            sampling_rate=sampling_rate,
            run_dir=run_dir,
            enhanced_root=enhanced_root,
            tag=mode.upper()
    )