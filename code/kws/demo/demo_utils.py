"""
demo_utils.py

Utilities for preparing demo and Gradio analysis files.

This module compares prediction results across
multiple evaluation modes and generates:

- fixed examples
- degraded examples
- still-wrong examples
- demo CSV files
- audio path mappings
"""
import pandas as pd
import os

def compare_prediction_modes(
    clean_csv,
    noisy_csv,
    enh_csv,
    output_dir
):
    """
    demo_utils.py

    Utilities for preparing demo and Gradio analysis files.

    This module compares prediction results across
    multiple evaluation modes and generates:

    - fixed examples
    - degraded examples
    - still-wrong examples
    - demo CSV files
    - audio path mappings
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
    """
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
    """

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
    """
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
    """
