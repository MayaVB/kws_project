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


# =====================================================
# HISTORICAL -> ARCHIVE AUDIO PATHS
# =====================================================
# interesting_examples.csv files written by earlier runs store absolute audio
# paths under the original data directory in the lab home. The data now lives
# in the verified GannotLab archive (reproducibility/DATA_LAYOUT.md). The CSV
# files are historical outputs and are never edited: paths are remapped in
# memory when a CSV is read.

HISTORICAL_DATA_ROOT = "/home/dsi/skopavi/Project/kws_project/data"
ARCHIVE_DATA_ROOT = "/dsi/gannot-lab/gannot-lab1/datasets/skopavi/kws_archive/datasets"

DEFAULT_PATH_MAP = (
    (f"{HISTORICAL_DATA_ROOT}/raw/data_new", f"{ARCHIVE_DATA_ROOT}/speech_commands_v0.01"),
    (f"{HISTORICAL_DATA_ROOT}/noisy_new", f"{ARCHIVE_DATA_ROOT}/noisy_new"),
    (f"{HISTORICAL_DATA_ROOT}/enhanced_new", f"{ARCHIVE_DATA_ROOT}/enhanced_new"),
)

AUDIO_PATH_COLUMNS = ("clean_path", "noisy_path", "enh_path")


def parse_path_map(items):
    """Parse ["OLD=NEW", ...] into ((OLD, NEW), ...)."""
    pairs = []
    for item in items or ():
        old, sep, new = item.partition("=")
        if not sep or not old or not new:
            raise ValueError(f"Invalid path mapping {item!r}; expected OLD_PREFIX=NEW_PREFIX")
        pairs.append((old, new))
    return tuple(pairs)


def remap_path(path, path_map=DEFAULT_PATH_MAP):
    """
    Replace the longest matching directory prefix of `path`.

    Prefixes match whole path components only (".../noisy_new" does not match
    ".../noisy_new_metadata.csv"). On equal length, the earlier mapping wins.
    Paths without a matching prefix are returned unchanged.
    """
    path = str(path)
    best = None
    for old, new in path_map:
        old = old.rstrip("/")
        if (path == old or path.startswith(old + "/")) and (best is None or len(old) > len(best[0])):
            best = (old, new.rstrip("/"))
    return path if best is None else best[1] + path[len(best[0]):]


def remap_audio_paths(df, path_map=DEFAULT_PATH_MAP, columns=AUDIO_PATH_COLUMNS):
    """Return a copy of `df` with the audio path columns remapped (the input is not modified)."""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = [remap_path(p, path_map) if isinstance(p, str) else p for p in out[col]]
    return out


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
