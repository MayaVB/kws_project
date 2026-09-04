"""
build_enhanced_dataset.py

Generate an enhanced speech dataset using an SGMSE checkpoint.

Pipeline:
1. Load noisy speech files
2. Run SGMSE enhancement
3. Save enhanced files
4. Generate synchronized metadata CSV

The script can be resumed: files that already exist in the output
directory are skipped.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from math import ceil
from multiprocessing import Pool
from pathlib import Path
import pandas as pd


# =====================================================
# ARGUMENTS
# =====================================================

def parse_args():
    default_sgmse_script = (Path(__file__).resolve().parents[2] / "sgmse" / "enhancement.py")
    parser = argparse.ArgumentParser(description="Generate an enhanced dataset using an SGMSE checkpoint.")
    parser.add_argument("--input_root", type=str, required=True, help="Directory containing the noisy test dataset.",)
    parser.add_argument("--metadata", type=str, required=True, help="Metadata CSV corresponding to the noisy dataset.",)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the SGMSE checkpoint.",)
    parser.add_argument("--output_root", type=str, required=True, help="Directory in which enhanced audio files will be saved.",)
    parser.add_argument("--output_metadata", type=str, required=True, help="Path for saving the enhanced dataset metadata CSV.",)
    parser.add_argument("--sgmse_script", type=str, default=str(default_sgmse_script), help="Path to the SGMSE enhancement.py script.",)
    parser.add_argument("--device", type=str, default="cuda", help="Device used for SGMSE inference (default: cuda).",)
    parser.add_argument("--chunk_size", type=int, default=10,  help="Number of files enhanced per SGMSE call (default: 10).",)
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers (default: 1).",)
    parser.add_argument("--tmp_root",  type=str, default="/tmp/enhanced_chunks", help="Directory for temporary enhancement files.",)
    return parser.parse_args()


# GLOBAL WORKER CONFIGURATION
CONFIG = {}
META_LOOKUP = {}

def init_worker(config, meta_lookup):
    global CONFIG, META_LOOKUP
    CONFIG = config
    META_LOOKUP = meta_lookup


# WORKER
def process_chunk(task):
    folder, files = task

    input_root = CONFIG["input_root"]
    output_root = CONFIG["output_root"]
    tmp_root = CONFIG["tmp_root"]

    tmp_in = os.path.join(tmp_root, f"in_{os.getpid()}")
    tmp_out = os.path.join(tmp_root, f"out_{os.getpid()}")

    os.makedirs(tmp_in, exist_ok=True)
    os.makedirs(tmp_out, exist_ok=True)

    try:
        # Copy the current chunk into a temporary input directory
        for filename in files:
            src = os.path.join(input_root, folder, filename)
            dst = os.path.join(tmp_in, filename)
            shutil.copy2(src, dst)

        # Run SGMSE enhancement
        cmd = [
            sys.executable,
            CONFIG["sgmse_script"],
            "--test_dir",
            tmp_in,
            "--enhanced_dir",
            tmp_out,
            "--ckpt",
            CONFIG["ckpt"],
            "--device",
            CONFIG["device"],
        ]

        subprocess.run(cmd, check=True)

        metadata_rows = []

        # Move enhanced files to the final dataset directory
        for filename in files:
            src = os.path.join(tmp_out, filename)
            dst = os.path.join(output_root, folder, filename)

            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)

                key = (folder, filename)
                row = META_LOOKUP.get(key)

                if row is not None:
                    metadata_rows.append(row)
                else:
                    print(f"Warning: metadata not found for {key}")
            else:
                print(f"Warning: enhanced file not found for {filename}")

        return metadata_rows

    finally:
        # Always remove temporary directories
        shutil.rmtree(tmp_in, ignore_errors=True)
        shutil.rmtree(tmp_out, ignore_errors=True)


# DATASET BUILD
def build(args):
    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    metadata_path = os.path.abspath(args.metadata)
    output_metadata = os.path.abspath(args.output_metadata)
    ckpt = os.path.abspath(args.ckpt)
    sgmse_script = os.path.abspath(args.sgmse_script)
    tmp_root = os.path.abspath(args.tmp_root)

    # Basic path validation
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input dataset directory not found: {input_root}")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if not os.path.isfile(sgmse_script):
        raise FileNotFoundError(f"SGMSE enhancement script not found: {sgmse_script}")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(tmp_root, exist_ok=True)

    output_metadata_dir = os.path.dirname(output_metadata)
    if output_metadata_dir:
        os.makedirs(output_metadata_dir, exist_ok=True)

    # -------------------------------------------------
    # Load metadata
    # -------------------------------------------------

    meta_df = pd.read_csv(metadata_path)

    # The input_root contains the noisy test set.
    if "split" in meta_df.columns:
        meta_df = meta_df[meta_df["split"] == "test"]

    meta_lookup = {
        (row["label"], row["filename"]): row
        for _, row in meta_df.iterrows()
    }

    config = {
        "input_root": input_root,
        "output_root": output_root,
        "tmp_root": tmp_root,
        "sgmse_script": sgmse_script,
        "ckpt": ckpt,
        "device": args.device,
    }

    # -------------------------------------------------
    # Dataset status
    # -------------------------------------------------

    print("\n========== CONFIGURATION ==========")
    print(f"Input dataset:   {input_root}")
    print(f"Checkpoint:      {ckpt}")
    print(f"Output dataset:  {output_root}")
    print(f"Output metadata: {output_metadata}")
    print(f"Device:          {args.device}")
    print(f"Chunk size:      {args.chunk_size}")
    print(f"Workers:         {args.n_workers}")

    print("\n========== DATASET STATUS ==========")

    folders = sorted(
        folder
        for folder in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, folder))
    )

    total_all = 0
    done_all = 0

    for folder in folders:
        in_dir = os.path.join(input_root, folder)
        out_dir = os.path.join(output_root, folder)

        all_files = [
            f for f in os.listdir(in_dir)
            if f.endswith(".wav")
        ]

        done_files = []

        if os.path.exists(out_dir):
            done_files = [
                f for f in os.listdir(out_dir)
                if f.endswith(".wav")
            ]

        total = len(all_files)
        done = len(done_files)
        remaining = total - done

        total_all += total
        done_all += done

        progress = 100 * done / total if total else 0.0

        print(
            f"{folder:15} | "
            f"{done}/{total} ({progress:.1f}%) | "
            f"remaining: {remaining}"
        )

    print(
        f"\nOverall: {done_all}/{total_all} "
        f"({100 * done_all / total_all if total_all else 0:.1f}%)"
    )

    # -------------------------------------------------
    # Build
    # -------------------------------------------------

    print("\n========== START BUILD ==========")

    global_start = time.time()

    for folder in folders:
        folder_start = time.time()

        in_dir = os.path.join(input_root, folder)
        out_dir = os.path.join(output_root, folder)

        os.makedirs(out_dir, exist_ok=True)

        all_files = [
            f for f in os.listdir(in_dir)
            if f.endswith(".wav")
        ]

        done_files = [
            f for f in os.listdir(out_dir)
            if f.endswith(".wav")
        ]

        total = len(all_files)
        done = len(done_files)

        print(f"\n=== {folder} ===")

        progress = 100 * done / total if total else 0.0
        print(f"Progress: {done}/{total} ({progress:.1f}%)")

        if done == total:
            print("Already done")
            continue

        remaining_files = [
            f for f in all_files
            if not os.path.exists(os.path.join(out_dir, f))
        ]

        total_chunks = ceil(
            len(remaining_files) / args.chunk_size
        )

        print(f"Remaining files: {len(remaining_files)}")
        print(
            f"Chunks: {total_chunks} "
            f"(up to {args.chunk_size} files each)\n"
        )

        folder_tasks = []

        for i in range(
            0,
            len(remaining_files),
            args.chunk_size,
        ):
            chunk = remaining_files[
                i:i + args.chunk_size
            ]
            folder_tasks.append((folder, chunk))

        processed_chunks = 0

        with Pool(
            args.n_workers,
            initializer=init_worker,
            initargs=(config, meta_lookup),
        ) as pool:

            for _ in pool.imap(
                process_chunk,
                folder_tasks,
            ):
                processed_chunks += 1

                elapsed = time.time() - folder_start
                avg_time = elapsed / processed_chunks

                remaining_chunks = (
                    total_chunks - processed_chunks
                )
                eta = remaining_chunks * avg_time

                processed_files = min(
                    processed_chunks * args.chunk_size,
                    len(remaining_files),
                )

                print(
                    f"[{folder}] "
                    f"{processed_chunks}/{total_chunks} chunks "
                    f"({processed_files}/{len(remaining_files)} files) | "
                    f"ETA: {eta / 60:.1f} min"
                )

        folder_time = time.time() - folder_start
        print(
            f"Finished {folder} "
            f"in {folder_time / 60:.1f} min"
        )

    # -------------------------------------------------
    # Final metadata synchronization
    # -------------------------------------------------

    print("\nFinal metadata sync...")

    valid_rows = []

    for _, row in meta_df.iterrows():
        label = row["label"]
        filename = row["filename"]

        enhanced_path = os.path.join(
            output_root,
            label,
            filename,
        )

        if os.path.exists(enhanced_path):
            valid_rows.append(row)

    if len(valid_rows) == 0:
        print("Warning: no enhanced files were found.")

    final_df = pd.DataFrame(valid_rows)

    if not final_df.empty:
        final_df = final_df.drop_duplicates(
            subset=["label", "filename"]
        )

    final_df.to_csv(
        output_metadata,
        index=False,
    )

    total_time = time.time() - global_start

    print(f"\nFinal metadata size: {len(final_df)}")
    print(f"Total runtime: {total_time / 60:.1f} minutes")
    print("\n========== DONE ==========\n")


def main():
    args = parse_args()
    build(args)


if __name__ == "__main__":
    main()