import os
import subprocess
import pandas as pd
import time
from multiprocessing import Pool
from math import ceil

# ================= CONFIG ================= #

INPUT_ROOT = "/home/dsi/skopavi/Project/kws_project/generated_noisy"
OUTPUT_ROOT = "/home/dsi/skopavi/Project/kws_project/generated_enhanced"

META_IN = "/home/dsi/skopavi/Project/kws_project/generated_noisy_metadata.csv"
META_OUT = "/home/dsi/skopavi/Project/kws_project/generated_enhanced_metadata.csv"

CKPT_PATH = "/home/dsi/skopavi/Project/kws_project/sgmse_repo/checkpoints/train_vb_29nqe0uh_epoch=115.ckpt"

CHUNK_SIZE = 10
N_WORKERS = 1

TMP_ROOT = "/tmp/sgmse_chunks"

# ========================================= #

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)

meta_df = pd.read_csv(META_IN)

meta_lookup = {
    (row["label"], row["filename"]): row
    for _, row in meta_df.iterrows()
}


# =====================================================
# WORKER
# =====================================================
def process_chunk(args):
    folder, files = args

    tmp_in = os.path.join(TMP_ROOT, f"in_{os.getpid()}")
    tmp_out = os.path.join(TMP_ROOT, f"out_{os.getpid()}")

    os.makedirs(tmp_in, exist_ok=True)
    os.makedirs(tmp_out, exist_ok=True)

    # copy
    for f in files:
        src = os.path.join(INPUT_ROOT, folder, f)
        dst = os.path.join(tmp_in, f)
        os.system(f"cp {src} {dst}")

    # run model
    cmd = f"""
    python sgmse_repo/sgmse/enhancement.py \
        --test_dir {tmp_in} \
        --enhanced_dir {tmp_out} \
        --ckpt {CKPT_PATH} \
        --device cuda
    """
    subprocess.run(cmd, shell=True)

    meta_rows = []

    for f in files:
        src = os.path.join(tmp_out, f)
        dst = os.path.join(OUTPUT_ROOT, folder, f)

        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.system(f"mv {src} {dst}")

            key = (folder, f)
            row = meta_lookup.get(key)

            if row is not None:
                meta_rows.append(row)

    # cleanup
    os.system(f"rm -rf {tmp_in}")
    os.system(f"rm -rf {tmp_out}")

    return meta_rows


# =====================================================
# MAIN
# =====================================================
def build():

    print("\n========== DATASET STATUS ==========")

    total_all = 0
    done_all = 0

    folder_stats = []

    folders = sorted([
        f for f in os.listdir(INPUT_ROOT)
        if os.path.isdir(os.path.join(INPUT_ROOT, f))
    ])

    for folder in folders:
        in_dir = os.path.join(INPUT_ROOT, folder)
        out_dir = os.path.join(OUTPUT_ROOT, folder)

        all_files = [f for f in os.listdir(in_dir) if f.endswith(".wav")]
        done_files = []

        if os.path.exists(out_dir):
            done_files = [f for f in os.listdir(out_dir) if f.endswith(".wav")]

        total = len(all_files)
        done = len(done_files)
        remaining = total - done

        total_all += total
        done_all += done

        folder_stats.append((folder, total, done, remaining))

        print(f"{folder:15} | {done}/{total} ({100*done/total:.1f}%) | remaining: {remaining}")
    print("\n========== START BUILD ==========")

    global_start = time.time()

    for folder in folders:

        folder_start = time.time()

        in_dir = os.path.join(INPUT_ROOT, folder)
        out_dir = os.path.join(OUTPUT_ROOT, folder)

        os.makedirs(out_dir, exist_ok=True)

        all_files = [f for f in os.listdir(in_dir) if f.endswith(".wav")]
        done_files = [f for f in os.listdir(out_dir) if f.endswith(".wav")]

        total = len(all_files)
        done = len(done_files)

        print(f"\n=== {folder} ===")
        print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

        if done == total:
            print("✔ Already done")
            continue

        remaining_files = [
            f for f in all_files
            if not os.path.exists(os.path.join(out_dir, f))
        ]

        total_chunks = ceil(len(remaining_files) / CHUNK_SIZE)

        print(f"Remaining files: {len(remaining_files)}")
        print(f"Chunks: {total_chunks} (each {CHUNK_SIZE} files)\n")

        # build chunks
        folder_tasks = []
        for i in range(0, len(remaining_files), CHUNK_SIZE):
            chunk = remaining_files[i:i + CHUNK_SIZE]
            folder_tasks.append((folder, chunk))

        # run with progress
        processed_chunks = 0
        all_rows = []

        with Pool(N_WORKERS) as p:
            for result in p.imap(process_chunk, folder_tasks):
                processed_chunks += 1
                all_rows.extend(result)

                # progress calc
                elapsed = time.time() - folder_start
                avg_time = elapsed / processed_chunks
                remaining = total_chunks - processed_chunks
                eta = remaining * avg_time

                print(
                    f"[{folder}] "
                    f"{processed_chunks}/{total_chunks} chunks "
                    f"({processed_chunks*CHUNK_SIZE}/{len(remaining_files)} files) | "
                    f"ETA: {eta/60:.1f} min"
                )

        folder_time = time.time() - folder_start
        print(f"✔ Finished {folder} in {folder_time/60:.1f} min")

    # =====================================================
    # FINAL METADATA
    # =====================================================
    print("\nFinal metadata sync...")

    valid_rows = []

    for _, row in meta_df.iterrows():
        label = row["label"]
        fname = row["filename"]

        path = os.path.join(OUTPUT_ROOT, label, fname)

        if os.path.exists(path):
            valid_rows.append(row)

    if len(valid_rows) == 0:
        print("WARNING: No files found!")

    final_df = pd.DataFrame(valid_rows)
    final_df = final_df.drop_duplicates(subset=["label", "filename"])
    final_df.to_csv(META_OUT, index=False)

    total_time = time.time() - global_start

    print(f"\nFinal metadata size: {len(final_df)}")
    print(f"Total runtime: {total_time/60:.1f} minutes")

    print("\n========== DONE ==========\n")


# =====================================================
if __name__ == "__main__":
    build()