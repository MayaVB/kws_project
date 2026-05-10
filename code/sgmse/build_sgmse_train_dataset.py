import os
import shutil
from tqdm import tqdm

# PATHS
CLEAN_ROOT = "/home/dsi/skopavi/Project/kws_project/data/raw/data_new"
NOISY_TRAIN_ROOT = "/home/dsi/skopavi/Project/kws_project/data/noisy_new/train"
NOISY_VAL_ROOT   = "/home/dsi/skopavi/Project/kws_project/data/noisy_new/val"
OUT_ROOT = "/home/dsi/skopavi/Project/kws_project/data/train_sgmse"

# COPY FUNCTION
def copy_split(noisy_root, subset_name):

    clean_out = os.path.join(OUT_ROOT, subset_name, "clean")
    noisy_out = os.path.join(OUT_ROOT, subset_name, "noisy")

    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(noisy_out, exist_ok=True)

    total_files = 0

    for label in os.listdir(noisy_root):
        noisy_label_dir = os.path.join(noisy_root, label)
        clean_label_dir = os.path.join(CLEAN_ROOT, label)

        if not os.path.isdir(noisy_label_dir):
            continue

        for fname in tqdm(os.listdir(noisy_label_dir), desc=f"{subset_name}-{label}"):

            if not fname.endswith(".wav"):
                continue

            noisy_path = os.path.join(noisy_label_dir, fname)
            clean_path = os.path.join(clean_label_dir, fname)

            if not os.path.exists(clean_path):
                print(f"⚠ Missing clean file: {clean_path}")
                continue

            # copy noisy
            shutil.copy2(noisy_path, os.path.join(noisy_out, fname))

            # copy clean
            shutil.copy2(clean_path, os.path.join(clean_out, fname))

            total_files += 1

    print(f"✔ {subset_name}: {total_files} files copied")


# MAIN
def main():

    print("\nBuilding SGMSE dataset...\n")

    # TRAIN
    copy_split(NOISY_TRAIN_ROOT, "train")

    # VALID
    copy_split(NOISY_VAL_ROOT, "valid")

    print("\n✅ Done! Dataset ready at:")
    print(OUT_ROOT)


if __name__ == "__main__":
    main()