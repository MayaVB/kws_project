import argparse
import os
import shutil

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build paired clean/noisy train and validation data for SGMSE.")
    parser.add_argument("--clean_root", type=str, required=True, help="Root directory of the clean Speech Commands dataset.",)
    parser.add_argument("--noisy_root", type=str, required=True,
        help=(
            "Root directory of the noisy dataset containing "
            "train/ and val/ subdirectories."
        ),)
    parser.add_argument("--output_root", type=str, required=True, help="Output directory for the paired SGMSE dataset.",)
    return parser.parse_args()


def copy_split(clean_root, noisy_root, output_root, subset_name):

    clean_out = os.path.join(output_root, subset_name, "clean")
    noisy_out = os.path.join(output_root, subset_name, "noisy")

    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(noisy_out, exist_ok=True)

    total_files = 0

    for label in os.listdir(noisy_root):
        noisy_label_dir = os.path.join(noisy_root, label)
        clean_label_dir = os.path.join(clean_root, label)

        if not os.path.isdir(noisy_label_dir):
            continue

        for fname in tqdm(
            os.listdir(noisy_label_dir),
            desc=f"{subset_name}-{label}",
        ):

            if not fname.endswith(".wav"):
                continue

            noisy_path = os.path.join(noisy_label_dir, fname)
            clean_path = os.path.join(clean_label_dir, fname)

            if not os.path.exists(clean_path):
                print(f"Missing clean file: {clean_path}")
                continue

            shutil.copy2(
                noisy_path,
                os.path.join(noisy_out, fname),
            )

            shutil.copy2(
                clean_path,
                os.path.join(clean_out, fname),
            )

            total_files += 1

    print(f"{subset_name}: {total_files} files copied")


def main():
    args = parse_args()

    clean_root = os.path.abspath(
        os.path.expanduser(args.clean_root)
    )
    noisy_root = os.path.abspath(
        os.path.expanduser(args.noisy_root)
    )
    output_root = os.path.abspath(
        os.path.expanduser(args.output_root)
    )

    print("\nBuilding SGMSE dataset...\n")

    copy_split(
        clean_root,
        os.path.join(noisy_root, "train"),
        output_root,
        "train",
    )

    copy_split(
        clean_root,
        os.path.join(noisy_root, "val"),
        output_root,
        "valid",
    )

    print("\nDone! Dataset ready at:")
    print(output_root)


if __name__ == "__main__":
    main()