"""
build_noise_dataset.py

Generate a noisy speech dataset from clean speech samples.

Pipeline:
1. Load clean speech files
2. Split files into train/val/test
3. Add random background noise
4. Save noisy audio files
5. Generate metadata CSV

This script is intended for dataset creation
and is not part of the training/evaluation pipeline.
"""

import argparse
import os
import random
import librosa
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from noise_utils import mix_with_noise_at_snr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a noisy Google Speech Commands dataset.")
    parser.add_argument("--clean_root", type=str, required=True, help="Path to the Google Speech Commands dataset directory.",)
    parser.add_argument("--output_root", type=str, required=True, help="Directory in which the noisy dataset will be created.",)
    parser.add_argument("--meta_path", type=str,  required=True,  help="Path for saving the generated metadata CSV file.",)
    return parser.parse_args()


def main():
    args = parse_args()

    clean_root = args.clean_root
    output_root = args.output_root
    meta_path = args.meta_path

    # Google Speech Commands background-noise directory
    noise_root = os.path.join(clean_root, "_background_noise_")

    os.makedirs(output_root, exist_ok=True)

    # Load noise bank
    noise_files = [f for f in os.listdir(noise_root) if f.endswith(".wav")]
    noise_bank = {}

    for f in noise_files:
        name = os.path.splitext(f)[0]
        y, _ = librosa.load( os.path.join(noise_root, f), sr=16000)
        noise_bank[name] = y

    # ----------------------------------------------------------
    # Collect clean speech files
    # ----------------------------------------------------------
    all_files = []

    folders = [
        f for f in os.listdir(clean_root)
        if os.path.isdir(os.path.join(clean_root, f))
        and f != "_background_noise_"
    ]

    folders = sorted(folders)

    for label in folders:
        in_dir = os.path.join(clean_root, label)

        files = [ f for f in os.listdir(in_dir) if f.endswith(".wav")]

        for fname in files:
            all_files.append((label, fname))

    # Train / validation / test split
    train_files, valtest_files = train_test_split(all_files, test_size=0.2, random_state=42)
    val_files, test_files = train_test_split( valtest_files, test_size=0.5, random_state=42,)
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")

    rows = []

    # Generate noisy samples
    def process(files_list, split_name, snr_choices):
        """
        Generate noisy versions of a list of audio files.

        For each sample:
        - select a random noise source
        - select a random SNR
        - mix clean and noise
        - save noisy waveform
        - record metadata
        """

        for label, fname in tqdm(files_list, desc=split_name):

            in_path = os.path.join(clean_root, label, fname)
            clean, _ = librosa.load(in_path, sr=16000)
            noise_name = random.choice(list(noise_bank.keys()))
            noise = noise_bank[noise_name]
            snr = random.choice(snr_choices)
            noisy = mix_with_noise_at_snr(clean, noise, snr)
            out_dir = os.path.join(output_root, split_name, label)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, fname)

            sf.write(out_path, noisy, 16000)
            rows.append({
                "filename": fname,
                "label": label,
                "snr": snr,
                "noise": noise_name,
                "split": split_name,
            })

    # Training and validation:
    # SNR = 0, 5, 10 dB
    process(
        train_files,
        "train",
        [0, 5, 10],
    )

    process(
        val_files,
        "val",
        [0, 5, 10],
    )

    # Test:
    # SNR = 2.5, 7.5, 12.5 dB
    process(
        test_files,
        "test",
        [2.5, 7.5, 12.5],
    )

    # Save metadata
    df_meta = pd.DataFrame(rows)
    df_meta.to_csv(meta_path, index=False)

    print("Dataset built successfully!")
    print(f"Noisy dataset: {output_root}")
    print(f"Metadata:      {meta_path}")


if __name__ == "__main__":
    main()