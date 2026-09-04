"""
kaggle_download_google_speech_commands.py

Download the Google Speech Commands dataset from Kaggle.
"""

import argparse
import os

import kagglehub


def parse_args():
    parser = argparse.ArgumentParser(description="Download the Google Speech Commands dataset.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory used by KaggleHub for storing the dataset. "
            "If not provided, the default KaggleHub cache is used."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is not None:
        os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(args.output_dir)

    path = kagglehub.dataset_download("neehakurelli/google-speech-commands")

    print(f"Dataset downloaded to: {path}")


if __name__ == "__main__":
    main()