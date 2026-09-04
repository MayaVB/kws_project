import argparse
import os

import kagglehub


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset used by the SGMSE pipeline."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Kaggle dataset identifier, for example "
            "'jweiqi/voicebank-demand-16k' or "
            "'aanhari/demand-dataset'."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional Kaggle cache/output directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is not None:
        output_dir = os.path.abspath(
            os.path.expanduser(args.output_dir)
        )
        os.makedirs(output_dir, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = output_dir

    path = kagglehub.dataset_download(args.dataset)

    print("Downloaded to:", path)


if __name__ == "__main__":
    main()
