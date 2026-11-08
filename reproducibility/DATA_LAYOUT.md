# Data layout: historical locations and the GannotLab archive

**Status (2026-11-08):** the verified copy in the GannotLab archive is the default
location for every dataset and model used by the final results. The original files
under the lab home directory still exist; nothing has been moved or deleted.

| Name used below | Path (lab machine) |
|---|---|
| `<repo>` (historical location) | `/home/dsi/skopavi/Project/kws_project`: where the experiments ran and the data was created |
| `<archive>` (current location) | `/dsi/gannot-lab/gannot-lab1/datasets/skopavi/kws_archive`: verified copy (mode 2750, group `dsi`) |

The other reproducibility documents give **historical** paths: they record where
each experiment read its inputs. Use the table below to find the same bytes in the archive.

## Where each artifact is

| Content | Historical path (`<repo>/…`) | Archive path (`<archive>/…`) | Files |
|---|---|---|---|
| Clean Google Speech Commands v0.01 | `data/raw/data_new/` | `datasets/speech_commands_v0.01/` | 64,732 |
| Noisy dataset (train/val/test) | `data/noisy_new/` | `datasets/noisy_new/` | 64,721 |
| Noisy metadata | `data/noisy_new_metadata.csv` | `datasets/noisy_new_metadata.csv` | 1 |
| Enhanced: pretrained SGMSE+ | `data/enhanced_new/baseline/` | `datasets/enhanced_new/baseline/` | 6,473 (12,946 in `enhanced_new/` together with the next row) |
| Enhanced: epoch-176 SGMSE+ | `data/enhanced_new/trained_ep176/` | `datasets/enhanced_new/trained_ep176/` | 6,473 |
| Enhanced metadata (identical pair) | `data/enhanced_pretrainted_new_metadata.csv`, `data/enhanced_trained_ep176_new_metadata.csv` | `datasets/` (same names) | 2 |
| SGMSE paired training data | `data/train_sgmse/` | `datasets/train_sgmse/` | 19,118 |
| SGMSE epoch-176 checkpoint (full training checkpoint) | `experiments/sgmse_logs/fcj02u93-kws_exp_9/epoch=176-valid_loss=558.3868.ckpt` | `checkpoints/sgmse/epoch=176-valid_loss=558.3868.ckpt` | 1 |
| Pretrained SGMSE+ (VoiceBank-DEMAND) | `code/sgmse/checkpoints/train_vb_29nqe0uh_epoch=115.ckpt` | `checkpoints/sgmse/train_vb_29nqe0uh_epoch=115.ckpt` | 1 |
| DS-CNN 30-class | `outputs/dscnn_full_dataset_best/` | `classifier_backups/dscnn/30class/` | 3 |
| DS-CNN 6-class A | `outputs/new_runs/2026-07-07_11-36-05/models/` | `classifier_backups/dscnn/6class_bed-bird-dog-go-happy-house/` | 3 |
| DS-CNN 6-class B | `outputs/new_runs/2026-06-08_12-14-57_presentation/models/` | `classifier_backups/dscnn/6class_go-no-off-on-stop-yes/` | 3 |
| All run outputs | `outputs/new_runs/` | `experiments_archive/outputs_new_runs/` | 353 |

Not archived: `experiments/sgmse_logs/fcj02u93-kws_exp_9/kws_sgmse_trained.ckpt`, a
byte-identical duplicate of the epoch-176 checkpoint. `<archive>/README.md` lists the
archive's contents and how to re-check them; `<archive>/manifests/` holds a per-file
SHA-256 manifest for every row above.

## How the code finds the data

No tracked code hard-codes a data path. Inputs come from the YAML config or from
command-line arguments. The one exception is the demo's mapping table from historical
to archive paths in `code/kws/demo/demo_utils.py`. Three local scripts ignored by Git
still contain absolute paths: `code/kws/dataset_generation/build_reverb_dataset.py`
(reads `<repo>/data/raw/data_new`), `code/kws/denoiser/noise_onthefly_dataset.py`
(paths that no longer exist) and `code/sgmse/model_visualization/sgmse_visualization.py`
(repository demo assets). None of them is used for the final results.

| Use | What to use |
|---|---|
| Re-run a final experiment | `reproducibility/configs/archive/*.yaml`: archive-location variants of the four historical configs (see [`README.md`](README.md#historical-and-archive-location-configs)) |
| Day-to-day training and evaluation on the lab machine | `code/kws/new_config.yaml` (local, ignored by Git). It points to the archive since 2026-11-08 |
| Browse `interesting_examples.csv` with the Gradio demo | Historical CSVs store `<repo>/data/...` paths. `gradio_demo.py` remaps them to the archive in memory; the CSVs are not edited. Use `--path-map OLD=NEW` for another location, `--no-path-remap` to disable |
| Another machine | Copy the archive layout and change the path values in a config copy |

`output_dir` still points to `<repo>/outputs/new_runs`. New results are written to
the working repository, never to the archive.

## Verification

- **Copy (Phase 5C):** every one of the 161,989 archived files has the same SHA-256 as
  its source. `rsync --checksum` dry runs found no differences. The 11 checkpoint and
  classifier hashes match [`CHECKPOINTS.md`](CHECKPOINTS.md).
- **Use (Phase 5D):** with every input read from the archive, the 30-class
  evaluation reproduces `metrics.txt`, `reports.txt` and `misclassified_report.txt`
  byte for byte, and both 6-class models reproduce 10,284 / 10,284 historical
  per-file predictions. A file-access trace (`strace`) found no read of the home data. Details in
  [`VERIFICATION.md`](VERIFICATION.md#archive-verification-phases-5c-and-5d-2026-11-08).

## Remaining risks and open items

- **No off-site copy.** The archive and the home originals are on the same Isilon
  cluster, which keeps snapshots every 12 h for 7 days. `noisy_new` cannot be
  regenerated (noise offsets were not recorded), and `enhanced_new` cannot be
  regenerated bit-exactly (unseeded sampler). Once the home originals are deleted,
  these exist in one place only until an off-site backup is made.
- **Kaggle upload staging.** `/home/dsi/skopavi/kws_kaggle_upload/` (outside the
  repository) holds 5 symlinks into `<repo>/data/`. They still point to the home
  originals and would dangle after deletion. They have not been changed.
- **Archive permissions** are 2750 (directories) and 640 (files), group `dsi`: the
  owner can write, members of `dsi` can read, other users have no access.
- The `.gitignore` pattern `data/` ignores **any** directory named `data` at any
  depth. Today that only hides `code/sgmse/data/` (an empty Kaggle download
  directory), but it would silently hide a future `data/` code package.
