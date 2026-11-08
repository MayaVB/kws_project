# Final KWS experiments: 30-class and 6-class

The final project includes both a 30-class experiment and 6-class experiments.
No run saved a copy of its configuration. The configs in `configs/` were
therefore reconstructed from evidence only, in this order of strength:

1. **Run outputs**: class lists in `reports.txt`, file counts, metric values
2. **Bracketing commits**: the committed `code/kws/new_config.yaml`
   immediately before and after the run (tracked in Git 2026-05-10 → 2026-09-04)
3. **Current code**: keys the entry points require today

Values that the evidence does not settle are marked in the config comments.
None were filled in by guesswork.

## Shared inputs (all experiments)

| Input | Path | Evidence |
|---|---|---|
| Clean speech (Google Speech Commands, Kaggle `neehakurelli/google-speech-commands`) | `data/raw/data_new` | 30 keyword folders + `_background_noise_` |
| Noisy dataset | `data/noisy_new/{train,val,test}` | `data/noisy_new_metadata.csv`: 51776 train / 6472 val / 6473 test rows; SNR ∈ {0, 2.5, 5, 7.5, 10, 12.5} dB; SHA-256 `5432a0b07f66333c58708f6923d239ecc71acaf2a69d083617e3dcc2fd769340` |
| Baseline enhancement (pretrained SGMSE+) | `data/enhanced_new/baseline` | 6473 files |
| Trained enhancement (epoch-176 SGMSE+) | `data/enhanced_new/trained_ep176` | 6473 files |
| Enhanced metadata | `data/enhanced_pretrainted_new_metadata.csv`, `data/enhanced_trained_ep176_new_metadata.csv` | Byte-identical to each other (both list the same 6473 test files); SHA-256 `20dc774ac4c57248e979bbe68531564f40e0b0c4066cdfd4c3bc7833cf0c645f` |

**The noisy dataset cannot be regenerated exactly** *(corrected 2026-11-08)*:

- *Historical fact:* `data/noisy_new` (2026-05-08) was generated without any seed.
  `noisy_new_metadata.csv` records the split, label, noise type and SNR of every
  file, but **not the noise-segment offset**. `noise_utils.mix_with_noise_at_snr`
  draws that offset with unseeded `np.random.randint`. Even with the metadata, the
  exact mixtures cannot be recreated. The generated wav files are the only exact
  copy, so `data/noisy_new` itself must be preserved.
- *Current working tree (not yet committed):* `build_noise_dataset.py` adds
  `random.seed(42)` and a sorted file listing. That fixes noise type and SNR
  selection for future generations, but it does not seed `np.random`, so offsets
  stay random.
- *Recommended for future datasets (not implemented):* also seed `np.random`, and
  record the offset in the metadata.

## 30-class experiment

| Item | Value |
|---|---|
| Training run | `outputs/new_runs/2026-05-26_18-01-43_all_folders` (HEAD then: `2b029a9`, plus any changes not committed at the time) |
| Evaluation run with full metrics | `outputs/new_runs/2026-05-29_19-14-17_all_annot_false` (evaluation only; same model) |
| Config | `configs/kws_30class_train.yaml`, `configs/kws_30class_eval.yaml` |
| Saved model | `outputs/dscnn_full_dataset_best/` (byte-identical to the 05-26 run's `models/`) |
| Classes | all 30 (`folder_start 0`, `folder_end 30`, no `selected_folders`) |
| Test files | 6473 |

| Mode | Accuracy | PESQ | ESTOI | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|---|
| clean | 0.9303 | – | – | – | – | – |
| noisy | 0.6484 | – | – | – | – | – |
| enh_baseline | 0.789 | 1.60 ± 0.45 | 0.64 ± 0.22 | 11.1 ± 5.7 | 30.2 ± 9.4 | 11.4 ± 5.6 |
| enh_trained (ep176) | 0.8248 | 2.00 ± 0.50 | 0.69 ± 0.21 | 15.3 ± 4.3 | 26.6 ± 7.1 | 15.9 ± 4.1 |

Source: `2026-05-29_19-14-17_all_annot_false/metrics_comparison.txt`.

Key evidence:
- `enh_trained` → `trained_ep176`. The metrics equal the `enh_trained_ep176` row of
  `2026-05-25_23-39-48_all_trained_modes/metrics_comparison.txt`. The config
  committed on 2026-05-20 still pointed at `trained_ep166`, so the bracket
  alone would not settle this.
- `epochs`: the bracketing configs say 100 and 150. `training_history.png` shows
  early stopping at about 91 epochs, so both give the same run.

## 6-class experiment A: bed, bird, dog, go, happy, house

| Item | Value |
|---|---|
| Run | `outputs/new_runs/2026-07-07_11-36-05` (HEAD then: `38cb187`, plus changes not committed at the time) |
| Config | `configs/kws_6class_bed-bird-dog-go-happy-house.yaml` |
| Model | `outputs/new_runs/2026-07-07_11-36-05/models/` |
| Test files | 1124 (= 183 + 180 + 165 + 245 + 173 + 178, the six test folders) |
| Class-list evidence | `reports.txt`; also the committed config of 2026-08-09 and the current local `new_config.yaml` |

| Mode | Accuracy | PESQ | ESTOI | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|---|
| clean | 0.95 | – | – | – | – | – |
| noisy | 0.70 | – | – | – | – | – |
| enh_baseline | 0.85 | 1.60 ± 0.46 | 0.62 ± 0.22 | 11.1 ± 5.8 | 30.4 ± 9.4 | 11.3 ± 5.7 |
| enh_trained | 0.89 | 2.04 ± 0.50 | 0.69 ± 0.20 | 15.5 ± 4.2 | 27.1 ± 7.1 | 16.0 ± 4.0 |

## 6-class experiment B: go, no, off, on, stop, yes

| Item | Value |
|---|---|
| Run | `outputs/new_runs/2026-06-08_12-14-57_presentation` (HEAD then: `1c3a5ae`, plus changes not committed at the time) |
| Config | `configs/kws_6class_go-no-off-on-stop-yes.yaml` |
| Model | `outputs/new_runs/2026-06-08_12-14-57_presentation/models/` |
| Test files | 1447 (equals the sum of the six test folders) |
| Class-list evidence | `reports.txt` only (no committed config has this list) |

| Mode | Accuracy | PESQ | ESTOI | SI-SDR | SI-SIR | SI-SAR |
|---|---|---|---|---|---|---|
| clean | 0.95 | – | – | – | – | – |
| noisy | 0.66 | – | – | – | – | – |
| enh_baseline | 0.84 | 1.64 ± 0.48 | 0.62 ± 0.22 | 11.4 ± 5.6 | 30.4 ± 9.6 | 11.7 ± 5.6 |
| enh_trained | 0.87 | 2.03 ± 0.51 | 0.67 ± 0.21 | 15.6 ± 4.4 | 27.1 ± 7.4 | 16.2 ± 4.2 |

## Limits on exact reproduction

- **KWS training is not seeded.** `train.set_seed()` exists but is never called.
  Weight initialization, training-set shuffling and dropout are therefore
  unseeded, so a retrained DS-CNN will give close but not identical accuracies.
  Use the saved models for exact numbers.
- **`random_state` is unused** *(corrected 2026-11-08)*. `run_training.py` reads
  `random_state` but never uses it. The train/val/test split is fixed by the
  `split` column of the metadata CSV, not computed at run time. The historical
  configs keep `random_state: 42` as it was recorded. The trailing comment
  "(train/val split seed)" in `configs/kws_30class_train.yaml` is inaccurate and
  is kept only because the historical configs are not edited. Seeding future runs
  needs a separate, explicit setting (proposed, not implemented).
- **The code has changed since the runs.** `code/kws/src` was modified in
  commits `92da893`, `5dd2ae5`, `96cc177` and `c655087`, after some or all of
  the runs. Whatever was not committed at run time is unknown. The configs
  target the current code.
- **SGMSE enhancement is stochastic and unseeded** (see `CHECKPOINTS.md`).
- **The interpreter used for the final KWS runs is not recorded**
  (see `environment/SYSTEM.md`).

## Runs not part of the final results

`2026-05-19_*`, `2026-05-20_*` (earlier class ranges), `2026-05-25_*` (checkpoint
selection across ep149 / ep149_ft / ep163 / ep166 / ep176), `2026-05-27_*` and
`2026-05-31_*` (bed / bird / cat), `2026-06-01_*` (three / tree / two),
`2026-06-04_*` (7 classes), `2026-06-23_*` and `2026-07-04_21-35-07` (go / no / on),
and `2026-07-04_21-43-53` (30 classes). The two 2026-07-04 runs report different
`enh_trained` metrics from the ep176 runs, which points to an enhanced dataset
that is no longer on disk. Their provenance is unresolved.
