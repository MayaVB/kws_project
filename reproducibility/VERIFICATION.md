# Clean-environment verification (2026-11-08)

A fresh main environment was built from the committed requirements and checked
against the historical results. All runs were CPU-only (`CUDA_VISIBLE_DEVICES=""`,
8 threads, `nice -n 10`). No historical data, checkpoint or result was modified.

## Environment

| Item | Value |
|---|---|
| Path (lab machine) | `/home/dsi/skopavi/envs/kws-main-py39` |
| Created with | `/usr/bin/python3.9 -m venv`, then `pip==26.0.1`, then `pip install --no-cache-dir -r requirements/requirements-main.txt -c environment/sgmse_env.pip-freeze.2026-11-08.txt` |
| Snapshot | [`environment/kws-main-py39.pip-freeze.2026-11-08.txt`](environment/kws-main-py39.pip-freeze.2026-11-08.txt): 103 packages, byte-identical to `pip freeze` of the installed environment |
| Key versions | Python 3.9.25, torch 2.8.0+cu128, torchaudio 2.8.0, pytorch-lightning 2.6.0, numpy 1.26.4, scipy 1.13.1, scikit-learn 1.6.1, librosa 0.11.0, pesq 0.0.4 (built from source), tabulate 0.9.0 |
| Relation to `sgmse_env` | All 102 shared packages have identical versions. Added: `tabulate`. Omitted: 39 packages not used by the pipeline (Jupyter/IPython, torchvision, pyarrow, h5py, torchsde, sdeint, pyroomacoustics, ninja, cython, ...) |
| Install | 28 min, no warnings or dependency conflicts; `pip check` clean |
| Size | 7.0 GiB of files (11 GB allocated on the NFS home filesystem) |

## Results

| Check | Result |
|---|---|
| Imports, versions, `pip check` | Pass |
| Unit tests (`python -m unittest discover -s tests -t .`) | 20 tests, all pass, none skipped |
| Four historical configs (`configs/`) | All parse; class counts and test-set sizes match their runs |
| Saved DS-CNN artifacts (3 sets) | SHA-256 as in `CHECKPOINTS.md`; strict weight loading; 13 MFCC features; 30 / 6 / 6 classes |
| DS-CNN predictions, both 6-class runs, clean / noisy / enh_baseline / enh_trained | **10,284 / 10,284 per-file predictions identical** to the saved `*_all_predictions.csv`; all 8 accuracies identical to 6 decimals |
| SGMSE checkpoint loading on CPU | Epoch 176: loads cleanly. Pretrained VoiceBank (Lightning 1.5.10): loads; Lightning upgrades the format in memory only. Both: 65.59 M parameters, EMA weights present, about 10 s, peak about 3.0 GiB RAM |
| SGMSE enhancement on CPU (3 noisy test files per checkpoint, default sampler: PC, reverse diffusion, ALD, N=30, snr 0.5) | Outputs 16 kHz, 16000 samples, finite; SI-SDR improves from 2.5–7.5 dB (noisy) to 14.5–19.3 dB. About 55 s per file with 8 threads, peak about 3.1 GiB. Compared with the historical enhanced files for the same 3 inputs: mean SI-SDR 15.2 vs 15.3 dB (pretrained) and 17.0 vs 16.9 dB (epoch 176); mean PESQ 1.52 vs 1.93 and 1.69 vs 1.82. The sampler is unseeded, so outputs are not expected to be identical, and 3 files are too few to separate sampling spread from a systematic difference |
| **30-class evaluation, end to end** (`run_evaluation_main.py --config <copy of configs/kws_30class_eval.yaml with output_dir redirected>`, no wrappers) | Completes in about 19 min, peak 1.6 GB. Accuracies **0.9303 / 0.6484 / 0.7890 / 0.8248**, equal to the 2026-05-29 run. `metrics.txt`, `reports.txt` and `misclassified_report.txt` are **byte-identical** to the 2026-05-29 run |

Details of the 30-class comparison with
`outputs/new_runs/2026-05-29_19-14-17_all_annot_false`:

- `metrics_comparison.txt`: all PESQ / ESTOI / SI-SDR / SI-SIR / SI-SAR cells are
  identical. Only the accuracy column differs in formatting (`0.93` instead of
  `0.9303`), because commit `96cc177` (2026-08-09) changed it from
  `round(acc, 4)` to `f"{acc:.2f}"`. The full-precision values are equal.
- Prediction CSVs (new output; the 2026-05-29 run predates them): 6473 rows per
  mode, exactly the test split; accuracies computed from them match;
  SNR/noise columns match the metadata.
- Plots: the six per-SNR and per-noise confusion plots are byte-identical. The
  confusion-matrix comparison differs only in presentation: the 2026-05-29 run
  had cell annotations switched off ("annot_false") and no figure title. The
  example-signal plot differs only in its title line (SNR and noise added in
  `96cc177`); the waveforms are pixel-identical. Four plots are new outputs of
  the current code.

Two regressions were found and fixed during this verification, in commits
`c5087d2` (metrics subprocess run twice) and `0ac1696` (standalone evaluation
could not complete). Both were introduced after the final experiments, so the
historical results are unaffected.

## Not verified

- Running the new environment on a GPU. The wheels are the CUDA 12.8 builds, but
  only CPU runs were made.
- SGMSE training, the fine-tuning path, and regenerating the full enhanced
  datasets (GPU-hours; the sampler is stochastic, so never bit-exact).
- The PESQ difference seen in the 3-file enhancement test (needs more files or
  repeated samples).
- W&B logging, Kaggle and gdown downloads (credentials and network), and the
  Hugging Face copy of the epoch-176 checkpoint.
- The optional Gradio demo environment (`demo_env`) was not rebuilt.
