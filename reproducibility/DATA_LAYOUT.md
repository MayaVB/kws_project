# Data layout: current state and proposed convention

**Status: proposal only.** No data has been moved. No existing absolute path in
code, configs or documentation has been changed.

## Current layout (2026-11-08)

Everything sits under the project root
`/home/dsi/skopavi/Project/kws_project`, and configs use absolute paths.

| Content | Current path | Size | Needed for final results |
|---|---|---|---|
| Clean Google Speech Commands | `data/raw/data_new/` | 5.8 GB | yes |
| Noisy dataset | `data/noisy_new/{train,val,test}/` | 6.5 GB | yes |
| Noisy metadata | `data/noisy_new_metadata.csv` | 3.2 MB | yes (split, noise type and SNR per file; no noise offsets, so the wav files in `noisy_new/` are the only exact copy) |
| Enhanced: pretrained SGMSE+ | `data/enhanced_new/baseline/` | part of 712 MB | yes |
| Enhanced: epoch-176 SGMSE+ | `data/enhanced_new/trained_ep176/` | part of 712 MB | yes |
| Enhanced metadata (identical pair) | `data/enhanced_pretrainted_new_metadata.csv`, `data/enhanced_trained_ep176_new_metadata.csv` | 0.3 MB each | yes |
| SGMSE paired training data | `data/train_sgmse/{train,valid}/{clean,noisy}/` | 1.1 GB | only to retrain SGMSE |
| SGMSE checkpoint (epoch 176) + identical copy | `experiments/sgmse_logs/fcj02u93-kws_exp_9/` | 2 × 1.25 GB | one copy |
| Pretrained SGMSE+ checkpoint | `code/sgmse/checkpoints/` | 1.25 GB | yes (re-downloadable) |
| DS-CNN models and run reports | `outputs/` | 72 MB | the three final model folders |

Absolute paths appear in: `code/kws/new_config.yaml` (local, ignored by Git),
`reproducibility/configs/*.yaml`, `code/sgmse/model_visualization/sgmse_visualization.py`,
and two experimental scripts ignored by Git (`code/kws/denoiser/noise_onthefly_dataset.py`,
`code/kws/dataset_generation/build_reverb_dataset.py`).

## Proposed convention for GannotLab storage

Use one root per artifact class, so the shared location can be set once and code
and configs stay free of personal home directories.

```text
<KWS_ROOT>/                                  # e.g. a GannotLab shared project directory (TBD)
├── data/
│   ├── speech_commands/                     # ← data/raw/data_new
│   ├── noisy/                               # ← data/noisy_new
│   │   └── metadata.csv                     # ← data/noisy_new_metadata.csv
│   ├── enhanced/
│   │   ├── sgmse_vbdmd_pretrained/          # ← data/enhanced_new/baseline
│   │   ├── sgmse_kws_ep176/                 # ← data/enhanced_new/trained_ep176
│   │   └── metadata.csv                     # ← either identical enhanced_*_metadata.csv
│   └── sgmse_train/                         # ← data/train_sgmse
├── checkpoints/
│   ├── sgmse/
│   │   ├── kws_ep176.ckpt                   # ← epoch=176-valid_loss=558.3868.ckpt (preferred source)
│   │   └── vbdmd_pretrained_ep115.ckpt      # ← train_vb_29nqe0uh_epoch=115.ckpt
│   └── dscnn/
│       ├── 30class/                         # ← outputs/dscnn_full_dataset_best
│       ├── 6class_bed-bird-dog-go-happy-house/   # ← 2026-07-07_11-36-05/models
│       └── 6class_go-no-off-on-stop-yes/         # ← 2026-06-08_12-14-57_presentation/models
└── outputs/                                 # new run outputs
```

Principles:

1. **Names describe content, not history**: `sgmse_kws_ep176` instead of `trained_ep176`,
   and `sgmse_vbdmd_pretrained` instead of `baseline`.
2. **Configs would reference `<KWS_ROOT>`** (substituted per machine) instead of
   `/home/dsi/skopavi/...`. This requires either a small loader change or
   per-machine config copies, which is a decision for a later phase.
3. **Verify every migrated artifact against the SHA-256 values** in
   `CHECKPOINTS.md` and `EXPERIMENTS.md`.
4. **Keep one copy of the epoch-176 checkpoint.** `kws_sgmse_trained.ckpt` is the
   publication name on Hugging Face, not a separate model.

## Open decisions (not taken)

- The GannotLab root path and its access permissions.
- Whether to copy data or move it, and when to retire the copies under `/home/dsi/skopavi`.
- Whether `<KWS_ROOT>` is resolved by an environment variable, a config key, or
  rewritten configs.
- Caution: the `.gitignore` pattern `data/` ignores **any** directory named `data`
  at any depth. Today that only hides `code/sgmse/data/`, which holds an empty
  Kaggle download directory, but it would silently hide any future `data/` code
  package.
