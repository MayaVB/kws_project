# Reproducibility record

This directory records the state in which the final KWS results were produced,
and how to rebuild the software environment. It was assembled on 2026-11-08
from read-only inspection of the lab machine. Every value is either observed
directly or marked as unverified.

| File | Contents |
|---|---|
| [`environment/SYSTEM.md`](environment/SYSTEM.md) | OS, GPU, driver, CUDA, Python, PyTorch, environment provenance, source revisions |
| [`environment/sgmse_env.pip-freeze.2026-11-08.txt`](environment/sgmse_env.pip-freeze.2026-11-08.txt) | Raw, unedited `pip freeze` of the main environment |
| [`environment/demo_env.pip-freeze.2026-11-08.txt`](environment/demo_env.pip-freeze.2026-11-08.txt) | Raw, unedited `pip freeze` of the demo environment |
| [`requirements/requirements-main.txt`](requirements/requirements-main.txt) | Pinned direct dependencies for KWS + SGMSE (observed versions, plus `tabulate` as a proposed addition) |
| [`requirements/requirements-demo.txt`](requirements/requirements-demo.txt) | Pinned dependencies for the optional Gradio demo |
| [`configs/`](configs/) | Reconstructed configs for the 30-class and both 6-class experiments, with per-key provenance |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | The final experiments, the evidence behind each config, results, limits on exact reproduction |
| [`CHECKPOINTS.md`](CHECKPOINTS.md) | SGMSE and DS-CNN checkpoint identities (SHA-256, metadata, inference settings) |
| [`DATA_LAYOUT.md`](DATA_LAYOUT.md) | Current data locations and a proposed convention for GannotLab storage |

## Verifying an environment

From the repository root, with the environment activated:

```bash
# 1. Versions
python -c "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda)"
#    expected: 3.9.x 2.8.0+cu128 12.8

# 2. Imports of every pipeline module (CPU only, no GPU or data needed)
CUDA_VISIBLE_DEVICES="" python - <<'EOF'
import sys
sys.path[:0] = ["code", "code/kws/src", "code/sgmse"]
import tabulate, analysis_utils, config, dataset, features, models, metrics, train, visualization
import new_noise_dataset, enhanced_dataset, kws.demo.demo_utils
from sgmse.model import ScoreModel
print("main environment OK")
EOF

# 3. Checkpoint identity (reads ~1.3 GB per file)
sha256sum path/to/kws_sgmse_trained.ckpt
#    expected: 90e19444b5e6c7d9a07cc49543804a02f57717d8d971e45832ee531b447f7af5
```

## Corrections log

| Date | File | Correction |
|---|---|---|
| 2026-11-08 | `CHECKPOINTS.md`, main README | The epoch-176 SGMSE run used **batch size 1 × gradient accumulation 4, weight decay 0** (from the checkpoint's loop counters and optimizer state), not batch 2 × accumulation 2 |
| 2026-11-08 | `EXPERIMENTS.md`, main README | `random_state` is read but **unused** by KWS training; the split comes from the metadata CSV |
| 2026-11-08 | `EXPERIMENTS.md`, `DATA_LAYOUT.md` | The noisy metadata does **not** record noise-segment offsets, so it cannot recreate the mixtures exactly; `data/noisy_new` is the only exact copy |
| 2026-11-08 | `configs/kws_30class_eval.yaml` (comment only, left unedited) | Its "usage limitation" note is obsolete: `run_evaluation_main.py` now accepts `--config` |

The four files in `configs/` are kept exactly as reconstructed. Corrections are
recorded here rather than by editing them.

## Snapshot policy

The `*.pip-freeze.*.txt` files are raw evidence and must not be edited. If an
environment changes, add a new dated snapshot next to the old one.
