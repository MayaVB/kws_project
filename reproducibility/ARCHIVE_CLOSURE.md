# KWS archive closure (2026-11-08)

Final state after moving the project's data and models to the GannotLab archive.
**Nothing has been deleted.** The home originals still exist; the cleanup plan in
section 8 needs explicit approval, path by path.

## 1. Canonical location

| | Path |
|---|---|
| Archive (canonical) | `/dsi/gannot-lab/gannot-lab1/datasets/skopavi/kws_archive` (also visible as `/private/gannot-lab/...`) |
| Storage | `psa.local.biu.ac.il:/ifs/a/private/gannot-lab` (Isilon). Directories 2750, files 640, owner `skopavi`, group `dsi` (readable by `dsi` members) |
| Contents | 14 items, 161,989 data files, 7,457 MiB content, 22,409 MiB allocated. Layout and historical ↔ archive mapping: [`DATA_LAYOUT.md`](DATA_LAYOUT.md); per-file manifests: `<archive>/manifests/` |
| Repository | `/home/dsi/skopavi/Project/kws_project` (lab machine), `https://github.com/MayaVB/kws_project.git` |

## 2. Verification status

| Check | Result |
|---|---|
| Copy (Phase 5C): source vs archive SHA-256 for every file, `rsync --checksum` dry runs | 14 / 14 identical |
| Use (Phase 5D): 30-class evaluation from the archive | `metrics.txt`, `reports.txt`, `misclassified_report.txt` byte-identical to the 2026-05-29 run |
| Use (Phase 5D): both 6-class models | 10,284 / 10,284 per-file predictions identical |
| Use (Phase 5D): both SGMSE checkpoints | Load and enhance on CPU; output format, alignment and level correct (sampler is stochastic) |
| Path audit (Phase 5D, `strace`) | No read of home data |
| Training reporting fix (5E-1B, `996531b`) | Tiny 3-epoch CPU training run on archived data writes every end-of-run output |
| Closure: every archived file vs frozen manifests (`reverify_all.sh`) | 161,989 OK, 0 extra files |
| Closure: 11 checkpoint/classifier hashes (`archive_checks_v2.py`, `checkpoint_hashes.sha256`) | 11 / 11 vs `CHECKPOINTS.md` at `700e943` and at `996531b` |
| Closure: CPU regression suite (`kws-main-py39`) | 33 / 33 pass; demo remap tests 9 / 9 in `demo_env` |
| Closure: active config, archive configs, 3 DS-CNN sets, 2 SGMSE checkpoints resolve and load from the archive | 101 / 101 checks pass |
| Closure: historical configs, `outputs/new_runs` (353 files), home checkpoints, public demo (`docs/`) | Unchanged |

## 3. Reproduction commands (lab machine)

```bash
cd /home/dsi/skopavi/Project/kws_project
source /home/dsi/skopavi/envs/kws-main-py39/bin/activate   # active environment

# tests (CPU)
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -t .

# 30-class evaluation of the saved model (reproduces the 2026-05-29 reports)
python code/kws/src/run_evaluation_main.py --config reproducibility/configs/archive/kws_30class_eval.yaml

# train + evaluate a final experiment (KWS training is not seeded: statistically, not bit-exactly, reproducible)
python code/kws/src/run_training.py --config reproducibility/configs/archive/kws_6class_bed-bird-dog-go-happy-house.yaml

# re-verify the archive
cd /dsi/gannot-lab/gannot-lab1/datasets/skopavi/kws_archive
sha256sum -c manifests/checkpoint_hashes.sha256
bash manifests/reverify_all.sh
```

Results go to `output_dir` (`<repo>/outputs/new_runs`), never to the archive.

## 4. Environments

| Environment | Status |
|---|---|
| `/home/dsi/skopavi/envs/kws-main-py39` | **Active.** Python 3.9.25, torch 2.8.0+cu128; built from `requirements/requirements-main.txt`; verified on CPU ([`VERIFICATION.md`](VERIFICATION.md)). GPU use not yet verified |
| `<repo>/demo_env` | Active for the local Gradio demo (`requirements/requirements-demo.txt`) |
| `<repo>/code/sgmse/sgmse_env` | Original environment, superseded; kept until a GPU check of `kws-main-py39` |
| `<repo>/kws_env` | Legacy, broken outside Docker, unused |

## 5. Git state

`main` at `996531b` ("Fix training crash at end of run: restore build_audio_paths")
plus this closure documentation commit. `origin/main` is at `0bf7739`; the later
commits are local until pushed. `stash@{0}` is untouched. `code/kws/new_config.yaml`
is local (ignored by Git) and points to the archive.

## 6. Demo and Kaggle paths

- **Gradio demo** (`code/kws/demo/gradio_demo.py --csv <interesting_examples.csv>`): historical
  CSVs store home paths. They are remapped in memory to the archive and the CSVs are
  unchanged; all 7 historical CSVs resolve.
  `--path-map OLD=NEW` / `--no-path-remap` override this.
- **Public demo** (GitHub Pages, `docs/`): self-contained, no dependency on data paths; unchanged.
- **Kaggle upload staging** `/home/dsi/skopavi/kws_kaggle_upload/` (no remote change made):

  | Link | Now points to | Before (rollback record `~/kws_kaggle_upload.links.before.tsv`) |
  |---|---|---|
  | `noisy` | `<archive>/datasets/noisy_new` | `<repo>/data/noisy_new` |
  | `enhanced` | `<archive>/datasets/enhanced_new` | `<repo>/data/enhanced_new` |
  | `metadata/noisy_new_metadata.csv` | `<archive>/datasets/noisy_new_metadata.csv` | `<repo>/data/noisy_new_metadata.csv` |
  | `metadata/enhanced_trained_ep176_metadata.csv` | `<archive>/datasets/enhanced_trained_ep176_new_metadata.csv` | `<repo>/data/` (same name) |
  | `metadata/enhanced_pretrained_metadata.csv` | `<archive>/datasets/enhanced_pretrainted_new_metadata.csv` | `<repo>/data/` (same name) |

## 7. Known limitations

- **Hugging Face `kws_sgmse_trained.ckpt`: identity unresolved.** Hugging Face reports
  594,804,736 bytes, while the original epoch-176 checkpoint is 1,312,973,992 bytes.
  The public file has not been downloaded or compared ([`CHECKPOINTS.md`](CHECKPOINTS.md)).
  The archived file is the reference.
- `noisy_new` cannot be regenerated (noise offsets unrecorded). `enhanced_new` cannot be
  regenerated bit-exactly (unseeded sampler). KWS training is unseeded.
- GPU runs with `kws-main-py39` are not verified.
- `<archive>/manifests/archive_checks.sh` works only against `CHECKPOINTS.md` at `700e943`;
  use `archive_checks_v2.py`.
- Every evaluation leaves about 0.6 GB per enhanced mode in `/tmp/kws_metrics_*` (local disk, not cleaned up).
- Local files ignored by Git with absolute paths: `code/kws/denoiser/noise_onthefly_dataset.py`
  (dead paths), `code/kws/avital_cheatsheet.txt` (personal notes).
  `build_reverb_dataset.py` now reads the archive; it still writes to `<repo>/data/reverb`.

## 8. Backup status and cleanup plan

**Off-site backup: BLOCKED.** The home directory (`psa:/ifs/a/home`) and the archive
(`psa:/ifs/a/private/gannot-lab`) are on the **same Isilon cluster**, which keeps snapshots
every 12 h for 7 days. No approved independent destination exists. Mounted alternatives
are not usable without approval: `/cortex` and `/home/lab` belong to another lab;
`/home/fast` is a different Isilon (`psf`) with no user directory; the local disk is node-only.

Deletion candidates (home allocated sizes measured 2026-11-08; `<repo>` = `/home/dsi/skopavi/Project/kws_project`).
"Copy" means the verified archive copy; no candidate has an independent backup.

| Cat. | Path | Allocated | Verified copy | Still depended on | Risk |
|---|---|---|---|---|---|
| A | (none: no independent backup exists) | — | — | — | — |
| B | `<repo>/experiments/sgmse_logs/fcj02u93-kws_exp_9/kws_sgmse_trained.ckpt` | 1,673 MiB | Byte-identical to the epoch-176 file at home **and** in the archive | No | Low (redundant duplicate) |
| B | `<repo>/code/sgmse/checkpoints/train_vb_29nqe0uh_epoch=115.ckpt` | 1,256 MiB | Archive (same Isilon); also re-downloadable from the SGMSE authors (`gdown`) | No | Low–medium |
| B | `<repo>/data/raw/data_new/` | 5,905 MiB | Archive (same Isilon); public original on Kaggle (not re-verified) | No | Medium |
| B | `<repo>/data/train_sgmse/` | 1,051 MiB | Archive (same Isilon) | No | Medium |
| B | `/home/dsi/skopavi/kws_phase4b2_scratch/` | 19 MiB | Archive (`experiments_archive/phase4b_verification`) | No | Low |
| B (unique) | `<repo>/data/noisy_new/` | 6,576 MiB | Archive only (same Isilon) | No (Kaggle links repointed) | **High: not regenerable** |
| B (unique) | `<repo>/data/enhanced_new/` | 712 MiB | Archive only (same Isilon) | No | **High: not bit-exactly regenerable** |
| B (unique) | `<repo>/experiments/sgmse_logs/fcj02u93-kws_exp_9/epoch=176-valid_loss=558.3868.ckpt` | 1,674 MiB | Archive only (Hugging Face file unresolved) | No | **High: only full training checkpoint** |
| B | `<repo>/data/{noisy_new_metadata,enhanced_pretrainted_new_metadata,enhanced_trained_ep176_new_metadata}.csv` | 7 MiB | Archive (same Isilon) | No | Medium (tiny; keep with `noisy_new`) |
| C | `<repo>/outputs/new_runs/` | 64 MiB | Archive | **Yes**: `output_dir`, tests, demo CSVs, docs | Keep |
| C | `<repo>/outputs/dscnn_full_dataset_best/` | 1 MiB | Archive | Docs (historical path) | Keep (negligible) |
| D | `<repo>/code/sgmse/sgmse_env/` | 9,117 MiB | Rebuildable (`kws-main-py39` verified on CPU) | No process; last change 2026-07-12 | Medium: run a GPU check of `kws-main-py39` first |
| D | `<repo>/kws_env/` | 963 MiB | Not needed (broken outside Docker) | No | Low |
| D | `<repo>/demo_env/` | 1,349 MiB | Rebuildable | **Yes**: Gradio demo, demo tests | Keep |
| E | `/home/dsi/skopavi/kws_phase5d_scratch/` | 195 MiB (141 MB of `strace` logs) | Results summarized in `VERIFICATION.md` | No | Low |
| E | `/home/dsi/skopavi/kws_phase5e1_scratch/`, `kws_phase5e1b_scratch/`, `kws_closure_scratch/` | 37 MiB | Results in this file | Rollback copies (`new_config.yaml`, `build_reverb_dataset.py`, link records) | Low; keep until push |
| E | `/home/dsi/skopavi/kaggle_symlink_test/` | 1 MiB | — | No | Low |
| E | `/tmp/kws_metrics_*` (8 directories) | 3.6 GB on the **local disk** | — | No | Low; **no home space recovered** |

Projected home-space recovery (allocated):

| Scope | MiB |
|---|---|
| B, lower risk (duplicate, re-downloadable checkpoint, raw speech data, SGMSE training pairs, 4B scratch) | 9,904 |
| B, unique data (`noisy_new`, `enhanced_new`, epoch-176 checkpoint, metadata): **wait for an off-site backup** | 8,969 |
| D (`sgmse_env`, `kws_env`) | 10,080 |
| E (home scratch) | 233 |
| **Total if everything above is approved** | **29,186 (≈ 28.5 GiB)** |

Recommended order once approved: (1) the duplicate checkpoint and `kws_env`; (2) `sgmse_env`
after a GPU check; (3) B lower-risk items; (4) unique data only after an off-site
backup is verified. Immediately before any deletion, re-run `manifests/reverify_all.sh`
(home snapshots last only 7 days).
