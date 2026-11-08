# System and environment record

Captured on 2026-11-08, read-only, on the machine that holds the final project
state (`/home/dsi/skopavi/Project/kws_project`). Every value below was observed
directly; nothing is inferred.

## Raw snapshots in this directory

| File | Produced by | SHA-256 |
|---|---|---|
| `sgmse_env.pip-freeze.2026-11-08.txt` | `code/sgmse/sgmse_env/bin/python -m pip freeze` (141 packages) | `a7b6e31cec3a7a3a4c0bd70dab9c137dceaf3feabac2486ac67bc49124bcfd81` |
| `demo_env.pip-freeze.2026-11-08.txt` | `demo_env/bin/python -m pip freeze` (78 packages) | `343b5fb01539caee21e3cac4ee3d674ac317358730275ea39e392b6655184304` |
| `kws-main-py39.pip-freeze.2026-11-08.txt` | `/home/dsi/skopavi/envs/kws-main-py39/bin/python -m pip freeze` (103 packages), the verified clean rebuild of the main environment | `0c2d98404bc25355b62980f1d325cdf71e980039b7cf35543c94d4c2260c8676` |

These files are unedited `pip freeze` output. Do not change them; record a new
dated snapshot instead.

## Host

| Item | Value |
|---|---|
| OS | Rocky Linux 9.8 (Blue Onyx), kernel 5.14.0-687.61.1.el9_8.0.1.x86_64 |
| CPU cores | 64 |
| GPUs | 2 × NVIDIA GeForce RTX 2080 Ti, 11 GB each (compute capability 7.5) |
| NVIDIA driver | 610.43.02 |
| System CUDA toolkit (`/usr/local/cuda`, nvcc) | 12.3 (V12.3.103). PyTorch does not use it: it ships its own CUDA 12.8 runtime wheels |
| Compiler | gcc 11.5.0 |
| System Python (`/usr/bin/python3`) | 3.9.25 |

## Environments

### Main environment: `code/sgmse/sgmse_env` (SGMSE + KWS)

| Item | Value |
|---|---|
| Type | `venv`, `include-system-site-packages = false` |
| Python | 3.9.25 (`bin/python` → `/usr/bin/python`) |
| PyTorch | 2.8.0+cu128 (`torch.version.cuda` = 12.8), torchaudio 2.8.0, torchvision 0.23.0 |
| CUDA runtime wheels | `nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cudnn-cu12==9.10.2.21` (full list in the freeze) |
| PyTorch Lightning | 2.6.0 |
| Size on disk | 9.0 GB |

Import check on 2026-11-08 (CPU only, `python -I`, no GPU): every KWS module
(`code/kws/src`) and `sgmse.model.ScoreModel` imports, **except
`analysis_utils`, which fails with `ModuleNotFoundError: tabulate`**. Both KWS
entry points (`run_training.py`, `run_evaluation_main.py`) import
`analysis_utils`, so this environment cannot run them unless `tabulate` is added.

### Verified clean rebuild: `/home/dsi/skopavi/envs/kws-main-py39` (2026-11-08)

| Item | Value |
|---|---|
| Type | `venv` from `/usr/bin/python3.9` (3.9.25), outside the repository |
| Installed from | `requirements/requirements-main.txt`, constrained by `sgmse_env.pip-freeze.2026-11-08.txt`, with `pip==26.0.1` and `--no-cache-dir` |
| Packages | 103: the 102 shared with `sgmse_env` at identical versions, plus `tabulate 0.9.0`; 39 unused packages omitted |
| PyTorch | 2.8.0+cu128 (same wheels as `sgmse_env`) |
| Size on disk | 7.0 GiB of files (11 GB allocated on the NFS home filesystem) |

All tests pass in this environment, and it reproduces the historical DS-CNN
predictions and the 30-class evaluation results. See
[`../VERIFICATION.md`](../VERIFICATION.md). `sgmse_env` has not been removed.

### Demo environment: `demo_env` (optional)

| Item | Value |
|---|---|
| Type | `venv`, `include-system-site-packages = false` |
| Python | 3.9.25 |
| Key packages | gradio 4.44.1, numpy 2.0.2, librosa 0.11.0, pandas 2.3.3. No torch |
| Size on disk | 1.4 GB |

Import check on 2026-11-08: `gradio_demo`, `export_demo_assets` and
`kws.demo.demo_utils` all import.

### Legacy: `kws_env` (not usable, not needed)

- Created inside Docker (`pyvenv.cfg`: `command = /usr/bin/python3 -m venv /kws_project/kws_env`), Python 3.12.3.
- On the host, `bin/python3` → `/usr/bin/python3` is 3.9, so the environment is broken outside the container.
- It has no torch, librosa or scikit-learn, so it cannot run any part of the pipeline.

## Installation provenance (what is and is not known)

- **sgmse_env** was created at `.../kws_project/sgmse_repo/sgmse_env` and later
  moved. Console scripts in `bin/` (for example `pip`) still have that stale
  shebang. Use `bin/python -m pip`. No install log or command history survives.
- The installed versions differ from both `code/sgmse/requirements.txt` (unpinned)
  and `code/sgmse/requirements_version.txt` (the upstream authors' pins: torch
  2.2.0, Lightning 2.1.4). The environment was therefore resolved by `pip` at
  install time, not installed from either file.
- **`tabulate`** exists only in the user site (`~/.local/lib/python3.9/site-packages`,
  version 0.9.0), which the venv cannot see. `docker_skopavi/requirements.txt`
  (ignored by Git) also pins `tabulate==0.9.0`. Which interpreter produced the
  final KWS reports is not recorded.
- **Docker** (`docker_skopavi/`, ignored by Git): based on `nvcr.io/nvidia/pytorch:25.03-py3`.
  `docker-compose.yml` builds from `~/docker/`, a different copy of the Dockerfile.
  It is not part of the documented setup.

## Source revisions

| Item | Revision |
|---|---|
| Repository | `https://github.com/MayaVB/kws_project.git`, branch `main` |
| HEAD at capture time | `2c57c54bb01621ceb88e1e0fdb6005df3f3bd6ce` ("Update SGMSE training and checkpoint documentation") |
| Working tree at capture time | 69 changed files not yet committed (demo assets, `index.html`, `export_demo_assets.py`, `.gitignore`, the seed fix in `build_noise_dataset.py`) and one stash (`stash@{0}`). See `../EXPERIMENTS.md` for the revisions in place when each experiment ran |
| SGMSE code | Vendored copy of `sp-uhh/sgmse` under `code/sgmse/`, with local changes (see the notes in `../CHECKPOINTS.md`) |
