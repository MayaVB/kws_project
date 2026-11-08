# Checkpoints and model weights

Verified on 2026-11-08. Hashes were computed with `sha256sum`. Metadata was read
on CPU with `torch.load(..., mmap=True)` from the checkpoint's stored
`hyper_parameters`, `epoch` and `global_step`.

## SGMSE enhancement models

### Project-trained SGMSE+ (epoch 176): used for every "enh_trained" result

| Item | Value |
|---|---|
| **Preferred permanent local copy** | `experiments/sgmse_logs/fcj02u93-kws_exp_9/epoch=176-valid_loss=558.3868.ckpt` |
| Size | 1,312,973,992 bytes |
| SHA-256 | `90e19444b5e6c7d9a07cc49543804a02f57717d8d971e45832ee531b447f7af5` |
| Stored metadata | `epoch` 176 (0-indexed, i.e. the 177th epoch), `global_step` 272226, written by PyTorch Lightning 2.6.0 |
| Training run | W&B run id `fcj02u93`, name `kws_exp_9` (from the directory name; the local W&B files are dangling symlinks) |
| Training data | `data/train_sgmse` (`base_dir` in hparams): 6150 train and 3409 valid clean/noisy pairs |
| Batching / optimizer (historical) | batch size 1, `accumulate_grad_batches` 4 (effective batch 4), Adam `lr` 1e-4, `weight_decay` 0 (see below) |
| Key hparams | `backbone ncsnpp`, `sde ouve`, `loss_type score_matching`, `lr 1e-4`, `n_fft 510`, `hop_length 128`, `num_frames 256`, `spec_factor 0.15`, `spec_abs_exponent 0.5`, `sigma_min 0.05`, `sigma_max 0.5`, `theta 1.5`, `t_eps 0.03`, `ema_decay 0.999`, `num_eval_files 5`, `sr 16000` |
| Selection | Checkpoint kept by a `ModelCheckpoint(monitor='valid_loss', mode='min')` callback (the filename embeds `valid_loss=558.3868`) |
| Enhanced output | `data/enhanced_new/trained_ep176/` (6473 wav files, generated 2026-05-21) |

**Identical copy:** `experiments/sgmse_logs/fcj02u93-kws_exp_9/kws_sgmse_trained.ckpt`
(created 2026-09-06). `cmp` on 2026-11-08 found it **byte-identical** to the
epoch-176 file. It is a separate file (different inode), not a link, so it uses
another 1.25 GB. Its filename matches the checkpoint the main README links on
Hugging Face:
[`avitalskop/kws-sgmse/kws_sgmse_trained.ckpt`](https://huggingface.co/avitalskop/kws-sgmse/blob/main/kws_sgmse_trained.ckpt).
**Not yet verified:** the Hugging Face copy was not downloaded or hashed during
this audit. It is only expected to match the SHA-256 above.

**Historical training settings: batch size 1, gradient accumulation 4, weight decay 0.**
*(Corrected 2026-11-08. An earlier version of this file wrongly concluded batch 2 × accumulation 2.)*

These values come from the checkpoint's own state:

| Evidence in the checkpoint | Value | Conclusion |
|---|---|---|
| `hyper_parameters.batch_size` | 1 | `ScoreModel` saves the CLI DataModule arguments as hparams and builds its DataLoader from them (`sgmse/model.py`, `save_hyperparameters` + `data_module_cls(**kwargs)`), so this is the batch size the DataLoader actually used |
| `loops…batch_progress.total.processed` | 1,088,550 = 177 × 6150 | 6150 training batches per epoch for 6150 pairs → **batch size 1** in every epoch |
| `loops…optimizer.step.total.completed` (= `global_step`) | 272,226 = 177 × 1538 | ceil(6150 / 4) = 1538 optimizer steps per epoch → **`accumulate_grad_batches` 4** (effective batch 4) |
| `loops…val_loop.batch_progress` | 3409 per epoch | validation also at batch size 1 (3409 pairs) |
| `optimizer_states[0].param_groups[0]` | `lr` 1e-4, `weight_decay` 0, Adam betas (0.9, 0.999) | **weight decay 0** (the `ScoreModel` default; not set on the command line) |

The `train.py` in use was the version from commit `e95959a` (2026-05-14). The
checkpoint's callback set matches that version exactly, including the
`'{epoch}-{valid_loss:.4f}'` filename and no ESTOI callback; the current `train.py`
names checkpoints differently.

Historical command consistent with this evidence. The batch, accumulation and
weight-decay values are confirmed; the other values come from the stored hparams
(`num_eval_files` 5, `backbone` ncsnpp, `sde` ouve), or from the main README, not from a log
(`--num_workers 2` is in hparams; `--max_epochs 200` is not recorded):

```bash
python train.py --base_dir <data/train_sgmse> --batch_size 1 --accumulate_grad_batches 4 \
    --backbone ncsnpp --num_eval_files 5 --num_workers 2 ...
```

The README's command (`--batch_size 2 --accumulate_grad_batches 2`) gives the same
effective batch of 4, but it is **not** what produced this checkpoint. The two
are not guaranteed to be numerically equivalent: each forward pass sees a
different batch shape (1 versus 2 examples), and the shuffling order differs.
Whether any layer depends on batch composition was not checked.

### Pretrained SGMSE+ (VoiceBank-DEMAND): "enh_baseline" and fine-tuning initialization

| Item | Value |
|---|---|
| Local path | `code/sgmse/checkpoints/train_vb_29nqe0uh_epoch=115.ckpt` |
| Size | 1,312,806,643 bytes |
| SHA-256 | `e3875747b5646092d5c556bae68e5af639e2c1f45f009c669f379cd4d415cbd8` |
| Stored metadata | `epoch` 116, `global_step` 41992, PyTorch Lightning 1.5.10, `base_dir /data/richter/VoiceBank/` |
| Source | Official SGMSE authors: `gdown 1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ` |
| Enhanced output | `data/enhanced_new/baseline/` (6473 wav files, generated 2026-05-10) |

### Enhancement (inference) settings used for both enhanced datasets

`build_enhanced_dataset.py` (in commit ff92488, 2026-05-20 and in HEAD) calls
`code/sgmse/enhancement.py` with only `--test_dir`, `--enhanced_dir`, `--ckpt`
and `--device cuda`. All sampler settings are therefore the script defaults,
which are unchanged between ff92488 and HEAD:

`sampler_type pc`, predictor `reverse_diffusion`, `corrector ald`,
`corrector_steps 1`, `snr 0.5`, `N 30`, `t_eps 0.03`.

The reverse diffusion is stochastic and `enhancement.py` sets no random seed.
Re-running enhancement gives statistically equivalent but not bit-identical audio.

### Local modifications to the vendored SGMSE code

Recorded in the personal notes and visible in the code: `calc_metrics.py` trims
both signals to `min(len(x), len(y))`, because enhanced outputs can be shorter
than 16000 samples. `sgmse/model.py` wraps PESQ in `try` (it fails on very short
audio) and counts ESTOI successes. These explain the "N=…/…" success counts in
the metric reports.

### Other SGMSE checkpoints referenced in results but not present locally

Run `2026-05-25_23-39-48_all_trained_modes` also reports `enh_trained_ep149`,
`enh_trained_ep149_ft`, `enh_trained_ep163` and `enh_trained_ep166`. Neither
those checkpoints nor their enhanced datasets exist on disk any more. They
belong to the checkpoint-selection history, not to the final results.

## DS-CNN keyword-spotting classifiers (all small; none in Git or on Hugging Face)

| Experiment | Location | best_model.pt SHA-256 | scaler.pkl SHA-256 | label_encoder.pkl SHA-256 |
|---|---|---|---|---|
| 30-class (from run 2026-05-26 `all_folders`) | `outputs/dscnn_full_dataset_best/` (byte-identical to `outputs/new_runs/2026-05-26_18-01-43_all_folders/models/`) | `37f9f425fe403ca6e395077cda1043c599cb71f70fdfc5d5c90e985d0a84ab00` | `e582839d99e8d5cef86ebf473beadaf6d9d47d6676980b1ec2dcc0374e8702ac` | `9a25b3d08613988848fcf451cf48048dee519825ce9fcace78bba1daae4ecff6` |
| 6-class A: bed, bird, dog, go, happy, house | `outputs/new_runs/2026-07-07_11-36-05/models/` | `506c95d46c4204616ab132d84c0713136dd63bcebebb49ab709782b33920f890` | `897b3c4e36622a1f0f84ba418b498ea5ab01d101f8d5f75c9e826615f1910239` | `5bb856dad61660c53ea82e15a7614745e0d52c58a1a14c2ba0f10038f5558063` |
| 6-class B: go, no, off, on, stop, yes | `outputs/new_runs/2026-06-08_12-14-57_presentation/models/` | `2345381266fb5987603f47cec20813d1863c958b2341d16a58f268642b528100` | `9b192493a995a7033247d864d61dbb066990de7e9d7372d9714241bd17bc64b7` | `9ab8989b27bf0472413e0a4f29f9be31a42257599efc5a000f6d837faacf22bf` |

The other `best_model.pt` files under `outputs/` belong to exploratory runs.

## Needed for what

| Artifact | Final inference / evaluation | Full re-training from scratch |
|---|---|---|
| epoch-176 SGMSE checkpoint | Needed to regenerate `enhanced_new/trained_ep176` | Can be retrained (see README) but not bit-exactly |
| Pretrained VB-DMD checkpoint | Needed to regenerate `enhanced_new/baseline` | Needed (re-downloadable) |
| DS-CNN artifacts | Needed to re-evaluate without retraining | Can be retrained with `reproducibility/configs/` |
| `kws_sgmse_trained.ckpt` local copy | Redundant (identical to epoch 176) | Redundant |
