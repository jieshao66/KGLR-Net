# Reproducibility Protocol

This release implements the limited-annotation fully supervised protocol used
by KGLR-Net. Training consumes only the selected target-domain images and their
pixel-level masks. It does not consume target-domain unlabeled images, manual
prompts, or held-out samples.

## Experimental matrix

The complete matrix contains four datasets, three annotation budgets, and five
independent fixed resamplings:

- Datasets: `ISIC2018`, `HAM10000`, `Kvasir-SEG`, and `CVC-ClinicDB`.
- Budgets: 5, 10, and 20 annotated training images.
- Folds: 1 through 5.

The version-controlled manifests are stored at:

```text
splits/<dataset>/<shot>_shot/fold_<fold>_labeled.txt
```

Each manifest contains exactly `shot` unique `.npz` basenames. The split seed
is `42 + 1000 * shot + fold`. The test partition is fixed and is not resampled.

## Data layout

The public datasets are not redistributed. After downloading them from their
official sources and running the provided preprocessing utility, arrange the
preprocessed files as follows:

```text
<data-root>/
  <dataset>/
    train/npz_data/*.npz
    test/npz_data/*.npz
```

Each `.npz` archive contains the arrays `image`, `mask`, `w_fg`, and `w_bg`.
Archives are loaded with pickling disabled. The data loader checks the selected
names against the fixed manifests and rejects any overlap with the held-out
partition.

## HAM10000 partition

For this study, HAM10000 is partitioned into an 8,012-image training pool and
a 2,003-image held-out test set. Fixed-shot samples are drawn only from the
training pool. The manifests are included in `splits/HAM10000` and can be
regenerated after preparing the data with:

```bash
python tools/generate_splits.py --data-root /path/to/preprocessed-data --overwrite
```

## Checkpoint selection

Training does not build an evaluation loader and does not evaluate held-out
samples. Checkpoint selection is determined entirely by the training schedule
or annotated-training statistics.

The default policy is `last`, which selects the final training epoch. The only
alternative is `train_loss`, which selects the minimum loss measured on the
annotated training split. The files are named:

```text
gpps_last.pth
gpps_best_train_loss.pth
lprs_last.pth
lprs_best_train_loss.pth
```

The selected files are copied to `gpps.pth` and `lprs.pth`. With the default
policy, these aliases are exact copies of the corresponding `*_last.pth`
files. Evaluation is a separate, explicit step performed only after the
checkpoints have been fixed.

## One experiment

Download the official MedSAM ViT-B checkpoint separately, then run:

```bash
python train.py \
  --data-root /path/to/preprocessed-data \
  --dataset ISIC2018 \
  --shot 10 \
  --fold 1 \
  --sam-checkpoint /path/to/medsam_vit_b.pth
```

Training writes the selected checkpoints to:

```text
results/<dataset>/<shot>_shot/fold_<fold>/weights/gpps.pth
results/<dataset>/<shot>_shot/fold_<fold>/weights/lprs.pth
```

Use `--checkpoint-selection train_loss` only when training-loss selection is
required. Use `--resume` to reuse a completed stage in the same result
directory.

## Complete matrix

Run all 60 combinations sequentially with:

```bash
python scripts/run_all_experiments.py \
  --data-root /path/to/preprocessed-data \
  --sam-checkpoint /path/to/medsam_vit_b.pth
```

Add `--dry-run` to inspect every command without starting training. Add
`--resume` to skip runs that already contain both selected checkpoint aliases
and to reuse a completed stage in a partially finished run.

## Determinism

The training command seeds Python, NumPy, PyTorch, CUDA, data-loader workers,
and shuffling generators. Deterministic cuDNN and PyTorch algorithms are used
by default, with warnings for operations that lack a deterministic
implementation. Exact floating-point identity can still depend on the GPU,
CUDA, cuDNN, PyTorch, and driver versions. Preserve the environment metadata
and all JSON training histories when reporting a run.
