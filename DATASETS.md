# Dataset preparation

This repository does **not** redistribute images, masks, or preprocessed arrays.
Download each dataset from its official source, review its license and terms of
use, and keep all data outside version control.

| Dataset | Fixed training pool | Held-out test set | Official source |
|---|---:|---:|---|
| ISIC2018 | 2,594 | 1,000 | [ISIC Challenge 2018](https://challenge.isic-archive.com/data/#2018) |
| HAM10000 | 8,012 | 2,003 | [Images and metadata (DOI: 10.7910/DVN/DBW86T)](https://doi.org/10.7910/DVN/DBW86T); [lesion masks](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations) |
| Kvasir-SEG | 800 | 100 | [Simula Research Laboratory](https://datasets.simula.no/kvasir-seg/) |
| CVC-ClinicDB | 550 | 62 | [CVC-ClinicDB challenge page](https://polyp.grand-challenge.org/CVCClinicDB/) |

The version-controlled manifests under `splits/<dataset>/` define the paper's
partitions by basename only:

- `train_pool.txt` lists every image eligible for fixed-shot sampling.
- `test.txt` lists the fixed, held-out test partition.
- `<shot>_shot/fold_<fold>_labeled.txt` lists the labeled training samples for
  5-, 10-, and 20-shot experiments and independent resamplings 1--5.

The manifests contain identifiers only; they do not contain dataset content.

## HAM10000 partition

For the experiments in this study, HAM10000 is partitioned into an 8,012-image
training pool and a 2,003-image held-out test set. The two sets are defined by
`HAM10000/train_pool.txt` and `HAM10000/test.txt`, respectively. Fixed-shot
samples are drawn only from the training pool. The held-out set is used only
for final evaluation, and no images are removed from a 5-, 10-, or 20-shot
labeled subset to construct a validation set.

## 1. Arrange the raw files

Use the stems in `train_pool.txt` and `test.txt` to place downloaded images and
masks into fixed partitions. Image and mask extensions may differ, but stems
must match. The preprocessor also recognizes the conventional `_mask` and
`_segmentation` mask suffixes.

```text
raw_data/
  ISIC2018/
    train/images/
    train/masks/
    test/images/
    test/masks/
  HAM10000/
    train/images/
    train/masks/
    test/images/
    test/masks/
  Kvasir-SEG/
    train/images/
    train/masks/
    test/images/
    test/masks/
  CVC-ClinicDB/
    train/images/
    train/masks/
    test/images/
    test/masks/
```

For every dataset, the train and test directories must match `train_pool.txt`
and `test.txt`, respectively, and must be disjoint.

## 2. Preprocess each partition

`tools/preprocess.py` resizes images and masks to 224 x 224. It writes one
compressed NumPy archive (`.npz`) per sample with four named arrays:

- `image`: RGB `uint8`, shape `[224, 224, 3]`;
- `mask`: binary `uint8`, shape `[224, 224]`;
- `w_fg`: foreground-oriented spatial weight, `float32`;
- `w_bg`: background-oriented spatial weight, `float32`.

Run the tool once for each dataset and partition. For example:

```bash
python tools/preprocess.py \
  --dataset ISIC2018 \
  --split train \
  --image-dir raw_data/ISIC2018/train/images \
  --mask-dir raw_data/ISIC2018/train/masks \
  --output-root Dataset

python tools/preprocess.py \
  --dataset ISIC2018 \
  --split test \
  --image-dir raw_data/ISIC2018/test/images \
  --mask-dir raw_data/ISIC2018/test/masks \
  --output-root Dataset
```

Repeat with `HAM10000`, `Kvasir-SEG`, and `CVC-ClinicDB`. The resulting layout
is:

```text
Dataset/
  <dataset>/
    train/npz_data/*.npz
    test/npz_data/*.npz
```

The tool refuses to overwrite an existing preprocessed partition unless
`--overwrite` is given.

## 3. Verify or regenerate the fixed-shot lists

The repository already includes all 60 fixed-shot files. To reproduce them from
the fixed manifests and locally prepared data, run:

```bash
python tools/generate_splits.py --data-root Dataset --overwrite
```

For each shot budget and fold, the random seed is exactly:

```text
42 + 1000 * shot + fold
```

Only labeled filenames are generated. The pipeline does not create or consume
unlabeled target-domain lists.

## 4. Validate the complete setup

The fast check verifies the partition counts, on-disk names, all 60 fixed-shot
files, per-file line counts, and zero overlap with every test manifest:

```bash
python tools/check_setup.py --data-root Dataset
```

To additionally load every array and validate keys, 224 x 224 shapes, binary
masks, finite values, and spatial-weight ranges, use:

```bash
python tools/check_setup.py --data-root Dataset --inspect-arrays
```

Each sample is stored with `numpy.savez_compressed`, and the loader always uses
`allow_pickle=False`. Raw data, preprocessed arrays, checkpoints, cached
predictions, and experiment
outputs must remain outside Git. The repository `.gitignore` excludes the
standard local directories used for these artifacts.
