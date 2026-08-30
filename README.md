# KGLR-Net

Official implementation of **KGLR-Net: Knowledge-Guided Global-to-Local
Refinement for Limited-Annotation Dermoscopic Lesion and Colonoscopic Polyp
Segmentation**.

KGLR-Net studies fully supervised segmentation with only a small number of
pixel-labeled target-domain images. It does not use target-domain unlabeled
images or manual prompts. The method has two stages:

- **Global Prior Prompting Segmenter (GPPS):** extracts modality-specific,
  computable medical imaging priors and uses them to prompt a SAM-based medical
  foundation model automatically.
- **Local Prior Refinement Segmenter (LPRS):** uses the global coarse mask as a
  localization constraint and the corresponding priors as local conditioning
  information to correct residual region and boundary errors.

The released training protocol covers ISIC2018, HAM10000, Kvasir-SEG, and
CVC-ClinicDB at 5, 10, and 20 annotated images over five independent
resamplings.

> **Experimental protocol.** The version-controlled manifests define disjoint
> training and test partitions. Training uses only the selected annotated
> samples and does not construct a validation or test loader. Checkpoints are
> fixed by the final training epoch by default, and held-out test sets are used
> only for final evaluation. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repository layout

```text
.
├── kglr_net/
│   ├── data.py             # fully supervised dataset loader
│   ├── metrics.py          # Dice, IoU, recall, precision, and HD95
│   ├── models.py           # priors, GPPS, LPRS, and sliding-window inference
│   ├── pipeline.py         # end-to-end GPPS -> LPRS inference
│   ├── runtime.py          # safe checkpoint and reproducibility utilities
│   └── training.py         # annotated-only two-stage training
├── splits/                 # fixed train/test manifests and 60 labeled subsets
├── tools/                  # preprocessing, split generation, and setup checks
├── scripts/                # complete-matrix training/evaluation and aggregation
├── train.py
├── evaluate.py
├── cross_dataset_eval.py
└── infer.py
```

Datasets, model weights, generated metric files, logs, and prediction masks are
intentionally excluded from version control.

## Installation

Python 3.10 is recommended. A CUDA-capable GPU is strongly recommended because
GPPS contains a ViT-B SAM backbone.

Create the supplied Conda environment:

```bash
conda env create -f environment.yml
conda activate kglr-net
pip install -e .
```

Alternatively, first install the PyTorch build matching your CUDA runtime and
then run:

```bash
pip install -r requirements.txt
pip install -e .
```

The dependency ranges define the supported environment for this release.
Record `python`, PyTorch, CUDA, cuDNN, GPU, and driver versions with every
experiment.

## External MedSAM initialization

Download the official MedSAM ViT-B checkpoint from the
[MedSAM repository](https://github.com/bowang-lab/MedSAM). Do not commit the
checkpoint to this repository. Pass its local path through
`--sam-checkpoint` in every command.

Segment Anything is installed as an external dependency and is not copied into
this repository. Licensing and attribution information is provided in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Dataset preparation

This repository does not redistribute the four public datasets. Follow
[DATASETS.md](DATASETS.md) to:

1. download and arrange images and masks;
2. reproduce the fixed 2,594/1,000, 8,012/2,003, 800/100, and 550/62
   train/test partitions;
3. preprocess images and masks to safe compressed `.npz` files at 224 x 224;
4. verify all partition manifests and 60 fixed-shot lists.

Before training, run the complete integrity check:

```bash
python tools/check_setup.py --data-root Dataset --inspect-arrays
```

The check fails on missing files, unexpected partition membership, duplicate
identifiers, incorrect shapes or fields, and any training-test overlap.

## Train one experiment

The following command trains all ten selected annotations and does not create a
test loader:

```bash
python train.py \
  --data-root Dataset \
  --dataset ISIC2018 \
  --shot 10 \
  --fold 1 \
  --sam-checkpoint checkpoints/medsam_vit_b.pth \
  --device cuda:0
```

The paper's full local-reconstruction configuration is the default:

- central valid patch: 56 x 56;
- one-sided halo: 8 pixels;
- sliding-window overlap: 0.5;
- Gaussian-weighted reconstruction.

The default checkpoint policy is the fixed final epoch (`last`). The optional
`--checkpoint-selection train_loss` policy uses only annotated-training loss.
Neither policy reads a validation or test set. Selected weights are written to:

```text
results/<dataset>/<shot>_shot/fold_<fold>/weights/gpps.pth
results/<dataset>/<shot>_shot/fold_<fold>/weights/lprs.pth
```

## Train and evaluate the complete matrix

Inspect all 60 training commands first:

```bash
python scripts/run_all_experiments.py \
  --data-root Dataset \
  --sam-checkpoint checkpoints/medsam_vit_b.pth \
  --dry-run
```

Remove `--dry-run` to execute them sequentially. Use `--resume` to skip
completed runs and reuse a completed stage.

After checkpoints have been fixed, evaluate the held-out test sets in a
separate step:

```bash
python scripts/evaluate_all.py \
  --data-root Dataset \
  --sam-checkpoint checkpoints/medsam_vit_b.pth \
  --skip-existing
```

Aggregate the five repeat-level means into the paper's mean and standard
deviation format:

```bash
python scripts/summarize_repeats.py \
  --result-root results \
  --dataset ISIC2018 \
  --shot 10 \
  --stage kglr_net
```

## Evaluate one experiment

```bash
python evaluate.py \
  --data-root Dataset \
  --dataset ISIC2018 \
  --sam-checkpoint checkpoints/medsam_vit_b.pth \
  --gpps-checkpoint results/ISIC2018/10_shot/fold_1/weights/gpps.pth \
  --lprs-checkpoint results/ISIC2018/10_shot/fold_1/weights/lprs.pth \
  --output-dir results/ISIC2018/10_shot/fold_1/metrics
```

The command writes per-image CSV files and a JSON summary for GPPS and the full
KGLR-Net. Metrics are Dice, IoU, recall, precision, and HD95 in pixels.

## Bidirectional cross-dataset evaluation

Cross-dataset testing uses a source model directly on the other dataset of the
same modality, without target-domain optimization or target-domain model
selection. The paper reports the 10-shot directions ISIC2018 to/from
HAM10000 and Kvasir-SEG to/from CVC-ClinicDB.

```bash
python cross_dataset_eval.py \
  --data-root Dataset \
  --source-dataset ISIC2018 \
  --target-dataset HAM10000 \
  --sam-checkpoint checkpoints/medsam_vit_b.pth \
  --gpps-checkpoint results/ISIC2018/10_shot/fold_1/weights/gpps.pth \
  --lprs-checkpoint results/ISIC2018/10_shot/fold_1/weights/lprs.pth \
  --output-dir results/ISIC2018/10_shot/fold_1/cross/ISIC2018_to_HAM10000
```

## Inference on new images

Choose a dataset name with the same modality as the input so that the correct
prior branch is used:

```bash
python infer.py \
  --input path/to/image_or_directory \
  --output-dir outputs \
  --dataset ISIC2018 \
  --sam-checkpoint checkpoints/medsam_vit_b.pth \
  --gpps-checkpoint path/to/gpps.pth \
  --lprs-checkpoint path/to/lprs.pth
```

Binary masks are restored to each input image's original resolution. Add
`--save-probabilities` to save the 224 x 224 probability maps locally; these
generated arrays are ignored by Git.

## Tests

Install development dependencies and run:

```bash
pip install -r requirements-dev.txt
pytest
python -m compileall -q .
```

The release-integrity tests verify fixed manifest counts, zero test overlap,
absence of test-loader construction in training, absence of private machine
paths, and absence of datasets or model binaries from the repository.

## Citation

Citation metadata is available in [CITATION.cff](CITATION.cff). The final
journal citation will be added after publication.

```bibtex
@article{wang2026kglrnet,
  title   = {KGLR-Net: Knowledge-Guided Global-to-Local Refinement for
             Limited-Annotation Dermoscopic Lesion and Colonoscopic Polyp
             Segmentation},
  author  = {Wang, Xijie and Jiang, Yun and Sun, Tao},
  year    = {2026}
}
```

## Funding

This work is supported by the National Natural Science Foundation of China
(Grant No. 62561049) and the Industrial Support Program Project of the Gansu
Provincial Department of Education (Grant No. 2025CYZC-008).

## License

The authors' code is released under the Apache License 2.0. External software,
pretrained weights, and datasets remain under their respective licenses and
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[DATASETS.md](DATASETS.md).
