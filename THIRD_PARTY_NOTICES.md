# Third-party software and model assets

KGLR-Net does not redistribute datasets, Segment Anything source code, or
pretrained model weights. Users must obtain each external asset from its
official source and comply with its license and terms.

## Segment Anything

The GPPS stage imports the `segment_anything` Python package at runtime.
Segment Anything is Copyright Meta Platforms, Inc. and affiliates and is
distributed under the Apache License 2.0.

- Repository: <https://github.com/facebookresearch/segment-anything>
- License: <https://github.com/facebookresearch/segment-anything/blob/main/LICENSE>
- Paper: Kirillov et al., *Segment Anything*, ICCV 2023.

## MedSAM

GPPS is initialized from the MedSAM ViT-B checkpoint. The checkpoint is not
part of this repository. Download it only from the official MedSAM project and
follow that project's license and citation instructions.

- Repository: <https://github.com/bowang-lab/MedSAM>
- License: <https://github.com/bowang-lab/MedSAM/blob/main/LICENSE>
- Paper: Ma et al., *Segment Anything in Medical Images*, Nature
  Communications, 2024.

## Methods implemented from the literature

The repository implements or uses ideas described in the following works but
does not vendor their source repositories:

- Kovesi, *Phase Congruency Detects Corners and Edges*, DICTA 2003.
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- Gabor filtering and discrete wavelet operations through PyTorch/SciPy
  primitives.

## Python dependencies

PyTorch, TorchVision, NumPy, SciPy, OpenCV, Pillow, and tqdm remain under their
respective upstream licenses. See `requirements.txt` and each package's
official distribution for the applicable license text.

## Datasets

ISIC2018, HAM10000, Kvasir-SEG, and CVC-ClinicDB are not distributed here.
Their sources and citations are listed in `DATASETS.md`.
