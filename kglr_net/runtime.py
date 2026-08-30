"""Runtime utilities shared by KGLR-Net command-line programs."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without changing the experiment protocol."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """Seed a data-loader worker from PyTorch's per-worker initial seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create a deterministically seeded CPU generator for a data loader."""

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(device: str) -> torch.device:
    """Resolve a requested device and reject unavailable CUDA devices early."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is unavailable.")
    return resolved


def load_checkpoint(
    path: Path | str, map_location: Any = "cpu"
) -> MutableMapping[str, torch.Tensor]:
    """Load a trusted local checkpoint and extract its model state dictionary.

    ``weights_only=True`` is used when supported. Older PyTorch releases fall
    back to the legacy loader. Only checkpoints downloaded from an official
    source or produced by this repository should be loaded.
    """

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, MutableMapping):
        raise TypeError(f"Checkpoint {checkpoint_path} does not contain a state dictionary.")
    keys = list(payload)
    if keys and all(isinstance(key, str) and key.startswith("module.") for key in keys):
        payload = {key[len("module.") :]: value for key, value in payload.items()}
    return payload


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write a deterministic UTF-8 JSON file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write records to UTF-8 CSV, preserving the first row's key order."""

    if not rows:
        raise ValueError("Cannot write an empty CSV file.")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "load_checkpoint",
    "make_generator",
    "resolve_device",
    "seed_everything",
    "seed_worker",
    "write_csv",
    "write_json",
]
