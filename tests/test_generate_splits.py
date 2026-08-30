import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_splits_for_test", ROOT / "tools" / "generate_splits.py"
)
assert SPEC is not None and SPEC.loader is not None
GENERATE_SPLITS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATE_SPLITS)


def _write_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _prepare_partition(tmp_path: Path, train_names: list[str], test_names: list[str]) -> tuple:
    data_root = tmp_path / "data"
    splits_root = tmp_path / "splits"
    for split, names in (("train", train_names), ("test", test_names)):
        directory = data_root / "ISIC2018" / split / "npz_data"
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).touch()
    _write_names(splits_root / "ISIC2018" / "train_pool.txt", ["train_a.npz", "train_b.npz"])
    _write_names(splits_root / "ISIC2018" / "test.txt", ["test_a.npz"])
    return data_root, splits_root


def test_validated_partition_accepts_exact_disjoint_directories(tmp_path, monkeypatch) -> None:
    data_root, splits_root = _prepare_partition(
        tmp_path,
        train_names=["train_a.npz", "train_b.npz"],
        test_names=["test_a.npz"],
    )
    monkeypatch.setitem(GENERATE_SPLITS.EXPECTED_COUNTS, "ISIC2018", (2, 1))
    candidates, test_names = GENERATE_SPLITS.validated_partition(data_root, splits_root, "ISIC2018")
    assert candidates == ["train_a.npz", "train_b.npz"]
    assert test_names == {"test_a.npz"}


def test_validated_partition_rejects_directory_overlap(tmp_path, monkeypatch) -> None:
    data_root, splits_root = _prepare_partition(
        tmp_path,
        train_names=["train_a.npz", "train_b.npz", "test_a.npz"],
        test_names=["test_a.npz"],
    )
    monkeypatch.setitem(GENERATE_SPLITS.EXPECTED_COUNTS, "ISIC2018", (2, 1))
    with pytest.raises(RuntimeError, match="overlap"):
        GENERATE_SPLITS.validated_partition(data_root, splits_root, "ISIC2018")
