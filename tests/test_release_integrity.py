from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ISIC2018", "HAM10000", "Kvasir-SEG", "CVC-ClinicDB")
EXPECTED_COUNTS = {
    "ISIC2018": (2594, 1000),
    "HAM10000": (8012, 2003),
    "Kvasir-SEG": (800, 100),
    "CVC-ClinicDB": (550, 62),
}


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_all_fixed_shot_manifests_are_complete_and_disjoint() -> None:
    checked = 0
    for dataset in DATASETS:
        train_count, test_count = EXPECTED_COUNTS[dataset]
        train = set(read_names(ROOT / "splits" / dataset / "train_pool.txt"))
        test = set(read_names(ROOT / "splits" / dataset / "test.txt"))
        assert len(train) == train_count
        assert len(test) == test_count
        assert train.isdisjoint(test)
        for shot in (5, 10, 20):
            for fold in range(1, 6):
                path = ROOT / "splits" / dataset / f"{shot}_shot" / f"fold_{fold}_labeled.txt"
                names = read_names(path)
                assert len(names) == shot
                assert len(names) == len(set(names))
                assert set(names) <= train
                assert set(names).isdisjoint(test)
                checked += 1
    assert checked == 60


def test_training_code_never_constructs_a_test_loader() -> None:
    files = [
        ROOT / "train.py",
        ROOT / "kglr_net" / "training.py",
        ROOT / "scripts" / "run_all_experiments.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "test_loader" not in text
        assert 'split="test"' not in text
        assert "split='test'" not in text


def test_python_sources_contain_no_cjk_or_private_machine_paths() -> None:
    cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
    private_path_patterns = (
        re.compile(r"[A-Za-z]:\\" + "Users" + r"\\", flags=re.IGNORECASE),
        re.compile("/" + "home" + r"/[^/]+/", flags=re.IGNORECASE),
        re.compile(r"/(?:mnt/)?" + "hd" + r"\d+t/", flags=re.IGNORECASE),
        re.compile("172" + r"\.18\.\d+\.\d+"),
    )
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert cjk.search(text) is None, path
        for pattern in private_path_patterns:
            assert pattern.search(text) is None, (path, pattern.pattern)


def test_repository_contains_no_large_binary_artifacts() -> None:
    forbidden_suffixes = {".pth", ".pt", ".ckpt", ".onnx", ".npy", ".npz"}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        assert path.suffix.lower() not in forbidden_suffixes, path
        assert path.stat().st_size < 5 * 1024 * 1024, path
