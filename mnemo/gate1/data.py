"""Permuted-MNIST data for Gate 1a (pure numpy; no torch here).

Raw MNIST IDX files are downloaded from a reliable mirror and cached OUTSIDE
the repo. Permuted-MNIST: T tasks sharing one 10-way label space; task 0 is the
identity permutation, tasks 1..T-1 apply fixed random pixel permutations. Two
orderings (TUNE / EVAL) use DISJOINT permutation seeds so tuning never leaks the
eval permutations.
"""
from __future__ import annotations
import gzip
import struct
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE_DIR = Path(tempfile.gettempdir()) / "mnemo_mnist"
_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}

# Disjoint permutation-seed bases for the two orderings.
TUNE_SEED_BASE = 1000
EVAL_SEED_BASE = 9000
N_TASKS = 5
N_TRAIN_SUB = 10_000
SUBSAMPLE_SEED = 4242


def _download(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if not path.exists() or path.stat().st_size == 0:
        urllib.request.urlretrieve(_MIRROR + name, path)
    return path


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad image magic {magic}"
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255.0


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad label magic {magic}"
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = _read_idx_images(_download(_FILES["train_x"]))
    train_y = _read_idx_labels(_download(_FILES["train_y"]))
    test_x = _read_idx_images(_download(_FILES["test_x"]))
    test_y = _read_idx_labels(_download(_FILES["test_y"]))
    return train_x, train_y, test_x, test_y


def make_permutations(ordering_seed_base: int, n_tasks: int = N_TASKS) -> list[np.ndarray]:
    """Task 0 = identity; tasks 1..n-1 = fixed random pixel permutations whose
    seeds are derived from ordering_seed_base (disjoint between orderings)."""
    perms = [np.arange(784)]
    for t in range(1, n_tasks):
        rng = np.random.default_rng(ordering_seed_base + t)
        perms.append(rng.permutation(784))
    return perms


@dataclass
class Task:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def build_tasks(ordering: str, n_tasks: int = N_TASKS,
                n_train_sub: int = N_TRAIN_SUB) -> list[Task]:
    """Build the T permuted tasks for an ordering ('tune' or 'eval').

    The train subsample (fixed seed, identical across methods) selects one set
    of n_train_sub images reused by every task; each task applies its pixel
    permutation. The full 10k test set is used per task.
    """
    base = {"tune": TUNE_SEED_BASE, "eval": EVAL_SEED_BASE}[ordering]
    perms = make_permutations(base, n_tasks)
    train_x, train_y, test_x, test_y = load_mnist()

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    idx = rng.choice(train_x.shape[0], size=n_train_sub, replace=False)
    sub_x, sub_y = train_x[idx], train_y[idx]

    tasks: list[Task] = []
    for p in perms:
        tasks.append(Task(
            train_x=sub_x[:, p].copy(), train_y=sub_y.copy(),
            test_x=test_x[:, p].copy(), test_y=test_y.copy(),
        ))
    return tasks
