"""Gate 0 data: masking + by-video split + example construction.

Pure numpy — no torch here. One example per query (segment) node: the node's
own audio is masked, along with its whole ancestor chain to root, and the
task is to reconstruct the query node's true audio_dbfs_avg (dBFS).
"""
from __future__ import annotations
import random
from dataclasses import dataclass

import numpy as np

from mnemo.model.featurize import load_corpus, FeaturizedTree, FEATURE_NAMES
from mnemo.gate0.structure import relation_matrix

IS_SEGMENT_IDX = FEATURE_NAMES.index("is_segment")
AUDIO_AVG_IDX = FEATURE_NAMES.index("audio_dbfs_avg")
AUDIO_PEAK_IDX = FEATURE_NAMES.index("audio_dbfs_peak")
N_BASE = len(FEATURE_NAMES)        # 19
INPUT_DIM = N_BASE + 2             # 21 (+ is_query, + audio_hidden)

SPLIT_SEED = 0                     # the video split is FIXED, independent of model seed


@dataclass
class Example:
    video_id: str
    tokens: np.ndarray        # [n_nodes, 21] standardized + indicators, audio masked
    query_idx: int            # index of the query node within tokens
    audio_hidden: np.ndarray  # [n_nodes] 0/1
    is_query: np.ndarray      # [n_nodes] 0/1
    label: float              # true (un-standardized) audio_dbfs_avg, dBFS
    # Precomputed permutation-invariant views (over real nodes):
    query_row: np.ndarray     # [21]
    mean_pool: np.ndarray     # [21]
    max_pool: np.ndarray      # [21]
    # Optional [n_nodes, n_nodes] relation matrix (parent/child/sibling/...).
    # None for tasks/rungs that don't use the structural attention bias.
    rel: np.ndarray | None = None


def split_videos(video_ids: list[str], seed: int = SPLIT_SEED,
                 n_train: int = 6, n_val: int = 2, n_test: int = 2):
    """Deterministic disjoint split of video ids into train/val/test."""
    ids = sorted(video_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    train = ids[:n_train]
    val = ids[n_train:n_train + n_val]
    test = ids[n_train + n_val:n_train + n_val + n_test]
    return train, val, test


def fit_standardizer(trees: list[FeaturizedTree], train_ids: list[str]):
    """Per-feature mean/std fit on TRAIN videos only (no leakage)."""
    train_X = np.vstack([t.X for t in trees if t.video_id in train_ids])
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def ancestors_of(ft: FeaturizedTree, idx: int) -> list[int]:
    """Indices from idx's parent up to root (exclusive of idx)."""
    chain: list[int] = []
    p = ft.parent_idx[idx]
    while p != -1:
        chain.append(p)
        p = ft.parent_idx[p]
    return chain


def build_examples_for_tree(ft: FeaturizedTree, mean: np.ndarray,
                            std: np.ndarray) -> list[Example]:
    x_std = (ft.X - mean) / std
    n = len(ft.node_ids)
    rel = relation_matrix(ft.parent_idx)  # tree structure; reused across queries
    examples: list[Example] = []
    for i in range(n):
        if ft.X[i, IS_SEGMENT_IDX] != 1.0:
            continue  # query nodes are segments only
        hidden = {i, *ancestors_of(ft, i)}
        audio_hidden = np.zeros(n, dtype=np.float32)
        is_query = np.zeros(n, dtype=np.float32)
        for h in hidden:
            audio_hidden[h] = 1.0
        is_query[i] = 1.0

        masked = x_std.copy()
        for h in hidden:
            masked[h, AUDIO_AVG_IDX] = 0.0   # standardized mean
            masked[h, AUDIO_PEAK_IDX] = 0.0
        tokens = np.concatenate(
            [masked, is_query[:, None], audio_hidden[:, None]], axis=1
        ).astype(np.float32)

        label = float(ft.X[i, AUDIO_AVG_IDX])  # true un-standardized dBFS
        examples.append(Example(
            video_id=ft.video_id, tokens=tokens, query_idx=i,
            audio_hidden=audio_hidden, is_query=is_query, label=label,
            query_row=tokens[i].copy(),
            mean_pool=tokens.mean(axis=0).astype(np.float32),
            max_pool=tokens.max(axis=0).astype(np.float32),
            rel=rel,
        ))
    return examples


def build_dataset(corpus_dir: str = "corpus", split_seed: int = SPLIT_SEED,
                  n_train: int = 6, n_val: int = 2, n_test: int = 2):
    """Load corpus, split by video, standardize on train, build examples."""
    trees = load_corpus(corpus_dir)
    ids = [t.video_id for t in trees]
    train_ids, val_ids, test_ids = split_videos(
        ids, seed=split_seed, n_train=n_train, n_val=n_val, n_test=n_test)
    mean, std = fit_standardizer(trees, train_ids)
    by_id = {t.video_id: t for t in trees}

    def make(idset: list[str]) -> list[Example]:
        out: list[Example] = []
        for vid in idset:
            out.extend(build_examples_for_tree(by_id[vid], mean, std))
        return out

    return {
        "trees": trees,
        "train": make(train_ids), "val": make(val_ids), "test": make(test_ids),
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "mean": mean, "std": std,
    }
