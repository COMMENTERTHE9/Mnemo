import numpy as np

from mnemo.gate1.data import (
    make_permutations, TUNE_SEED_BASE, EVAL_SEED_BASE, N_TASKS,
)


def test_permutations_deterministic_and_disjoint():
    # Deterministic per seed base.
    a = make_permutations(TUNE_SEED_BASE)
    b = make_permutations(TUNE_SEED_BASE)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # Task 0 is identity in both orderings.
    tune = make_permutations(TUNE_SEED_BASE)
    ev = make_permutations(EVAL_SEED_BASE)
    assert np.array_equal(tune[0], np.arange(784))
    assert np.array_equal(ev[0], np.arange(784))
    # TUNE vs EVAL permutations (tasks 1..) are disjoint (never identical).
    for t in range(1, N_TASKS):
        assert not np.array_equal(tune[t], ev[t])


def test_replay_buffer_never_exceeds_budget():
    from mnemo.gate1.methods import ReplayBuffer
    buf = ReplayBuffer(per_task=200)
    rng = np.random.default_rng(0)
    x = np.zeros((10_000, 784), dtype=np.float32)
    y = np.zeros(10_000, dtype=np.int64)
    for _ in range(N_TASKS):
        buf.add_task(x, y, rng)
    assert buf.total() <= 200 * N_TASKS
    assert all(len(a) <= 200 for a in buf.xs)
    # sampling never returns more than requested
    s = buf.sample(50, rng)
    assert s is not None and len(s[0]) == 50


def test_ewc_penalty_zero_at_anchor_positive_away():
    import torch
    from mnemo.gate1.methods import MLP, EWC
    model = MLP()
    x = np.random.default_rng(0).random((64, 784)).astype(np.float32)
    y = np.random.default_rng(1).integers(0, 10, size=64).astype(np.int64)
    ewc = EWC(lam=1.0, params_fn=lambda m: list(m.parameters()))
    ewc.consolidate(model, x, y, n_samples=64)
    # At the anchor the params are unchanged -> penalty exactly 0.
    assert float(ewc.penalty(model).detach()) == 0.0
    # Move away -> penalty strictly positive.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5)
    assert float(ewc.penalty(model).detach()) > 0.0


def test_rmatrix_bookkeeping():
    from mnemo.gate1.run import metrics
    # Hand-built R: diagonal (new-task) high, off-diagonal-below shows forgetting.
    R = np.array([
        [0.90, 0.10, 0.10, 0.10, 0.10],
        [0.50, 0.90, 0.10, 0.10, 0.10],
        [0.40, 0.50, 0.90, 0.10, 0.10],
        [0.30, 0.40, 0.50, 0.90, 0.10],
        [0.20, 0.30, 0.40, 0.50, 0.90],
    ])
    m = metrics(R)
    # final ACC = mean of last row
    assert abs(m["final_acc"] - np.mean([0.20, 0.30, 0.40, 0.50, 0.90])) < 1e-9
    # new-task = mean diagonal = 0.90
    assert abs(m["newtask"] - 0.90) < 1e-9
    # BWT = mean_{j<4} (R[4,j] - R[j,j]) = mean(0.20-0.90, 0.30-0.90, 0.40-0.90, 0.50-0.90)
    expected_bwt = np.mean([0.20 - 0.90, 0.30 - 0.90, 0.40 - 0.90, 0.50 - 0.90])
    assert abs(m["bwt"] - expected_bwt) < 1e-9
    assert m["bwt"] < 0  # forgetting is negative
