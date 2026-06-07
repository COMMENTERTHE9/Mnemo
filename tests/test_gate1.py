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


# ── Gate 1a' commitment mechanism ────────────────────────────────────────────
def _toy_tasks(n_tasks=5, n=256):
    from mnemo.gate1.data import Task
    rng = np.random.default_rng(0)
    tasks = []
    for _ in range(n_tasks):
        x = rng.random((n, 784)).astype(np.float32)
        y = rng.integers(0, 10, size=n).astype(np.int64)
        tasks.append(Task(train_x=x, train_y=y, test_x=x[:64], test_y=y[:64]))
    return tasks


def test_commitment_budget_and_cap_respected():
    from mnemo.gate1.commitment import train_arm, HARDEN_FRAC, COMMIT_CAP
    _, st = train_arm(_toy_tasks(), seed=0, penalty=True, gating=True,
                      harden_mode="topk", lam=100.0)
    # need a model to know layer sizes -> reconstruct from c buffers
    for boundary in st.harden_history:
        for nm, cnt in boundary.items():
            size = st.c[nm].numel()
            assert cnt <= int(HARDEN_FRAC * size)   # per-boundary 5% budget
    for nm, frac in st.per_layer_committed().items():
        assert frac <= COMMIT_CAP + 1e-9            # 40% cap


def test_harden_only_touches_uncommitted_and_c_monotonic():
    import torch
    from mnemo.gate1.commitment import CommitmentState
    from mnemo.gate1.methods import MLP
    torch.manual_seed(0)
    model = MLP()
    st = CommitmentState(model, lam=1.0)
    nm0 = next(iter(st.c))
    # pre-commit index 0 with a sentinel anchor
    st.c[nm0].view(-1)[0] = 1.0
    st.a[nm0].view(-1)[0] = 5.0
    omega = {nm: torch.rand_like(p) for nm, p in model.named_parameters()}
    vol = {nm: torch.rand_like(p) for nm, p in model.named_parameters()}
    counts = st.harden_topk(model, omega, vol)
    # committed weight was NOT rescored/re-anchored (score never on committed)
    assert st.c[nm0].view(-1)[0] == 1.0
    assert st.a[nm0].view(-1)[0] == 5.0
    # c is binary and never decreased
    for c in st.c.values():
        uniq = set(torch.unique(c).tolist())
        assert uniq <= {0.0, 1.0}
    # budget honored
    for nm, p in model.named_parameters():
        assert counts[nm] <= int(0.05 * p.numel())


def test_shuffled_matches_learned_counts():
    from mnemo.gate1.commitment import train_arm
    _, learned = train_arm(_toy_tasks(), seed=1, penalty=True, gating=True,
                           harden_mode="topk", lam=100.0)
    _, shuffled = train_arm(_toy_tasks(), seed=1, penalty=True, gating=True,
                            harden_mode="random", lam=100.0)
    assert len(learned.harden_history) == len(shuffled.harden_history)
    for hl, hs in zip(learned.harden_history, shuffled.harden_history):
        assert hl == hs  # identical per-layer counts per boundary


def test_c0_step_bit_identical_to_naive():
    import torch
    import torch.nn.functional as F
    from mnemo.gate1.commitment import CommitmentState, MOMENTUM
    from mnemo.gate1.methods import MLP, BASE_LR
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    torch.manual_seed(0)
    m1 = MLP()
    torch.manual_seed(0)
    m2 = MLP()
    # naive step
    o1 = torch.optim.SGD(m1.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    o1.zero_grad()
    F.cross_entropy(m1(x), y).backward()
    o1.step()
    # c=0 mechanism step (penalty + gating, but c==0 -> no-op)
    st = CommitmentState(m2, lam=100.0)
    o2 = torch.optim.SGD(m2.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    o2.zero_grad()
    loss = F.cross_entropy(m2(x), y) + st.penalty(m2)
    loss.backward()
    st.gate_grads(m2)
    o2.step()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_committed_weight_frozen_under_gating():
    import torch
    import torch.nn.functional as F
    from mnemo.gate1.commitment import CommitmentState, MOMENTUM
    from mnemo.gate1.methods import MLP, BASE_LR
    torch.manual_seed(0)
    model = MLP()
    st = CommitmentState(model, lam=100.0)
    nm0 = next(iter(st.c))
    params = dict(model.named_parameters())
    val = float(params[nm0].view(-1)[0].item())
    st.c[nm0].view(-1)[0] = 1.0
    st.a[nm0].view(-1)[0] = val  # anchor = current value
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    x = torch.rand(8, 784)
    y = torch.randint(0, 10, (8,))
    opt.zero_grad()
    loss = F.cross_entropy(model(x), y) + st.penalty(model)
    loss.backward()
    st.gate_grads(model)
    opt.step()
    assert params[nm0].view(-1)[0].item() == val  # frozen (grad gated to 0)


def test_commit_replay_budget_and_freeze_on_replay_batch():
    import torch
    import torch.nn.functional as F
    from mnemo.gate1.methods import MLP, ReplayBuffer, BASE_LR
    from mnemo.gate1.commitment import (
        CommitmentState, MOMENTUM, _zero_committed_momentum,
    )
    # 1) ring buffer budget holds (<= 200/task)
    buf = ReplayBuffer(per_task=200)
    rng = np.random.default_rng(0)
    x = np.zeros((5000, 784), dtype=np.float32)
    y = np.zeros(5000, dtype=np.int64)
    for _ in range(5):
        buf.add_task(x, y, rng)
    assert buf.total() <= 200 * 5
    assert all(len(a) <= 200 for a in buf.xs)

    # 2) a c=1 weight does not move on a replay-mixed batch
    torch.manual_seed(0)
    model = MLP()
    st = CommitmentState(model, lam=100.0)
    nm0 = next(iter(st.c))
    params = dict(model.named_parameters())
    val = float(params[nm0].view(-1)[0].item())
    st.c[nm0].view(-1)[0] = 1.0
    st.a[nm0].view(-1)[0] = val
    opt = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM)
    _zero_committed_momentum(opt, model, st)
    # current + replay samples concatenated, exactly as train_arm mixes them
    xb = torch.cat([torch.rand(8, 784), torch.rand(8, 784)])
    yb = torch.randint(0, 10, (16,))
    opt.zero_grad()
    loss = F.cross_entropy(model(xb), yb) + st.penalty(model)
    loss.backward()
    st.gate_grads(model)
    opt.step()
    assert params[nm0].view(-1)[0].item() == val  # frozen on the replay batch


def test_spearman_instrument():
    import torch
    from mnemo.gate1.diagnostic import spearman
    x = torch.arange(100, dtype=torch.float64)
    assert abs(spearman(x, x) - 1.0) < 1e-9            # perfect rank agreement
    assert abs(spearman(x, -x) + 1.0) < 1e-9           # perfect anti-agreement
    assert abs(spearman(x, x ** 3) - 1.0) < 1e-9       # monotone -> rho 1
    g = torch.Generator().manual_seed(0)
    a = torch.rand(5000, generator=g)
    b = torch.rand(5000, generator=g)
    assert abs(spearman(a, b)) < 0.05                  # independent -> ~0


def test_oracle_harden_lowest_selects_lowest_and_budgeted():
    import torch
    from mnemo.gate1.commitment import CommitmentState, HARDEN_FRAC
    from mnemo.gate1.methods import MLP
    torch.manual_seed(0)
    model = MLP()
    st = CommitmentState(model, lam=10.0)
    # scores = 0,1,2,... per layer -> the n lowest are indices 0..n-1
    scores = {nm: torch.arange(p.numel(), dtype=torch.float64).reshape(p.shape)
              for nm, p in model.named_parameters()}
    counts = st.harden_lowest(model, scores)
    for nm, p in model.named_parameters():
        n = counts[nm]
        assert n <= int(HARDEN_FRAC * p.numel())          # 5% budget
        c = st.c[nm].view(-1)
        assert int((c > 0).sum()) == n                    # exactly n committed
        if n > 0:
            assert bool((c[:n] == 1.0).all())             # the lowest-score ones
            assert bool((c[n:] == 0.0).all())


def test_naive_snapshots_deterministic_and_no_oracle_input():
    from mnemo.gate1.commitment import naive_snapshots
    tasks = _toy_tasks()
    R1, fm1 = naive_snapshots(tasks, seed=0)
    R2, fm2 = naive_snapshots(tasks, seed=0)
    # depends only on (tasks, seed): identical across calls (no external leakage)
    assert (R1 == R2).all()
    assert len(fm1) == len(tasks) - 1                     # one per boundary
    for d1, d2 in zip(fm1, fm2):
        for nm in d1:
            assert bool((d1[nm] == d2[nm]).all())
            assert bool((d1[nm] >= 0).all())              # |movements| non-negative
