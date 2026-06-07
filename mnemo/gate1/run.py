"""Gate 1a protocol: tune on the TUNE ordering, freeze winners, evaluate on the
disjoint EVAL ordering over 3 seeds. torch pulled in via methods.py.
"""
from __future__ import annotations

import numpy as np

from mnemo.gate1.data import build_tasks
from mnemo.gate1.methods import run_sequential, run_joint

EVAL_SEEDS = [0, 1, 2]
TUNE_SEED = 0
GRID_LAMBDA = [1e1, 1e2, 1e3]
GRID_C = [0.1, 1.0, 10.0]
GRID_ALPHA = [0.1, 0.3, 1.0]


def metrics(R: np.ndarray) -> dict:
    T = R.shape[0]
    final_acc = float(R[-1].mean())
    bwt = float(np.mean([R[-1, j] - R[j, j] for j in range(T - 1)]))
    newtask = float(np.mean([R[i, i] for i in range(T)]))
    return {"final_acc": final_acc, "bwt": bwt, "newtask": newtask}


def tune(tasks) -> dict:
    """Grid-search on the TUNE ordering, 1 seed; select by final ACC."""
    ewc = [(lam, metrics(run_sequential("ewc", tasks, TUNE_SEED, lam=lam))["final_acc"])
           for lam in GRID_LAMBDA]
    lam_star = max(ewc, key=lambda kv: kv[1])[0]
    si = [(c, metrics(run_sequential("si", tasks, TUNE_SEED, c=c))["final_acc"])
          for c in GRID_C]
    c_star = max(si, key=lambda kv: kv[1])[0]
    rec = [(a, metrics(run_sequential("recipe", tasks, TUNE_SEED, lam=lam_star, alpha=a))["final_acc"])
           for a in GRID_ALPHA]
    alpha_star = max(rec, key=lambda kv: kv[1])[0]
    return {"lambda": lam_star, "c": c_star, "alpha": alpha_star,
            "grids": {"ewc": ewc, "si": si, "recipe": rec}}


def evaluate(tasks, hp: dict) -> dict:
    """3-seed eval on the EVAL ordering. Returns per-method metric lists + the
    seed-mean recipe R matrix."""
    out = {m: {"final_acc": [], "bwt": [], "newtask": []}
           for m in ("naive", "replay", "ewc", "si", "recipe")}
    joint_finals = []
    recipe_Rs = []
    for seed in EVAL_SEEDS:
        joint_finals.append(run_joint(tasks, seed))
        runs = {
            "naive": run_sequential("naive", tasks, seed),
            "replay": run_sequential("replay", tasks, seed),
            "ewc": run_sequential("ewc", tasks, seed, lam=hp["lambda"]),
            "si": run_sequential("si", tasks, seed, c=hp["c"]),
            "recipe": run_sequential("recipe", tasks, seed,
                                     lam=hp["lambda"], alpha=hp["alpha"]),
        }
        for m, R in runs.items():
            mt = metrics(R)
            for k in ("final_acc", "bwt", "newtask"):
                out[m][k].append(mt[k])
        recipe_Rs.append(runs["recipe"])
    return {"methods": out, "joint_final": joint_finals,
            "recipe_R": np.mean(np.stack(recipe_Rs), axis=0)}


def _ms(xs):
    return float(np.mean(xs)), float(np.std(xs))


def run_gate1a() -> dict:
    tune_tasks = build_tasks("tune")
    hp = tune(tune_tasks)
    eval_tasks = build_tasks("eval")
    ev = evaluate(eval_tasks, hp)
    return {"hp": hp, "eval": ev}


def print_report(res: dict) -> None:
    hp = res["hp"]
    ev = res["eval"]
    M = ev["methods"]
    jm, js = _ms(ev["joint_final"])
    print("GATE 1a — Permuted-MNIST, T=5, single-head, 3 seeds")
    print(f"TUNED (frozen): lambda={hp['lambda']:g} c={hp['c']:g} alpha={hp['alpha']:g}")

    def fa(m):
        mean, std = _ms(M[m]["final_acc"])
        return f"{mean:.3f}±{std:.3f}"
    print("FINAL ACC : naive=%s joint=%.3f±%.3f replay=%s ewc=%s si=%s recipe=%s"
          % (fa("naive"), jm, js, fa("replay"), fa("ewc"), fa("si"), fa("recipe")))

    def bw(m):
        mean, std = _ms(M[m]["bwt"])
        return f"{mean:+.3f}±{std:.3f}"
    print("BWT       : naive=%s replay=%s ewc=%s si=%s recipe=%s"
          % (bw("naive"), bw("replay"), bw("ewc"), bw("si"), bw("recipe")))
    nt_naive = _ms(M["naive"]["newtask"])[0]
    nt_recipe = _ms(M["recipe"]["newtask"])[0]
    print(f"NEW-TASK R[i,i] mean: naive={nt_naive:.3f} recipe={nt_recipe:.3f}  (statue check)")
    print("R MATRIX (recipe, seed-mean):")
    for row in ev["recipe_R"]:
        print("  " + " ".join(f"{v:.3f}" for v in row))

    joint_top = jm >= max(_ms(M[m]["final_acc"])[0] for m in M) - 1e-9
    naive_forgets = _ms(M["naive"]["bwt"])[0] < -0.05
    print(f"SANITY: joint on top? {'Y' if joint_top else 'N'}   "
          f"naive forgets? {'Y' if naive_forgets else 'N'}")

    rec = _ms(M["recipe"]["final_acc"])[0]
    best_other = max(_ms(M[m]["final_acc"])[0] for m in ("replay", "ewc", "si"))
    beats_naive = rec > _ms(M["naive"]["final_acc"])[0] + 0.02
    competitive = rec >= best_other - 0.02
    alive = nt_recipe > 0.80
    print(f"VERDICT vs plan §2a: beats naive clearly? {'Y' if beats_naive else 'N'}  "
          f"competitive with best of {{replay,ewc,si}}? {'Y' if competitive else 'N'}  "
          f"new-task learning alive? {'Y' if alive else 'N'}")


# ── Gate 1a': learned per-weight commitment + attribution lattice ────────────
def _agg(runs_by_arm: dict) -> dict:
    out = {}
    for arm, Rs in runs_by_arm.items():
        ms = [metrics(R) for R in Rs]
        out[arm] = {k: (float(np.mean([m[k] for m in ms])),
                        float(np.std([m[k] for m in ms])))
                    for k in ("final_acc", "bwt", "newtask")}
    return out


def run_gate1aprime() -> dict:
    from mnemo.gate1.commitment import train_arm, train_ewc, train_joint

    # TUNE lambda via the prot+rec (.2) arm's final ACC, 1 seed, TUNE ordering.
    tune_tasks = build_tasks("tune")
    grid = []
    for lam in GRID_LAMBDA:
        R, _ = train_arm(tune_tasks, TUNE_SEED, penalty=True, gating=True,
                         harden_mode="topk", lam=lam)
        grid.append((lam, metrics(R)["final_acc"]))
    lam_star = max(grid, key=lambda kv: kv[1])[0]

    eval_tasks = build_tasks("eval")
    # committed fraction f + per-layer health from a .2 run (deterministic counts)
    _, st = train_arm(eval_tasks, EVAL_SEEDS[0], penalty=True, gating=True,
                      harden_mode="topk", lam=lam_star)
    f = st.committed_fraction()

    runs = {a: [] for a in ("naive", "ewc", "prot1", "prot2",
                            "shuffled", "uniform", "c0")}
    joint_finals = []
    for seed in EVAL_SEEDS:
        joint_finals.append(train_joint(eval_tasks, seed))
        runs["naive"].append(train_arm(eval_tasks, seed, penalty=False,
                             gating=False, harden_mode=None)[0])
        runs["ewc"].append(train_ewc(eval_tasks, seed, lam_star))
        runs["prot1"].append(train_arm(eval_tasks, seed, penalty=True,
                             gating=False, harden_mode="topk", lam=lam_star)[0])
        runs["prot2"].append(train_arm(eval_tasks, seed, penalty=True,
                             gating=True, harden_mode="topk", lam=lam_star)[0])
        runs["shuffled"].append(train_arm(eval_tasks, seed, penalty=True,
                                gating=True, harden_mode="random", lam=lam_star)[0])
        runs["uniform"].append(train_arm(eval_tasks, seed, penalty=True,
                               gating=True, harden_mode=None, uniform_c=f,
                               lam=lam_star)[0])
        runs["c0"].append(train_arm(eval_tasks, seed, penalty=True, gating=True,
                          harden_mode=None, lam=lam_star)[0])

    agg = _agg(runs)
    jm, js = float(np.mean(joint_finals)), float(np.std(joint_finals))
    return {"lam": lam_star, "agg": agg, "joint": (jm, js),
            "committed_fraction": f, "health": st.per_layer_committed(),
            "harden_history": st.harden_history}


def print_aprime_report(res: dict) -> None:
    agg = res["agg"]
    jm, js = res["joint"]
    print(f"GATE 1a' — Permuted-MNIST T=5, single-head, SGD+momentum, 3 seeds, lambda={res['lam']:g}")

    def fa(a):
        m, s = agg[a]["final_acc"]
        return f"{m:.3f}±{s:.3f}"

    def bw(a):
        m, s = agg[a]["bwt"]
        return f"{m:+.3f}±{s:.3f}"
    print("FINAL ACC : naive=%s joint=%.3f±%.3f ewc=%s prot(.1)=%s prot+rec(.2)=%s"
          % (fa("naive"), jm, js, fa("ewc"), fa("prot1"), fa("prot2")))
    print("            shuffled=%s uniform=%s c0=%s"
          % (fa("shuffled"), fa("uniform"), fa("c0")))
    print("BWT       : naive=%s ewc=%s prot(.1)=%s prot+rec(.2)=%s"
          % (bw("naive"), bw("ewc"), bw("prot1"), bw("prot2")))
    print("            shuffled=%s uniform=%s c0=%s"
          % (bw("shuffled"), bw("uniform"), bw("c0")))
    print("NEW-TASK R[i,i]: naive=%.3f .1=%.3f .2=%.3f"
          % (agg["naive"]["newtask"][0], agg["prot1"]["newtask"][0],
             agg["prot2"]["newtask"][0]))
    d_shuf = agg["prot2"]["final_acc"][0] - agg["shuffled"]["final_acc"][0]
    d_prot = agg["prot2"]["final_acc"][0] - agg["prot1"]["final_acc"][0]
    print(f"ATTRIBUTION: learned(.2) vs shuffled delta = {d_shuf:+.3f}   .2 vs .1 delta = {d_prot:+.3f}")
    print("HEALTH (final committed frac/layer; plastic=1-committed; middle=0 by binary spec):")
    for nm, frac in res["health"].items():
        print(f"  {nm}: committed={frac:.3f} plastic={1 - frac:.3f}")
    print(f"  committed_fraction(global)={res['committed_fraction']:.3f}")
    # sanity
    accs = {a: agg[a]["final_acc"][0] for a in agg}
    joint_top = jm >= max(accs.values()) - 1e-9
    naive_forgets = agg["naive"]["bwt"][0] < -0.05
    c0_eq_naive = abs(accs["c0"] - accs["naive"]) < 1e-6
    print(f"SANITY: joint on top? {'Y' if joint_top else 'N'}   "
          f"naive forgets? {'Y' if naive_forgets else 'N'}   "
          f"c0 == naive? {'Y' if c0_eq_naive else 'N'}")
    # verdict (all four)
    forgetting_reduced = agg["prot2"]["bwt"][0] > agg["naive"]["bwt"][0] + 0.01
    new_task_alive = agg["prot2"]["newtask"][0] > 0.80
    beats_shuffled = d_shuf > 0.01
    budgeted = all(frac <= 0.40 + 1e-9 for frac in res["health"].values())
    print(f"VERDICT: forgetting reduced vs naive? {'Y' if forgetting_reduced else 'N'}  "
          f"new-task alive? {'Y' if new_task_alive else 'N'}  "
          f"learned beats shuffled? {'Y' if beats_shuffled else 'N'}  "
          f"commitment healthy/budgeted? {'Y' if budgeted else 'N'}")


# ── Gate 1a'-R: commitment x replay completion cell (lambda frozen at 10) ─────
def run_gate1aprime_replay() -> dict:
    from mnemo.gate1.commitment import train_arm
    lam = 10.0  # FROZEN (tuned for the no-replay arm; not re-tuned)
    eval_tasks = build_tasks("eval")
    runs = {a: [] for a in ("replay", "prot2_replay", "shuffled_replay")}
    for seed in EVAL_SEEDS:
        runs["replay"].append(train_arm(eval_tasks, seed, penalty=False,
                              gating=False, harden_mode=None, replay=True)[0])
        runs["prot2_replay"].append(train_arm(eval_tasks, seed, penalty=True,
                                    gating=True, harden_mode="topk",
                                    replay=True, lam=lam)[0])
        runs["shuffled_replay"].append(train_arm(eval_tasks, seed, penalty=True,
                                       gating=True, harden_mode="random",
                                       replay=True, lam=lam)[0])
    return {"lam": lam, "agg": _agg(runs)}


def print_aprime_replay_report(res: dict) -> None:
    agg = res["agg"]
    print(f"GATE 1a'-R — Permuted-MNIST T=5, SGD, 3 seeds, lambda={res['lam']:g} (frozen)")

    def fa(a):
        m, s = agg[a]["final_acc"]
        return f"{m:.3f}±{s:.3f}"

    def bw(a):
        m, s = agg[a]["bwt"]
        return f"{m:+.3f}±{s:.3f}"
    print("FINAL ACC : replay=%s .2+replay=%s shuffled+replay=%s   (refs: naive=0.626 joint=0.946)"
          % (fa("replay"), fa("prot2_replay"), fa("shuffled_replay")))
    print("BWT       : replay=%s .2+replay=%s shuffled+replay=%s"
          % (bw("replay"), bw("prot2_replay"), bw("shuffled_replay")))
    print(f"NEW-TASK R[i,i]: .2+replay={agg['prot2_replay']['newtask'][0]:.3f}  (statue check)")
    r_mean, r_std = agg["replay"]["final_acc"]
    p_mean, p_std = agg["prot2_replay"]["final_acc"]
    s_mean, _ = agg["shuffled_replay"]["final_acc"]
    d_replay = p_mean - r_mean
    d_shuf = p_mean - s_mean
    print(f"DELTAS: (.2+replay - replay) = {d_replay:+.3f}   (.2+replay - shuffled+replay) = {d_shuf:+.3f}")
    noise = max(r_std, p_std)
    adds = (d_replay > noise) and (d_replay > 0.0) and (d_shuf > 0.0)
    alive = agg["prot2_replay"]["newtask"][0] > 0.80
    verdict = "ADDS" if (adds and alive) else "RETIRED"
    print(f"PRE-REGISTERED VERDICT: {verdict}  "
          f"(beats replay outside noise [{d_replay:+.3f} vs std {noise:.3f}]? "
          f"{'Y' if d_replay > noise else 'N'}; beats shuffled+replay? {'Y' if d_shuf > 0 else 'N'}; "
          f"new-task alive? {'Y' if alive else 'N'})")


# ── Gate 1a' oracle-freeze kill-shot (perfect-foresight selection) ───────────
def run_oracle_killshot() -> dict:
    """Oracle arm = .2 mechanics but freezing the lowest-future-movement
    uncommitted weights, where future movement is read from a seed-matched
    naive run (perfect foresight). If oracle <= shuffled, value-pinning during
    drift is dead even with foresight."""
    from mnemo.gate1.commitment import train_arm, naive_snapshots
    eval_tasks = build_tasks("eval")
    runs = {"naive": [], "oracle": []}
    for seed in EVAL_SEEDS:
        nR, fmoves = naive_snapshots(eval_tasks, seed)
        oR, _ = train_arm(eval_tasks, seed, penalty=True, gating=True,
                          harden_mode="oracle", oracle_moves=fmoves, lam=10.0)
        runs["naive"].append(nR)
        runs["oracle"].append(oR)
    return {"agg": _agg(runs)}


def print_oracle_report(res: dict) -> None:
    agg = res["agg"]

    def fmt(a):
        fa = agg[a]["final_acc"]
        bw = agg[a]["bwt"]
        nt = agg[a]["newtask"]
        return f"ACC={fa[0]:.3f}±{fa[1]:.3f} BWT={bw[0]:+.3f} R[i,i]={nt[0]:.3f}"
    print("GATE 1 — oracle-freeze kill-shot (1a' harness, EVAL ordering, 3 seeds)")
    print(f"  oracle      : {fmt('oracle')}")
    print(f"  naive(in-run): {fmt('naive')}")
    print("  committed refs: naive=0.626  shuffled=0.641  learned(.2)=0.596")
    o = agg["oracle"]["final_acc"][0]
    o_std = agg["oracle"]["final_acc"][1]
    closed = o <= 0.641 + max(o_std, 0.005)  # oracle <= shuffled (within noise)
    verdict = ("CLOSED (oracle <= shuffled — value-pinning during drift dead even "
               "with perfect foresight)" if closed else
               "ORACLE-ONLY-LIFE (oracle > shuffled & naive; no real signal has foresight)")
    print(f"VERDICT (pre-registered): {verdict}")


if __name__ == "__main__":
    print_report(run_gate1a())
