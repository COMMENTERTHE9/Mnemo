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


if __name__ == "__main__":
    print_report(run_gate1a())
