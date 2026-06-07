# Mnemo: Experimental Plan (Gate Program)

*Companion to `mnemo_design_proposal.md`. Turns the proposal's hypotheses into pass/fail experiments.*

> **Status.** Decisions locked in design session 2026-05-30; **results recorded 2026-06** (substrate axis complete; perception axis Gate 0 passed). §2b/§2c below are closed/superseded — see RESULT blocks. The live work is the perception-axis output model (rung 1 proceeding); any revived substrate mechanism must first pass the single-episode consolidation screen (§2c).

---

## 0. Framing

**A gate is the cheapest possible falsification.** Its only virtue is being fast and cheap enough to tell you whether to keep going. A gate is *not* a design object to polish — turning a gate into a research program is the failure mode this plan exists to prevent. Keep each gate at the smallest resolution that yields a real pass/fail.

**The gates are parallel, not sequential.** There are two orthogonal axes, and neither result is an input to the other:

| Axis | Question | Gate |
|---|---|---|
| Perception | Are the perception trees usable as model input? | Gate 0 |
| Substrate | Does the continual-learning mechanism reduce forgetting? | Gate 1 (1a → 1a′ → 1b) |

Gate 1 does **not** require the trees — it tests the substrate on standard CL benchmarks. Gate 0 does **not** test the mechanism. They can run independently, in either order or at once.

**Necessary vs. sufficient.** Gate 0 is *necessary but not sufficient*. A red Gate 0 kills the project regardless of how clean the mechanism is (the prerequisite no architecture removes — proposal §8.6). A green Gate 0 proves only that the input is usable; it says nothing about the moving-number thesis. A green Gate 1 is mechanism insurance, not project validation.

**Mapping to the proposal's tiers.** Gate 0 = Tier 1 tracer-bullet. Gate 1 = an early extraction of the Tier 2 substrate and Tier 3 continual-learning work, pulled forward so the differentiating bet gets tested sooner rather than deferred behind an ever-shinier perception gate.

---

## 1. Gate 0 — Tree Learnability Smoke Test

**Purpose.** Prove the perception-tree representation is usable model input. This is a *smoke test*, not the proof of Mnemo. Keep it to one screen.

**Input.** Tree nodes from the perception layer, one token per node. Each token carries: importance, level, relative timing, motion scalar(s), audio scalar(s), visual scalar(s), action multi-hot. Include parent/child structure and level metadata; optional tree-position encoding.

**Task.** Masked-node reconstruction — mask one continuous feature of a node (motion, audio loudness, or importance) and predict it from surrounding tree context. One task, not four.

**Baseline ladder.**
1. mean / majority predictor
2. linear model
3. small MLP over pooled node features
4. tiny Transformer over node tokens

**Split.** By **video**, not by node. Train / validation / test must contain disjoint videos. At ~10 videos this is a pipeline smoke test — debug the harness, do not treat the number as evidence. Widen the corpus before believing it.

**Pass.** A tiny model beats the trivial and simple baselines on held-out videos.
**Fail.** Repair the featurizer, tree structure, or node-feature design before any of the mind matters.

**Non-goals (explicitly out of scope here):** ternary weights, moving-number clouds, continual learning, the mold, identity governance, language generation.

> **RESULT — PASS (2026-06).** 40-video corpus, split by video, masked audio-loudness reconstruction: tiny Transformer MAE 2.59 dBFS / R² 0.912 vs. mean baseline 9.33, consistent across all 6 held-out videos. Controls: identity floor (plumbing) and a synthetic sibling-mean task that required a parent-index relational attention bias. On the real target the relational-bias ablation **tied** — loudness is content/context-shaped, not edge-shaped; the structure mechanism is validated and shelved. Trees are usable model input; the §8.6 prerequisite is retired.

---

## 2. Gate 1 — Continual-Learning Substrate (parallel track)

Runs on standard CL benchmarks (Permuted-MNIST / Split-MNIST / synthetic), **full precision**, **no trees required**. Tests the substrate, not perception. Quantization is deliberately excluded here — it is a separate question (Gate 1b).

### 2a. Gate 1a — Conservative CLS recipe *(the cheap mechanism gate)*

**Purpose.** Test whether the proposal's conservative §5 build reduces forgetting at all: parameter isolation (slow/protected core + small full-precision plastic head) + EWC-style protection + replay. This is nearly off-the-shelf — Permuted-MNIST + EWC is the benchmark EWC was introduced on — which is exactly why it is the right *gate*: the cheapest falsification of "can this system do continual learning."

**Baselines (equal params, equal compute).** naive sequential fine-tune (floor); joint all-tasks training (oracle ceiling); replay; EWC; SI. Hyperparameters tuned on a *separate* task ordering, then frozen for the eval ordering.

**Metrics (from the accuracy matrix R[i,j]).** final ACC; backward transfer / forgetting; new-task accuracy R[i,i] (anti-statue — catches "protect everything, learn nothing"); 3+ seeds, mean ± std.

**Pass.** Reduces forgetting clearly vs. naive and is competitive with the best of {replay, EWC, SI} at equal budget, without collapsing new-task learning.
**Fail.** The conservative recipe doesn't survive forgetting on a standard benchmark — fix it before anything more ambitious.

> **RESULT — PASS on all three criteria (2026-06).** Permuted-MNIST T=5 single-head, tune-then-freeze, 3 seeds: recipe 0.837±0.003 vs. naive 0.789 (BWT −0.023 vs. −0.088, ≈4× less forgetting); competitive with replay (0.843); R[i,i] 0.855 ≈ naive (no statue). Honest read: **replay is the workhorse**; EWC/SI alone ≈ naive in this single-head domain-incremental setting. Trees-as-replay is promoted from ingredient to backbone.

### 2b. Gate 1a′ — Learned per-weight commitment *(the "raise plasticity" step — NOT the gate)*

This is the ambitious, thesis-adjacent mechanism. It is **not** the cheap gate — it belongs to the "raise plasticity gradually" step of §5, run *after* 1a. (It was briefly mistaken for Gate 1a during design; the revert puts the conservative recipe first and demotes this to 1a′. The proposal §9 item 7 records the same ordering.)

**Mechanism.** Each weight `w_i` carries a commitment scalar `c_i ∈ [0,1]`. Two **distinct** primitives — do not conflate them:
- **Anchor penalty = protection:** `λ · c_i · (w_i − a_i)²` resists drift of committed weights without redirecting where new learning goes.
- **LR-gating = recruitment:** `lr_i = lr_base · (1 − c_i)` concentrates new learning into uncommitted weights. (Note: LR-gating *is itself* a recruitment mechanism, so it must not appear in a "protection-only" arm.)

**Sub-gates (clean attribution).**
- **1a′.0 — harness sanity:** naive / joint / EWC reference; verify the benchmark forgets, the model learns, the metric code is correct.
- **1a′.1 — protection only:** anchor penalty alone (no LR-gating). Does protecting committed weights reduce forgetting without blocking new learning?
- **1a′.2 — protection + recruitment:** add LR-gating. Does routing new learning into plastic weights improve the stability–plasticity tradeoff?
- **1a′.3 — commitment dynamics:** per-layer, budget-bounded hardening.

**Commitment update.** `commit_score = importance × stability` (SI-style path importance × low late-training volatility), computed **only on currently-uncommitted weights** (scoring already-protected weights creates a fake-stability feedback loop → saturation). Per-layer percentile hardening, budget-bounded (e.g. 5% of uncommitted per boundary, ≤40% committed per layer). **No decay** in this experiment.

**Attribution ablation lattice (required).** learned `c` vs. **shuffled** `c` (same budget, wrong targets) vs. **uniform** `c` (no structure, global pressure) vs. `c = 0` (off → should fall to naive). **Learned must beat shuffled** — otherwise `c` is just a global regularizer, not weight-specific information.

**Health metric.** Bimodality, per layer — core fraction (`c > 0.8`), plastic fraction (`c < 0.2`), middle fraction — **not** mean `c`. A healthy run is a small stable core plus a large preserved plastic reserve, with little mass in between. (Mean `c` ≈ 0.5 can be a degenerate "everything lukewarm" model.)

**Pass requires all four:** forgetting reduced vs. naive; new-task learning alive (no statue); learned `c` beats shuffled `c`; commitment distribution healthy/bimodal (no saturation, no collapse).

> **RESULT — FAIL, 2 of 4, well-attributed (2026-06). CLOSED.** Sanity held (joint 0.946 top; naive forgets −0.393; `c=0` == naive exactly). Protection-only 0.604 and protection+recruitment 0.596 both landed **below naive** 0.626; **shuffled-c 0.641 beat learned-c** — the learned selection chose worse freeze targets than chance; recruitment added nothing (−0.008); plasticity alive; budgets/health clean; uniform collapsed as expected. **1a′-R (mechanism + replay): RETIRED** — replay-alone 0.923 ≈ replay+commitment 0.923 (Δ −0.000 vs. seed noise 0.001; hardening confirmed live at committed fraction 0.200); commitment is inert on top of replay, which sits near the joint ceiling.
>
> **Corrected attribution.** A pre-registered retrodiction measured ρ(commit_score, next-task weight movement) = **−0.052 ± 0.010** (random control 0.000 exactly): the score is **orthogonal** to next-task demand, falsifying "important-to-A ≈ demanded-by-B" as the failure mechanism. With selection orthogonal to demand yet learned < shuffled, the failure is forced onto what the score *does* track — A-sensitivity: **importance-weighted partial-freeze miscalibration**. Pinning A's highest-curvature weights to stale values while co-adapted neighbors drift yields a configuration worse for A than full drift. Retention is a property of coherent configurations, not individual weight values; value-space protection is the wrong abstraction, and replay wins because it protects in **function space**. **Oracle kill-shot (run, pre-registered, CLOSED):** an oracle-freeze arm with seed-matched perfect foresight (lowest future-movement selection, 1a′ mechanics) scored 0.639±0.010 vs. shuffled 0.641 (naive 0.626) — oracle ≤ shuffled, so value-pinning during drift fails even with perfect selection; selection quality was never the variable. The wake-phase protection line is **permanently closed**; configurations-not-values now stands on three independent legs (orthogonality retrodiction, learned < shuffled, oracle ≤ shuffled).

### 2c. Gate 1b — Ternary 0-as-abstain *(thesis tier — only after 1a′)*

**Purpose.** Test whether the ternary realization recovers the 1a′ benefit: a weight held as a distribution over {−1, 0, +1} (categorical posterior via softmax logits), trained with straight-through / shadow-weight machinery (the QAT trick the proposal cites in §7), collapsed at consolidation.

**Load-bearing correction — `0` is not the abstain marker.** Abstain (uncommitted, recruitable) is a property of the posterior's **entropy/concentration**, not of mass on 0. A maximally-undecided symmetric posterior has expected value 0 — which is why 0 *looks* like abstention — but the converse fails: a posterior peaked on 0 also has expectation 0 and is the **opposite** of abstaining (a confidently-pruned weight). `E[w] = 0` is necessary, not sufficient. The commitment dial is **entropy**, never `|E[w]|` or `P(0)`. `0` is only the do-nothing value an open weight contributes while it waits.

**Decisive test.** A recruitment/protection asymmetry: when a new task arrives, updates flow preferentially into high-entropy weights while low-entropy committed weights stay protected. **Ablate it** (force uniform updates); forgetting should return. If it doesn't, `0` was sparsity wearing a costume, not abstention.

> **SUPERSEDED AS SPECCED (2026-06).** Gate 1b as written inherits the falsified delivery vehicle — per-weight value-pinning during wake-phase training — which 1a′ showed harms retention via partial-freeze miscalibration regardless of selection. The entropy dial itself is untested, not falsified, but its site of action moves: **commitment is a property of consolidation events, not of weights during wake.** Successor design: **entropy-gated consolidation** — wake stays the validated §2a shape (plastic head + replay doing retention), and entropy accumulated across the replay distribution gates only how strongly each core weight absorbs the distilled update during sleep. This dodges both failure causes: consolidation is a global coordinated fit (no chimera), and replay-distribution entropy is a cross-task signal (no single-task retrospection).
>
> **Entry exam (pre-registrable) — the single-episode consolidation screen.** The mechanism's single action is one distillation event, so test that event in isolation, in function space. Core trained on A; wake learns B via the validated recipe; one consolidation event distills wake behavior into the core under the candidate gate. Four arms: **ungated** distill, **entropy-gated** distill, **shuffled-gate** (same gate mass, random targets — the control that caught 1a′), **all-frozen-core** floor. Measure post-consolidation teacher–student disagreement (or accuracy) on A-replay and on B separately. **Pass:** learned gate strictly dominates shuffled on the A/B tradeoff, at ≤ ε cost on B-fit vs. ungated. **Failure readings:** learned ≈ shuffled ⇒ the gate carries no weight-specific information; learned worse than ungated on both axes ⇒ the gate is a pure capacity tax (protected weights block paths the coordinated fit needs). One episode, no sequences, afternoon-scale. No new arena is purchased until a mechanism passes this screen; arenas (restricted replay budgets, long sequences, shared-structure tasks) are then sized to model deployment constraints, not to rescue a pattern.

---

## 3. Coherence findings (locked this session)

1. **0-as-abstain category error** (above) — patched into proposal §4.2: entropy is the dial; 0 is the expected resting value of an undecided posterior, not the marker of openness.
2. **Loop closure.** Gate 1b's metric (posterior entropy/concentration) *is* the spec for Gate 1a′'s commitment scalar. The same correction seen from two ends: the commitment scalar `c_i` should track entropy, not `|E[w]|`, not `P(0)`.

---

## 4. Methodological principles (locked)

- **Split by video, not by node** — node-split leaks structure across train/test.
- **Wide signal spread is necessary, not sufficient.** Varied features prove the signals aren't dead; they don't prove anything is predictable. That's what the gates test.
- **Never combine ternary + clouds + CL + EWC + mold in one experiment.** If a confounded blob fails, you learn nothing. Build order: full-precision tiny model → learn tree structure → output → quantize → uncertainty → plastic head → replay → consolidation → governance.
- **Always include baselines; always include attribution ablations.** Beating a baseline shows it works; the ablation lattice shows it works *for the right reason*.
- **A design pass earns its place only by surfacing a contradiction or a category error.** When a pass stops producing those, stop. (This plan is that stop.)

---

## 5. Current state (2026-06)

**Substrate axis: complete.** Gate 1a passed (conservative build validated; replay is the backbone). Gate 1a′ failed with full attribution (partial-freeze miscalibration); 1a′-R retired. Gate 1b superseded as specced; successor (entropy-gated consolidation) is queued behind the consolidation screen, deferred behind the perception axis.

**Perception axis: live.** Gate 0 passed; the output model is the active work. **Rung 1 (templated verbalization): plumbing validated** — conditioned beats unconditioned on every slot (exact-match 0.743 vs. 0.069; motion/position/action > 0.90); the loudness slot narrowly missed its pre-registered bar (0.840 < 0.90) via a bin-edge artifact (hand-picked dBFS edges clustered near corpus mass under the by-video split) — recorded, not re-run; rung 2 moves to quantile bins. **Next rung: masked-node verbalization** (describe a segment whose own features are hidden, from tree context) — free and inferential, run before the paid supervision decision because it measures how much descriptive ability tree context provides for free. Pre-registrations: per-slot bars derived from Gate 0's measured predictability ceilings (not a uniform 0.90); adjacent-bin/soft accuracy reported alongside exact-match; control = context-conditioned vs. priors-only; split by video. Transcript census: ~36% coverage, non-speech-dominated by design ⇒ transcripts are auxiliary signal, not primary supervision; the rung-2 source decision follows masked-rung results.
