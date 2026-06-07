# Mnemo: A Design Proposal for a Tiny, Continually-Learning Perceptual AI

*Moving-number representations, ternary weights, and a commit-governed "mold."*

> **Status.** This is a design proposal, not a results paper. The perception layer (§2) is built and validated end-to-end on real video. The "mind" (§3–§6) is designed but **unbuilt**, and its continual-learning core (§5) touches genuinely open research problems. This document exists to be argued with — §10 lists the questions where critique is most useful.

---

## 1. What this is

Mnemo is an attempt to build a *small, self-contained AI that perceives video and reasons about what it perceived* — built from scratch rather than wrapped around a pretrained large model. It is one component of a larger local-first agent stack:

- **Mnemo** is the *mind* — the model itself.
- **A harness** (internally, "KERN") supplies autonomy: the runtime loop, memory, the persistent body the model lives in. (Roughly: the harness is to Mnemo what an agent framework is to a chat model.)
- **A tool layer** (a suite of local MCP servers for analysis tasks) is what the mind *calls*.

This paper concerns only the mind. Four constraints are treated as hard requirements:

1. **Tiny** — "bit-small," must fit a small runtime.
2. **Fast.**
3. **Efficient.**
4. **Dynamic** — the load-bearing constraint, and the one most easily misread.

A clarification on (4), because it changes everything downstream. *Dynamic here does **not** mean a frozen model that allocates its fixed compute conditionally at inference* (e.g. Mixture-of-Depths, early-exit, mixture-of-experts routing). It means a model that **keeps learning after deployment** — whose parameters, and ideally whose structure, change over its operational lifetime. This is continual / lifelong learning, and it is the hard part of the proposal.

---

## 2. Perception: video as a small "tree of meaning" (built)

Before the mind can reason about video, video must be made small enough to reason *about*. A token-budget-constrained model cannot ingest thousands of raw frames.

Mnemo's perception is a bio-inspired hierarchical compressor. Per-second signals (visual sharpness, body-pose motion magnitude, audio loudness) roll up into ~5-second **segments**, segments into ~30-second **scenes**, scenes into **chapters**, chapters into one **meta** node. A 19-second clip becomes ~7 nodes; a 5-minute video, ~70. The result is a *tree of meaning* at multiple zoom levels.

This layer is built and validated. The three input signals were specifically hardened so they carry honest, varying information rather than noise (e.g. motion is normalized to mean-per-joint displacement so camera shake doesn't dominate; action labels fire only when the relevant body landmarks are well-tracked; audio is real RMS/dBFS loudness). On a first corpus of 10 diverse public videos the signals show wide spread (segment importance 0.17–0.85, motion across ~2.5 orders of magnitude, audio −80 to −10 dBFS), which is the precondition for there being anything learnable.

**Input representation to the mind.** Each tree node becomes **one token**, embedded from a small feature vector (importance, level, relative timing, aggregated motion/audio/visual scalars, action multi-hot). A 70-node tree is 70 tokens — not thousands of tokens of serialized text. The compression that makes vision affordable to a tiny model is realized again at the input layer: the model spends no capacity parsing a serialization format.

---

## 3. The problem: dynamic learning

A neural network stores everything it knows in one shared set of weights — there is no per-memory "folder." Training on something new moves the weights to reduce current error, and in doing so overwrites the configurations that encoded older knowledge. The network has no incentive to preserve the old. This is **catastrophic forgetting**: learn task B and performance on task A collapses.

Underneath it sits the **stability–plasticity dilemma**: to learn, weights must be free to change (plasticity); to remember, they must stay put (stability). These conflict directly. Every continual-learning method is a different compromise on that axis; none wins both.

Conventional large models sidestep this entirely by *not* learning continually: train once on a fixed corpus, freeze, deploy. That is the correct engineering choice for a static assistant — but it is exactly what constraint (4) forbids. A model that lives in a persistent harness and is meant to accumulate experience cannot be a frozen statue. So Mnemo has to confront the dilemma rather than dodge it.

---

## 4. Proposed substrate

### 4.1 Ternary weights ("bit-small")

Weights are constrained to {−1, 0, +1} (≈1.58 bits each), as in the BitNet b1.58 line. Matrix multiplication reduces to additions, memory footprint drops ~10× versus FP16, and a 10–50M-parameter model lands at ~2–10 MB. The host stack already has a custom AVX2 ternary matmul kernel, so the hardest low-level piece exists. This answers *tiny*, and most of *fast/efficient*, on the inference side.

### 4.2 Moving numbers (values as clouds)

The novel representational claim. A value need not be a fixed point; it can be a *range of possibility* — a distribution (e.g. mean + variance) that stays open during computation and collapses to a concrete value only when the system must commit (act, store, or self-modify). Four properties define the semantics:

- **Graded uncertainty.** Variance is the "how-committed" dial. Low variance ≈ hardened; high variance ≈ still a cloud.
- **Contagion (propagation).** If a computation depends on an unresolved value, the result is unresolved too. A conclusion standing on a cloud is itself a cloud. This is the mechanism by which fluidity propagates through reasoning instead of being silently dropped.
- **Commit at boundaries.** The system does not block waiting for resolution. It computes fluidly and, at boundaries (output, storage, self-modification), commits whatever resolved and carries the rest forward as still-open. The *commit policy* — when exactly a cloud collapses, and what is allowed to modify already-committed state — is the central engineering question (see §8).
- **Conservation.** Uncertainty is never silently lost. An operation over partial state must leave a trace that the result is partial (an audit receipt). This is the basis for the reversible, logged transitions in §6.

A speculative but appealing unification — stated carefully, because the obvious version is a category error. It is tempting to call **0 the abstain state** in a ternary weight: hold the weight as a distribution over {−1, 0, +1} during learning, collapse it at consolidation, and an uncommitted weight seems to "land on 0." The intuition is seductive for a real reason — a maximally-undecided (symmetric) posterior over {−1, 0, +1} has **expected value 0**, so an abstaining weight does read as ~0 *in expectation*. But the converse fails: a posterior sharply *peaked* on 0 also has expected value 0 and is the **opposite** of abstaining — a confidently-pruned weight. Expected-value-0 is necessary, not sufficient. Abstain is therefore a property of the posterior's **entropy/concentration**, not its value; 0 is merely the do-nothing value an open weight contributes while it waits, not the marker of openness. With that correction the unification holds — moving numbers, ternary substrate, and unknown-value semantics fuse into one mechanism graded by posterior entropy, resting at 0 when undecided — and the commitment dial used anywhere (including the §5 periphery) is entropy, never |expected value| or P(0). It is unverified.

### 4.3 Intellectual ancestry of the moving-number idea

These semantics were first worked out, independently, at the *programming-language* level in a separate systems language (Cx). There, any value of any type can hold an explicit `unknown` state (`?`): it propagates through operations (contagion), resolves at scope boundaries without blocking, conserves itself (an unknown cannot be consumed by arithmetic, so it leaves a remainder), and uses three-valued (Kleene) logic for booleans via a dedicated ternary `{false, true, unknown}` representation. Mnemo borrows these *semantics*, not the language — lifting a value-level uncertainty model up into the internal representation of a model. Note the difference: Cx's unknown is **binary** (resolved or not); Mnemo's cloud is **graded** (a width). Cx supplies the state machine and propagation rules; the mind adds the dial.

---

## 5. Continual learning via complementary systems

The proposal does not claim to *solve* catastrophic forgetting. It proposes to *manage* it using the brain's own strategy — Complementary Learning Systems — and to exploit structural advantages specific to this system.

**The split:**

- **A stable core** (the "neocortex") — ternary, slow-changing, holds consolidated knowledge, runs inference cheaply. Where *tiny/fast* lives.
- **A small plastic periphery** (the "hippocampus") — full-precision, fast, mutable. Where continual learning actually happens. Small enough to stay efficient. May *grow* capacity when it meets something it cannot represent — the non-static-structure requirement.
- **Consolidation ("sleep")** — periodically distills the periphery into the core and frees it, with regularization (EWC-style importance weighting) protecting what the core already knows.

This also resolves a tension between constraints (1) and (4): ternary weights cannot be smoothly nudged, so a fully-ternary model cannot learn online. Keeping only the small periphery plastic and full-precision sidesteps that — the cheap bulk stays ternary; learning happens in a small mutable head.

**The structural advantage:** continual-learning research routinely struggles to obtain past data to replay without storing everything. Mnemo *produces compressed memories as a byproduct of perceiving* — the trees from §2. Those are the replay buffer. Replaying them to keep the core sharp on the past is essentially free and built-in. Compressed-representation replay is exactly what the memory tree is.

**Starting conservative:** the first build does not attempt full online plasticity. Freeze the core, let only a tiny plastic head adapt, replay trees in batches ("sleep"), protect the core with EWC, and measure *learn-vs-forget*. Then raise plasticity gradually and find how far it goes before forgetting bites.

---

## 6. The "mold": commit-governed stability

A biological mind has a body holding it stable; a model has no such container. It needs an artificial one — but the requirement is subtle:

- Too fluid, and it collapses (no stable identity, learns and forgets noise).
- Too hard, and it cannot adapt.
- Never removed from the mold, and it cannot be used.

The proposed resolution is a **spectrum of timescales**, not a binary fluid/hard split:

| Layer | Fluidity | Persistence |
|---|---|---|
| Activations | fully fluid | never stored (per-thought) |
| Plastic periphery / short-term memory | fast | recent experience |
| Consolidated core | slow | protected |
| Identity / invariants | near-frozen | changed only via audited transitions |

This mirrors working memory → hippocampus → neocortex → core personality.

The "mold" itself is **not a single component** but an *active governance layer* — closer to homeostasis than to a skull. It watches the fluid state, decides what is ready to commit, performs consolidation, protects the core, and **logs every transition so it is reversible**. Concretely: harness + consolidation/sleep policy + audit log + protected invariants, coordinated. Each hardening event is treated like a **database transaction** — fluid in the working set, atomic on commit, logged, versioned, reversible.

Stated plainly, this is a *constitution for a mind*: a protected core, an amendment process, and a separation of the entrenched from the revisable.

---

## 7. Grounding in prior work

The components are individually real; the proposal's bet is their *unification*.

- **Extreme quantization:** BitNet / BitNet b1.58 (Ma et al., 2024, arXiv:2402.17764; 2B-parameter native model, 2025). Ternary {−1,0,+1} matches full-precision quality at scale.
- **Quantization-aware training:** standard practice keeps a full-precision "shadow" weight (the cloud) and quantizes on the forward pass (the hardening) — the moving-number idea already lives inside how quantized models are trained.
- **Uncertainty as representation:** Bayesian neural networks (weights as distributions); Fisher-information weighting. EWC (Kirkpatrick et al., PNAS 2017, doi:10.1073/pnas.1611835114) protects weights in proportion to importance — i.e. it hardens the important and frees the rest, which is the moving-number idea applied to continual learning.
- **Complementary Learning Systems:** McClelland, McNaughton & O'Reilly (1995), *Psychological Review* — the hippocampus/neocortex division of labor; the basis for replay-and-consolidate methods.
- **Continual learning families:** regularization (EWC/SI), replay/rehearsal, and architectural/parameter-isolation methods (see surveys, 2021–2025). All are compromises on stability–plasticity.
- **What this is *not*:** dynamic *compute* allocation — Mixture-of-Depths (Raposo et al., Google DeepMind, 2024, arXiv:2404.02258), the conditional-computation survey (arXiv:2403.07965), the dynamic-networks survey (Han et al., arXiv:2102.04906). These optimize a *frozen* model's inference cost and are orthogonal; they could be layered in for efficiency but do not provide the *learning* dynamism that is the goal.

The genuinely novel element is treating uncertainty, weights, memory, goals, and identity under **one** principle — fluid-until-commit, with differential hardening rates by timescale, governed by a commit/audit "mold" — rather than applying these ideas piecemeal and only to weights.

---

## 8. Honest open problems

1. **It manages the dilemma; it does not solve it.** Separating fluid from hard regions and gating transitions turns "balance learning and remembering inside one lump of weights" into "decide a policy for promoting fluid state into the protected core." That is a better fight — explicit, inspectable, reversible — but it is still a fight. The **promotion policy is the unsolved center.**
2. **Defining "identity" computationally.** "Important weights" is computable (Fisher information). "The numbers that *are* the core self" is not. The mold can protect a set once defined; *defining* the protected set is open and partly philosophical.
3. **Cost of clouds.** A distribution is ≥2 numbers where a point is one — in tension with "bit-small." Mitigation: fluidity is itself differential — clouds only in the plastic layers; the hardened core is points/ternary. Unverified that this is enough.
4. **Reversibility vs. true commitment.** Logged transitions allow rollback only if the log stores enough to reverse a consolidation — which costs storage. Tension between "genuinely hardened" and "reversible."
5. **Unproven combination at scale.** Each piece is validated alone (ternary at ~2B params; CLS in neuroscience and small models; replay broadly). Tiny + ternary + graded-uncertainty + continual, *together*, at a "bit-small" scale, is untested territory.
6. **The prerequisite no architecture removes.** None of this matters until a model can learn *anything* from the trees and produce output at all. That is ordinary supervised/self-supervised ML and comes first (§9).

---

## 9. What needs to be done

In strict dependency order. Almost everything sits downstream of one gate. *(The experiments that operationalize these tiers — exact tasks, baselines, splits, pass/fail — are specified in the companion `mnemo_experimental_plan.md`.)*

**Tier 1 — prove the ground holds (ordinary ML; tractable; immediate):**
1. *Featurizer* — tree → feature-token tensors. (In progress.)
2. *Tracer-bullet* — a tiny **ordinary** model on a self-supervised tree task, beating a trivial baseline. **The gate.** If a model cannot find structure in the trees, the approach is wrong, learned cheaply.
3. *Output model* — expand the corpus; train a model that produces language about what it saw. Mnemo "talks back," basically. Known work.

**Tier 2 — the substrate (design-led; buildable in layers):**
4. Ternary weights (BitNet-style; AVX2 kernel exists).
5. Moving-number / uncertainty semantics (the four properties of §4.2; test the ternary-0-as-uncommitted hunch).
6. Composition: *a weight is a cloud — a distribution over {−1,0,+1} — that collapses at consolidation.*

**Tier 3 — the dynamic mind (research frontier; managed, not solved):**
7. Continual learning: CLS split, trees as replay buffer, EWC-protected core; start with frozen core + tiny plastic head, measure learn-vs-forget, raise plasticity gradually. *(RESULT 2026-06: the conservative build is **validated** (≈4× less forgetting than naive, competitive with replay, no statue); replay is the workhorse. The ambitious per-weight wake-phase commitment mechanism is **falsified**: learned freezing landed below shuffled freezing, and a retrodiction showed the importance×stability score is **orthogonal** to next-task demand (ρ ≈ −0.05) — so the failure is **importance-weighted partial-freeze miscalibration**: pinning A's most sensitive weights to stale values while co-adapted neighbors drift harms A more than full drift. Retention is a property of coherent configurations, not individual weight values; replay wins because it protects in function space. Consequence: commitment, if revived, acts at **consolidation time** (entropy-gated absorption during the sleep/distill step), never as per-weight freezing during wake; entry exam is the single-episode consolidation screen in the companion plan.)*
8. The mold: commit-as-transaction, audit log, protected core, timescale spectrum.

Tier 1 is the next one or two work units. Tier 2 is design-led but buildable. Tier 3 is a long arc with real uncertainty. The moving-number idea is the soul of the proposal, but it lives in Tier 2 and is not built until Tier 1's gate opens.

---

## 10. Questions for discussion

The points where outside critique would be most valuable:

1. **The promotion policy (§8.1).** Is there a principled rule for *when a fluid value should be allowed to modify the protected core* — beyond confidence thresholds, action-forcing, and scheduled consolidation? This is the crux.
2. **Graded clouds on a ternary substrate.** Is "a weight is a distribution over {−1,0,+1} that collapses at consolidation" coherent and trainable, or does the discreteness of the target defeat the point of carrying a distribution? Does the *0-as-abstain* interpretation hold up?
3. **Trees-as-replay-buffer.** Compressed, lossy, perception-derived memories as the *sole* replay source for consolidation — sufficient to prevent forgetting, or does the lossiness of the compression bias what the core retains?
4. **Defining the protected "identity" set (§8.2).** Is there any tractable, computable handle on *which parts of a model constitute identity* that a governance layer could protect — or is this irreducibly a design-time hand-specification?
5. **Is the unification real or cosmetic?** Treating values, weights, memory, goals, and identity under one fluid-until-commit principle is elegant. Does the elegance buy anything engineering-wise, or do the layers actually want different mechanisms and the unification is just a satisfying frame?
6. **Scope sanity.** Given the genuinely unsolved core, is this the right *shape* of ambition for a small independent project — or is there a leaner version that captures most of the value (e.g. a static perception+reasoning model with bolt-on episodic memory, deferring continual learning entirely)?

---

*Prepared as a discussion document. Treat all claims about the unbuilt mind (§3–§8) as hypotheses, not findings.*
