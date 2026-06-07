"""Gate 2 — the "talks back" decoding gate. Rung 1: a tiny decoder conditioned
on tree tokens verbalizes a segment with a programmatic (templated) target, so
accuracy should be near-ceiling and the decisive control is an UNCONDITIONED
twin. torch is imported only inside model.py / run.py.
"""
