"""Gate 3 — ternarization (bit-small thesis, accuracy only).

Quantize the validated rung-1.5 masked-verbalization model's weights to
{-1,0,+1} via QAT (shadow FP master + straight-through), and measure the cost
against two arms (iso-shape and iso-footprint — the thesis is *bits*, not
parameters). A declared FP-kept list is enforced by audit. No kernel/throughput
numbers here (separate follow-up). torch stays inside the gate package.
"""
