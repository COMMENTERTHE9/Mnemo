"""Tree-structure relations derived from parent_idx (numpy only).

The relation matrix is a function of tree topology alone, so it is precomputed
per tree and reused across the K z-draws in Control B. It feeds the rung-4
transformer's structural attention bias — a general parent/child/sibling
mechanism, not a query-specific shortcut.
"""
from __future__ import annotations

import numpy as np

REL_SELF = 0
REL_PARENT = 1     # j is i's parent
REL_CHILD = 2      # j is i's child
REL_SIBLING = 3    # i, j share a (non-root) parent
REL_OTHER = 4
N_RELATIONS = 5


def relation_matrix(parent_idx: list[int]) -> np.ndarray:
    """[n, n] int matrix rel[i, j] in {self, parent, child, sibling, other}.

    parent_idx[k] == -1 marks a root. Precedence: self > parent > child >
    sibling > other.
    """
    n = len(parent_idx)
    rel = np.full((n, n), REL_OTHER, dtype=np.int64)
    for i in range(n):
        pi = parent_idx[i]
        for j in range(n):
            if i == j:
                rel[i, j] = REL_SELF
            elif pi == j:
                rel[i, j] = REL_PARENT
            elif parent_idx[j] == i:
                rel[i, j] = REL_CHILD
            elif pi != -1 and pi == parent_idx[j]:
                rel[i, j] = REL_SIBLING
    return rel
