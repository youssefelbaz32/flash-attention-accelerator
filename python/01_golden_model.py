"""
Milestone 1: Floating-point golden model for single-head scaled dot-product attention.

Target operation:
    S = Q @ K^T / sqrt(d)
    P = softmax(S)          # row-wise: each row of P sums to 1
    O = P @ V

Toy dimensions for hand-checkable debugging:
    N   = 4    # number of tokens (rows)
    d   = 4    # query/key feature dimension (gets contracted away in S)
    d_v = 4    # value feature dimension (survives into O)

This file is the "golden truth". Every later version (fixed-point, CUDA, RTL)
will be diffed against the numbers this script produces.

Write the code CHUNK BY CHUNK. Do not fill everything in at once.
"""

# =============================================================================
# CHUNK 1: imports + deterministic seed
#   - import NumPy with its conventional alias
#   - fix the random seed so Q/K/V are identical on every run
# -----------------------------------------------------------------------------
# TODO(you): write the two lines here
import numpy as np
np.random.seed(0)



# =============================================================================
# CHUNK 2: dimensions  (coming next)
#   N, d, d_v
# -----------------------------------------------------------------------------
N = 4 # number of tokens (rows)
d = 4 # query/key feature dimension (gets contracted away in S)
d_v = 4 # value feature dimension (survives into O)



# =============================================================================
# CHUNK 3: generate Q, K, V  (coming later)
# Q : (N, d)     K : (N, d)     V : (N, d_v)

# -----------------------------------------------------------------------------
Q = np.random.randn(N, d)
K = np.random.randn(N, d)
V = np.random.randn(N, d_v)


# =============================================================================
# CHUNK 4: scores S = Q @ K^T / sqrt(d)  (coming later)
# -----------------------------------------------------------------------------
S = Q @ K.T / np.sqrt(d)



# =============================================================================
# CHUNK 5: softmax -> P  (row-wise, with row-max subtraction)  (coming later)
# -----------------------------------------------------------------------------
max_ = np.max(S, axis=1, keepdims=True) # (N, 1)
exp_ = np.exp(S - max_) # (N, N)
sum_ = np.sum(exp_, axis=1, keepdims=True) # (N, 1)
P = exp_ / sum_ # (N, N)


# =============================================================================
# CHUNK 6: output O = P @ V  + sanity asserts + prints  (coming later)
# -----------------------------------------------------------------------------
O = P @ V
assert np.allclose(P.sum(axis=1), 1.0)
print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)
print("S:\n", S)
print("P:\n", P)
print("O:\n", O)