"""
Milestone 3: Export data for CUDA kernel acceleration

Goal: float32 CUDA kernel will be used to accelerate the attention computation. We need to export the input data (Q, K, V) and the output data (O) to files so that the CUDA kernel can read them.

Write CHUNK BY CHUNK. Do not fill everything at once.
"""

import numpy as np

# Same inputs as the golden model (identical seed => identical Q,K,V).
np.random.seed(0)
#make/ensure data/ directory exists
import os
if not os.path.exists('data'):
    os.makedirs('data')



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

#cast to float32
Q = Q.astype(np.float32)
K = K.astype(np.float32)
V = V.astype(np.float32)
O = O.astype(np.float32)
#save to data/
np.save('data/Q.npy', Q)
np.save('data/K.npy', K)
np.save('data/V.npy', V)
np.save('data/O_golden.npy', O)
