"""
Milestone 2: Fixed-point model of single-head attention.

Goal: reproduce the integer arithmetic the FPGA will do, in Python, and measure
how far the fixed-point result drifts from the M1 float64 golden model.

Strategy:
  - Pick a Q-format (FRAC fraction bits). Start with Q8.8 -> FRAC = 8.
  - quantize(x)   : real  -> stored integer   (round(x * 2^FRAC))
  - dequantize(q) : integer -> real           (q / 2^FRAC)
  - Run the SAME attention math, but quantize after each stage so values live
    on the fixed-point grid, then compare O_fixed vs O_golden.

We reuse the golden model's seed/inputs so both see identical Q,K,V.

Write CHUNK BY CHUNK. Do not fill everything at once.
"""

import numpy as np

# Same inputs as the golden model (identical seed => identical Q,K,V).
np.random.seed(0)
N, d, d_v = 4, 4, 4
Q = np.random.randn(N, d)
K = np.random.randn(N, d)
V = np.random.randn(N, d_v)

# =============================================================================
# CHUNK 1: fixed-point format + quantize / dequantize helpers
#   FRAC  = number of fraction bits (8 for Q8.8)
#   SCALE = 2 ** FRAC
#   quantize(x)   = round(x * SCALE) as integer
#   dequantize(q) = q / SCALE
# -----------------------------------------------------------------------------
# TODO(you): FRAC, SCALE, then the two helper functions
FRAC = 8
SCALE = 2 ** FRAC


# =============================================================================
# CHUNK 2: (coming next) saturation helper — clamp integers to the format range
# -----------------------------------------------------------------------------
BITS = 16 #total bits
INT_MIN = -2**(BITS-1)
INT_MAX = 2**(BITS-1) - 1

def saturate(q):
    return np.clip(q, INT_MIN, INT_MAX) #saturates at min or at max if out of range

def quantize(x):
    return saturate(np.round(x * SCALE).astype(int))

def dequantize(q):
    return q / SCALE



# =============================================================================
# CHUNK 3: (later) fixed-point attention: S, softmax, O on the fixed grid
# -----------------------------------------------------------------------------
Qq = quantize(Q)            # (N,d) integers, Q8.8
Kq = quantize(K)            # (N,d) integers, Q8.8
S_raw = Qq @ Kq.T           # integer matmul -> Q16.16 results (wide accumulator)
S_fixed = (S_raw / (SCALE * SCALE)) / np.sqrt(d)

m = np.max(S_fixed, axis=1, keepdims=True)
e = np.exp(S_fixed - m)
P_fixed = e / np.sum(e, axis=1, keepdims=True)
Vq = quantize(V)
Pq = quantize(P_fixed)

O_raw = Pq @ Vq  # integer matmul -> Q16.16 results (wide accumulator)

O = O_raw / (SCALE * SCALE) # back to float for comparison

# =============================================================================
# CHUNK 4: (later) golden float attention for reference + error metrics
# -----------------------------------------------------------------------------
S = Q @ K.T / np.sqrt(d)
P = np.exp(S - np.max(S, axis=1, keepdims=True))
P /= np.sum(P, axis=1, keepdims=True)
O_golden = P @ V

err = np.abs(O - O_golden)
max_abs = np.max(err)
mean_abs = np.mean(err)
#print each label here
print(f"Max absolute error: {max_abs}")
print(f"Mean absolute error: {mean_abs}")
