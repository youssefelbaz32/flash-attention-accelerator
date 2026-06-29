# 01 — Python Floating-Point Golden Model

## Milestone
- **Milestone:** M1 — Python float attention (the "golden truth")
- **Date:** 2026-06-13
- **Goal:** Compute single-head scaled dot-product attention in float64 for toy
  dimensions, print all intermediates, and self-check the softmax property.
  This output is the reference every later level (fixed-point, CUDA, RTL) is
  diffed against.

## Operation
```
S = Q @ K^T / sqrt(d)     # (N,d)·(d,N) -> (N,N)   contracts d (features)
P = softmax(S)            # row-wise, each row sums to 1
O = P @ V                 # (N,N)·(N,d_v) -> (N,d_v) contracts N (tokens)
```

## Dimensions
| Symbol | Value | Meaning |
|--------|-------|---------|
| N   | 4 | number of tokens (rows) |
| d   | 4 | query/key feature dim (contracted away in S) |
| d_v | 4 | value feature dim (survives into O) |

## Shapes
| Tensor | Shape | Notes |
|--------|-------|-------|
| Q | (4,4) | queries |
| K | (4,4) | keys |
| V | (4,4) | values |
| S | (4,4) | scores; entry S[i][j] = "token i attends to token j" |
| P | (4,4) | softmax(S); **each row sums to 1** |
| O | (4,4) | attention output; each row = weighted blend of V rows |

## Files
- `python/01_golden_model.py`

## How to run
```bash
/Users/yelhagra/anaconda3/bin/python3 python/01_golden_model.py
```
- Seed: `np.random.seed(0)` (fixed → reproducible inputs; this is the contract
  that makes cross-level diffs meaningful).

## Self-check (first test in the project)
```python
assert np.allclose(P.sum(axis=1), 1.0)   # softmax rows must sum to 1
```
- **Pass criterion:** script exits without AssertionError.
- **Result:** PASS (2026-06-13).

## Verification by inspection
- Row sums of P ≈ 1.0  ✓
- Softmax preserves rank: largest score in a row → largest probability  ✓
  (e.g. S[1] argmax = idx 2 → P[1] argmax = idx 2)
- O leans toward the V row that P weights most (O[1] pulled toward V[2])  ✓

## Key concepts locked in
- A matmul **eats its shared inner dimension** and keeps the two outer ones.
- `Q @ K.T` contracts `d` → token×token scores. `Q.T @ K` would contract `N`
  → feature×feature (covariance-like), NOT attention.
- `1/sqrt(d)` scaling: dot products grow ~`sqrt(d)`; scaling keeps softmax from
  saturating (numerical conditioning).
- Stable softmax subtracts the **row max** before exp; mathematically identical
  (`softmax(x) == softmax(x-c)`) but prevents `exp()` overflow. This is the
  motivation for the future RTL `row_max_engine`.
- `keepdims=True` keeps reductions as `(N,1)` so they broadcast **down rows**.
  Dropping it gives `(N,)`, which broadcasts against the wrong axis silently.
- `axis=1` reduces across the row (per-token). `axis=0` would be the classic
  softmax-axis bug (columns sum to 1 instead of rows).

## Common failure modes (watch for these later)
- Softmax on wrong axis → caught by the row-sum assert.
- Missing transpose → wrong values (at N=d=4 the shape won't even warn you).
- Missing `1/sqrt(d)` → right shape, wrong magnitudes.
- Forgetting `keepdims` → silently-wrong P, no crash.

## Next
- M2: Python fixed-point model (Q8.8 / Q4.12), compare against this golden
  output, track error stats.
