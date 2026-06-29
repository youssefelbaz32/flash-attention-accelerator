# 02 — Python Fixed-Point Model

## Milestone
- **Milestone:** M2 — Python fixed-point attention (the FPGA arithmetic, in Python)
- **Date:** 2026-06-28
- **Goal:** Reproduce the integer arithmetic the FPGA will do, on a Q8.8 grid,
  and measure how far the fixed-point output drifts from the M1 float64 golden
  model. Same seed → identical Q/K/V, so every difference is real quantization
  error, never input drift.

## Operation (same math, but on the grid)
```
Qq,Kq = quantize(Q), quantize(K)         # real -> Q8.8 integers (saturated)
S_raw = Qq @ Kq.T                        # integer matmul -> Q16.16 accumulator
S_fixed = S_raw / 2^16 / sqrt(d)         # dequantize, then scale
P_fixed = softmax(S_fixed)               # row-max subtract, exp, normalize (float)
Vq,Pq = quantize(V), quantize(P_fixed)   # values + probs onto the grid
O_raw = Pq @ Vq                          # integer matmul -> Q16.16 accumulator
O      = O_raw / 2^16                     # dequantize back to real
```

## Fixed-point format
| Param | Value | Meaning |
|-------|-------|---------|
| FRAC  | 8 | fraction bits → Q8.8 |
| SCALE | 256 (`2^8`) | real→int multiplier (`round(x*SCALE)`) |
| BITS  | 16 | total stored width |
| INT_MIN / INT_MAX | −32768 / 32767 | saturation range (signed 16-bit) |

- **quantize(x)** = `saturate(round(x * SCALE))` → integer on the grid.
- **dequantize(q)** = `q / SCALE` → back to real.
- A Q8.8 × Q8.8 product is **Q16.16** (scale `2^16` = `SCALE*SCALE`); the matmul
  sum keeps that scale, so we divide the accumulator by `SCALE*SCALE` to get back
  to a real value. This is the single most important bookkeeping rule of the file.

## Files
- `python/02_fixed_point_model.py`

## How to run
```bash
/Users/yelhagra/anaconda3/bin/python3 python/02_fixed_point_model.py
```

## Result — drift vs golden (M1)
| Metric | Value | Reading |
|--------|-------|---------|
| Max absolute error  | **0.00631** | worst single output element (~0.6% of output scale) |
| Mean absolute error | **0.00205** | typical output element |

This is the **M2 headline number** — the error budget the CUDA and RTL versions
will be diffed against once they also quantize.

## The FPGA bug we deliberately surfaced
Quantizing a perfectly normalized softmax row breaks normalization:
```
Pq row sums (Q8.8): [256, 257, 255, 256]   ← should all be 256 (= 1.0)
```
- Each tiny probability gets rounded onto the integer lattice; the ±1 errors
  don't cancel, so the row no longer sums to exactly 1.
- Here it's ±1/256 ≈ 0.4% because N=4. **At N=128 you sum 128 rounding errors —
  it compounds.** Real designs fight this with more fraction bits or an explicit
  renormalization stage.

## Key concepts locked in
- **Grid bookkeeping:** multiplying two QA.B numbers doubles the fraction bits.
  You must divide the accumulator by `SCALE^2` after every fixed×fixed matmul.
- **Saturation before storage:** clamp to the 16-bit range *before* writing, or a
  large value silently wraps. `quantize` saturates internally so callers can't
  forget.
- **`P @ V` needs no transpose** (unlike `Q @ K.T`). P is (queries×keys), V is
  (keys×d_v); the shared *keys* axis is already aligned, so the matmul contracts
  it directly. `Q@K.T` needed `.T` because both were (tokens×d) and we contract d.
- **Quantization is lossy where values are small:** a prob of 0.003 → `round(0.003*256)=1`
  → 0.0039. Small attention weights are the first casualties.
- **Scope choice:** softmax `exp` is still computed in float (Chunk 3). Only
  QK^T and PV are truly on the grid. A later pass can move `exp` to a LUT to
  mirror the RTL exp table.

## Common failure modes (bugs hit + ones to watch)
- **Missing softmax normalization (HIT):** golden P was written as `exp(S-max)`
  with no `/sum`, so rows summed to ~1.7–2.9 and `O_golden` came out ~2× too big.
  Caught by checking `P.sum(axis=1)`. Fix: add the `/= np.sum(..., keepdims=True)`.
- **Forgetting to divide by `SCALE^2`** after a fixed×fixed matmul → output off
  by a factor of 256.
- **Quantizing V with the raw float V** instead of `Vq` → the PV stage isn't
  actually on the grid (the earlier draft did this; now fixed).
- **Dropping `keepdims`** in the row-max / row-sum → silently-wrong broadcast.
- **Overflow without saturation** → wrap-around to a huge negative number.

## Next
- M3: CUDA naive kernel — same toy dims, diffed against M1 golden and sanity-
  checked against this M2 error budget.
