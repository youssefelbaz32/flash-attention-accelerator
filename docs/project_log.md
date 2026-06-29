# Project Log — Attention Accelerator (Python → CUDA → RTL)

Running log of what we built, what passed/failed, bugs, and lessons.

---

## 2026-06-13 — M1: Python float golden model ✅

**What we built:** `python/01_golden_model.py` — float64 single-head scaled
dot-product attention for N=d=d_v=4, built chunk by chunk:
1. imports + `np.random.seed(0)`
2. dimensions N, d, d_v
3. Q, K, V via `np.random.randn`
4. `S = Q @ K.T / np.sqrt(d)`
5. stable softmax (row-max subtract → exp → normalize) → P
6. `O = P @ V` + `assert np.allclose(P.sum(axis=1), 1.0)` + prints

**What passed:** assert passed; row sums = 1; softmax preserves rank; O leans
toward most-weighted V row. PASS.

**Bugs:** none this milestone.

**Assumptions:** random toy inputs (no trained weights); diagonal of S carries
no special structure in a random model.

**What I understand now:**
- matmul contracts the shared inner dim; `.T` placement decides which axis dies
- `Q@K.T` (token×token) vs `Q.T@K` (feature×feature)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
- `1/sqrt(d)` is numerical conditioning for softmax
- row-max subtraction = the reason `row_max_engine.sv` will exist
- `keepdims`/`axis=1` broadcasting traps in softmax

**Next step:** M2 — Python fixed-point model; compare to golden; error stats.

**Portfolio note:** "Started from a self-checking float reference so every lower
level has an unambiguous oracle — mismatches downstream are real bugs, never
input drift (fixed seed)."

---

## 2026-06-28 — M2: Python fixed-point model ✅

**What we built:** `python/02_fixed_point_model.py` — single-head attention on a
Q8.8 integer grid, diffed against the M1 golden model. Same seed → same Q/K/V.
Chunk by chunk:
1. format + helpers: `FRAC=8`, `SCALE=256`, `quantize`/`dequantize`
2. saturation: 16-bit range, `saturate()` clamps before storing
3. fixed-point forward path: `Qq,Kq → S_raw (int matmul) → S_fixed → softmax →
   P_fixed`, then `Vq,Pq → O_raw (int matmul) → O`
4. golden float `O_golden` recomputed locally (chose this over importing 01 —
   the `01_` filename isn't a valid module name) + error metrics

**What passed:** runs standalone; golden P rows sum to 1 after the fix.
- **Max abs error 0.00631**, **mean abs error 0.00205** vs golden. This is the
  M2 error budget.

**Bugs:**
- *Missing softmax normalization in golden* — wrote `P = exp(S-max)` with no
  `/sum`; golden P rows summed to 1.7–2.9 and `O_golden` was ~2× too big.
  Caught by inspecting `P.sum(axis=1)`. Fixed with `P /= np.sum(..., keepdims=True)`.
- *Earlier draft used raw float `V` (not `Vq`) in PV* — PV wasn't on the grid;
  fixed by quantizing both P and V before the integer matmul.

**FPGA issue surfaced (on purpose):** quantized softmax rows no longer sum to 1 —
`Pq` row sums = [256, 257, 255, 256] instead of all 256. ±1 rounding errors don't
cancel; compounds at large N. Renormalization / more frac bits is the real fix.

**What I understand now:**
- QA.B × QA.B → Q2A.2B; divide the accumulator by `SCALE^2` after each fixed×fixed matmul
- saturate *before* storing or values wrap silently
- `P @ V` needs no `.T` (keys axis already aligned); `Q @ K.T` does (contract d)
- small probabilities are the first casualties of quantization
- softmax `exp` is still float here — only QK^T and PV are truly on the grid

**Next step:** M3 — CUDA naive kernel; diff vs M1 golden, sanity-check against
this M2 error budget.

**Portfolio note:** "Built a bit-accurate fixed-point twin of the float model so
I could quote a concrete error budget (0.6% worst-case at Q8.8) before writing a
line of HDL — and caught the classic 'quantized softmax stops summing to 1' issue
in Python where it's cheap to see."
