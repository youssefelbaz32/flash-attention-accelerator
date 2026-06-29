# Attention Accelerator — from float Python to FPGA RTL

A single-head **scaled dot-product attention** engine, rebuilt from scratch at
five levels of abstraction — each one diffed against the level above so every
optimization stays provably correct.

```
S = Q·Kᵀ / √d      →      P = softmax(S)      →      O = P·V
```

The goal is **interview-level understanding and self-debugging**, not raw speed.
Correctness comes before optimization at every step: a self-checking float
reference is the "golden truth," and every lower level must reproduce its output
within a measured error budget.

## Why this project

Most attention tutorials stop at a NumPy one-liner. This one follows the number
all the way down to the hardware: the same `softmax` that's one line in Python
becomes a row-max engine and an exp lookup table in RTL. Building each level
against a fixed oracle means any downstream mismatch is a *real bug*, never input
drift.

## Build order (correctness before optimization)

| # | Level | Status | Diffed against |
|---|-------|--------|----------------|
| M1 | Python float64 golden model | ✅ done | self-check (softmax rows = 1) |
| M2 | Python fixed-point (Q8.8) | ✅ done | M1 golden |
| M3 | CUDA naive kernel | 🔜 next | M1 golden |
| M4 | CUDA tiled (shared memory) | ⬜ planned | M1 golden + M3 |
| M5 | SystemVerilog RTL (FPGA) | ⬜ planned | M1 golden |
| M6 | FlashAttention-lite (online softmax) | ⬜ planned | M1 golden |

**Toy dimensions:** `N = d = d_v = 4` — small enough to hand-check every
intermediate. Scales to 16 → 32/64/128 later.

## Results so far

| Level | Metric | Value |
|-------|--------|-------|
| M1 float | softmax row-sum self-check | PASS |
| M2 Q8.8 fixed-point | max abs error vs golden | **0.0063** (~0.6%) |
| M2 Q8.8 fixed-point | mean abs error vs golden | **0.0021** |

M2 also surfaces a classic hardware gotcha: once softmax probabilities are
quantized, the rows no longer sum to exactly 1 (`Pq` row sums came out
`[256, 257, 255, 256]` instead of all `256`) — a normalization-drift issue that
compounds at large `N`.

## Repository layout

```
python/
  01_golden_model.py        # M1 — float64 reference (the oracle)
  02_fixed_point_model.py   # M2 — Q8.8 fixed-point twin + drift metrics
docs/
  01_python_golden_model.md # per-milestone write-ups
  02_python_fixed_point_model.md
  project_log.md            # running build log: what passed, bugs, lessons
```

## Running it

```bash
python3 python/01_golden_model.py     # prints Q,K,V,S,P,O; asserts softmax rows = 1
python3 python/02_fixed_point_model.py # prints max/mean abs error vs golden
```

Requires Python 3.11+ and NumPy. Inputs are seeded (`np.random.seed(0)`) so runs
are fully reproducible.

## Key ideas locked in along the way

- A matmul **eats its shared inner dimension**; `.T` placement decides which axis
  is contracted (`Q·Kᵀ` contracts features → token×token scores).
- `1/√d` scaling keeps softmax from saturating (numerical conditioning).
- **Stable softmax** subtracts the row max before `exp` — identical math, no
  overflow — which is exactly why an RTL `row_max_engine` will exist.
- Fixed-point bookkeeping: a `QA.B × QA.B` product is `Q2A.2B`; you divide the
  accumulator by `SCALE²` after every fixed×fixed matmul.
- Quantization hits **small values hardest** — tiny attention weights are the
  first casualties on the grid.

---

*Built as a learning project, milestone by milestone, with a full
[build log](docs/project_log.md) of bugs and lessons.*
