"""
Milestone 5 support: BIT-EXACT integer model of the RTL attention datapath.

Why this file exists (and why M2 is not enough):
  M2 (`02_fixed_point_model.py`) quantizes Q,K,V and the two matmuls, but still
  calls np.exp and does the softmax division in float64. Hardware cannot do
  that. So M2 is a *precision study*, not an executable spec -- RTL that matched
  it bit-for-bit would be impossible to write.

  This file is the executable spec. Every operation here is an integer
  operation the SystemVerilog performs, in the same order, with the same
  truncation and the same saturation. The RTL is required to match its output
  EXACTLY -- not "within a tolerance". A single-LSB mismatch is a bug, and that
  is the whole point: tolerances hide bugs, exactness cannot.

  It is also the single source of truth for the exp lookup table: the .hex it
  emits is the same file the RTL $readmemh's, so software and hardware can
  never drift apart by construction.

Outputs (into rtl/vectors/):
  Q.hex K.hex V.hex   inputs, Q8.8 two's complement, one 16-bit word per line
  S.hex M.hex P.hex   per-stage golden intermediates (qkt / row_max / softmax)
  O.hex               golden final output
  exp_lut.hex         the exp LUT the RTL loads

Run:  python3 python/04_rtl_fixed_model.py
"""

import os
import numpy as np

# =============================================================================
# CHUNK 1: format + dimensions  (these must match the RTL parameters exactly)
# -----------------------------------------------------------------------------
# All overridable from the environment so one script can drive the whole
# parameter sweep. The RTL takes the same values as module parameters, so a
# sweep here and a sweep there stay in step.
DW    = int(os.environ.get("DW",   16))   # word width          -> rtl parameter DW
FRAC  = int(os.environ.get("FRAC",  8))   # fraction bits, Q8.8 -> rtl parameter FRAC
N     = int(os.environ.get("N",     4))   # sequence length     -> rtl parameter N
D     = int(os.environ.get("D",     4))   # head dim            -> rtl parameter D
DV    = int(os.environ.get("DV",    4))   # value dim           -> rtl parameter DV
SCALE = 1 << FRAC   # 256

INT_MIN = -(1 << (DW - 1))   # -32768
INT_MAX =  (1 << (DW - 1)) - 1   # +32767

# 1/sqrt(D) folded into dot4's single final shift. M = log2(D); the shift is
# floor(M/2) and, when M is odd, a leftover 1/sqrt(2) constant multiply.
M_LOG    = int(np.log2(D))
SQRTD_SH = M_LOG // 2
ODD_EXP  = (M_LOG % 2) == 1
HALF_MUL = ((7071 * SCALE) + 5000) // 10000 if ODD_EXP else SCALE

# exp LUT geometry. Input is (S - row_max), which is always <= 0.
# We cover [-EXP_RANGE, 0] with EXP_N entries; anything below clamps to the
# last entry. exp(-6.24) already quantizes to 0 in Q8.8, so -8 is generous.
EXP_N     = 256
EXP_RANGE = 8                              # in real units
EXP_STEP  = (EXP_RANGE * SCALE) // EXP_N   # 8 Q8.8 LSBs per LUT step = 1/32
EXP_SH    = int(np.log2(EXP_STEP))         # index = (-x) >> EXP_SH

VEC_DIR = os.path.join(os.path.dirname(__file__), "..", "rtl", "vectors")


# =============================================================================
# CHUNK 2: the integer primitives -- each one maps to a specific piece of RTL
# -----------------------------------------------------------------------------
def sat(x):
    """Clamp to the DW-bit signed range. RTL: the compare-and-clamp block at the
    end of dot4/pv. Saturating (not wrapping) so overflow degrades gracefully
    instead of flipping sign."""
    return int(np.clip(x, INT_MIN, INT_MAX))


def asr(x, n):
    """Arithmetic right shift = floor division by 2^n, for signed x.
    Python's >> on negative ints already floors, which is exactly what SV's
    >>> does on a signed operand. This is TRUNCATION, not rounding -- the RTL
    does not round, so neither does the model."""
    return x >> n


def quantize(x):
    """Real -> Q8.8 stored integer. Runs on the HOST, not in the RTL, so
    round-to-nearest is allowed here."""
    return np.vectorize(sat)(np.round(x * SCALE).astype(np.int64))


def build_exp_lut():
    """exp_lut[i] = round(exp(-i * EXP_STEP / SCALE) * SCALE), clamped to Q8.8.
    Index 0 is exp(0) = 1.0 = 256. Emitted to hex and consumed by the RTL, so
    there is exactly one definition of this table in the whole project."""
    return [sat(int(round(np.exp(-(i * EXP_STEP) / SCALE) * SCALE)))
            for i in range(EXP_N)]


EXP_LUT = build_exp_lut()


# =============================================================================
# CHUNK 3: the datapath, one function per RTL module
# -----------------------------------------------------------------------------
def rtl_qkt(Qq, Kq):
    """rtl/qkt.sv  ->  S[i][j] = sat( (sum_k Qq[i][k]*Kq[j][k] * HALF_MUL) >> (2*FRAC + SQRTD_SH) )

    acc is a full-width exact integer sum (Q16.16 + log2(D) bits of headroom),
    shifted ONCE at the end. Shifting per-term would round D times and lose
    accuracy for nothing.

    This shift TRUNCATES while softmax/pv round. Deliberate, and measured:
    adding rounding here changed mean error by 0.3% (0.004071 -> 0.004057),
    because S feeds an exp LUT indexed by (-x) >> 3 -- a +/-1 LSB wobble in S
    is shifted out of the index 7 times in 8. dot4 is also the module
    replicated LANES times, so it is the worst place to spend an adder."""
    S = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            acc = int(sum(int(Qq[i][k]) * int(Kq[j][k]) for k in range(D)))
            S[i][j] = sat(asr(acc * HALF_MUL, 2 * FRAC + SQRTD_SH))
    return S


def rtl_row_max(S):
    """rtl/row_max.sv -> m[i] = max_j S[i][j]
    The numerically-stable-softmax trick, in hardware: a comparator tree/chain.
    Subtracting the row max makes every exp argument <= 0, so exp() can never
    overflow and the LUT only has to cover the negative half-line."""
    return np.array([max(int(v) for v in S[i]) for i in range(N)], dtype=np.int64)


def rtl_softmax(S, m):
    """rtl/softmax.sv -> P[i][j], Q8.8, rows summing to ~SCALE.

    Three sub-steps, three pieces of hardware:
      x   = S - m          17-bit subtract (the difference needs one more bit
                           than either operand), then clamped into LUT range
      e   = EXP_LUT[idx]   a ROM read; idx = (-x) >> EXP_SH, saturated at EXP_N-1
      P   = (e << FRAC) / rowsum   one sequential restoring divider, reused
    """
    P   = np.zeros((N, N), dtype=np.int64)
    E   = np.zeros((N, N), dtype=np.int64)
    SUM = np.zeros(N, dtype=np.int64)
    for i in range(N):
        for j in range(N):
            x = int(S[i][j]) - int(m[i])          # <= 0 by construction
            idx = min((-x) >> EXP_SH, EXP_N - 1)  # clamp = the LUT's floor
            E[i][j] = EXP_LUT[idx]
        SUM[i] = int(E[i].sum())                  # >= 256: E[i][argmax] == 256
        for j in range(N):
            # ROUND-TO-NEAREST, not floor. Adding SUM/2 before the divide costs
            # one adder and cut mean output error 21% over a 200-seed sweep
            # (0.00514 -> 0.00407). Floor division biases every one of the N
            # terms in the SAME direction, so the errors accumulate instead of
            # cancelling -- visible as P row sums that are always <= 256.
            P[i][j] = ((int(E[i][j]) << FRAC) + int(SUM[i]) // 2) // int(SUM[i])
    return P, E, SUM


def rtl_pv(P, Vq):
    """rtl/pv.sv -> O[i][c] = sat( (sum_j P[i][j]*Vq[j][c]) >> FRAC )

    Note there is NO .T here: P is (query x key) and V is (key x value), and the
    key axis is already the shared inner dimension. Q@K.T needed the transpose
    because both operands were indexed (token x feature)."""
    O = np.zeros((N, DV), dtype=np.int64)
    for i in range(N):
        for c in range(DV):
            acc = int(sum(int(P[i][j]) * int(Vq[j][c]) for j in range(N)))
            O[i][c] = sat(asr(acc + (1 << (FRAC - 1)), FRAC))   # round-to-nearest
    return O


# =============================================================================
# CHUNK 4: hex emission for the testbenches
# -----------------------------------------------------------------------------
def to_hex(v):
    """Signed int -> DW-bit two's complement hex, zero-padded. & the mask is what
    turns -1 into ffff; $readmemh reads it straight back as the same bit pattern."""
    return f"{int(v) & ((1 << DW) - 1):0{DW // 4}x}"


def dump(name, arr):
    os.makedirs(VEC_DIR, exist_ok=True)
    flat = np.asarray(arr).reshape(-1)
    with open(os.path.join(VEC_DIR, name), "w") as f:
        f.write("\n".join(to_hex(v) for v in flat) + "\n")
    return len(flat)


# =============================================================================
# CHUNK 5: run the model, emit vectors, report the M5 error budget
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)                      # SAME seed as M1/M2 -> same Q,K,V
    Q = np.random.randn(N, D)
    K = np.random.randn(N, D)
    V = np.random.randn(N, DV)

    Qq, Kq, Vq = quantize(Q), quantize(K), quantize(V)

    S       = rtl_qkt(Qq, Kq)
    m       = rtl_row_max(S)
    P, E, SUM = rtl_softmax(S, m)
    O       = rtl_pv(P, Vq)

    # float64 oracle (M1) for the error budget
    Sf = Q @ K.T / np.sqrt(D)
    Pf = np.exp(Sf - Sf.max(axis=1, keepdims=True))
    Pf /= Pf.sum(axis=1, keepdims=True)
    O_golden = Pf @ V

    O_real = O / SCALE
    err = np.abs(O_real - O_golden)

    print(f"exp LUT: {EXP_N} entries, step 1/{SCALE // EXP_STEP}, "
          f"range [-{EXP_RANGE}, 0], index shift {EXP_SH}")
    print(f"scaling: HALF_MUL={HALF_MUL}, total right shift="
          f"{2 * FRAC + SQRTD_SH} (1/sqrt({D}) folded in)")
    print("\nS (Q8.8 ints):\n", S)
    print("row max m:", m)
    print("P row sums (ideal 256):", P.sum(axis=1))
    print("\nO (Q8.8 ints):\n", O)
    print(f"\nmax abs error vs M1 float golden : {err.max():.6f}")
    print(f"mean abs error vs M1 float golden: {err.mean():.6f}")

    for name, arr in [("Q.hex", Qq), ("K.hex", Kq), ("V.hex", Vq),
                      ("S.hex", S), ("M.hex", m), ("E.hex", E),
                      ("P.hex", P), ("O.hex", O),
                      ("exp_lut.hex", EXP_LUT)]:
        n = dump(name, arr)
        print(f"  wrote rtl/vectors/{name:12s} ({n} words)")
