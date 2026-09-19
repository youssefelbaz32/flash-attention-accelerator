"""
Milestone 6: bit-exact integer model of ONLINE (streaming) softmax.
FlashAttention-lite -- the spec for rtl/flash_top.sv.

WHY. M5 measured its own bottleneck: with LANES=4 the softmax stage was 449 of
515 cycles, 87% of runtime, and completely immune to adding multipliers. The
naive datapath also materializes the full N x N matrices S and P. Both problems
have the same root: softmax is defined over a whole row, so the naive design
waits for the whole row to exist.

THE TRICK. Softmax can be computed incrementally if you carry a running max and
retroactively fix up what you already accumulated. Streaming key block b:

    m_new = max(m_run, max_j s_j)
    corr  = exp(m_run - m_new)                 <= 1, the "forgetting factor"
    l_new = l_run * corr + sum_j exp(s_j - m_new)
    acc_c = acc_c * corr + sum_j exp(s_j - m_new) * V[j][c]

and only at the very end,  O[i][c] = acc_c / l.

`corr` is the whole idea. When a later block contains a bigger score, everything
accumulated under the old max is now scaled wrong -- by exactly exp(m_old-m_new),
a constant. So one multiply retroactively rebases the entire history. That is
why softmax, which looks irreducibly global, is actually streamable.

WHAT IT BUYS, concretely:
  storage  O(N^2) for S and P      ->  O(DV) accumulator + 2 scalars per row
  divides  N*N (one per P element) ->  N (one reciprocal per row)
At N=4 that is 4x fewer divides. At N=128 it is 128x, and the O(N^2) buffers
that would not have fit on the FPGA at all simply never exist.

The SAME exp LUT serves both the score exponentials and the correction factor,
so M6 adds no new ROM -- only a multiplier and two accumulators.

Run:  python3 python/05_online_softmax_model.py
"""

import os
import importlib.util
import numpy as np

# Reuse the M5 primitives so the two models cannot disagree about the format,
# the LUT, or the saturation rules. (The 04_ filename is not a legal module
# name, so load it by path rather than importing it.)
_spec = importlib.util.spec_from_file_location(
    "m5", os.path.join(os.path.dirname(__file__), "04_rtl_fixed_model.py"))
m5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m5)

DW, FRAC, N, D, DV = m5.DW, m5.FRAC, m5.N, m5.D, m5.DV
SCALE, EXP_LUT, EXP_SH, EXP_N = m5.SCALE, m5.EXP_LUT, m5.EXP_SH, m5.EXP_N
sat, asr = m5.sat, m5.asr

# Key-block size: how many keys are folded into the running state per step.
# BLK=1 is pure streaming; BLK=N degenerates to the naive one-shot softmax.
# In RTL this is the same knob as LANES -- one dot4 per key in the block.
BLK = int(os.environ.get("BLK", 2))   # overridable so the RTL sweep can
                                       # regenerate the spec per block size

# Reciprocal precision. r = round(2^RECIP_SH / l), then O = (acc * r) >> RECIP_SH.
# One divide per ROW instead of one per element; RECIP_SH must be generous
# enough that the reciprocal's own quantization stays below the output LSB.
RECIP_SH = 24

VEC_DIR = m5.VEC_DIR


def exp_lut_lookup(x):
    """Shared by scores and corrections. x is clamped to <= 0 first: the LUT only
    covers the negative half-line, which is the row-max trick paying rent."""
    if x > 0:
        x = 0
    idx = min((-x) >> EXP_SH, EXP_N - 1)
    return EXP_LUT[idx]


def online_attention(Qq, Kq, Vq):
    """One query row at a time; keys streamed in blocks of BLK.

    Running state per row -- this is ALL the storage the algorithm needs:
      m_run : running max of the scores seen so far          (Q8.8, 1 word)
      l_run : running sum of exp(s - m_run)                  (Q8.8, 1 word)
      acc   : running sum of exp(s - m_run) * V[j][:]        (Q16.16, DV words)
    """
    O = np.zeros((N, DV), dtype=np.int64)
    S_dbg = np.zeros((N, N), dtype=np.int64)     # scores, for the testbench
    l_dbg = np.zeros(N, dtype=np.int64)
    m_dbg = np.zeros(N, dtype=np.int64)

    for i in range(N):
        m_run = -(1 << (DW - 1))          # -32768: the identity for signed max
        l_run = 0                         # Q8.8
        acc   = [0] * DV                  # Q16.16

        for b in range(0, N, BLK):
            block = range(b, min(b + BLK, N))

            # --- scores for this block: identical hardware to M5's qkt --------
            s = {}
            for j in block:
                a = int(sum(int(Qq[i][k]) * int(Kq[j][k]) for k in range(D)))
                s[j] = sat(asr(a * m5.HALF_MUL, 2 * FRAC + m5.SQRTD_SH))
                S_dbg[i][j] = s[j]

            # --- the rebase ---------------------------------------------------
            m_new = max([m_run] + [s[j] for j in block])
            corr  = exp_lut_lookup(m_run - m_new)   # Q8.8, <= 256, == 256 if unchanged

            # Rescale the history. Rounding here (not truncation) matters: corr
            # is applied once per BLOCK, so a biased error compounds N/BLK times
            # down the row -- the one place in the design where an error feeds
            # back into itself.
            l_run = asr(l_run * corr + (1 << (FRAC - 1)), FRAC)
            for c in range(DV):
                acc[c] = asr(acc[c] * corr + (1 << (FRAC - 1)), FRAC)

            # --- fold the new block in ----------------------------------------
            for j in block:
                e = exp_lut_lookup(s[j] - m_new)     # Q8.8
                l_run += e                            # Q8.8
                for c in range(DV):
                    acc[c] += e * int(Vq[j][c])       # Q8.8 * Q8.8 -> Q16.16

            m_run = m_new

        # --- ONE divide for the whole row ------------------------------------
        # r is Q(RECIP_SH) reciprocal of a Q8.8 value; acc is Q16.16.
        # (Q16.16 * r) >> RECIP_SH lands back in Q8.8.
        r = ((1 << RECIP_SH) + (l_run >> 1)) // l_run
        for c in range(DV):
            O[i][c] = sat(asr(acc[c] * r + (1 << (RECIP_SH - 1)), RECIP_SH))

        m_dbg[i], l_dbg[i] = m_run, l_run

    return O, S_dbg, m_dbg, l_dbg


if __name__ == "__main__":
    np.random.seed(0)
    Q = np.random.randn(N, D)
    K = np.random.randn(N, D)
    V = np.random.randn(N, DV)
    Qq, Kq, Vq = m5.quantize(Q), m5.quantize(K), m5.quantize(V)

    O_flash, S_dbg, m_dbg, l_dbg = online_attention(Qq, Kq, Vq)

    # M5 naive path, same inputs
    S5 = m5.rtl_qkt(Qq, Kq)
    m5_max = m5.rtl_row_max(S5)
    P5, _, _ = m5.rtl_softmax(S5, m5_max)
    O_naive = m5.rtl_pv(P5, Vq)

    # M1 float oracle
    Sf = Q @ K.T / np.sqrt(D)
    Pf = np.exp(Sf - Sf.max(axis=1, keepdims=True)); Pf /= Pf.sum(axis=1, keepdims=True)
    O_golden = Pf @ V

    e_flash = np.abs(O_flash / SCALE - O_golden)
    e_naive = np.abs(O_naive / SCALE - O_golden)

    print(f"BLK={BLK}  RECIP_SH={RECIP_SH}  (one divide per row, not per element)")
    print(f"\nrow max m   : {m_dbg}   (matches M5 row_max: {np.array_equal(m_dbg, m5_max)})")
    print(f"row sum l   : {l_dbg}     Q8.8, ideal ~256 x rows")
    print(f"\nO online    :\n{O_flash}")
    print(f"O naive (M5):\n{O_naive}")
    print(f"max |online - naive| in LSBs: {np.abs(O_flash - O_naive).max()}")
    print(f"\nvs M1 float golden   online: max={e_flash.max():.6f} mean={e_flash.mean():.6f}")
    print(f"                      naive : max={e_naive.max():.6f} mean={e_naive.mean():.6f}")

    for name, arr in [("flash_O.hex", O_flash), ("flash_l.hex", l_dbg),
                      ("flash_m.hex", m_dbg)]:
        n = m5.dump(name, arr)
        print(f"  wrote rtl/vectors/{name:14s} ({n} words)")
