"""
Milestone 8 (Triton): the same online-softmax attention, written in Triton, plus a
PyTorch custom op and a head-to-head benchmark against
torch.nn.functional.scaled_dot_product_attention (SDPA).

THE POINT OF THIS FILE is the third implementation of one algorithm:

    rtl/flash_top.sv          online softmax in fixed-point hardware
    cuda/06_attention_flash.cu  online softmax in CUDA C, thread-block per query
    python/06_triton_attention.py  online softmax in Triton, tile-per-program

Same recurrence in all three:
    m_new = max(m_run, max_j s_j)
    corr  = exp(m_run - m_new)
    l_run = l_run*corr + sum_j exp(s_j - m_new)
    acc   = acc*corr   + sum_j exp(s_j - m_new) * V_j

What differs is only who owns the running state and how the reduction happens:
  RTL    : one register file, a comparator chain, explicit cycle-by-cycle FSM
  CUDA   : one thread block, warp shuffles, explicit __syncthreads()
  Triton : one program, tl.max over a tile axis, barriers implied by the
           block-level type system

Triton buys back the thing CUDA makes you do by hand -- a `tl.load` of a 2-D
block is one statement and the compiler picks the vectorization, the layout and
the barriers. What it does NOT buy back is the algorithm: if you write the
two-pass softmax here it will be just as memory-bound as M3 was.

HONEST CLAIM POLICY: this benchmarks against SDPA, which dispatches to the real
FlashAttention-2 / cuDNN kernels. Do NOT expect to beat it. The number worth
reporting is "% of SDPA", and anything above ~60% for a hand-written kernel at
these shapes is a respectable result. Reaching parity needs tensor cores
(WMMA/MMA), which this does not use.

Run on the GPU box:
  pip install triton
  python3 python/06_triton_attention.py            # correctness + benchmark
  python3 python/06_triton_attention.py --check    # correctness only
"""

import argparse
import math
import sys

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False


# =============================================================================
# CHUNK 1: the Triton kernel
# -----------------------------------------------------------------------------
if HAVE_TRITON:

    @triton.jit
    def _flash_fwd(
        Q, K, V, O,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, M, Nkv,
        sm_scale,
        BLOCK_M: tl.constexpr,      # query rows per program
        BLOCK_N: tl.constexpr,      # key columns per streamed tile
        BLOCK_D: tl.constexpr,      # head dim (power of two, padded)
        CAUSAL: tl.constexpr,
    ):
        """One program computes BLOCK_M query rows for one (batch, head) pair.

        The running state (m_i, l_i, acc) lives in REGISTERS the whole time --
        that is the entire memory argument for online softmax. The N x N score
        matrix is never written anywhere; only a BLOCK_M x BLOCK_N tile of it
        exists, and only inside the loop body.
        """
        start_m = tl.program_id(0)
        off_zh  = tl.program_id(1)
        off_z   = off_zh // H
        off_h   = off_zh % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = (Q + off_z*stride_qz + off_h*stride_qh
                  + offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qk)
        # Load the query tile ONCE and keep it resident for every key tile.
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

        # ---- running state, in registers -----------------------------------
        m_i   = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i   = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc   = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Causal: this program's rows are all < start_m*BLOCK_M + BLOCK_M, so
        # every key tile beyond that is entirely masked. Stop early rather than
        # compute-and-discard -- that is the ~2x causal speedup.
        hi = tl.minimum((start_m + 1) * BLOCK_M, Nkv) if CAUSAL else Nkv

        for start_n in range(0, hi, BLOCK_N):
            n_idx  = start_n + offs_n
            k_ptrs = (K + off_z*stride_kz + off_h*stride_kh
                      + n_idx[:, None]*stride_kn + offs_d[None, :]*stride_kk)
            v_ptrs = (V + off_z*stride_vz + off_h*stride_vh
                      + n_idx[:, None]*stride_vn + offs_d[None, :]*stride_vk)
            k = tl.load(k_ptrs, mask=n_idx[:, None] < Nkv, other=0.0)
            v = tl.load(v_ptrs, mask=n_idx[:, None] < Nkv, other=0.0)

            # scores for this tile: [BLOCK_M, BLOCK_N]
            s = tl.dot(q, tl.trans(k)) * sm_scale
            s = tl.where(n_idx[None, :] < Nkv, s, float("-inf"))
            if CAUSAL:
                s = tl.where(offs_m[:, None] >= n_idx[None, :], s, float("-inf"))

            # ---- the rebase, exactly as in the RTL and the CUDA kernel -------
            m_new = tl.maximum(m_i, tl.max(s, 1))
            corr  = tl.exp(m_i - m_new)          # <= 1; 0 on the first tile
            p     = tl.exp(s - m_new[:, None])

            l_i   = l_i * corr + tl.sum(p, 1)
            acc   = acc * corr[:, None] + tl.dot(p.to(v.dtype), v)
            m_i   = m_new

        # ---- ONE division, at the end ---------------------------------------
        acc = acc / l_i[:, None]

        o_ptrs = (O + off_z*stride_oz + off_h*stride_oh
                  + offs_m[:, None]*stride_om + offs_d[None, :]*stride_ok)
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < M)


# =============================================================================
# CHUNK 2: the PyTorch-facing wrapper (the "custom op")
# -----------------------------------------------------------------------------
def flash_attention(q, k, v, causal=False):
    """Drop-in for F.scaled_dot_product_attention (forward only).

    Shapes: (Z, H, M, D) -- batch, heads, sequence, head dim. Same convention
    as SDPA so the two can be swapped in a benchmark without reshaping, which
    is how you avoid accidentally timing a transpose.
    """
    if not HAVE_TRITON:
        raise RuntimeError("triton is not installed; run this on the GPU box")
    assert q.is_cuda and q.dim() == 4, "expected a CUDA tensor of shape (Z,H,M,D)"
    Z, H, M, D = q.shape
    Nkv = k.shape[2]
    o = torch.empty_like(q)

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(M, BLOCK_M), Z * H)

    _flash_fwd[grid](
        q, k, v, o,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(),
        Z, H, M, Nkv,
        1.0 / math.sqrt(D),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o


# =============================================================================
# CHUNK 3: correctness vs SDPA
# -----------------------------------------------------------------------------
def check():
    torch.manual_seed(0)
    fails = 0
    print(f"{'Z':>3} {'H':>3} {'M':>6} {'D':>5} {'causal':>7} {'dtype':>8} "
          f"{'max_err':>11}  result")
    for dtype, tol in ((torch.float16, 2e-2), (torch.float32, 2e-4)):
        for (Z, H, M, D) in ((1, 1, 128, 64), (2, 4, 512, 64),
                             (1, 8, 1024, 128), (1, 2, 300, 64)):
            for causal in (False, True):
                q = torch.randn(Z, H, M, D, device="cuda", dtype=dtype)
                k = torch.randn(Z, H, M, D, device="cuda", dtype=dtype)
                v = torch.randn(Z, H, M, D, device="cuda", dtype=dtype)
                mine = flash_attention(q, k, v, causal=causal)
                ref = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=causal)
                err = (mine.float() - ref.float()).abs().max().item()
                ok = err < tol
                fails += (not ok)
                print(f"{Z:>3} {H:>3} {M:>6} {D:>5} {str(causal):>7} "
                      f"{str(dtype).split('.')[-1]:>8} {err:>11.3e}  "
                      f"{'PASS' if ok else 'FAIL'}")
    # M=300 is deliberately not a multiple of BLOCK_M: the masked tail tile is
    # where boundary bugs live, and it is the case a clean power-of-two-only
    # test sweep would never catch.
    return fails


# =============================================================================
# CHUNK 4: benchmark vs SDPA
# -----------------------------------------------------------------------------
def bench():
    print(f"\n{'M':>6} {'D':>5} {'causal':>7} {'triton ms':>11} {'sdpa ms':>10} "
          f"{'% of sdpa':>10} {'TFLOP/s':>9}")
    for (Z, H) in ((1, 16),):
        for M in (128, 512, 2048, 4096):
            for D in (64, 128):
                for causal in (False, True):
                    q = torch.randn(Z, H, M, D, device="cuda", dtype=torch.float16)
                    k = torch.randn_like(q)
                    v = torch.randn_like(q)

                    def t(fn):
                        # median of triton.testing.do_bench, which handles
                        # warm-up and L2 flushing -- timing a cold cache or a
                        # JIT compile is the classic way to publish a wrong
                        # speedup.
                        return triton.testing.do_bench(fn, warmup=25, rep=100)

                    ms_t = t(lambda: flash_attention(q, k, v, causal=causal))
                    ms_s = t(lambda: torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, is_causal=causal))

                    pairs = M*(M+1)/2 if causal else M*M
                    flops = 2 * 2 * Z * H * pairs * D      # QK^T + PV
                    tflops = flops / (ms_t * 1e-3) / 1e12
                    print(f"{M:>6} {D:>5} {str(causal):>7} {ms_t:>11.4f} "
                          f"{ms_s:>10.4f} {100*ms_s/ms_t:>9.1f}% {tflops:>9.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="correctness only")
    args = ap.parse_args()

    if not HAVE_TRITON:
        sys.exit("triton not installed. On the GPU box: pip install triton")
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible to torch")

    print(f"torch {torch.__version__}  triton {triton.__version__}  "
          f"gpu {torch.cuda.get_device_name(0)}\n")
    fails = check()
    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILURES'}")
    if not args.check:
        bench()
