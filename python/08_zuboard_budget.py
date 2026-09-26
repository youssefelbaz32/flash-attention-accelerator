"""
M6/M7 planning for the actual target: Avnet ZUBoard 1CG (AMD XCZU1CG MPSoC).

Device facts this is built on (Avnet product brief / hardware user guide):
  PL   : ~81K logic cells, 216 DSP48E2 slices, 3.8 Mb block RAM
  PS   : dual Cortex-A53 + dual Cortex-R5F, 1 GB LPDDR4
  Boot : microSD or QSPI; official PYNQ v3.0.1 image from Avnet
  I/O  : gigabit Ethernet, USB 2.0 host, microUSB JTAG/UART

Two questions this answers, both of which decide M6:
  1. Does the link still dominate once the host is an A53 on the same die?
  2. At what sequence length does each datapath run out of block RAM?

Run:  python3 python/08_zuboard_budget.py
"""

DW_BYTES = 2
F_PL     = 150e6          # a realistic PL clock for ZU+ at this size

# ---- device limits -----------------------------------------------------------
DSP_TOTAL  = 216
BRAM_BITS  = 3.8 * 1024 * 1024        # 3.8 Mb
BRAM_BYTES = BRAM_BITS / 8            # ~497 KB

# ---- cycle model, validated against the RTL testbenches ----------------------
def flash_cycles(N, D, DV, BLK, RECIP_SH=24, FOLD_PAR=0):
    # +2 per block since the lane max was pipelined for timing (MAXR, CORR).
    # FOLD_PAR=1 rebases in 1 cycle and folds one key per cycle.
    rebase, fold = (1, BLK) if FOLD_PAR else (DV, BLK * DV)
    per_block = 1 + (D + 2) + 2 + rebase + BLK + fold
    return N * ((N // BLK) * per_block + (RECIP_SH + 1) + DV)

def naive_cycles(N, D, DV, LANES, NUMW=25):
    qkt = (N * N // LANES) * (D + 2)
    rm  = N * N
    sm  = N * (N + N * (NUMW + 3))
    pv  = (N * DV // LANES) * (N + 2)
    return qkt + rm + sm + pv

# ---- storage ----------------------------------------------------------------
def naive_bram(N, D, DV):
    """Q,K,V + the two N x N intermediates + O."""
    return DW_BYTES * (2*N*D + N*DV + 2*N*N + N*DV)

def flash_bram(N, D, DV, BLK):
    """Q,K,V + O + the running state. No N^2 term at all."""
    return DW_BYTES * (2*N*D + N*DV + N*DV + DV + 2)

def max_n(fn, D, DV, extra=()):
    n = 4
    while fn(n*2, D, DV, *extra) < BRAM_BYTES and n < 8192:
        n *= 2
    # linear refine
    while fn(n+1, D, DV, *extra) < BRAM_BYTES and n < 8192:
        n += 1
    return n

# ---- links ------------------------------------------------------------------
LINKS = [
    ("UART 115.2k",      115200/10),
    ("UART 3M",        3_000_000/10),
    ("AXI-DMA 64b@150", 150e6 * 8 * 0.6),   # 64-bit HP port, 60% sustained
]

CASES = [(4,4,4,4), (16,16,16,4), (64,64,64,8), (128,64,64,8), (512,64,64,8)]

if __name__ == "__main__":
    print("ZUBoard 1CG / XCZU1CG")
    print(f"  {DSP_TOTAL} DSP48E2   {BRAM_BITS/1024/1024:.1f} Mb BRAM "
          f"(~{BRAM_BYTES/1024:.0f} KB)   PL @ {F_PL/1e6:.0f} MHz assumed\n")

    print("=== 1. does the link still dominate? (link time / compute time) ===")
    print(f"{'shape':>16} {'bytes':>9} {'compute':>10}   " +
          "  ".join(f"{n:>16}" for n, _ in LINKS))
    for N, D, DV, BLK in CASES:
        b = DW_BYTES * (2*N*D + N*DV + N*DV)
        cs = flash_cycles(N, D, DV, BLK) / F_PL
        cells = []
        for _, Bps in LINKS:
            r = (b / Bps) / cs
            cells.append(f"{(b/Bps)*1e3:7.1f}ms {r:6.0f}x")
        print(f"{f'N={N} D={D}':>16} {b:>9,} {cs*1e3:>9.2f}ms   " +
              "  ".join(f"{c:>16}" for c in cells))

    print("\n=== 2. where does each datapath run out of block RAM? ===")
    for D, DV in ((64, 64), (128, 128)):
        nn = max_n(naive_bram, D, DV)
        nf = max_n(lambda n, d, dv: flash_bram(n, d, dv, 8), D, DV)
        print(f"  D=DV={D:<4} naive tops out at N={nn:<5} "
              f"({naive_bram(nn,D,DV)/1024:.0f} KB)   "
              f"streaming reaches N={nf:<5} ({flash_bram(nf,D,DV,8)/1024:.0f} KB)"
              f"   {nf/nn:.1f}x further")

    print("\n=== 3. how many MAC lanes fit in 216 DSPs? ===")
    # flash_top: BLK MACs + rebase + fold + output-scale multipliers
    for BLK in (1, 2, 4, 8, 16, 32, 64):
        dsp = BLK + 3
        pct = 100.0 * dsp / DSP_TOTAL
        note = "" if dsp <= DSP_TOTAL else "  DOES NOT FIT"
        print(f"  BLK={BLK:<3} ~{dsp:>3} DSP ({pct:4.1f}% of the device){note}")
    print("\n  A 16x16 signed multiply maps to one DSP48E2, so the lane count is")
    print("  nowhere near DSP limited on this part. BRAM and fmax bind first.")
