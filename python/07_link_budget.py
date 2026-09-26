"""
M6 planning: how big is the host-to-FPGA transfer, and does the link or the
accelerator set the wall clock?

The answer decides the whole M6 design, so it is worth computing before writing
a single line of UART RTL. Everything below is either measured from the RTL
testbenches or derived from a cycle formula that reproduces those measurements.

Run:  python3 python/07_link_budget.py
"""

DW_BYTES = 2          # Q8.8 -> 2 bytes per word
F_CLK    = 100e6      # a conservative fabric clock for a 7-series part

# flash_top cycle model, derived from the FSM and checked against the sim:
#   per key block : 1 issue + (D+2) MAC + 2 max/corr + DV rebase + BLK exp + BLK*DV fold
#   per row       : (N/BLK) blocks + RECIP_SH+1 divide + DV scale-out
# Measured: N=4,D=DV=4,BLK=4 -> 248 cy (model 248). N=16,D=DV=16,BLK=4 -> 7328 cy
# (model 7376, the 0.7% gap is handshake overhead the model does not track).
# With FOLD_PAR=1: N=16 BLK=4 -> 2528 cy (model 2576).
def flash_cycles(N, D, DV, BLK, RECIP_SH=24, FOLD_PAR=0):
    # +2 per block since the lane max was pipelined for timing (MAXR, CORR).
    # FOLD_PAR=1 rebases in 1 cycle and folds one key per cycle.
    rebase, fold = (1, BLK) if FOLD_PAR else (DV, BLK * DV)
    per_block = 1 + (D + 2) + 2 + rebase + BLK + fold
    per_row   = (N // BLK) * per_block + (RECIP_SH + 1) + DV
    return N * per_row


def payload_bytes(N, D, DV):
    """Q, K and V in; O back out."""
    inp = DW_BYTES * (N * D + N * D + N * DV)
    out = DW_BYTES * (N * DV)
    return inp, out


# effective bytes/sec. UART 8N1 spends 10 bit-times per byte, so divide by 10.
LINKS = [
    ("UART 115.2 kbaud",   115200 / 10),
    ("UART 1 Mbaud",      1_000_000 / 10),
    ("UART 3 Mbaud",      3_000_000 / 10),
    ("USB-FS bulk 12M",       1.0e6),        # ~1 MB/s realistic
    ("AXI-DMA @100MHz x32",  400e6 * 0.5),   # 32-bit @100MHz, 50% efficiency
]

CASES = [
    #  N,   D,  DV, BLK
    (   4,   4,   4, 4),
    (  16,  16,  16, 4),
    (  64,  64,  64, 8),
    ( 128,  64,  64, 8),
    ( 512,  64,  64, 8),
]

if __name__ == "__main__":
    print(f"{'shape':>18} {'bytes':>9} {'compute':>10}   " +
          "  ".join(f"{n.split()[0][:12]:>12}" for n, _ in LINKS))
    print(f"{'':>18} {'(in+out)':>9} {'@100MHz':>10}   " +
          "  ".join(f"{n.split(maxsplit=1)[1][:12]:>12}" for n, _ in LINKS))
    print("-" * 110)
    for N, D, DV, BLK in CASES:
        inp, out = payload_bytes(N, D, DV)
        total = inp + out
        cy = flash_cycles(N, D, DV, BLK)
        compute_s = cy / F_CLK
        row = f"{f'N={N} D={D} BLK={BLK}':>18} {total:>9,} {compute_s*1e3:>9.2f}ms   "
        cells = []
        for _, Bps in LINKS:
            link_s = total / Bps
            ratio = link_s / compute_s
            cells.append(f"{link_s*1e3:>8.1f}ms{'':1}" if link_s < 1 else f"{link_s:>9.2f}s ")
        print(row + "  ".join(f"{c:>12}" for c in cells))

    print("\nlink time / compute time  (>1 means the wire is the bottleneck)")
    print(f"{'shape':>18}   " + "  ".join(f"{n.split()[0][:12]:>12}" for n, _ in LINKS))
    print("-" * 110)
    for N, D, DV, BLK in CASES:
        inp, out = payload_bytes(N, D, DV)
        total = inp + out
        compute_s = flash_cycles(N, D, DV, BLK) / F_CLK
        cells = [f"{(total/Bps)/compute_s:>11.0f}x" for _, Bps in LINKS]
        print(f"{f'N={N} D={D} BLK={BLK}':>18}   " + "  ".join(f"{c:>12}" for c in cells))
