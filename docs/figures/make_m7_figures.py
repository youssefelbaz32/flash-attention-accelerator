"""Regenerate the M7 FPGA figure from the Vivado 2024.1 sweep.

    python docs/figures/make_m7_figures.py

Writes m7_fmax.svg next to this file. The numbers are the worst setup path of
each build in fpga/build/<tag>/timing.txt, also tabulated in fpga/README.md and
docs/07_fpga_timing_and_optimizations.md. Styling is shared with the M8 figures.
"""
from make_m8_figures import OUT, BLUE, AMBER, GRAY, svg

# BLK, fmax (MHz), logic levels on the critical path, LUTs. N = D = DV = 16.
SWEEP = [(2, 111.4, 25, 11159), (4, 101.8, 24, 11735),
         (8, 95.5, 36, 12771), (16, 65.0, 50, 13642)]
TARGET = 100.0


def fmax():
    W, H = 1120, 460
    x0, x1, y0, y1 = 190, 960, 380, 100          # plot box, y0 is the bottom
    fmin, fmax_ = 50.0, 125.0
    xs = {blk: x0 + (x1 - x0) * k / (len(SWEEP) - 1) for k, (blk, *_) in enumerate(SWEEP)}

    def y(f):
        return y0 - (f - fmin) / (fmax_ - fmin) * (y0 - y1)

    b = [
        '  <text class="t" x="32" y="42">More lanes, lower clock: fmax against BLK on the ZUBoard</text>',
        '  <text class="s" x="32" y="62">Attention accelerator, N = D = 16, xczu1cg, Vivado 2024.1. '
        'fmax from worst-path slack at a 100 MHz target. Labels give logic levels on the critical path.</text>',
    ]
    for f in (50, 75, 100, 125):
        b.append(f'  <line class="grid" x1="{x0}" y1="{y(f):.1f}" x2="{x1}" y2="{y(f):.1f}"/>')
        b.append(f'  <text class="m" x="{x0-12}" y="{y(f)+4:.1f}" text-anchor="end">{f} MHz</text>')
    b.append(f'  <line x1="{x0}" y1="{y(TARGET):.1f}" x2="{x1+60}" y2="{y(TARGET):.1f}" '
             f'stroke="{AMBER}" stroke-width="1.5" stroke-dasharray="6 5"/>')
    b.append(f'  <text class="a" x="{x1+60}" y="{y(TARGET)-8:.1f}" text-anchor="end" fill="{AMBER}">'
             f'100 MHz PL clock</text>')
    pts = " ".join(f"{xs[blk]:.1f},{y(f):.1f}" for blk, f, *_ in SWEEP)
    b.append(f'  <polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for blk, f, lv, lut in SWEEP:
        ok = f >= TARGET
        col = BLUE if ok else GRAY
        b.append(f'  <circle cx="{xs[blk]:.1f}" cy="{y(f):.1f}" r="6" fill="{col}" stroke="#dce4ee" stroke-width="1"/>')
        dy = -16 if blk in (2, 4) else 26
        b.append(f'  <text class="l" x="{xs[blk]:.1f}" y="{y(f)+dy:.1f}" text-anchor="middle">'
                 f'{f:.0f} MHz, {lv} levels</text>')
        b.append(f'  <text class="m" x="{xs[blk]:.1f}" y="{y0+24}" text-anchor="middle">BLK = {blk}</text>')
        b.append(f'  <text class="s" x="{xs[blk]:.1f}" y="{y0+40}" text-anchor="middle">'
                 f'{lut:,} LUTs{"" if ok else ", misses timing"}</text>')
    b.append(f'  <line class="axis" x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}"/>')
    (OUT / "m7_fmax.svg").write_text(svg(W, H,
        "Line chart of fmax against BLK for N=D=16 on the ZUBoard: 111 MHz at BLK 2, "
        "102 at BLK 4, 96 at BLK 8 and 65 at BLK 16, against a 100 MHz target. "
        "Critical-path logic levels grow from 25 to 50.", b), encoding="utf-8")


if __name__ == "__main__":
    fmax()
    print("wrote", OUT / "m7_fmax.svg")
