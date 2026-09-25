"""Regenerate the M8 GPU figures from the measured numbers.

    python docs/figures/make_m8_figures.py

Writes m8_journey.svg, m8_scaling.svg and m8_roofline.svg next to this file.
Every number below was measured on an RTX 4070 Laptop GPU (sm_89) and appears
in the README tables; the roofline ceilings come from the microbenchmarks
described there. Styling matches the portfolio's existing dark SVG charts.

Series colors were checked with a CVD validator against the #141b24 panel:
blue #3987e5, amber #d97706, teal #0d9488 all sit in the lightness band, and the
worst adjacent pair is dE 12.5 under protanopia. Every series is also named in
text, so identity never rests on color alone.
"""
import math
from pathlib import Path

OUT = Path(__file__).parent

BLUE, AMBER, TEAL, GRAY = "#3987e5", "#d97706", "#0d9488", "#8493a4"
STYLE = """  <style>
    .t{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px;font-weight:600;fill:#dce4ee}
    .s{font-family:system-ui,sans-serif;font-size:11px;fill:#8493a4}
    .a{font-family:system-ui,sans-serif;font-size:12px;fill:#8493a4}
    .l{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;fill:#dce4ee}
    .m{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;fill:#8493a4}
    .bg{fill:#141b24;stroke:#33404f;stroke-width:1.5}
    .grid{stroke:#26313d;stroke-width:1}
    .axis{stroke:#33404f;stroke-width:1}
  </style>"""


def svg(w, h, label, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="{label}">\n{STYLE}\n'
            f'  <rect class="bg" x="4" y="4" width="{w-8}" height="{h-8}" rx="12"/>\n'
            + "\n".join(body) + "\n</svg>\n")


def fmt_ms(v):
    return f"{v:.2f} ms" if v < 10 else f"{v:.1f} ms"


# ---------------------------------------------------------------------------
# 1. the journey: N=4096, D=128, one bar per step
# ---------------------------------------------------------------------------
def journey():
    steps = [  # label, detail, ms, color, speedup note
        ("Original", "fp32, one block per query row", 28.34, BLUE, ""),
        ("Coalesced K", "stage K through shared memory", 19.44, BLUE, "1.46x"),
        ("Query-tiled", "16 rows share each K/V tile", 7.59, AMBER, "3.7x"),
        ("Tensor cores", "fp16 WMMA, fp32 accumulate", 2.23, TEAL, "12.7x"),
        ("cuDNN (reference)", "PyTorch SDPA, fp16", 0.289, GRAY, ""),
    ]
    W, H = 1120, 430
    x0, x1 = 300, 960
    scale = (x1 - x0) / 30.0
    b = [
        '  <text class="t" x="32" y="42">Following the profile: 28.3 ms to 2.23 ms</text>',
        '  <text class="s" x="32" y="62">Fused attention forward, N = 4096, D = 128, one head. RTX 4070 Laptop GPU. Shorter is faster.</text>',
    ]
    for ms in (0, 10, 20, 30):
        x = x0 + ms * scale
        b.append(f'  <line class="grid" x1="{x:.1f}" y1="92" x2="{x:.1f}" y2="{H-50}"/>')
        b.append(f'  <text class="m" x="{x:.1f}" y="{H-30}" text-anchor="middle">{ms} ms</text>')
    for k, (name, detail, ms, col, sp) in enumerate(steps):
        y = 104 + k * 56
        w = max(ms * scale, 3)
        dash = ""
        b.append(f'  <text class="l" x="32" y="{y+18}">{name}</text>')
        b.append(f'  <text class="s" x="32" y="{y+34}">{detail}</text>')
        b.append(f'  <rect x="{x0}" y="{y}" width="{w:.1f}" height="36" rx="4" '
                 f'fill="{col}" fill-opacity="0.85" stroke="{col}"{dash}/>')
        label = fmt_ms(ms) + (f"  ·  {sp} faster" if sp else "")
        b.append(f'  <text class="l" x="{x0 + w + 10:.1f}" y="{y+23}">{label}</text>')
    b.append(f'  <line class="axis" x1="{x0}" y1="92" x2="{x0}" y2="{H-50}"/>')
    (OUT / "m8_journey.svg").write_text(svg(W, H,
        "Bar chart of kernel time at N=4096, D=128: original 28.3 ms, coalesced 19.4 ms, "
        "query-tiled 7.59 ms, tensor cores 2.23 ms, against cuDNN at 0.29 ms", b), encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. scaling: time vs N at D=128, log-log
# ---------------------------------------------------------------------------
def scaling():
    Ns = [128, 512, 2048, 4096]
    series = [  # name, color, dashed, times (ms)
        ("Original, fp32", BLUE, False, [0.0360, 0.4790, 7.0820, 28.3400]),
        ("Query-tiled, fp32", AMBER, False, [0.0478, 0.1806, 2.3286, 7.5870]),
        ("Tensor cores, fp16", TEAL, False, [0.0328, 0.1242, 0.6392, 2.2269]),
        ("Best SDPA, fp16", GRAY, True, [0.0209, 0.0366, 0.1207, 0.2893]),
    ]
    W, H = 1120, 500
    px0, px1, py0, py1 = 110, 830, 440, 96          # plot box
    lx0, lx1 = math.log10(100), math.log10(5000)
    ly0, ly1 = math.log10(0.01), math.log10(50)
    X = lambda n: px0 + (math.log10(n) - lx0) / (lx1 - lx0) * (px1 - px0)
    Y = lambda t: py0 - (math.log10(t) - ly0) / (ly1 - ly0) * (py0 - py1)
    b = [
        '  <text class="t" x="32" y="42">Time against sequence length</text>',
        '  <text class="s" x="32" y="62">D = 128, one head, non-causal. Both axes logarithmic. Each fp32 step is still verified against float64.</text>',
    ]
    for t in (0.01, 0.1, 1, 10):
        y = Y(t)
        b.append(f'  <line class="grid" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        b.append(f'  <text class="m" x="{px0-10}" y="{y+4:.1f}" text-anchor="end">{t:g} ms</text>')
    for n in Ns:
        x = X(n)
        b.append(f'  <text class="m" x="{x:.1f}" y="{py0+22}" text-anchor="middle">{n}</text>')
    b.append(f'  <text class="a" x="{(px0+px1)/2:.0f}" y="{py0+44}" text-anchor="middle">sequence length N</text>')
    b.append(f'  <line class="axis" x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}"/>')
    ends = []
    for name, col, dashed, ts in series:
        pts = " ".join(f"{X(n):.1f},{Y(t):.1f}" for n, t in zip(Ns, ts))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        b.append(f'  <polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        for n, t in zip(Ns, ts):
            b.append(f'  <circle cx="{X(n):.1f}" cy="{Y(t):.1f}" r="4.5" fill="{col}" '
                     f'stroke="#141b24" stroke-width="2"><title>{name}, N={n}: {t:g} ms</title></circle>')
        ends.append([Y(ts[-1]), name, col, ts[-1]])
    ends.sort()
    for k in range(1, len(ends)):                    # keep end labels 18 px apart
        ends[k][0] = max(ends[k][0], ends[k-1][0] + 18)
    for y, name, col, t in ends:
        b.append(f'  <rect x="{px1+14}" y="{y-6:.1f}" width="10" height="10" rx="2" fill="{col}"/>')
        b.append(f'  <text class="l" x="{px1+30}" y="{y+3:.1f}">{name} · {fmt_ms(t)}</text>')
    (OUT / "m8_scaling.svg").write_text(svg(W, H,
        "Log-log line chart of kernel time against N at D=128 for the original fp32 kernel, "
        "the query-tiled fp32 kernel, the fp16 tensor-core kernel and the best PyTorch SDPA", b), encoding="utf-8")


# ---------------------------------------------------------------------------
# 3. roofline against the measured ceilings, intensity per L2 byte
# ---------------------------------------------------------------------------
def roofline():
    # L2: read-only ld.cg microbenchmark, 1.30 TB/s. (A copy benchmark reads 847
    # GB/s because it splits the bandwidth with writes; attention only reads.)
    L2_GBS, FP32, TC = 1300.0, 11800.0, 39200.0     # GB/s, GFLOP/s, GFLOP/s
    pts = [  # name, AI (FLOP/L2 byte), GFLOP/s, color, measured?
        ("Original", 0.44, 303, BLUE, True),
        ("Coalesced K", 0.44, 442, BLUE, False),
        ("Query-tiled", 0.44 * 16, 1132, AMBER, False),
        ("Tensor cores", 16.0, 3857, TEAL, False),
    ]
    W, H = 1120, 520
    px0, px1, py0, py1 = 110, 860, 460, 96
    lx0, lx1 = math.log10(0.1), math.log10(200)
    ly0, ly1 = math.log10(100), math.log10(60000)
    X = lambda a: px0 + (math.log10(a) - lx0) / (lx1 - lx0) * (px1 - px0)
    Y = lambda g: py0 - (math.log10(g) - ly0) / (ly1 - ly0) * (py0 - py1)
    b = [
        '  <text class="t" x="32" y="42">Where each kernel sits on the roofline</text>',
        '  <text class="s" x="32" y="62">Ceilings measured on this card. N = 4096, D = 128. Filled: intensity counted by Nsight. Hollow: modeled from the access pattern.</text>',
    ]
    for g in (100, 1000, 10000):
        y = Y(g)
        lab = f"{g/1000:g} TFLOP/s" if g >= 1000 else f"{g} GFLOP/s"
        b.append(f'  <line class="grid" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        b.append(f'  <text class="m" x="{px0-10}" y="{y+4:.1f}" text-anchor="end">{lab}</text>')
    for a in (0.1, 1, 10, 100):
        x = X(a)
        b.append(f'  <line class="grid" x1="{x:.1f}" y1="{py1}" x2="{x:.1f}" y2="{py0}"/>')
        b.append(f'  <text class="m" x="{x:.1f}" y="{py0+22}" text-anchor="middle">{a:g}</text>')
    b.append(f'  <text class="a" x="{(px0+px1)/2:.0f}" y="{py0+44}" text-anchor="middle">arithmetic intensity, FLOP per byte from L2</text>')
    # roofs: min(L2 bandwidth * AI, compute ceiling)
    for ceil, name in ((FP32, "FP32 CUDA cores 11.8 TFLOP/s"), (TC, "fp16 tensor cores 39.2 TFLOP/s")):
        ridge = ceil / L2_GBS
        b.append(f'  <polyline points="{X(0.1):.1f},{Y(L2_GBS*0.1):.1f} {X(ridge):.1f},{Y(ceil):.1f} '
                 f'{X(200):.1f},{Y(ceil):.1f}" fill="none" stroke="#5b6b7d" stroke-width="2"/>')
        b.append(f'  <text class="m" x="{X(200)-6:.1f}" y="{Y(ceil)-8:.1f}" text-anchor="end">{name}</text>')
    b.append(f'  <text class="m" x="{X(0.16):.1f}" y="{Y(L2_GBS*0.16)-10:.1f}" '
             f'transform="rotate(-33 {X(0.16):.1f} {Y(L2_GBS*0.16)-10:.1f})">L2 read 1.30 TB/s</text>')
    offs = {"Original": (14, 18), "Coalesced K": (14, -10), "Query-tiled": (12, 4), "Tensor cores": (12, 4)}
    for name, ai, g, col, measured in pts:
        x, y = X(ai), Y(g)
        fill = col if measured else "#141b24"
        b.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{fill}" stroke="{col}" stroke-width="2.5">'
                 f'<title>{name}: {ai:g} FLOP/byte, {g} GFLOP/s</title></circle>')
        dx, dy = offs[name]
        val = f"{g/1000:.2f} TFLOP/s" if g >= 1000 else f"{g} GFLOP/s"
        b.append(f'  <text class="l" x="{x+dx:.1f}" y="{y+dy:.1f}">{name} · {val}</text>')
    (OUT / "m8_roofline.svg").write_text(svg(W, H,
        "Log-log roofline with measured L2 bandwidth, FP32 and fp16 tensor-core ceilings, "
        "showing the original kernel at 0.44 FLOP per byte and 303 GFLOP/s rising to the "
        "tensor-core kernel at 3.86 TFLOP/s", b), encoding="utf-8")


if __name__ == "__main__":
    journey(); scaling(); roofline()
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("m8_*.svg"))))
