# LinkedIn post — draft

## Option A (recommended): the counterintuitive measurement

> I quadrupled the multipliers in my attention accelerator and got 1.28× faster.
>
> That number is the most useful thing I learned this year.
>
> I've been building single-head scaled dot-product attention at five levels of
> abstraction — NumPy → Q8.8 fixed-point → CUDA → SystemVerilog → Triton — with
> every level diffed against the one above it. Most of them bit-exactly, not
> within a tolerance.
>
> When the RTL datapath worked, I did the obvious thing: parameterized the MAC
> count and swept it. 1 → 2 → 4 multipliers. The matmuls got 3.4× faster. The
> pipeline got 1.28× faster.
>
> The profile said why. The sequential divider inside softmax is 449 cycles no
> matter what you do to the multipliers — 68% of runtime at 1 lane, 87% at 4.
> Adding hardware to the matmul just made the divider a larger share of the
> problem. Amdahl's law, with a waveform attached.
>
> So I stopped adding multipliers and changed the algorithm instead — online
> softmax, the core idea behind FlashAttention. Carry a running max, and when a
> later block contains a bigger one, rebase everything you've already
> accumulated by exp(m_old − m_new). It's a constant, so one multiply fixes the
> entire history. That's why softmax, which looks irreducibly global, is
> actually streamable.
>
> Result: one multiplier, 360 cycles. Beating the four-multiplier naive design
> at 515 — at a quarter of the arithmetic area. And the O(N²) score buffers
> disappear entirely: at N=128, D=64 that's ~64 KB of BRAM replaced by ~132
> bytes, independent of sequence length. On an FPGA that's the difference
> between fitting and not fitting.
>
> Two things I'd tell my earlier self:
>
> 1. Build an executable spec, not a reference. My fixed-point Python model
> still called np.exp — fine as a precision study, useless as a spec, because no
> RTL can match it. I rewrote it so every operation is an integer operation the
> hardware performs, in the same order, with the same truncation. Then the
> testbenches compare with zero tolerance. A tolerance would have hidden the
> exact bug it later caught: a divider producing results exactly 2× too large.
> (A clean power-of-two error is always a shift-count bug.)
>
> 2. Never make a numerics decision from one random seed. Twice I reached a
> conclusion from seed 0 and a 200-seed sweep reversed it. Once about rounding
> modes; once about whether online softmax is more accurate than naive. It
> isn't — it's a memory and divide win, and its correction factor actually
> compounds error the harder you stream. I'd have published both backwards.
>
> Everything reproduces with one command — CI runs the Python models, every
> SystemVerilog testbench across the parameter sweeps, and the CUDA kernel under
> a CPU emulation harness so correctness is checkable with no GPU attached.
>
> Repo: github.com/youssefelbaz32/flash-attention-accelerator
>
> Still to come: FPGA bring-up and the synthesis Pareto curve.
>
> #FPGA #CUDA #SystemVerilog #GPU #MachineLearning #ComputerArchitecture

---

## Option B: shorter, lead with the vertical slice

> Most attention tutorials stop at a NumPy one-liner. I followed the number all
> the way down to the gates.
>
> Single-head attention, built five times — NumPy, Q8.8 fixed-point, CUDA,
> SystemVerilog, Triton — each level diffed against the one above, most of them
> bit-exactly.
>
> The best measurement I got: 4× the multipliers in the RTL bought 1.28×.
> Switching from naive to online (FlashAttention-style) softmax bought 1.83× at
> identical area, and deleted the O(N²) buffers — ~64 KB of BRAM → ~132 bytes at
> N=128.
>
> The `softmax` that's one line in Python turns out to be a 256-entry exp ROM, a
> comparator chain and a restoring divider — and that divider, not the matmul,
> is 87% of the runtime.
>
> One command reproduces every number. CI runs the RTL testbenches and the CUDA
> kernel under a CPU emulator, so it's all checkable without a GPU.
>
> github.com/youssefelbaz32/flash-attention-accelerator
>
> #FPGA #CUDA #SystemVerilog #ComputerArchitecture

---

## Notes before posting

- Add a screenshot: the `run_all.sh` per-stage table, or a GTKWave shot of the
  flash_top handshake. A terminal screenshot of "M8 COMPLETE — BIT-EXACTLY"
  reads well.
- Do NOT claim GPU speedups until `./build/flash bench` has actually run. The
  post above deliberately makes no GPU performance claim.
- If the Vivado numbers land before posting, add one line: "X LUTs, Y DSPs,
  Z MHz" — synthesis numbers are what separate this from a simulation project.
