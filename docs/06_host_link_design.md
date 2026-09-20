# M6: host link on the ZUBoard 1CG

Target is the Avnet ZUBoard 1CG, AMD XCZU1CG MPSoC. That is a Zynq UltraScale+
part, not a plain fabric board, which changes the answer completely. An earlier
version of this document planned a UART plus packet FSM plus CRC stack. On this
board that is the wrong design and most of it is unnecessary.

## The device

| | |
|---|---|
| PL | ~81K logic cells, **216 DSP48E2**, **3.8 Mb block RAM** (~486 KB) |
| PS | dual Cortex-A53, dual Cortex-R5F, 1 GB LPDDR4 |
| Boot | microSD or QSPI. Avnet publishes an official **PYNQ v3.0.1** image |
| I/O | gigabit Ethernet, USB 2.0 host, microUSB JTAG/UART |

The PS is the important part. The host is not a laptop on the end of a cable, it
is two ARM cores on the same die with a coherent path into the same DDR the PL
can read.

## Finding 1: with DMA the link stops mattering

`python/08_zuboard_budget.py`, link time against measured compute time:

| shape | UART 115.2k | UART 3M | AXI-DMA 64b @150MHz |
|---|---|---|---|
| N=64 D=64 | 2844 ms (1259x) | 109 ms (48x) | 0.04 ms (**<1x**) |
| N=128 D=64 | 5689 ms (635x) | 219 ms (24x) | 0.07 ms (**<1x**) |
| N=512 D=64 | 22756 ms (160x) | 874 ms (6x) | 0.28 ms (**<1x**) |

Over a 64-bit HP port at 150 MHz, moving an N=128 problem takes about 0.07 ms
against 8.96 ms of compute. **The accelerator becomes the bottleneck, which is
the whole point.** All the UART machinery, the framing, the CRC, the byte
assembler, the RTS/CTS question, exists only to survive a link that is 600x too
slow. On this board none of it is needed.

The on-chip cycle counter is still worth having, because a cycle count is the
number you actually quote and it is immune to what Linux is doing on the A53.
But it is now a convenience rather than the only way to get a real measurement.

## Finding 2: block RAM is what limits sequence length, and it is where the streaming design pays

| head dim | naive path | streaming path | |
|---|---|---|---|
| D=DV=64 | tops out at **N=294** | reaches **N=972** | 3.3x further |
| D=DV=128 | tops out at **N=247** | reaches **N=486** | 2.0x further |

The naive path stores S and P, which is 2N² words, so block RAM is consumed
quadratically. The streaming path stores DV+2 words of running state, so its
BRAM use is linear in N and comes entirely from holding Q, K and V.

That is the concrete version of the argument in the M8 writeup, stated in the
units this board actually has. And the streaming path can go further still by
keeping Q/K/V in DDR and streaming them in, which the naive path cannot do
because it needs random access to all of S while computing row maxima.

## Finding 3: DSPs are not the constraint

A 16x16 signed multiply maps to one DSP48E2. `flash_top` needs about BLK+3.

| BLK | DSPs | % of device |
|---|---|---|
| 8 | ~11 | 5.1% |
| 16 | ~19 | 8.8% |
| 32 | ~35 | 16.2% |
| 64 | ~67 | 31.0% |

216 DSPs is a lot for this design. The lane count will be limited by fmax and by
BRAM port count long before it is limited by multipliers, which means the Pareto
sweep should go much wider than the 1/2/4 currently simulated.

## The architecture

```
A53 running PYNQ (Python)
  |
  |  numpy arrays in pynq.allocate() buffers, physically contiguous
  v
AXI DMA   MM2S  ---->  AXI4-Stream, tdata[63:0], tvalid/tready, tlast
                            |
                            v
                       qkv_loader        unpacks 64-bit beats into Q8.8 words,
                            |            writes q_mem / k_mem / v_mem
                            v
                       flash_top         unchanged arithmetic, new input port
                            |
                            v
                       o_packer          o_mem back into 64-bit beats
                            |
AXI DMA   S2MM  <----  AXI4-Stream
  |
  v
numpy array on the A53, diffed in-process
```

Plus a small AXI4-Lite slave for control and status: start, done, the cycle
counter, and a read-only register reporting the synthesized N/D/DV/BLK so a host
built for one shape cannot silently talk to a bitstream built for another.

### What this means for the host program

The host is Python on the A53, so it can `import` the bit-exact model directly:

```python
from pynq import Overlay, allocate
import numpy as np

ol  = Overlay("attention.bit")
dma = ol.axi_dma_0

qkv = allocate(shape=(3*N*D,), dtype=np.int16)
out = allocate(shape=(N*DV,),  dtype=np.int16)
qkv[:] = np.concatenate([Qq.ravel(), Kq.ravel(), Vq.ravel()])

dma.sendchannel.transfer(qkv); dma.recvchannel.transfer(out)
dma.sendchannel.wait();        dma.recvchannel.wait()

assert np.array_equal(out, O_spec.ravel())      # bit-exact, on the board
cycles = ol.attention_0.read(CYCLES_REG)
```

**The golden model becomes the on-board self-test.** `python/04_rtl_fixed_model.py`
already runs anywhere numpy runs, including on the A53, so the same integer spec
that the simulation is checked against also checks the silicon, in the same
process, with no serialization format in between. There is no protocol to get
wrong because there is no protocol.

That deletes the packet FSM, the CRC, the byte-to-word assembler, the UART
transmitter and receiver, and the backpressure question that came with them.

## The one piece of RTL work that is genuinely required

`flash_top` takes all of Q, K and V in a single beat on flat buses. That does not
scale:

| shape | width of `Q_flat` |
|---|---|
| N=4, D=4 | 256 bits |
| N=64, D=64 | 65,536 bits |
| N=128, D=64 | 131,072 bits |

There is no 131,072-bit port. The flat interface was right for reaching
bit-exactness quickly and has to become a streaming write port before this meets
a board. On this design that port is AXI4-Stream, which is where it wanted to go
anyway.

This is the only change to already-verified RTL, so it happens first and gets
re-diffed against the golden vectors immediately, at both N=4 and N=16, before
anything else moves.

## Order of work

1. **Streaming input port on `flash_top`.** Replace the flat beat with an AXIS
   write port feeding `q_mem`/`k_mem`/`v_mem`. Re-run `tb_flash_top` at N=4 and
   N=16 and confirm still bit-exact. No board involved.
2. **AXIS output port**, same treatment.
3. **AXI4-Lite control slave**: start, done, cycle count, and the
   parameter-report register.
4. **Package as an IP**, build the block design: DMA, interconnect, the
   accelerator, PS AXI HP port.
5. **PYNQ image** from Avnet's `ZUBoard_1CG-PYNQ` repo, then the notebook that
   loads the overlay, runs one N=4 case, and diffs against
   `python/04_rtl_fixed_model.py`. Smallest possible thing that proves the whole
   chain.
6. **Scale up** N and sweep BLK, recording cycle counts from the counter and
   LUT/FF/DSP/BRAM plus WNS from Vivado at each point. That is the Pareto curve.

## What to measure once it runs

- Cycles from the on-chip counter, against the simulated count. They should
  match closely; a gap means DMA stall, not arithmetic.
- fmax from the Vivado timing report, and therefore latency in microseconds.
- LUT / FF / DSP / BRAM at BLK of 4, 8, 16, 32, 64.
- The BRAM crossover: the N where the naive build stops fitting and the
  streaming build still does. Predicted at N≈294 for D=DV=64. Demonstrating that
  on hardware is a much stronger claim than computing it.
