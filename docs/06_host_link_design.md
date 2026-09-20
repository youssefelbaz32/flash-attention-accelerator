# M6: how the host actually talks to the FPGA

Design notes written before any RTL, because two measurements changed what the
answer should be.

## Finding 1: the wire dominates, by two to three orders of magnitude

`python/07_link_budget.py` computes the payload and compares it against the
measured accelerator cycle counts. Link time divided by compute time:

| shape | UART 115.2k | UART 1M | UART 3M | USB-FS bulk | AXI-DMA 32b |
|---|---|---|---|---|---|
| N=4 D=4 | 4630x | 533x | 178x | 53x | <1x |
| N=16 D=16 | 2453x | 283x | 94x | 28x | <1x |
| N=64 D=64 | 839x | 97x | 32x | 10x | <1x |
| N=128 D=64 | 423x | 49x | 16x | 5x | <1x |
| N=512 D=64 | 106x | 12x | 4x | 1x | <1x |

At N=128 with a head dim of 64 the accelerator computes in 13.5 ms and a
115200-baud UART spends 5.7 seconds moving the data. Every "throughput" number
measured end to end over that link is a measurement of an FTDI chip.

This is not a reason to abandon UART. It is a reason to be clear about what UART
is for: **UART is a correctness transport, not a performance transport.**

## Finding 2: the current top-level interface does not scale

Both `attention_top` and `flash_top` take all of Q, K and V in a single beat on
flat buses. That is fine at the toy size and impossible at any real one:

| shape | width of `Q_flat` |
|---|---|
| N=4, D=4 | 256 bits |
| N=16, D=16 | 4,096 bits |
| N=64, D=64 | 65,536 bits |
| N=128, D=64 | 131,072 bits |

There is no 131,072-bit port. The flat interface was the right call for getting
to bit-exact quickly, and it has to go before the design meets a board. This is
M6 work regardless of which link wins, so it comes first.

## The design that follows from both findings

**Measure on-chip, transport off-chip.** Put a free-running cycle counter in the
fabric, start it when the accelerator accepts its input beat, stop it on the
output beat, and return the count in the response packet. The host then gets a
real accelerator latency in clock cycles over a link that is 400x too slow to
measure it any other way. Cost is roughly one 32-bit counter and a comparator.

That single decision decouples the two problems. UART becomes a perfectly good
way to prove the hardware computes the right answer on real silicon, which is
what M7 actually needs, while the performance claim rests on a cycle count plus
a timing-closure fmax rather than on a stopwatch around a serial port.

## Proposed stack

```
Python host (pyserial)
  |
  |  framed packets over /dev/tty.usbserial
  v
uart_rx / uart_tx        oversampling receiver, 16x, one byte at a time
  |
  v
packet_fsm               SOF / TYPE / LEN / PAYLOAD / CRC16, both directions
  |
  v  AXI4-Stream, tdata[7:0], tvalid/tready, tlast = end of packet
byte_to_word             2 bytes -> one Q8.8 word, little endian first
  |
  v  AXI4-Stream, tdata[15:0]
qkv_loader               writes q_mem / k_mem / v_mem, counts words, raises start
  |
  v
flash_top                unchanged, except its input side becomes a write port
  |
  v
o_reader + word_to_byte  drains o_mem back into a response packet
```

### Packet format

```
host -> fpga            fpga -> host
  SOF    1B  0xA5         SOF    1B  0x5A
  TYPE   1B               TYPE   1B  request type | 0x80
  LEN    2B  LE           LEN    2B  LE
  PAYLOAD LEN B           PAYLOAD LEN B
  CRC16  2B  CCITT        CRC16  2B  CCITT
```

CRC16-CCITT over TYPE through PAYLOAD. It is eight lines of RTL and it turns a
flaky cable from "wrong answers" into "a retry", which matters a lot when the
thing you are trying to prove is that the arithmetic is correct.

| TYPE | meaning | payload |
|---|---|---|
| 0x01 | LOAD_Q | N*D Q8.8 words |
| 0x02 | LOAD_K | N*D words |
| 0x03 | LOAD_V | N*DV words |
| 0x04 | START | none |
| 0x05 | READ_O | none, response carries N*DV words + cycle count |
| 0x06 | PING | none, response echoes build ID and the synthesized N/D/DV/BLK |

`PING` returning the synthesized parameters is worth having on day one. The most
confusing possible bug is a host built for one set of dimensions talking to a
bitstream built for another, and it presents as garbage arithmetic rather than as
a protocol error.

### Backpressure, and the one place it bites

Internally every module already speaks valid/ready, so backpressure propagates
for free. The exception is the UART receiver: bytes arrive on the wire whether or
not anything downstream is ready, and a plain 8N1 link has no way to say stop.
Two options, and the choice is not obvious:

- **RTS/CTS hardware flow control.** Correct, and costs two pins plus the host
  agreeing to use them. pyserial supports it with `rtscts=True`.
- **Size the sink so it cannot stall.** The loader writes straight into
  `q_mem`/`k_mem`/`v_mem`, which are always ready during a load, so there is
  nowhere to stall as long as the host does not send a second packet before the
  first is consumed.

The second is simpler and sufficient here, but it is a protocol invariant rather
than a hardware guarantee, so it belongs in an assertion: if a byte arrives while
the loader is not ready, raise an error flag and report it in the next response
rather than silently dropping it. Silent drop on a serial link is exactly the
failure mode that costs a day.

## The board question, which is still open

The stack above assumes a plain fabric part with a USB-UART bridge, which covers
Basys3, Arty A7, Nexys A7 and similar. If the board is a Zynq (Pynq-Z2, Zybo,
Kria) the better answer is different: the accelerator becomes an AXI4-Stream
peripheral hanging off an AXI-DMA, the ARM core runs a small Python or C program
in Linux, and the transfer stops being the bottleneck entirely. That is more work
to set up and much more representative of how an accelerator is actually driven.

Deciding this changes maybe 60% of the M6 RTL, so it is worth settling before
writing any of it.

## Order of work

1. Replace the flat input beat with a streaming write port on `flash_top`.
   Needed on every path, and it is the only change to already-verified RTL, so it
   happens first and gets re-diffed against the golden vectors immediately.
2. Cycle counter and the `PING`/parameter-report path. Smallest useful bitstream,
   proves the toolchain and the link before any arithmetic is involved.
3. `uart_rx`/`uart_tx` plus a loopback test at the pins.
4. `packet_fsm` with CRC, tested in simulation against a Python model of the same
   framing so both sides are checked against one definition.
5. Full path, host sends Q/K/V and diffs the returned O against
   `python/04_rtl_fixed_model.py`. Same golden vectors as the simulation, so a
   hardware mismatch is immediately distinguishable from a model mismatch.
