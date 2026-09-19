# OpenAI Hardware Interview Prep — RTL, Custom Silicon, and Edge Infrastructure

OpenAI hardware, custom-silicon, and edge-infrastructure roles focus heavily on
custom AI accelerators, tensor processing units, matrix-multiplication engines,
and low-latency interconnects. Register-transfer level (RTL) design and hardware
verification interviews commonly target the following areas.

## 1. Matrix Multiplication and Tensor Core Design

Large language models rely heavily on GEMM (general matrix multiplication), so
expect questions about hardware that executes it efficiently.

- **Systolic arrays:** Design a 4×4 weight-stationary systolic array for matrix
  multiplication. Show synthesizable RTL for one processing element containing
  a multiply-accumulate unit, and explain how the stream avoids data bubbles.
- **Sparsity:** Design an RTL block that skips zero-valued weights—structured or
  unstructured—to save power and cycles. Discuss metadata, load balancing, and
  the overhead that can erase sparsity gains.
- **Quantization and bit widths:** Implement a reconfigurable datapath capable of
  one FP16 MAC or two INT8 MAC operations per cycle. Explain packing, accumulator
  width, rounding, saturation, and mode-control timing.

## 2. High-Bandwidth Memory and SRAM Caching

Serving model weights and the key-value cache requires memory systems that avoid
fragmentation, bank conflicts, starvation, and wasted bandwidth.

- **FIFO and flow control:** Design a parameterized asynchronous FIFO for tensor
  data crossing unrelated clock domains. Explain Gray-coded pointers,
  synchronizers, and race-free full/empty flag generation.
- **Multi-bank SRAM conflicts:** Design an arbiter for an eight-bank SRAM cache
  where several tensor pipelines can request the same bank. Maximize throughput
  while guaranteeing fairness.
- **Memory alignment:** Write a SystemVerilog module that accepts unaligned data
  from a 512-bit bus and emits a sequence of variable-length tokens. Track byte
  enables, offsets, carry-over data, and transaction boundaries.

## 3. Static Timing Analysis and Clock-Domain Crossing

At advanced process nodes, physical constraints often determine the practical
microarchitecture.

- **Metastability:** Safely transfer a multi-bit control bus between asynchronous
  clock domains. Compare Gray coding, bundled-data handshakes, pulse/toggle
  synchronizers, and asynchronous FIFOs.
- **Setup and hold violations:** Given a large setup violation through activation
  logic such as a GeLU or Softmax approximation, propose RTL-level fixes. Explain
  the latency, area, control, and verification trade-offs of pipelining and
  retiming. Know that adding pipeline stages does not directly solve hold issues.
- **Clock gating:** Implement or instantiate an RTL-level integrated clock-gating
  scheme that reduces dynamic power while a tensor pipeline is stalled. Explain
  glitch prevention, scan/test enable, and why ad hoc combinational clock gating
  is unsafe.

## 4. Direct Coding and SystemVerilog Gotchas

Live coding usually emphasizes concise, synthesizable, race-free logic.

- **Blocking vs. nonblocking assignments:** Identify a race caused by using `=`
  versus `<=`, explain simulation scheduling, and distinguish behavioral races
  from the hardware synthesis may infer.
- **Inferred latches:** Write a complete combinational instruction decoder. Be
  ready to explain how missing defaults or incomplete branches infer storage and
  complicate timing and verification.
- **Combinational loops:** Draw an accidental feedback loop and write the
  corresponding RTL. Explain how lint, elaboration, synthesis, and static timing
  tools detect or report it.

## 5. Architecture and 10× Scale-Up

Expect follow-ups that move a small correct design toward production scale.

- **Interconnects:** Connect eight custom accelerator dies on an interposer.
  Define link-layer features for lossless, low-latency credit-based flow control:
  virtual channels, buffering, replay, CRC, ordering, deadlock avoidance,
  backpressure, and link initialization/recovery.
- **Fault tolerance:** Design a low-latency streaming wrapper for single-error
  correction and double-error detection (SECDED ECC). Address codeword layout,
  syndrome calculation, correction, error reporting, scrubbing, and pipeline
  placement.

## Recommended Preparation

- **Coding practice:** Use HDLBits to build speed and accuracy on combinational
  and sequential SystemVerilog exercises.
- **Digital design:** Review *Digital Design and Computer Architecture* by Harris
  and Harris.
- **Computer architecture:** Revisit pipelining, memory hierarchy, coherence,
  interconnects, performance modeling, and relevant Georgia Tech/Udacity course
  material.
- **Deep-learning hardware:** Study the microarchitectures of Google TPUs,
  Tenstorrent processors, and NVIDIA Tensor Cores. Focus on dataflow, memory
  hierarchy, utilization, precision, sparsity, and communication—not just peak
  TOPS.

## Useful Next Deep Dives

- A 4×4 systolic array, including cycle-by-cycle wavefront timing and PE RTL
- A production-style asynchronous FIFO with assertions and a CDC explanation
- Banked-SRAM arbitration with fairness and throughput analysis
- FP16/INT8 shared multiplier architecture
- A verification plan for any of the above, including assertions, scoreboards,
  constrained-random stimulus, coverage, and corner cases

When tailoring preparation, choose the target level explicitly: architecture,
RTL design, or design verification (DV).
