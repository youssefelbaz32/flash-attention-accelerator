// M5 · module 1/5: dot-product MAC unit  (the atom inside QK^T)
//
// Computes score = sum_{k=0..D-1} a[k]*b[k]  for two length-D Q8.8 vectors.
// One TIME-MULTIPLEXED signed multiplier reused over D cycles (area-efficient),
// driven by a small FSM. Output is Q8.8. (/sqrt(d) scaling is applied later, at
// the QK^T level — this unit is a pure dot product.)
//
// Fixed-point: Q8.8 (DW=16, FRAC=8). Q8.8 * Q8.8 = Q16.16 (32b). Accumulate raw
// Q16.16 in a WIDE acc (ACCW) for headroom, then >>> FRAC + saturate to DW once.
//
// Interface: Avalon-ST-style valid/ready. Whole vector accepted in one input beat
// (a_flat/b_flat are D elements packed LSB-first). Streaming per-element comes at M6.
//
// Target: ModelSim (sim) + Quartus (synth), Intel FPGA. Synthesizable SV subset.
// Write CHUNK BY CHUNK. Do not fill everything at once.

// =============================================================================
// CHUNK 1: module declaration + parameters + ports
//   params: DW=16, FRAC=8, D=4, ACCW=40
//   ports :
//     clk, rst_n                         (active-LOW sync reset)
//     in_valid  (in) , in_ready  (out)   input handshake
//     a_flat, b_flat (in, [D*DW-1:0])    two packed Q8.8 vectors, elem k = [k*DW +: DW]
//     out_valid (out), out_ready (in)    output handshake
//     s         (out, signed [DW-1:0])   Q8.8 score
// -----------------------------------------------------------------------------
// TODO(you): module dot4 #(...) ( ... );

module dot4 #(
  parameter DW=16, FRAC=8, D=4, ACCW=40
)(
  input  logic clk,
  input  logic rst_n,

  input  logic in_valid,
  output logic in_ready,
  input  logic [D*DW-1:0] a_flat,
  input  logic [D*DW-1:0] b_flat,

  output logic out_valid,
  input  logic out_ready,
  output logic signed [DW-1:0] s
);










// =============================================================================
// CHUNK 2: internal signals
//   state enum (IDLE/LOAD/MAC/DONE), a_reg[D], b_reg[D], idx counter, acc (ACCW)
// -----------------------------------------------------------------------------
// TODO(you)


// =============================================================================
// CHUNK 3: FSM  (state register + next-state logic)
// -----------------------------------------------------------------------------
// TODO(you)


// =============================================================================
// CHUNK 4: datapath  (load latch, MAC accumulate, >>>FRAC + saturate)
// -----------------------------------------------------------------------------
// TODO(you)


// =============================================================================
// CHUNK 5: handshake outputs  (in_ready, out_valid, s)
// -----------------------------------------------------------------------------
// TODO(you)

// endmodule

endmodule