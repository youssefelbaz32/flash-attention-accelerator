`timescale 1ns/1ps
// Testbench for qkt -- checks the FULL S matrix against the bit-exact Python
// model (python/04_rtl_fixed_model.py), not against hand-computed constants.
//
// Why vectors instead of inline expectations: a testbench that recomputes the
// expected value in SystemVerilog can repeat the DUT's own bug (same author,
// same misreading of the spec). Loading golden data produced by a DIFFERENT
// implementation in a DIFFERENT language makes that class of shared-bug
// impossible. The comparison is EXACT -- no tolerance -- because the Python
// model performs the identical integer operations in the identical order.
//
// Run from the project root (paths below are relative to it):
//   python3 python/04_rtl_fixed_model.py
//   iverilog -g2012 -o build/qkt.vvp rtl/dot4.sv rtl/qkt.sv rtl/tb_qkt.sv
//   vvp build/qkt.vvp
// Sweep the lane count:  iverilog -g2012 -PLANES=4 ... (or edit the localparam)
module tb_qkt;
  localparam int DW = 16, FRAC = 8, N = 4, D = 4;
`ifdef LANES_OVERRIDE
  localparam int LANES = `LANES_OVERRIDE;
`else
  localparam int LANES = 1;
`endif

  logic clk = 0, rst_n;
  logic in_valid, in_ready, out_valid, out_ready;
  logic [N*D*DW-1:0] Q_flat, K_flat;
  logic [N*N*DW-1:0] S_flat;
  int fails = 0;
  int got, exp;                // iverilog rejects `automatic` inside a
                               // for-loop body, so hoist the temporaries
  int cycles = 0;            // latency counter -> feeds the LANES Pareto table

  qkt #(.DW(DW), .FRAC(FRAC), .N(N), .D(D), .LANES(LANES)) dut (.*);

  always #5 clk = ~clk;                       // 10ns period
  always @(posedge clk) if (rst_n) cycles++;

  // golden vectors, one 16-bit word per line, row-major
  logic [DW-1:0] q_hex [N*D];
  logic [DW-1:0] k_hex [N*D];
  logic [DW-1:0] s_hex [N*N];

  initial begin #50000; $display("TIMEOUT - qkt handshake stalled"); $finish; end

  initial begin
    $readmemh("rtl/vectors/Q.hex", q_hex);
    $readmemh("rtl/vectors/K.hex", k_hex);
    $readmemh("rtl/vectors/S.hex", s_hex);

    // pack row-major: element (i,k) sits at flat word (i*D + k)
    for (int w = 0; w < N*D; w++) begin
      Q_flat[w*DW +: DW] = q_hex[w];
      K_flat[w*DW +: DW] = k_hex[w];
    end

    rst_n = 0; in_valid = 0; out_ready = 1;
    repeat (3) @(negedge clk); rst_n = 1;

    $display("qkt tests  (N=%0d D=%0d LANES=%0d, Q8.8)", N, D, LANES);

    // --- input beat, hostile producer ------------------------------------
    @(negedge clk); in_valid = 1;
    while (!in_ready) @(negedge clk);         // check-then-wait, never do-while
    @(posedge clk);                           // THE transfer edge
    #1 in_valid = 0; Q_flat = '1; K_flat = '1;   // legal: scramble immediately
    cycles = 0;

    // --- backpressure on the output: stall 5 cycles, result must hold -----
    out_ready = 0;
    while (!out_valid) @(negedge clk);
    repeat (5) @(negedge clk);

    // --- compare every score exactly --------------------------------------
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) begin
        got = $signed(S_flat[(i*N+j)*DW +: DW]);
        exp = $signed(s_hex[i*N+j]);
        if (got !== exp) begin
          $display("  S[%0d][%0d] got=%6d exp=%6d  FAIL", i, j, got, exp);
          fails++;
        end
      end

    out_ready = 1; @(posedge clk); @(negedge clk);

    if (out_valid) begin
      $display("  out_valid still high after the accept beat  FAIL");
      fails++;
    end

    $display("  latency: %0d cycles for %0d scores (%0d cy/score)",
             cycles, N*N, cycles / (N*N));
    if (fails == 0) $display("  ALL %0d SCORES MATCH GOLDEN -- PASS", N*N);
    else            $display("  %0d FAIL(S)", fails);
    $finish;
  end
endmodule
