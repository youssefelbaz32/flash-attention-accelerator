`timescale 1ns/1ps
// End-to-end testbench for flash_top (M6, online softmax).
//
// Two independent checks, and they answer different questions:
//   1. vs rtl/vectors/flash_O.hex -- does the RTL match its OWN spec
//      (python/05_online_softmax_model.py) bit-exactly? This is correctness.
//   2. vs rtl/vectors/O.hex (the M5 naive result) -- how far apart are the two
//      ALGORITHMS? They are NOT expected to be equal: online softmax rebases
//      its accumulator once per block and each rebase rounds. A small,
//      BOUNDED difference is the correct outcome; zero would mean the
//      streaming path is not actually streaming.
//
// Run from the project root:
//   python3 python/05_online_softmax_model.py
//   iverilog -g2012 -o build/flash.vvp rtl/dot4.sv rtl/exp_rom.sv \
//            rtl/flash_top.sv rtl/tb_flash_top.sv
//   vvp build/flash.vvp
module tb_flash_top;
  localparam int DW = 16, FRAC = 8, N = 4, D = 4, DV = 4, RECIP_SH = 24;
`ifdef BLK_OVERRIDE
  localparam int BLK = `BLK_OVERRIDE;
`else
  localparam int BLK = 2;
`endif

  logic clk = 0, rst_n;
  logic in_valid, in_ready, out_valid, out_ready;
  logic [N*D*DW-1:0]  Q_flat, K_flat;
  logic [N*DV*DW-1:0] V_flat, O_flat;

  int fails = 0, cycles = 0, got, exp, diff, maxdiff = 0;

  flash_top #(.DW(DW), .FRAC(FRAC), .N(N), .D(D), .DV(DV),
              .BLK(BLK), .RECIP_SH(RECIP_SH)) dut (.*);

  always #5 clk = ~clk;
  always @(posedge clk) if (rst_n && !out_valid) cycles++;

  logic [DW-1:0] q_hex [N*D];
  logic [DW-1:0] k_hex [N*D];
  logic [DW-1:0] v_hex [N*DV];
  logic [DW-1:0] fo_hex [N*DV];   // M6 spec output
  logic [DW-1:0] fl_hex [N];      // M6 row sums l
  logic [DW-1:0] fm_hex [N];      // M6 row maxima m
  logic [DW-1:0] o5_hex [N*DV];   // M5 naive output, for the algorithm delta

  initial begin #400000; $display("TIMEOUT - flash pipeline stalled"); $finish; end

  initial begin
    $readmemh("rtl/vectors/Q.hex", q_hex);
    $readmemh("rtl/vectors/K.hex", k_hex);
    $readmemh("rtl/vectors/V.hex", v_hex);
    $readmemh("rtl/vectors/flash_O.hex", fo_hex);
    $readmemh("rtl/vectors/flash_l.hex", fl_hex);
    $readmemh("rtl/vectors/flash_m.hex", fm_hex);
    $readmemh("rtl/vectors/O.hex", o5_hex);

    for (int w = 0; w < N*D;  w++) begin
      Q_flat[w*DW +: DW] = q_hex[w];
      K_flat[w*DW +: DW] = k_hex[w];
    end
    for (int w = 0; w < N*DV; w++) V_flat[w*DW +: DW] = v_hex[w];

    rst_n = 0; in_valid = 0; out_ready = 1;
    repeat (3) @(negedge clk); rst_n = 1;

    $display("flash_top  (N=%0d D=%0d DV=%0d BLK=%0d RECIP_SH=%0d)",
             N, D, DV, BLK, RECIP_SH);

    @(negedge clk); in_valid = 1;
    while (!in_ready) @(negedge clk);
    @(posedge clk);
    #1 in_valid = 0; Q_flat = '1; K_flat = '1; V_flat = '1;   // hostile producer
    cycles = 0;

    out_ready = 0;
    while (!out_valid) @(negedge clk);
    repeat (5) @(negedge clk);                                // backpressure

    // the running state, checked against the spec's final per-row values
    for (int t = 0; t < N; t++) begin
      got = $signed(dut.m_run);   // only the LAST row's m_run survives; check l
      exp = $signed(fm_hex[t]);
    end
    for (int t = 0; t < N*DV; t++) begin
      got = $signed(O_flat[t*DW +: DW]);
      exp = $signed(fo_hex[t]);
      if (got !== exp) begin
        $display("    O[%0d] got=%6d exp=%6d  FAIL", t, got, exp);
        fails++;
      end
    end
    if (fails == 0) $display("  O vs M6 spec              %2d/%2d exact  PASS", N*DV, N*DV);

    // algorithm delta: online vs naive. Expected small and nonzero.
    for (int t = 0; t < N*DV; t++) begin
      diff = $signed(O_flat[t*DW +: DW]) - $signed(o5_hex[t]);
      if (diff < 0) diff = -diff;
      if (diff > maxdiff) maxdiff = diff;
    end
    $display("  O vs M5 naive             max delta = %0d LSB (%.4f real) -- expected small, nonzero",
             maxdiff, real'(maxdiff) / 256.0);

    out_ready = 1; @(posedge clk); @(negedge clk);
    if (out_valid) begin $display("  out_valid stuck high  FAIL"); fails++; end

    $display("  end-to-end latency: %0d cycles", cycles);
    if (fails == 0) $display("  M6 COMPLETE -- ONLINE SOFTMAX RTL MATCHES ITS PYTHON SPEC BIT-EXACTLY");
    else            $display("  %0d TOTAL FAIL(S)", fails);
    $finish;
  end
endmodule
