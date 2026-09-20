`timescale 1ns/1ps
// End-to-end testbench for attention_top.
//
// Checks EVERY STAGE BOUNDARY against the bit-exact Python model, not just the
// final O. If only O were checked, a failure would say "the accelerator is
// wrong" and nothing else; probing S, m and P turns one red light into a
// bisection -- the first stage that mismatches is the one with the bug, and
// everything downstream of it is just carrying the error.
//
// Run from the project root:
//   python3 python/04_rtl_fixed_model.py
//   iverilog -g2012 -o build/top.vvp rtl/dot4.sv rtl/qkt.sv rtl/row_max.sv \
//            rtl/softmax.sv rtl/pv.sv rtl/attention_top.sv rtl/tb_attention_top.sv
//   vvp build/top.vvp
module tb_attention_top;
  // Dimensions come from the command line so one testbench covers the whole
  // parameter sweep. They must match whatever python/04_rtl_fixed_model.py was
  // run with, because that is what produced rtl/vectors/*.hex.
  localparam int DW = 16, FRAC = 8;
`ifdef N_OVERRIDE
  localparam int N = `N_OVERRIDE, D = `D_OVERRIDE, DV = `DV_OVERRIDE;
`else
  localparam int N = 4, D = 4, DV = 4;
`endif
`ifdef LANES_OVERRIDE
  localparam int LANES = `LANES_OVERRIDE;
`else
  localparam int LANES = 1;
`endif

  logic clk = 0, rst_n;
  logic in_valid, in_ready, out_valid, out_ready;
  logic [N*D*DW-1:0]  Q_flat, K_flat;
  logic [N*DV*DW-1:0] V_flat, O_flat;

  int fails = 0, cycles = 0, got, exp;

  attention_top #(.DW(DW), .FRAC(FRAC), .N(N), .D(D), .DV(DV), .LANES(LANES))
    dut (.*);

  always #5 clk = ~clk;
  always @(posedge clk) if (rst_n && !out_valid) cycles++;

  // Per-stage occupancy: count the cycles each stage spends NOT in its IDLE
  // state. Hierarchical probes into the DUT -- simulation-only, and the reason
  // a testbench can profile a design without adding a single gate to it.
  int cy_qkt = 0, cy_rm = 0, cy_sm = 0, cy_pv = 0;
  always @(posedge clk) if (rst_n) begin
    if (dut.u_qkt.curr_state     != dut.u_qkt.IDLE)     cy_qkt++;
    if (dut.u_rowmax.curr_state  != dut.u_rowmax.IDLE)  cy_rm++;
    if (dut.u_softmax.curr_state != dut.u_softmax.IDLE) cy_sm++;
    if (dut.u_pv.curr_state      != dut.u_pv.IDLE)      cy_pv++;
  end

  logic [DW-1:0] q_hex [N*D];
  logic [DW-1:0] k_hex [N*D];
  logic [DW-1:0] v_hex [N*DV];
  logic [DW-1:0] s_hex [N*N];
  logic [DW-1:0] m_hex [N];
  logic [DW-1:0] p_hex [N*N];
  logic [DW-1:0] o_hex [N*DV];

  // One scalar comparison. Deliberately NOT a task taking an unsized array
  // argument -- passing buses of different widths (m_bus is N*DW, S is N*N*DW)
  // to one such task crashes iverilog 13. Keep testbench plumbing boring.
  int stage_fails;
  task automatic check1(input string label, input int idx,
                        input int g, input int e);
    if (g !== e) begin
      $display("    %s[%0d] got=%6d exp=%6d  FAIL", label, idx, g, e);
      stage_fails++;
      fails++;
    end
  endtask

  task automatic report(input string label, input int count);
    if (stage_fails == 0) $display("  %-28s %2d/%2d exact  PASS", label, count, count);
    else                  $display("  %-28s %0d FAIL", label, stage_fails);
  endtask

  initial begin #400000; $display("TIMEOUT - pipeline stalled"); $finish; end

  initial begin
    $readmemh("rtl/vectors/Q.hex", q_hex);
    $readmemh("rtl/vectors/K.hex", k_hex);
    $readmemh("rtl/vectors/V.hex", v_hex);
    $readmemh("rtl/vectors/S.hex", s_hex);
    $readmemh("rtl/vectors/M.hex", m_hex);
    $readmemh("rtl/vectors/P.hex", p_hex);
    $readmemh("rtl/vectors/O.hex", o_hex);

    for (int w = 0; w < N*D;  w++) begin
      Q_flat[w*DW +: DW] = q_hex[w];
      K_flat[w*DW +: DW] = k_hex[w];
    end
    for (int w = 0; w < N*DV; w++) V_flat[w*DW +: DW] = v_hex[w];

    rst_n = 0; in_valid = 0; out_ready = 1;
    repeat (3) @(negedge clk); rst_n = 1;

    $display("attention_top  (N=%0d D=%0d DV=%0d LANES=%0d, Q8.8)", N, D, DV, LANES);

    // hostile producer: drop Q/K/V the instant the beat is taken
    @(negedge clk); in_valid = 1;
    while (!in_ready) @(negedge clk);
    @(posedge clk);
    #1 in_valid = 0; Q_flat = '1; K_flat = '1; V_flat = '1;
    cycles = 0;

    // backpressure the output for 5 cycles before reading it
    out_ready = 0;
    while (!out_valid) @(negedge clk);
    repeat (5) @(negedge clk);

    stage_fails = 0;
    for (int t = 0; t < N*N; t++)
      check1("S", t, $signed(dut.s_reg[t*DW +: DW]), $signed(s_hex[t]));
    report("S  (qkt)", N*N);

    stage_fails = 0;
    for (int t = 0; t < N; t++)
      check1("m", t, $signed(dut.m_bus[t*DW +: DW]), $signed(m_hex[t]));
    report("m  (row_max)", N);

    // P is a probability in [0, 256] -- unsigned by construction
    stage_fails = 0;
    for (int t = 0; t < N*N; t++)
      check1("P", t, dut.p_bus[t*DW +: DW], p_hex[t]);
    report("P  (softmax)", N*N);

    stage_fails = 0;
    for (int t = 0; t < N*DV; t++)
      check1("O", t, $signed(O_flat[t*DW +: DW]), $signed(o_hex[t]));
    report("O  (pv)", N*DV);

    out_ready = 1; @(posedge clk); @(negedge clk);
    if (out_valid) begin
      $display("  out_valid stuck high after accept beat  FAIL");
      fails++;
    end

    $display("  end-to-end latency: %0d cycles", cycles);
    $display("  per-stage: qkt=%0d  row_max=%0d  softmax=%0d  pv=%0d   (softmax = %0d%% of total)",
             cy_qkt, cy_rm, cy_sm, cy_pv, (cy_sm * 100) / cycles);
    if (fails == 0) $display("  M5 COMPLETE -- RTL MATCHES PYTHON MODEL BIT-EXACTLY");
    else            $display("  %0d TOTAL FAIL(S)", fails);
    $finish;
  end
endmodule
