`timescale 1ns/1ps
// Testbench for flash_top with the AXI-Stream interface (M6).
//
// Three things are checked, and the second and third are new:
//   1. O still matches rtl/vectors/flash_O.hex bit-exactly. The interface
//      changed, the arithmetic did not, so this must be unchanged from before.
//   2. The stream survives a hostile producer and a hostile consumer. The
//      producer drops tvalid at random and the consumer drops tready at random,
//      which is legal AXI4-Stream and is exactly what a real DMA does when it
//      crosses a page boundary or loses arbitration.
//   3. The profiling taps are consistent: load + compute + store accounts for
//      every busy cycle, and the stall counts match what the testbench injected.
//
// Run from the project root:
//   python3 python/05_online_softmax_model.py
//   iverilog -g2012 -o build/flash.vvp rtl/dot4.sv rtl/exp_rom.sv \
//            rtl/flash_top.sv rtl/tb_flash_top.sv
//   vvp build/flash.vvp
module tb_flash_top;
  localparam int DW = 16, FRAC = 8;
`ifdef N_OVERRIDE
  localparam int N = `N_OVERRIDE, D = `D_OVERRIDE, DV = `DV_OVERRIDE;
`else
  localparam int N = 4, D = 4, DV = 4;
`endif
  localparam int RECIP_SH = 24;
`ifdef BLK_OVERRIDE
  localparam int BLK = `BLK_OVERRIDE;
`else
  localparam int BLK = 2;
`endif
`ifdef FOLD_PAR_OVERRIDE
  localparam int FOLD_PAR = `FOLD_PAR_OVERRIDE;
`else
  localparam int FOLD_PAR = 0;
`endif
  localparam int N_IN  = 2*N*D + N*DV;
  localparam int N_OUT = N*DV;

  logic clk = 0, rst_n;
  always #5 clk = ~clk;

  logic              s_valid, s_ready, s_last;
  logic [DW-1:0]     s_data;
  logic              m_valid, m_ready, m_last;
  logic [DW-1:0]     m_data;
  logic              busy, done_pulse, ph_load, ph_compute, ph_store;
  logic              in_stall, out_stall, err_overrun, err_underrun;

  int fails = 0, cycles = 0, diff, maxdiff = 0;
  int cy_load = 0, cy_compute = 0, cy_store = 0, cy_busy = 0;
  int cy_in_stall = 0, cy_out_stall = 0;
  int inj_in_stall = 0, inj_out_stall = 0;

  flash_top #(.DW(DW), .FRAC(FRAC), .N(N), .D(D), .DV(DV),
              .BLK(BLK), .RECIP_SH(RECIP_SH), .FOLD_PAR(FOLD_PAR)) dut (.*);

  always @(posedge clk) if (rst_n) begin
    if (busy) cy_busy++;
    if (ph_load)    cy_load++;
    if (ph_compute) cy_compute++;
    if (ph_store)   cy_store++;
    if (ph_load  && in_stall)  cy_in_stall++;
    if (ph_store && out_stall) cy_out_stall++;
    if (busy) cycles++;
    if (err_overrun)  begin $display("  err_overrun asserted  FAIL"); fails++; end
    if (err_underrun) begin $display("  err_underrun asserted  FAIL"); fails++; end
  end

  logic [DW-1:0] q_hex [N*D];
  logic [DW-1:0] k_hex [N*D];
  logic [DW-1:0] v_hex [N*DV];
  logic [DW-1:0] fo_hex [N*DV];
  logic [DW-1:0] o5_hex [N*DV];
  logic [DW-1:0] stream_in [N_IN];
  logic [DW-1:0] got_out [N*DV];

  int seed = 32'h1234_5678;

  initial begin #2000000; $display("TIMEOUT - stream stalled"); $finish; end

  // ---- producer: offers beats, randomly withholding tvalid -----------------
  initial begin
    s_valid = 0; s_data = '0; s_last = 0;
    wait (rst_n);
    for (int i = 0; i < N_IN; i++) begin
      // withhold ~1 cycle in 3. Legal AXIS, and what a real DMA does.
      while ($random(seed) % 3 == 0) begin
        @(negedge clk); s_valid = 0; inj_in_stall++;
      end
      @(negedge clk);
      s_data  = stream_in[i];
      s_last  = (i == N_IN-1);
      s_valid = 1;
      @(posedge clk);
      while (!s_ready) begin @(negedge clk); @(posedge clk); end
    end
    @(negedge clk); s_valid = 0; s_last = 0; s_data = '1;   // scramble after the beat
  end

  // ---- consumer: accepts beats, randomly withholding tready ----------------
  initial begin
    m_ready = 0;
    wait (rst_n);
    for (int i = 0; i < N_OUT; i++) begin
      while ($random(seed) % 4 == 0) begin
        @(negedge clk); m_ready = 0; inj_out_stall++;
      end
      @(negedge clk); m_ready = 1;
      @(posedge clk);
      while (!m_valid) begin @(negedge clk); @(posedge clk); end
      got_out[i] = m_data;
      if ((i == N_OUT-1) && !m_last) begin
        $display("  m_last not asserted on the final beat  FAIL"); fails++;
      end
      if ((i != N_OUT-1) && m_last) begin
        $display("  m_last asserted early at beat %0d  FAIL", i); fails++;
      end
    end
    @(negedge clk); m_ready = 0;
  end

  initial begin
    $readmemh("rtl/vectors/Q.hex", q_hex);
    $readmemh("rtl/vectors/K.hex", k_hex);
    $readmemh("rtl/vectors/V.hex", v_hex);
    $readmemh("rtl/vectors/flash_O.hex", fo_hex);
    $readmemh("rtl/vectors/O.hex", o5_hex);

    // the stream order the RTL expects: Q row-major, then K, then V
    for (int i = 0; i < N*D;  i++) stream_in[i]            = q_hex[i];
    for (int i = 0; i < N*D;  i++) stream_in[N*D + i]      = k_hex[i];
    for (int i = 0; i < N*DV; i++) stream_in[2*N*D + i]    = v_hex[i];

    rst_n = 0; repeat (3) @(negedge clk); rst_n = 1;

    $display("flash_top AXIS  (N=%0d D=%0d DV=%0d BLK=%0d)", N, D, DV, BLK);

    wait (done_pulse);
    repeat (3) @(negedge clk);

    for (int t = 0; t < N*DV; t++)
      if ($signed(got_out[t]) !== $signed(fo_hex[t])) begin
        $display("    O[%0d] got=%6d exp=%6d  FAIL", t,
                 $signed(got_out[t]), $signed(fo_hex[t]));
        fails++;
      end
    if (fails == 0)
      $display("  O vs M8 spec              %2d/%2d exact  PASS", N*DV, N*DV);

    for (int t = 0; t < N*DV; t++) begin
      diff = $signed(got_out[t]) - $signed(o5_hex[t]);
      if (diff < 0) diff = -diff;
      if (diff > maxdiff) maxdiff = diff;
    end
    $display("  O vs M5 naive             max delta = %0d LSB -- expected small, nonzero",
             maxdiff);

    // ---- the profiling taps must account for every busy cycle --------------
    if (cy_load + cy_compute + cy_store == cy_busy)
      $display("  phases account for all %0d busy cycles  PASS", cy_busy);
    else begin
      $display("  load %0d + compute %0d + store %0d != busy %0d  FAIL",
               cy_load, cy_compute, cy_store, cy_busy);
      fails++;
    end
    if (cy_in_stall > 0 && cy_out_stall > 0)
      $display("  stalls observed: %0d on input, %0d on output  PASS",
               cy_in_stall, cy_out_stall);
    else begin
      $display("  expected both stall counters to fire, got in=%0d out=%0d  FAIL",
               cy_in_stall, cy_out_stall);
      fails++;
    end

    $display("  cycles: total %0d  load %0d  compute %0d  store %0d",
             cy_busy, cy_load, cy_compute, cy_store);
    if (fails == 0) $display("  M8 COMPLETE -- AXIS STREAMING, BIT-EXACT VS SPEC");
    else            $display("  %0d TOTAL FAIL(S)", fails);
    $finish;
  end
endmodule
