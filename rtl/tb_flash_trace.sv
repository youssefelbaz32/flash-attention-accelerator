`timescale 1ns/1ps
// Debug harness: prints every flash_top state TRANSITION with the loop counters,
// so you can watch the two nested loops (query rows outside, key blocks inside)
// actually run. Not a pass/fail test, so it is not in run_all.sh.
//
// This is the fastest way to answer "what is BLK doing", because j_base is
// printed on every transition:
//   BLK=1 -> j_base = 0,1,2,3   four blocks of one key
//   BLK=2 -> j_base = 0,2       two blocks of two keys
//   BLK=4 -> j_base = 0         one block of four keys
//
//   iverilog -g2012 -DBLK_OVERRIDE=2 -o build/tr.vvp \
//     rtl/dot4.sv rtl/exp_rom.sv rtl/flash_top.sv rtl/tb_flash_trace.sv
//   vvp build/tr.vvp
//
// Vectors must have been generated at matching BLK:
//   BLK=2 python3 python/05_online_softmax_model.py
module tb_flash_trace;
  localparam int DW=16, FRAC=8, N=4, D=4, DV=4, RECIP_SH=24;
`ifdef BLK_OVERRIDE
  localparam int BLK = `BLK_OVERRIDE;
`else
  localparam int BLK = 2;
`endif
  localparam int N_IN = 2*N*D + N*DV, N_OUT = N*DV;
  logic clk=0, rst_n; always #5 clk=~clk;
  logic s_valid, s_ready, s_last; logic [DW-1:0] s_data;
  logic m_valid, m_ready, m_last; logic [DW-1:0] m_data;
  logic busy, done_pulse, ph_load, ph_compute, ph_store, in_stall, out_stall,
        err_overrun, err_underrun;
  flash_top #(.DW(DW),.FRAC(FRAC),.N(N),.D(D),.DV(DV),.BLK(BLK),.RECIP_SH(RECIP_SH)) dut(.*);

  logic [DW-1:0] q[N*D], k[N*D], v[N*DV], si[N_IN];
  int cyc = 0;
  string prev = "";
  string st;

  always @(posedge clk) if (rst_n) cyc++;

  // name the state, and print only on a CHANGE so the trace stays readable
  always @(posedge clk) if (rst_n) begin
    case (dut.curr_state)
      dut.IDLE:        st = "IDLE";
      dut.LOAD:        st = "LOAD";
      dut.SCORE_ISSUE: st = "SCORE_ISSUE";
      dut.SCORE_WAIT:  st = "SCORE_WAIT";
      dut.REBASE:      st = "REBASE";
      dut.EXPF:        st = "EXPF";
      dut.FOLD:        st = "FOLD";
      dut.RECIP_LOAD:  st = "RECIP_LOAD";
      dut.RECIP_ITER:  st = "RECIP_ITER";
      dut.SCALE_OUT:   st = "SCALE_OUT";
      dut.DRAIN:       st = "DRAIN";
      default:         st = "???";
    endcase
    if (st != prev) begin
      $display("cy %4d  %-12s   row_i=%0d j_base=%0d  (keys %0d..%0d)",
               cyc, st, dut.row_i, dut.j_base, dut.j_base, dut.j_base+BLK-1);
      prev = st;
    end
  end

  initial begin #200000; $finish; end
  initial begin
    $readmemh("rtl/vectors/Q.hex",q); $readmemh("rtl/vectors/K.hex",k); $readmemh("rtl/vectors/V.hex",v);
    for(int i=0;i<N*D;i++) si[i]=q[i];
    for(int i=0;i<N*D;i++) si[N*D+i]=k[i];
    for(int i=0;i<N*DV;i++) si[2*N*D+i]=v[i];
    rst_n=0; s_valid=0; m_ready=1; s_last=0; s_data=0;
    repeat(3) @(negedge clk); rst_n=1;
    $display("=== BLK=%0d : N=%0d keys, so %0d block(s) per query row ===", BLK, N, N/BLK);
    for(int i=0;i<N_IN;i++) begin
      @(negedge clk); s_data=si[i]; s_last=(i==N_IN-1); s_valid=1;
      @(posedge clk); while(!s_ready) begin @(negedge clk); @(posedge clk); end
    end
    @(negedge clk); s_valid=0;
    wait(done_pulse); repeat(2) @(posedge clk);
    $display("cy %4d  done.", cyc);
    $finish;
  end
endmodule
