`timescale 1ns/1ps
// STRICT testbench for dot4 — protocol + corner cases that tb_dot4.sv misses.
//
// tb_dot4.sv is a *polite* producer: it holds a_flat/b_flat steady well past the
// accept beat. That politeness hid a real bug (data captured one cycle after the
// valid&&ready transfer). This tb is a maximally-hostile-but-LEGAL producer:
// data and in_valid are dropped the instant after the transfer edge, which is
// exactly what the handshake contract permits. A DUT that captures late reads
// garbage and fails every test here.
//
// Four things this covers beyond the basic tb:
//   1. hostile producer timing  — data scrambled right after the transfer edge
//   2. negative saturation      — clamp to -32768 (basic tb only tests +32767)
//   3. output backpressure      — out_ready held low; result must stay stable
//   4. back-to-back beats       — no idle gap between scores
// Reuse this shape for row_max / softmax / pv — same four cases apply to each.
//
// HANDSHAKE SAMPLING RULE (learned the hard way): check-then-wait
//     while (!in_ready) @(negedge clk);     // correct
//     do @(negedge clk); while (!in_ready); // WRONG - sleeps through the beat
// The do-while form advances a negedge before looking, so if the DUT already has
// in_ready high it misses the transfer at the intervening posedge and then waits
// forever. That makes the tb silently encode the DUT's latency: shave a cycle off
// the design and a passing test starts hanging.
//
// Run (Vivado XSim handles dot4.sv as-is; local iverilog additionally needs the
// '{default:0} array reset in dot4.sv rewritten as a for-loop — tool quirk only):
//   iverilog -g2012 -o dot4_strict.vvp rtl/dot4.sv rtl/tb_dot4_strict.sv
//   vvp dot4_strict.vvp
module tb_dot4_strict;
  // must match the DUT parameters
  localparam DW=16, FRAC=8, D=4;   // ACCW is derived inside the DUT

  logic clk=0, rst_n;
  logic in_valid, in_ready, out_valid, out_ready;
  logic [D*DW-1:0] a_flat, b_flat;
  logic signed [DW-1:0] s;
  int fails = 0;

  dot4 #(.DW(DW), .FRAC(FRAC), .D(D)) dut (.*);

  always #5 clk = ~clk;   // 10ns period

  // pack 4 Q8.8 ints into the flat bus, LSB-first: elem0 in the LOW bits.
  function automatic logic [D*DW-1:0] pack(input int e0, e1, e2, e3);
    pack = { e3[DW-1:0], e2[DW-1:0], e1[DW-1:0], e0[DW-1:0] };
  endfunction

  // $signed(s) so the 16-bit result compares correctly against a 32-bit int
  // expectation (matters for the -32768 case).
  task automatic check(input int exp, input string name);
    if ($signed(s) == exp)
      $display("  %-38s s=%7d  exp=%7d  PASS", name, $signed(s), exp);
    else begin
      $display("  %-38s s=%7d  exp=%7d  FAIL", name, $signed(s), exp);
      fails++;
    end
  endtask

  // One score, driven by a hostile producer.
  //   bp = 0 : out_ready stays high (no backpressure)
  //   bp > 0 : out_ready held LOW for bp cycles after out_valid rises, then
  //            released - the result must survive unchanged across the stall.
  task automatic run_strict(input logic [D*DW-1:0] av, bv, input int exp,
                            input int bp, input string name);
    @(negedge clk);
    a_flat = av; b_flat = bv; in_valid = 1;
    out_ready = (bp == 0);

    while (!in_ready) @(negedge clk);   // check-then-wait (see rule above)
    @(posedge clk);                     // <-- THE transfer edge: valid && ready
    #1 in_valid = 0; a_flat = '1; b_flat = '1;   // legal: beat is done, drop it all

    while (!out_valid) @(negedge clk);
    if (bp > 0) begin
      repeat (bp) @(negedge clk);       // hold the result under backpressure
      check(exp, {name, $sformatf(" (held %0dcy)", bp)});
      out_ready = 1;
    end
    else check(exp, name);

    @(posedge clk); @(negedge clk); out_ready = 1;   // retire the output beat
  endtask

  // watchdog: never let a broken handshake hang the sim forever
  initial begin #20000; $display("TIMEOUT - handshake stalled"); $finish; end

  initial begin
    // dumps to the CURRENT directory (an unwritable path aborts the sim in
    // iverilog); move the .vcd into docs/waveforms/ when you want to keep one
    $dumpfile("tb_dot4_strict.vcd");
    $dumpvars(0, tb_dot4_strict);

    rst_n = 0; in_valid = 0; out_ready = 1; a_flat = 0; b_flat = 0;
    repeat (3) @(negedge clk); rst_n = 1;          // release reset

    $display("dot4 STRICT tests (Q8.8, scale=256):");

    // S1 basic, hostile timing: [1,2,3,4].[1,1,1,1] = 10.0 -> 2560
    run_strict(pack(256,512,768,1024), pack(256,256,256,256), 2560, 0,
               "S1 [1,2,3,4].[1,1,1,1] = 10");
    // S2 signed: [-1,-2,3,4].[1,1,1,1] = 4.0 -> 1024
    run_strict(pack(-256,-512,768,1024), pack(256,256,256,256), 1024, 0,
               "S2 signed dot = 4");
    // S3 positive overflow: 4 * 100*100 = 40000 > 127.996 -> clamp +32767
    run_strict(pack(25600,25600,25600,25600), pack(25600,25600,25600,25600),
               32767, 0, "S3 pos overflow -> +32767");
    // S4 negative overflow: -40000 < -128 -> clamp -32768 (the saturation half
    // the basic tb never exercises; an unsigned/asymmetric clamp fails here)
    run_strict(pack(-25600,-25600,-25600,-25600), pack(25600,25600,25600,25600),
               -32768, 0, "S4 neg overflow -> -32768");
    // S5 backpressure: consumer stalls 4 cycles; s must not move
    run_strict(pack(256,512,768,1024), pack(256,256,256,256), 2560, 4,
               "S5 backpressure");
    // S6/S7 back-to-back with no idle gap between beats
    run_strict(pack(128,128,128,128), pack(256,256,256,256), 512, 0,
               "S6 back-to-back A ([.5]x4.[1]x4=2)");
    run_strict(pack(-128,256,-384,512), pack(256,256,256,256), 256, 0,
               "S7 back-to-back B ([-.5,1,-1.5,2].[1]x4=1)");

    if (fails == 0) $display("ALL PASS");
    else            $display("%0d FAIL(S)", fails);
    $finish;
  end
endmodule
