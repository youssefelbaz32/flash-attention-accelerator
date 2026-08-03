`timescale 1ns/1ps
// Testbench for dot4 — drives known Q8.8 vector pairs through the valid/ready
// handshake and checks the dot-product result (basic, signed, saturation).
// Run in Vivado XSim (handles dot4.sv as-is). Local iverilog needs the
// '{default:0} reset rewritten to a loop — tool quirk only.
module tb_dot4;
  // must match the DUT parameters
  localparam DW=16, FRAC=8, D=4, ACCW=40;

  // DUT interface signals
  logic clk=0, rst_n;
  logic in_valid, in_ready, out_valid, out_ready;
  logic [D*DW-1:0] a_flat, b_flat;
  logic signed [DW-1:0] s;

  // instantiate the DUT (.* connects same-named signals automatically)
  dot4 #(.DW(DW), .FRAC(FRAC), .D(D), .ACCW(ACCW)) dut (.*);

  // free-running clock: 10ns period
  always #5 clk = ~clk;

  // pack 4 Q8.8 ints into the flat bus, LSB-first: elem0 in the LOW bits.
  // (concatenation lists MSB-left, so e0 goes rightmost.)
  function automatic logic [D*DW-1:0] pack(input int e0, e1, e2, e3);
    pack = { e3[DW-1:0], e2[DW-1:0], e1[DW-1:0], e0[DW-1:0] };
  endfunction

  // Drive one dot product and check it. Full valid/ready handshake:
  //   present a,b + in_valid -> wait for the accept beat (in_ready) -> drop
  //   in_valid -> wait for out_valid -> sample s -> compare.
  task automatic run(input logic [D*DW-1:0] av, bv, input int exp, input string name);
    @(negedge clk); a_flat = av; b_flat = bv; in_valid = 1;
    forever begin @(posedge clk); if (in_ready) break; end   // input beat taken
    @(negedge clk); in_valid = 0;
    forever begin @(posedge clk); if (out_valid) break; end  // result produced
    #1 $display("  %-34s s=%6d  exp=%6d  %s",
                name, s, exp, (s === exp) ? "PASS" : "FAIL");
  endtask

  // watchdog: never let a broken handshake hang the sim forever
  initial begin #20000; $display("TIMEOUT"); $finish; end

  // stimulus
  initial begin
    rst_n = 0; in_valid = 0; out_ready = 1; a_flat = 0; b_flat = 0;
    repeat (3) @(negedge clk); rst_n = 1;          // release reset

    $display("dot4 tests (Q8.8, scale=256):");
    // T1 basic:  [1,2,3,4].[1,1,1,1] = 10.0  -> 10*256 = 2560
    run(pack(256,512,768,1024), pack(256,256,256,256), 2560,
        "T1 [1,2,3,4].[1,1,1,1] = 10");
    // T2 signed: [-1,-2,3,4].[1,1,1,1] = 4.0  -> 1024
    run(pack(-256,-512,768,1024), pack(256,256,256,256), 1024,
        "T2 signed dot = 4");
    // T3 saturate: [100..].[100..] = 40000 > 127.996 -> clamp to 32767
    run(pack(25600,25600,25600,25600), pack(25600,25600,25600,25600), 32767,
        "T3 overflow -> saturate 32767");

    $display("done"); $finish;
  end
endmodule
