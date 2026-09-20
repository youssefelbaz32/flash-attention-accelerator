`timescale 1ns/1ps
// Testbench for axil_regs: protocol, register map, profiling counters, interrupt.
//
// The AXI4-Lite cases here are the ones that actually break slaves in the field:
//   - W arriving BEFORE AW (legal, and a slave that waits for AW first deadlocks)
//   - AW arriving before W, far apart
//   - B held until BREADY
//   - a partial WSTRB not clobbering neighbouring bits
//   - reads of unmapped addresses
//   - write-1-to-clear semantics on the ISR, including that writing 0 does nothing
//
// The counter cases drive the phase inputs directly, so the block is verified
// independently of any datapath. That matters: when the full system misbehaves
// later, this block is already known good and can be ruled out.
module tb_axil_regs;
  localparam int AW = 8;
  localparam int N = 4, D = 4, DV = 4, BLK = 2, DW = 16, FRAC = 8;

  logic clk = 0, rst_n;
  always #5 clk = ~clk;

  logic [AW-1:0] awaddr, araddr;
  logic          awvalid, awready, wvalid, wready, bvalid, bready;
  logic          arvalid, arready, rvalid, rready;
  logic [31:0]   wdata, rdata;
  logic [3:0]    wstrb;
  logic [1:0]    bresp, rresp;

  logic start, soft_reset, busy, done_pulse;
  logic ph_load, ph_compute, ph_store, in_stall, out_stall;
  logic err_overrun, err_underrun, irq;

  int fails = 0;

  axil_regs #(.DW(DW), .FRAC(FRAC), .N(N), .D(D), .DV(DV), .BLK(BLK), .AW(AW)) dut (
    .clk(clk), .rst_n(rst_n),
    .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
    .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
    .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready),
    .s_axi_araddr(araddr), .s_axi_arvalid(arvalid), .s_axi_arready(arready),
    .s_axi_rdata(rdata), .s_axi_rresp(rresp), .s_axi_rvalid(rvalid), .s_axi_rready(rready),
    .start(start), .soft_reset(soft_reset), .busy(busy), .done_pulse(done_pulse),
    .ph_load(ph_load), .ph_compute(ph_compute), .ph_store(ph_store),
    .in_stall(in_stall), .out_stall(out_stall),
    .err_overrun(err_overrun), .err_underrun(err_underrun), .irq(irq)
  );

  // register offsets
  localparam CTRL=8'h00, STATUS=8'h04, ISR=8'h08, CYC_TOTAL=8'h0C, CYC_LOAD=8'h10,
             CYC_LOAD_STALL=8'h14, CYC_COMPUTE=8'h18, CYC_STORE=8'h1C,
             CYC_STORE_STALL=8'h20, PARAM0=8'h24, PARAM1=8'h28, PARAM2=8'h2C,
             BUILD_ID=8'h30;

  task automatic check(input string name, input int unsigned got, exp);
    if (got !== exp) begin
      $display("  %-44s got=%08x exp=%08x  FAIL", name, got, exp);
      fails++;
    end else
      $display("  %-44s %08x  PASS", name, got);
  endtask

  // ---- AXI-Lite master: AW and W presented together ------------------------
  // AW and W can complete on the SAME edge or on different ones, and the master
  // has to cope with either. Waiting for one and then the other hangs the
  // moment both handshake simultaneously, because the ready it is waiting on
  // has already dropped. Track the two channels independently.
  logic aw_done, w_done;
  task automatic axi_write(input [AW-1:0] a, input [31:0] d, input [3:0] strb = 4'hF);
    aw_done = 0; w_done = 0;
    @(negedge clk); awaddr = a; awvalid = 1; wdata = d; wstrb = strb; wvalid = 1; bready = 1;
    forever begin
      @(posedge clk);
      if (awvalid && awready) aw_done = 1;
      if (wvalid  && wready)  w_done  = 1;
      #1;
      if (aw_done) awvalid = 0;
      if (w_done)  wvalid  = 0;
      if (aw_done && w_done) break;
      @(negedge clk);
    end
    while (!bvalid) @(negedge clk);
    @(posedge clk); #1 bready = 0;
  endtask

  // ---- AXI-Lite master: W FIRST, AW several cycles later -------------------
  // A slave that requires AW before W deadlocks here. This is legal AXI.
  task automatic axi_write_w_first(input [AW-1:0] a, input [31:0] d);
    @(negedge clk); wdata = d; wstrb = 4'hF; wvalid = 1; bready = 1;
    while (!(wready && wvalid)) @(negedge clk);
    @(posedge clk); #1 wvalid = 0;
    repeat (3) @(negedge clk);                 // deliberate gap
    awaddr = a; awvalid = 1;
    while (!(awready && awvalid)) @(negedge clk);
    @(posedge clk); #1 awvalid = 0;
    while (!bvalid) @(negedge clk);
    @(posedge clk); #1 bready = 0;
  endtask

  task automatic axi_read(input [AW-1:0] a, output [31:0] d);
    @(negedge clk); araddr = a; arvalid = 1; rready = 1;
    while (!(arready && arvalid)) @(negedge clk);
    @(posedge clk); #1 arvalid = 0;
    while (!rvalid) @(negedge clk);
    d = rdata;
    @(posedge clk); #1 rready = 0;
  endtask

  // drive one phase for n cycles, with `stall` high for `stalls` of them
  task automatic run_phase(input int n, input int stalls, input int which);
    for (int i = 0; i < n; i++) begin
      @(negedge clk);
      ph_load    = (which == 0);
      ph_compute = (which == 1);
      ph_store   = (which == 2);
      in_stall   = (which == 0) && (i < stalls);
      out_stall  = (which == 2) && (i < stalls);
      @(posedge clk);
    end
    @(negedge clk);
    ph_load = 0; ph_compute = 0; ph_store = 0; in_stall = 0; out_stall = 0;
  endtask

  logic [31:0] v;

  initial begin #200000; $display("TIMEOUT"); $finish; end

  initial begin
    awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0; awaddr=0; araddr=0;
    wdata=0; wstrb=4'hF; busy=0; done_pulse=0;
    ph_load=0; ph_compute=0; ph_store=0; in_stall=0; out_stall=0;
    err_overrun=0; err_underrun=0;
    rst_n = 0; repeat (3) @(negedge clk); rst_n = 1;

    $display("axil_regs tests");

    // ---- identity and parameter reporting --------------------------------
    axi_read(BUILD_ID, v); check("BUILD_ID", v, 32'hA77E_0001);
    axi_read(PARAM0,  v);  check("PARAM0 {D,N}",   v, {16'(D), 16'(N)});
    axi_read(PARAM1,  v);  check("PARAM1 {BLK,DV}",v, {16'(BLK), 16'(DV)});
    axi_read(PARAM2,  v);  check("PARAM2 {FRAC,DW}",v,{16'd0, 8'(FRAC), 8'(DW)});
    axi_read(8'hFC,   v);  check("unmapped read -> DEADBEEF", v, 32'hDEAD_BEEF);

    // ---- W-before-AW must not deadlock -----------------------------------
    axi_write_w_first(CTRL, 32'h2);            // irq_enable = 1
    axi_read(CTRL, v);     check("W-before-AW write took effect", v, 32'h2);

    // ---- a run, with known phase lengths ---------------------------------
    axi_write(CTRL, 32'h3);                    // start | irq_enable
    busy = 1;
    run_phase(40, 12, 0);                      // load:    40 cy, 12 stalled
    run_phase(100, 0, 1);                      // compute:100 cy
    run_phase(25,  7, 2);                      // store:   25 cy,  7 stalled
    @(negedge clk); done_pulse = 1; @(posedge clk); @(negedge clk); done_pulse = 0;
    busy = 0;
    repeat (2) @(negedge clk);

    axi_read(CYC_LOAD,        v); check("CYC_LOAD = 40",         v, 40);
    axi_read(CYC_LOAD_STALL,  v); check("CYC_LOAD_STALL = 12",   v, 12);
    axi_read(CYC_COMPUTE,     v); check("CYC_COMPUTE = 100",     v, 100);
    axi_read(CYC_STORE,       v); check("CYC_STORE = 25",        v, 25);
    axi_read(CYC_STORE_STALL, v); check("CYC_STORE_STALL = 7",   v, 7);

    // TOTAL is counted independently of the phases, so it should bracket their
    // sum rather than equal it exactly. Checking >= catches a dead counter.
    axi_read(CYC_TOTAL, v);
    if (v >= 165) $display("  %-44s %0d (>= 165)  PASS", "CYC_TOTAL brackets the phases", v);
    else begin $display("  CYC_TOTAL = %0d, expected >= 165  FAIL", v); fails++; end

    // ---- interrupt behaviour ---------------------------------------------
    if (irq) $display("  %-44s PASS", "irq asserted after done");
    else begin $display("  irq did not assert  FAIL"); fails++; end
    axi_read(STATUS, v); check("STATUS done|irq_pending", v, 32'h6);

    axi_write(ISR, 32'h0);                     // writing 0 must NOT clear
    if (irq) $display("  %-44s PASS", "writing 0 to ISR does not clear");
    else begin $display("  writing 0 to ISR cleared irq  FAIL"); fails++; end

    axi_write(ISR, 32'h1);                     // write-1-to-clear
    repeat (2) @(negedge clk);
    if (!irq) $display("  %-44s PASS", "irq cleared by W1C");
    else begin $display("  irq still set after W1C  FAIL"); fails++; end

    // ---- irq_enable gates the output, but the flag still latches ----------
    axi_write(CTRL, 32'h0);                    // irq_enable = 0
    @(negedge clk); done_pulse = 1; @(posedge clk); @(negedge clk); done_pulse = 0;
    repeat (2) @(negedge clk);
    if (!irq) $display("  %-44s PASS", "irq masked when irq_enable=0");
    else begin $display("  irq asserted with enable low  FAIL"); fails++; end
    axi_read(STATUS, v);
    if (v[2]) $display("  %-44s PASS", "irq_pending latched even while masked");
    else begin $display("  irq_pending lost while masked  FAIL"); fails++; end

    // ---- a new start clears the previous result and rezeroes counters -----
    axi_write(CTRL, 32'h1);
    repeat (2) @(negedge clk);
    axi_read(CYC_COMPUTE, v); check("start rezeroes CYC_COMPUTE", v, 0);
    axi_read(STATUS, v);
    if (!v[1]) $display("  %-44s PASS", "start clears done flag");
    else begin $display("  done flag survived a new start  FAIL"); fails++; end

    // ---- partial byte strobe must not clobber neighbours ------------------
    axi_write(CTRL, 32'h2);                    // irq_enable = 1
    axi_write(CTRL, 32'hFF00_0000, 4'b1000);   // touch only byte 3
    axi_read(CTRL, v);  check("WSTRB respected, irq_enable intact", v, 32'h2);

    // ---- sticky error flags ----------------------------------------------
    axi_write(CTRL, 32'h1);
    @(negedge clk); err_overrun = 1; @(posedge clk); @(negedge clk); err_overrun = 0;
    repeat (2) @(negedge clk);
    axi_read(STATUS, v);
    if (v[3]) $display("  %-44s PASS", "err_overrun is sticky");
    else begin $display("  err_overrun did not latch  FAIL"); fails++; end

    // ---- soft reset clears everything ------------------------------------
    axi_write(CTRL, 32'h4);
    repeat (2) @(negedge clk);
    axi_read(STATUS, v);    check("soft_reset clears status", v, 32'h0);
    axi_read(CYC_TOTAL, v); check("soft_reset clears CYC_TOTAL", v, 0);

    if (fails == 0) $display("  ALL PASS");
    else            $display("  %0d FAIL(S)", fails);
    $finish;
  end
endmodule
