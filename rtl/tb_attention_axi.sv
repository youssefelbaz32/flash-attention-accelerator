`timescale 1ns/1ps
// Integration testbench for attention_axi: the packaged IP as the board will
// see it. AXI4-Lite control, AXI4-Stream data, interrupt, profiling counters.
//
// This is the test that matters most before hardware, because it is the first
// one where the control path and the data path have to agree. The unit tests
// proved each side in isolation; this proves that what axil_regs REPORTS
// matches what flash_top actually DID, which is the claim the whole profiling
// story rests on. If these numbers drift, every measurement quoted later is
// wrong and nothing else would have caught it.
module tb_attention_axi;
  localparam int DW = 16, FRAC = 8, N = 4, D = 4, DV = 4, BLK = 2, AW = 8;
  localparam int N_IN = 2*N*D + N*DV, N_OUT = N*DV;

  logic aclk = 0, aresetn;
  always #5 aclk = ~aclk;

  logic [AW-1:0] awaddr, araddr;
  logic [2:0]    awprot = 0, arprot = 0;
  logic          awvalid, awready, wvalid, wready, bvalid, bready;
  logic          arvalid, arready, rvalid, rready;
  logic [31:0]   wdata, rdata;
  logic [3:0]    wstrb;
  logic [1:0]    bresp, rresp;

  logic [31:0] s_tdata, m_tdata;
  logic [3:0]  s_tkeep = 4'hF, m_tkeep;
  logic        s_tlast, s_tvalid, s_tready, m_tlast, m_tvalid, m_tready;
  logic        irq;

  int fails = 0;
  int obs_load = 0, obs_compute = 0, obs_store = 0, obs_busy = 0;
  int obs_in_stall = 0, obs_out_stall = 0;
  logic [DW-1:0] got_out [N*DV];

  attention_axi #(.DW(DW), .FRAC(FRAC), .N(N), .D(D), .DV(DV), .BLK(BLK),
                  .EXP_FILE("rtl/vectors/exp_lut.hex"),
                  .C_S_AXI_LITE_ADDR_WIDTH(AW)) dut (
    .aclk(aclk), .aresetn(aresetn),
    .s_axi_lite_awaddr(awaddr), .s_axi_lite_awprot(awprot),
    .s_axi_lite_awvalid(awvalid), .s_axi_lite_awready(awready),
    .s_axi_lite_wdata(wdata), .s_axi_lite_wstrb(wstrb),
    .s_axi_lite_wvalid(wvalid), .s_axi_lite_wready(wready),
    .s_axi_lite_bresp(bresp), .s_axi_lite_bvalid(bvalid), .s_axi_lite_bready(bready),
    .s_axi_lite_araddr(araddr), .s_axi_lite_arprot(arprot),
    .s_axi_lite_arvalid(arvalid), .s_axi_lite_arready(arready),
    .s_axi_lite_rdata(rdata), .s_axi_lite_rresp(rresp),
    .s_axi_lite_rvalid(rvalid), .s_axi_lite_rready(rready),
    .s_axis_tdata(s_tdata), .s_axis_tkeep(s_tkeep), .s_axis_tlast(s_tlast),
    .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
    .m_axis_tdata(m_tdata), .m_axis_tkeep(m_tkeep), .m_axis_tlast(m_tlast),
    .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready),
    .irq(irq)
  );

  localparam CTRL=8'h00, STATUS=8'h04, ISR=8'h08, CYC_TOTAL=8'h0C, CYC_LOAD=8'h10,
             CYC_LOAD_STALL=8'h14, CYC_COMPUTE=8'h18, CYC_STORE=8'h1C,
             CYC_STORE_STALL=8'h20, PARAM0=8'h24, BUILD_ID=8'h30;

  // observe what the datapath really does, independently of what it reports
  always @(posedge aclk) if (aresetn) begin
    if (dut.busy)       obs_busy++;
    if (dut.ph_load)    obs_load++;
    if (dut.ph_compute) obs_compute++;
    if (dut.ph_store)   obs_store++;
    if (dut.ph_load  && dut.in_stall)  obs_in_stall++;
    if (dut.ph_store && dut.out_stall) obs_out_stall++;
  end

  logic aw_done, w_done;
  task automatic axi_write(input [AW-1:0] a, input [31:0] d);
    aw_done = 0; w_done = 0;
    @(negedge aclk); awaddr=a; awvalid=1; wdata=d; wstrb=4'hF; wvalid=1; bready=1;
    forever begin
      @(posedge aclk);
      if (awvalid && awready) aw_done = 1;
      if (wvalid  && wready)  w_done  = 1;
      #1;
      if (aw_done) awvalid = 0;
      if (w_done)  wvalid  = 0;
      if (aw_done && w_done) break;
      @(negedge aclk);
    end
    while (!bvalid) @(negedge aclk);
    @(posedge aclk); #1 bready = 0;
  endtask

  task automatic axi_read(input [AW-1:0] a, output [31:0] d);
    @(negedge aclk); araddr=a; arvalid=1; rready=1;
    while (!(arready && arvalid)) @(negedge aclk);
    @(posedge aclk); #1 arvalid = 0;
    while (!rvalid) @(negedge aclk);
    d = rdata;
    @(posedge aclk); #1 rready = 0;
  endtask

  task automatic check(input string name, input int got, exp);
    if (got !== exp) begin
      $display("  %-42s got=%0d exp=%0d  FAIL", name, got, exp); fails++;
    end else
      $display("  %-42s %0d  PASS", name, got);
  endtask

  logic [DW-1:0] q_hex[N*D], k_hex[N*D], v_hex[N*DV], fo_hex[N*DV], si[N_IN];
  logic [31:0] v32;
  int seed = 32'hCAFE_1234;

  initial begin #500000; $display("TIMEOUT"); $finish; end

  // DMA-like producer: withholds tvalid sometimes, as a real DMA does
  initial begin
    s_tvalid = 0; s_tdata = 0; s_tlast = 0;
    wait (aresetn); @(negedge aclk);
    for (int i = 0; i < N_IN; i++) begin
      while ($random(seed) % 3 == 0) begin @(negedge aclk); s_tvalid = 0; end
      @(negedge aclk); s_tdata = {16'd0, si[i]}; s_tlast = (i == N_IN-1); s_tvalid = 1;
      @(posedge aclk);
      while (!s_tready) begin @(negedge aclk); @(posedge aclk); end
    end
    @(negedge aclk); s_tvalid = 0; s_tlast = 0; s_tdata = 32'hFFFF_FFFF;
  end

  // DMA-like consumer
  initial begin
    m_tready = 0;
    wait (aresetn); @(negedge aclk);
    for (int i = 0; i < N_OUT; i++) begin
      while ($random(seed) % 4 == 0) begin @(negedge aclk); m_tready = 0; end
      @(negedge aclk); m_tready = 1;
      @(posedge aclk);
      while (!m_tvalid) begin @(negedge aclk); @(posedge aclk); end
      got_out[i] = m_tdata[DW-1:0];
    end
    @(negedge aclk); m_tready = 0;
  end

  initial begin
    $readmemh("rtl/vectors/Q.hex", q_hex);
    $readmemh("rtl/vectors/K.hex", k_hex);
    $readmemh("rtl/vectors/V.hex", v_hex);
    $readmemh("rtl/vectors/flash_O.hex", fo_hex);
    for (int i = 0; i < N*D;  i++) si[i]         = q_hex[i];
    for (int i = 0; i < N*D;  i++) si[N*D + i]   = k_hex[i];
    for (int i = 0; i < N*DV; i++) si[2*N*D + i] = v_hex[i];

    awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0; wstrb=4'hF;
    aresetn = 0; repeat (3) @(negedge aclk); aresetn = 1;

    $display("attention_axi integration  (N=%0d D=%0d DV=%0d BLK=%0d)", N, D, DV, BLK);

    axi_read(BUILD_ID, v32); check("BUILD_ID", v32, 32'hA77E0001);
    axi_read(PARAM0,   v32); check("PARAM0 {D,N}", v32, {16'(D), 16'(N)});

    // arm the measurement and enable the interrupt, then let the stream run
    axi_write(CTRL, 32'h3);

    wait (irq);                         // the whole point of the interrupt
    $display("  %-42s PASS", "irq fired on completion");
    repeat (2) @(negedge aclk);

    // ---- what the hardware REPORTS must match what it DID ----------------
    axi_read(CYC_LOAD,        v32); check("CYC_LOAD matches observed",        v32, obs_load);
    axi_read(CYC_LOAD_STALL,  v32); check("CYC_LOAD_STALL matches observed",  v32, obs_in_stall);
    axi_read(CYC_COMPUTE,     v32); check("CYC_COMPUTE matches observed",     v32, obs_compute);
    axi_read(CYC_STORE,       v32); check("CYC_STORE matches observed",       v32, obs_store);
    axi_read(CYC_STORE_STALL, v32); check("CYC_STORE_STALL matches observed", v32, obs_out_stall);

    axi_read(CYC_TOTAL, v32);
    if (v32 >= obs_busy)
      $display("  %-42s %0d >= %0d busy  PASS", "CYC_TOTAL brackets busy", v32, obs_busy);
    else begin
      $display("  CYC_TOTAL %0d < observed busy %0d  FAIL", v32, obs_busy); fails++;
    end

    // ---- the answer is still right ----------------------------------------
    for (int t = 0; t < N*DV; t++)
      if ($signed(got_out[t]) !== $signed(fo_hex[t])) begin
        $display("    O[%0d] got=%0d exp=%0d  FAIL", t,
                 $signed(got_out[t]), $signed(fo_hex[t])); fails++;
      end
    if (fails == 0) $display("  %-42s %0d/%0d exact  PASS", "O vs spec", N*DV, N*DV);

    // ---- interrupt clears, status is sane ---------------------------------
    axi_read(STATUS, v32);
    if (v32[1] && v32[2] && !v32[3] && !v32[4])
      $display("  %-42s PASS", "STATUS done+pending, no errors");
    else begin $display("  STATUS = %08x unexpected  FAIL", v32); fails++; end

    axi_write(ISR, 32'h1);
    repeat (2) @(negedge aclk);
    if (!irq) $display("  %-42s PASS", "irq cleared by W1C");
    else begin $display("  irq stuck after W1C  FAIL"); fails++; end

    if (fails == 0) $display("  IP INTEGRATION PASS -- reported cycles match reality");
    else            $display("  %0d FAIL(S)", fails);
    $finish;
  end
endmodule
