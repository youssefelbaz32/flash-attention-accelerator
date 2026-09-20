// M6 · the packaged IP: attention accelerator with AXI4-Lite control and
// AXI4-Stream data, ready for Vivado IP packaging and a Zynq block design.
//
// Port names matter here in a way they do not elsewhere. Vivado's IP packager
// infers AXI interfaces from signal names, so `s_axi_lite_*`, `s_axis_*` and
// `m_axis_*` are recognised and bundled automatically. Rename them and you get
// a pile of loose scalar ports that you then have to map by hand, which is both
// tedious and easy to get wrong.
//
// WHAT CONNECTS TO WHAT ON THE BOARD (ZUBoard 1CG):
//   s_axi_lite   <- PS M_AXI_HPM0_LPD, through an AXI interconnect
//   s_axis       <- AXI DMA MM2S  (PS DDR -> PL)
//   m_axis       -> AXI DMA S2MM  (PL -> PS DDR)
//   irq          -> PS pl_ps_irq0[0]
//
// A NOTE ON START. flash_top begins as soon as the first stream beat arrives,
// so CTRL.start is not a go signal for the datapath, it arms the measurement:
// it zeroes the counters and opens the total-time window. The consequence is
// useful rather than awkward. Cycles between the host writing START and the DMA
// delivering the first beat belong to no phase, so they show up as the
// "unaccounted" gap between CYC_TOTAL and the sum of the phases, which is
// exactly the DMA setup latency and is worth seeing.

module attention_axi #(
  parameter int DW       = 16,
  parameter int FRAC     = 8,
  parameter int N        = 4,
  parameter int D        = 4,
  parameter int DV       = 4,
  parameter int BLK      = 2,
  parameter int RECIP_SH = 24,
  parameter     EXP_FILE = "exp_lut.hex",   // relative to the IP's source dir
  parameter int C_S_AXI_LITE_ADDR_WIDTH = 8,
  parameter int C_S_AXI_LITE_DATA_WIDTH = 32
)(
  input  logic        aclk,
  input  logic        aresetn,

  // ---- AXI4-Lite control -------------------------------------------------
  input  logic [C_S_AXI_LITE_ADDR_WIDTH-1:0] s_axi_lite_awaddr,
  input  logic [2:0]  s_axi_lite_awprot,
  input  logic        s_axi_lite_awvalid,
  output logic        s_axi_lite_awready,
  input  logic [31:0] s_axi_lite_wdata,
  input  logic [3:0]  s_axi_lite_wstrb,
  input  logic        s_axi_lite_wvalid,
  output logic        s_axi_lite_wready,
  output logic [1:0]  s_axi_lite_bresp,
  output logic        s_axi_lite_bvalid,
  input  logic        s_axi_lite_bready,
  input  logic [C_S_AXI_LITE_ADDR_WIDTH-1:0] s_axi_lite_araddr,
  input  logic [2:0]  s_axi_lite_arprot,
  input  logic        s_axi_lite_arvalid,
  output logic        s_axi_lite_arready,
  output logic [31:0] s_axi_lite_rdata,
  output logic [1:0]  s_axi_lite_rresp,
  output logic        s_axi_lite_rvalid,
  input  logic        s_axi_lite_rready,

  // ---- AXI4-Stream in ----------------------------------------------------
  // 32 bits on the wire because AXI DMA is happiest on a power-of-two byte
  // width and the PS HP ports are 32/64/128. Each beat carries ONE Q8.8 word in
  // the low half; the upper half is ignored. That wastes half the bandwidth and
  // is the right trade for now: the link budget says transfer is ~0.1ms against
  // ~9ms of compute at N=128, so there is nothing to win by packing two words
  // per beat, and unpacking them would add a mux in the load path.
  input  logic [31:0] s_axis_tdata,
  input  logic [3:0]  s_axis_tkeep,
  input  logic        s_axis_tlast,
  input  logic        s_axis_tvalid,
  output logic        s_axis_tready,

  // ---- AXI4-Stream out ---------------------------------------------------
  output logic [31:0] m_axis_tdata,
  output logic [3:0]  m_axis_tkeep,
  output logic        m_axis_tlast,
  output logic        m_axis_tvalid,
  input  logic        m_axis_tready,

  // ---- to the PS GIC -----------------------------------------------------
  output logic        irq
);

  logic        start, soft_reset, busy, done_pulse;
  logic        ph_load, ph_compute, ph_store, in_stall, out_stall;
  logic        err_overrun, err_underrun;
  logic [DW-1:0] core_s_data, core_m_data;
  logic        core_s_valid, core_s_ready, core_s_last;
  logic        core_m_valid, core_m_ready, core_m_last;

  // ---- arming ---------------------------------------------------------------
  // The DMA can legitimately start pushing before the host has written START,
  // and nothing stops it: MM2S runs as soon as its descriptor is submitted.
  // Without a gate, those beats get consumed, the load phase begins, and then
  // START zeroes the counters underneath a run already in progress. The counters
  // then under-report by however many cycles the host was late, which is
  // exactly the kind of quietly-wrong measurement that makes every number
  // downstream untrustworthy. The integration testbench caught this by checking
  // reported cycles against observed ones rather than trusting either alone.
  //
  // Gating tready on `armed` makes the ordering a hardware guarantee instead of
  // a documented convention: data simply waits at the door until the host is
  // ready to measure it.
  logic armed;
  always_ff @(posedge aclk) begin
    if (!aresetn)        armed <= 1'b0;
    else if (soft_reset) armed <= 1'b0;
    else if (start)      armed <= 1'b1;
    else if (done_pulse) armed <= 1'b0;
  end

  // 32-bit stream to DW-bit core. Sign is irrelevant here, these are just bits.
  assign core_s_data   = s_axis_tdata[DW-1:0];
  assign core_s_valid  = s_axis_tvalid & armed;
  assign core_s_last   = s_axis_tlast;
  assign s_axis_tready = core_s_ready & armed;

  assign m_axis_tdata  = {{(32-DW){core_m_data[DW-1]}}, core_m_data}; // sign extend
  assign m_axis_tkeep  = 4'hF;
  assign m_axis_tvalid = core_m_valid;
  assign m_axis_tlast  = core_m_last;
  assign core_m_ready  = m_axis_tready;

  flash_top #(
    .DW(DW), .FRAC(FRAC), .N(N), .D(D), .DV(DV),
    .BLK(BLK), .RECIP_SH(RECIP_SH), .EXP_FILE(EXP_FILE)
  ) u_core (
    .clk(aclk), .rst_n(aresetn && !soft_reset),
    .s_valid(core_s_valid), .s_ready(core_s_ready),
    .s_data(core_s_data),   .s_last(core_s_last),
    .m_valid(core_m_valid), .m_ready(core_m_ready),
    .m_data(core_m_data),   .m_last(core_m_last),
    .busy(busy), .done_pulse(done_pulse),
    .ph_load(ph_load), .ph_compute(ph_compute), .ph_store(ph_store),
    .in_stall(in_stall), .out_stall(out_stall),
    .err_overrun(err_overrun), .err_underrun(err_underrun)
  );

  axil_regs #(
    .DW(DW), .FRAC(FRAC), .N(N), .D(D), .DV(DV), .BLK(BLK),
    .AW(C_S_AXI_LITE_ADDR_WIDTH)
  ) u_regs (
    .clk(aclk), .rst_n(aresetn),
    .s_axi_awaddr(s_axi_lite_awaddr),   .s_axi_awvalid(s_axi_lite_awvalid),
    .s_axi_awready(s_axi_lite_awready), .s_axi_wdata(s_axi_lite_wdata),
    .s_axi_wstrb(s_axi_lite_wstrb),     .s_axi_wvalid(s_axi_lite_wvalid),
    .s_axi_wready(s_axi_lite_wready),   .s_axi_bresp(s_axi_lite_bresp),
    .s_axi_bvalid(s_axi_lite_bvalid),   .s_axi_bready(s_axi_lite_bready),
    .s_axi_araddr(s_axi_lite_araddr),   .s_axi_arvalid(s_axi_lite_arvalid),
    .s_axi_arready(s_axi_lite_arready), .s_axi_rdata(s_axi_lite_rdata),
    .s_axi_rresp(s_axi_lite_rresp),     .s_axi_rvalid(s_axi_lite_rvalid),
    .s_axi_rready(s_axi_lite_rready),
    .start(start), .soft_reset(soft_reset), .busy(busy), .done_pulse(done_pulse),
    .ph_load(ph_load), .ph_compute(ph_compute), .ph_store(ph_store),
    .in_stall(in_stall), .out_stall(out_stall),
    .err_overrun(err_overrun), .err_underrun(err_underrun),
    .irq(irq)
  );

  // awprot/arprot are part of the AXI4-Lite signal set and must be present for
  // the packager to recognise the interface, but this slave has no privilege
  // model, so they are deliberately unused.
  logic _unused;
  assign _unused = &{1'b0, s_axi_lite_awprot, s_axi_lite_arprot,
                     s_axis_tkeep, s_axis_tdata[31:DW]};

endmodule
