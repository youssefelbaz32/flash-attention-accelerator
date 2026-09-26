// M6 · AXI4-Lite control, status, profiling and interrupt block.
//
// This is the only thing the A53 talks to directly. Everything else on the PL
// side moves through AXI-DMA as a stream; this block starts a run, reports when
// it finished, says how long each phase took, and raises an interrupt so the
// host does not have to poll.
//
// WHY THE COUNTERS ARE SPLIT THE WAY THEY ARE. "How fast is the accelerator" is
// not one number on a system like this. A run has three phases and two distinct
// ways of being slow:
//
//   CYC_LOAD        streaming Q,K,V in over AXIS
//   CYC_LOAD_STALL    ...of which we sat waiting because tvalid was low
//   CYC_COMPUTE     the accelerator actually computing
//   CYC_STORE       streaming O back out
//   CYC_STORE_STALL   ...of which we were backpressured because tready was low
//
// The stall counters are the ones that will earn their keep. If CYC_LOAD is
// large and CYC_LOAD_STALL is nearly as large, the DMA is starving the datapath
// and widening the MAC array will do nothing. If CYC_LOAD_STALL is near zero
// then the transfer really is transfer and the fix is a wider stream. Without
// splitting them you cannot tell those two cases apart, and they have opposite
// fixes.
//
// CYC_TOTAL is measured independently rather than summed, so a discrepancy
// between TOTAL and LOAD+COMPUTE+STORE is itself a signal: it means the
// datapath spent cycles in a state nobody is accounting for.
//
// Register map (32-bit registers, byte addressed):
//
//   0x00  CTRL            RW   [0] start (write 1, self clearing)
//                              [1] irq_enable
//                              [2] soft_reset (clears counters and error flags)
//   0x04  STATUS          RO   [0] busy   [1] done   [2] irq_pending
//                              [3] err_overrun  [4] err_underrun
//   0x08  ISR             W1C  [0] done interrupt, write 1 to clear
//   0x0C  CYC_TOTAL       RO
//   0x10  CYC_LOAD        RO
//   0x14  CYC_LOAD_STALL  RO
//   0x18  CYC_COMPUTE     RO
//   0x1C  CYC_STORE       RO
//   0x20  CYC_STORE_STALL RO
//   0x24  PARAM0          RO   {D[15:0], N[15:0]}
//   0x28  PARAM1          RO   {BLK[15:0], DV[15:0]}
//   0x2C  PARAM2          RO   {15'0, FOLD_PAR, FRAC[7:0], DW[7:0]}
//   0x30  BUILD_ID        RO   0xA77E0001
//
// PARAM0..2 exist because the single most confusing possible failure is a host
// built for one shape talking to a bitstream built for another. It does not
// present as a protocol error, it presents as wrong arithmetic, and you can
// lose a day to it. The host reads these first and refuses to run on a
// mismatch.

module axil_regs #(
  parameter int DW    = 16,
  parameter int FRAC  = 8,
  parameter int FOLD_PAR = 0,
  parameter int N     = 4,
  parameter int D     = 4,
  parameter int DV    = 4,
  parameter int BLK   = 2,
  parameter int AW    = 8,            // 8 bits -> 64 registers of address space
  parameter logic [31:0] BUILD_ID = 32'hA77E_0001
)(
  input  logic        clk,
  input  logic        rst_n,          // active low, synchronous

  // ---- AXI4-Lite slave ----------------------------------------------------
  input  logic [AW-1:0] s_axi_awaddr,
  input  logic          s_axi_awvalid,
  output logic          s_axi_awready,
  input  logic [31:0]   s_axi_wdata,
  input  logic [3:0]    s_axi_wstrb,
  input  logic          s_axi_wvalid,
  output logic          s_axi_wready,
  output logic [1:0]    s_axi_bresp,
  output logic          s_axi_bvalid,
  input  logic          s_axi_bready,
  input  logic [AW-1:0] s_axi_araddr,
  input  logic          s_axi_arvalid,
  output logic          s_axi_arready,
  output logic [31:0]   s_axi_rdata,
  output logic [1:0]    s_axi_rresp,
  output logic          s_axi_rvalid,
  input  logic          s_axi_rready,

  // ---- to/from the datapath ----------------------------------------------
  output logic        start,          // one-cycle pulse
  output logic        soft_reset,     // one-cycle pulse
  input  logic        busy,
  input  logic        done_pulse,     // one cycle when a run completes

  input  logic        ph_load,        // phase indicators, expected one-hot-ish
  input  logic        ph_compute,
  input  logic        ph_store,
  input  logic        in_stall,       // load phase and the stream had no data
  input  logic        out_stall,      // store phase and the consumer was not ready
  input  logic        err_overrun,    // data arrived when the sink was not ready
  input  logic        err_underrun,   // datapath wanted data and the run ended

  // ---- to the PS ----------------------------------------------------------
  output logic        irq             // level, held until ISR is written
);

  localparam int RW = 6;              // register index width, addr[7:2]

  // ---- profiling counters --------------------------------------------------
  // Saturating, not wrapping. A wrapped counter reports a plausible small number
  // for a very long run, which is worse than obviously pegging at max.
  logic [31:0] cyc_total, cyc_load, cyc_load_stall,
               cyc_compute, cyc_store, cyc_store_stall;

  function automatic logic [31:0] bump(input logic [31:0] c, input logic en);
    bump = (en && (c != 32'hFFFF_FFFF)) ? c + 32'd1 : c;
  endfunction

  logic run_active;                   // total-time window: start .. done

  always_ff @(posedge clk) begin
    if (!rst_n || soft_reset) begin
      cyc_total       <= '0;
      cyc_load        <= '0;
      cyc_load_stall  <= '0;
      cyc_compute     <= '0;
      cyc_store       <= '0;
      cyc_store_stall <= '0;
      run_active      <= 1'b0;
    end else begin
      if (start)           run_active <= 1'b1;
      else if (done_pulse) run_active <= 1'b0;

      // a fresh start zeroes the counters, so back-to-back runs do not accumulate
      if (start) begin
        cyc_total       <= '0;
        cyc_load        <= '0;
        cyc_load_stall  <= '0;
        cyc_compute     <= '0;
        cyc_store       <= '0;
        cyc_store_stall <= '0;
      end else begin
        cyc_total       <= bump(cyc_total,       run_active);
        cyc_load        <= bump(cyc_load,        ph_load);
        cyc_load_stall  <= bump(cyc_load_stall,  ph_load  && in_stall);
        cyc_compute     <= bump(cyc_compute,     ph_compute);
        cyc_store       <= bump(cyc_store,       ph_store);
        cyc_store_stall <= bump(cyc_store_stall, ph_store && out_stall);
      end
    end
  end

  // ---- status, sticky error flags, interrupt ------------------------------
  // The write path REQUESTS an interrupt clear; it does not perform it.
  // irq_pending must have exactly one driver or it is a multi-driver net, so
  // this block owns it and isr_clear (driven from the write channel) is only a
  // combinational request. Declared here because it is USED here: SystemVerilog
  // requires declaration before use, and some tools will not look ahead.
  logic isr_clear;
  logic done_flag, irq_pending, irq_enable;
  logic err_ovr_sticky, err_und_sticky;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      done_flag <= 1'b0; irq_pending <= 1'b0;
      err_ovr_sticky <= 1'b0; err_und_sticky <= 1'b0;
    end else if (soft_reset) begin
      done_flag <= 1'b0; irq_pending <= 1'b0;
      err_ovr_sticky <= 1'b0; err_und_sticky <= 1'b0;
    end else begin
      if (start) begin
        done_flag      <= 1'b0;      // a new run clears the previous result
        err_ovr_sticky <= 1'b0;
        err_und_sticky <= 1'b0;
      end else if (done_pulse) begin
        done_flag   <= 1'b1;
        irq_pending <= 1'b1;
      end else if (isr_clear) begin
        // done_pulse deliberately wins if both land on the same cycle: losing a
        // completion is far worse than servicing one interrupt twice.
        irq_pending <= 1'b0;
      end
      if (err_overrun)  err_ovr_sticky <= 1'b1;
      if (err_underrun) err_und_sticky <= 1'b1;
    end
  end

  // Level interrupt, gated by the enable. Level rather than pulse because a
  // pulse can be missed if the PS is servicing something else, and the GIC on
  // Zynq is configured for level-sensitive PL interrupts by default.
  assign irq = irq_pending & irq_enable;

  logic [31:0] status_reg;
  assign status_reg = {27'd0, err_und_sticky, err_ovr_sticky,
                       irq_pending, done_flag, busy};

  // ---- AXI4-Lite write channel --------------------------------------------
  // AW and W are captured INDEPENDENTLY. A master is allowed to present them in
  // either order, or far apart, and a slave that only accepts them together can
  // deadlock against one that waits for awready before driving wvalid.
  logic            aw_hold, w_hold;
  logic [RW-1:0]   aw_addr_q;
  logic [31:0]     w_data_q;
  logic [3:0]      w_strb_q;

  assign s_axi_awready = !aw_hold;
  assign s_axi_wready  = !w_hold;

  logic do_write;
  assign do_write = aw_hold && w_hold && !s_axi_bvalid;


  // byte-enable expansion, so a partial-strobe write does not clobber the rest
  logic [31:0] w_mask;
  always_comb begin
    for (int b = 0; b < 4; b++) w_mask[b*8 +: 8] = {8{w_strb_q[b]}};
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      aw_hold <= 1'b0; w_hold <= 1'b0;
      aw_addr_q <= '0; w_data_q <= '0; w_strb_q <= '0;
      s_axi_bvalid <= 1'b0; s_axi_bresp <= 2'b00;
      irq_enable <= 1'b0;
      start <= 1'b0; soft_reset <= 1'b0;
    end else begin
      start      <= 1'b0;             // both are single-cycle pulses
      soft_reset <= 1'b0;

      if (s_axi_awvalid && s_axi_awready) begin
        aw_addr_q <= s_axi_awaddr[RW+1:2];
        aw_hold   <= 1'b1;
      end
      if (s_axi_wvalid && s_axi_wready) begin
        w_data_q <= s_axi_wdata;
        w_strb_q <= s_axi_wstrb;
        w_hold   <= 1'b1;
      end

      if (do_write) begin
        case (aw_addr_q)
          6'h00: begin                                   // CTRL
            if (w_mask[0]) start      <= w_data_q[0];
            if (w_mask[1]) irq_enable <= w_data_q[1];
            if (w_mask[2]) soft_reset <= w_data_q[2];
          end
          6'h02: ;                                       // ISR: see isr_clear below
          default: ;                                     // read-only, ignore
        endcase
        aw_hold <= 1'b0;
        w_hold  <= 1'b0;
        s_axi_bvalid <= 1'b1;
        s_axi_bresp  <= 2'b00;                           // always OKAY
      end else if (s_axi_bvalid && s_axi_bready) begin
        s_axi_bvalid <= 1'b0;
      end
    end
  end

  assign isr_clear = do_write && (aw_addr_q == 6'h02) && w_mask[0] && w_data_q[0];

  // ---- AXI4-Lite read channel ----------------------------------------------
  logic [RW-1:0] ar_addr_q;
  assign s_axi_arready = !s_axi_rvalid;

  always_comb begin
    case (ar_addr_q)
      6'h00:   s_axi_rdata = {30'd0, irq_enable, 1'b0};
      6'h01:   s_axi_rdata = status_reg;
      6'h02:   s_axi_rdata = {31'd0, irq_pending};
      6'h03:   s_axi_rdata = cyc_total;
      6'h04:   s_axi_rdata = cyc_load;
      6'h05:   s_axi_rdata = cyc_load_stall;
      6'h06:   s_axi_rdata = cyc_compute;
      6'h07:   s_axi_rdata = cyc_store;
      6'h08:   s_axi_rdata = cyc_store_stall;
      6'h09:   s_axi_rdata = {16'(D),    16'(N)};
      6'h0A:   s_axi_rdata = {16'(BLK),  16'(DV)};
      6'h0B:   s_axi_rdata = {15'd0, FOLD_PAR != 0, 8'(FRAC), 8'(DW)};
      6'h0C:   s_axi_rdata = BUILD_ID;
      default: s_axi_rdata = 32'hDEAD_BEEF;   // obviously wrong beats plausibly wrong
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s_axi_rvalid <= 1'b0;
      s_axi_rresp  <= 2'b00;
      ar_addr_q    <= '0;
    end else begin
      if (s_axi_arvalid && s_axi_arready) begin
        ar_addr_q    <= s_axi_araddr[RW+1:2];
        s_axi_rvalid <= 1'b1;
        s_axi_rresp  <= 2'b00;
      end else if (s_axi_rvalid && s_axi_rready) begin
        s_axi_rvalid <= 1'b0;
      end
    end
  end

endmodule
