// M5 · top level: single-head scaled dot-product attention, end to end.
//
//   Q,K,V  ->  qkt  ->  S  ->  row_max  ->  m  ->  softmax  ->  P  ->  pv  ->  O
//
// There is NO top-level FSM. Each stage already speaks valid/ready, so the
// chain is built by connecting each producer's output handshake to the next
// consumer's input handshake. Backpressure then propagates backwards for free:
// if the consumer of O stalls, pv stalls, which stalls softmax, and so on to
// the Q/K/V port. That is the payoff for having put a handshake on every module
// instead of hard-wiring cycle counts between them.
//
// TWO THINGS ARE LATCHED AT THIS LEVEL, and both for the same reason -- a value
// is needed LATER than the stage that produced it:
//   S_reg : softmax needs S, but row_max consumes it first
//   V_reg : pv needs V, which arrived N*N*(D+2)+... cycles earlier
// Everything else flows straight through, producer to consumer.
//
// THIS IS THE NAIVE (NOT FLASH) DATAFLOW: every stage runs to completion and
// materializes its full N x N intermediate before the next one starts. S and P
// both exist in full. That O(N^2) storage is exactly what FlashAttention
// removes, and removing it is M6 -- so this module is the baseline that M6 has
// to beat on area while matching on numbers.
//
// Latency (N=D=DV=4, LANES=1):
//   qkt      (N*N/LANES)*(D+2) + overhead    ~102 cy
//   row_max  N*N                              ~18 cy
//   softmax  N*(N + N*(NUMW+3))               ~440 cy   <- the divider dominates
//   pv       (N*DV/LANES)*(N+2) + overhead    ~102 cy
// The divider being 4x everything else is the measurement that justifies the
// reciprocal-multiply variant discussed in softmax.sv.

module attention_top #(
  parameter int DW    = 16,
  parameter int FRAC  = 8,
  parameter int N     = 4,
  parameter int D     = 4,
  parameter int DV    = 4,
  parameter int LANES = 1,
  parameter     EXP_FILE = "rtl/vectors/exp_lut.hex"
)(
  input  logic clk,
  input  logic rst_n,

  input  logic                in_valid,
  output logic                in_ready,
  input  logic [N*D*DW-1:0]   Q_flat,
  input  logic [N*D*DW-1:0]   K_flat,
  input  logic [N*DV*DW-1:0]  V_flat,

  output logic                out_valid,
  input  logic                out_ready,
  output logic [N*DV*DW-1:0]  O_flat
);

  // inter-stage buses
  logic [N*N*DW-1:0]  s_bus, s_reg, p_bus;
  logic [N*DW-1:0]    m_bus;
  logic [N*DV*DW-1:0] v_reg;

  logic qkt_ov, rm_ir, rm_ov, sm_ir, sm_ov, pv_ir;

  // ---- stage 1: QK^T / sqrt(d) ----------------------------------------------
  qkt #(.DW(DW), .FRAC(FRAC), .N(N), .D(D), .LANES(LANES)) u_qkt (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_ready(in_ready),
    .Q_flat(Q_flat), .K_flat(K_flat),
    .out_valid(qkt_ov), .out_ready(rm_ir), .S_flat(s_bus)
  );

  // ---- stage 2: row max (the stability trick) --------------------------------
  row_max #(.DW(DW), .N(N)) u_rowmax (
    .clk(clk), .rst_n(rst_n),
    .in_valid(qkt_ov), .in_ready(rm_ir), .S_flat(s_bus),
    .out_valid(rm_ov), .out_ready(sm_ir), .m_flat(m_bus)
  );

  // ---- stage 3: exp LUT + normalize ------------------------------------------
  // Fed from s_reg, not s_bus: qkt is free to start the NEXT tile the moment
  // row_max takes S, so s_bus must not be trusted past that beat.
  softmax #(.DW(DW), .FRAC(FRAC), .N(N), .EXP_FILE(EXP_FILE)) u_softmax (
    .clk(clk), .rst_n(rst_n),
    .in_valid(rm_ov), .in_ready(sm_ir), .S_flat(s_reg), .m_flat(m_bus),
    .out_valid(sm_ov), .out_ready(pv_ir), .P_flat(p_bus)
  );

  // ---- stage 4: P.V -----------------------------------------------------------
  pv #(.DW(DW), .FRAC(FRAC), .N(N), .DV(DV), .LANES(LANES)) u_pv (
    .clk(clk), .rst_n(rst_n),
    .in_valid(sm_ov), .in_ready(pv_ir), .P_flat(p_bus), .V_flat(v_reg),
    .out_valid(out_valid), .out_ready(out_ready), .O_flat(O_flat)
  );

  // ---- the two skip-ahead registers -------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s_reg <= '0;
      v_reg <= '0;
    end else begin
      // V arrives with Q and K but is not consumed until the very last stage.
      if (in_valid && in_ready)  v_reg <= V_flat;
      // Capture S on the beat row_max takes it, for softmax to use afterwards.
      if (qkt_ov && rm_ir)       s_reg <= s_bus;
    end
  end

endmodule
