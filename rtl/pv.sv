// M5 · module 5/5: P·V engine
//
// O[i][c] = sum_j P[i][j] * V[j][c]   -> the attention output, N x DV.
//
// THE SAME HARDWARE AS QK^T. Both stages are "contract one shared axis of two
// matrices", so both are built out of the same dot4 MAC atom, sequenced by the
// same FSM shape. Only three things differ:
//   1. the contracted axis is N (keys) here, D (features) in qkt
//   2. SCALE_EN=0 -- there is no 1/sqrt(d) on this side
//   3. ROUND_EN=1 -- measured worth it here, not worth it in qkt
// Saying "attention is two matmuls with a softmax between them" is cheap;
// having one multiplier module serve both is what makes it true in silicon.
//
// NO TRANSPOSE ON P. P is (query x key) and V is (key x value): the key axis is
// already the shared inner dimension, so P.V contracts directly. Q.K^T needed
// the transpose only because both operands were indexed (token x feature).
//
// V *IS* transposed -- but for free. dot4 wants its second operand as one
// contiguous length-N vector, and V is stored row-major as V[j][c], so column c
// is strided. CHUNK 3 below gathers it with a generate loop: in hardware a
// transpose of a fixed-size array is not an operation at all, it is just which
// wire goes where. This is the cheapest thing in the design and the most
// expensive thing in the CUDA version (uncoalesced access).
//
// Interface: valid/ready. One input beat takes P and V; one output beat gives O.

module pv #(
  parameter int DW    = 16,
  parameter int FRAC  = 8,
  parameter int N     = 4,    // keys -- the axis being contracted
  parameter int DV    = 4,    // value dim -- columns of the output
  parameter int LANES = 1     // parallel dot4s; must divide DV
)(
  input  logic clk,
  input  logic rst_n,

  input  logic                in_valid,
  output logic                in_ready,
  input  logic [N*N*DW-1:0]   P_flat,   // P[i][j] at [(i*N+j)*DW +: DW]
  input  logic [N*DV*DW-1:0]  V_flat,   // V[j][c] at [(j*DV+c)*DW +: DW]

  output logic                out_valid,
  input  logic                out_ready,
  output logic [N*DV*DW-1:0]  O_flat    // O[i][c] at [(i*DV+c)*DW +: DW]
);

  localparam int IW  = (N  > 1) ? $clog2(N)  : 1;
  localparam int CW  = (DV > 1) ? $clog2(DV) : 1;

  initial if (DV % LANES != 0) $error("LANES=%0d must divide DV=%0d", LANES, DV);

  typedef enum logic [1:0] { IDLE, ISSUE, WAIT, DONE } state_t;
  state_t curr_state, next_state;

  logic [N*DW-1:0]      p_mem [N];       // one query row of P, ready for dot4
  logic [N*DW-1:0]      v_col [DV];      // one COLUMN of V, gathered (see below)
  logic signed [DW-1:0] o_mem [N*DV];

  logic [IW-1:0] row_i;                  // output row  (query)
  logic [CW-1:0] c_base;                 // output column block (value dim)

  logic                 d4_in_valid, d4_out_ready;
  logic [LANES-1:0]     d4_in_ready, d4_out_valid;
  logic signed [DW-1:0] d4_s [LANES];
  logic d4_all_in_ready, d4_all_out_valid;
  assign d4_all_in_ready  = &d4_in_ready;
  assign d4_all_out_valid = &d4_out_valid;

  logic last_c, last_pair;
  assign last_c    = (c_base >= CW'(DV - LANES));
  assign last_pair = last_c && (row_i == IW'(N - 1));

  // -- CHUNK 3a: the free transpose -------------------------------------------
  // v_col[c] is V's column c as a contiguous length-N vector. Pure wiring: the
  // indices are compile-time constants, so this synthesizes to zero logic.
  genvar c, j;
  generate
    for (c = 0; c < DV; c++) begin : gen_vcol
      for (j = 0; j < N; j++) begin : gen_vrow
        assign v_col[c][j*DW +: DW] = V_flat[(j*DV + c)*DW +: DW];
      end
    end
  endgenerate

  // -- CHUNK 3b: the MAC lanes ------------------------------------------------
  // D(N): the dot product runs over the KEY axis, length N.
  // SCALE_EN(0): net shift becomes exactly >>> FRAC -- HALF_MUL is 2^FRAC and
  //              the total shift is 2*FRAC, so (acc * 2^FRAC) >> 2*FRAC.
  // ROUND_EN(1): +half an LSB first. This is the 21% mean-error win.
  genvar l;
  generate
    for (l = 0; l < LANES; l++) begin : gen_lane
      dot4 #(.DW(DW), .FRAC(FRAC), .D(N), .SCALE_EN(0), .ROUND_EN(1)) u_dot (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (d4_in_valid),
        .in_ready  (d4_in_ready[l]),
        .a_flat    (p_mem[row_i]),               // broadcast: one P row
        .b_flat    (v_col[c_base + CW'(l)]),     // per-lane: one V column
        .out_valid (d4_out_valid[l]),
        .out_ready (d4_out_ready),
        .s         (d4_s[l])
      );
    end
  endgenerate

  // -- CHUNK 4: controller FSM (identical shape to qkt) ------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) curr_state <= IDLE;
    else        curr_state <= next_state;
  end

  always_comb begin
    next_state = curr_state;
    case (curr_state)
      IDLE:  if (in_valid) next_state = ISSUE;
      ISSUE: if (d4_all_in_ready) next_state = WAIT;
      WAIT:  if (d4_all_out_valid) begin
               if (last_pair) next_state = DONE;
               else           next_state = ISSUE;
             end
      DONE:  if (out_ready) next_state = IDLE;
      default: next_state = IDLE;
    endcase
  end

  // -- CHUNK 5: datapath -------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      row_i <= '0; c_base <= '0;
      for (int t = 0; t < N;    t++) p_mem[t] <= '0;
      for (int t = 0; t < N*DV; t++) o_mem[t] <= '0;
    end else begin
      if (in_valid && in_ready) begin
        // P row i is already contiguous in P_flat -- slice it whole.
        for (int t = 0; t < N; t++) p_mem[t] <= P_flat[t*N*DW +: N*DW];
        row_i <= '0; c_base <= '0;
      end
      if (curr_state == WAIT && d4_all_out_valid) begin
        for (int t = 0; t < LANES; t++)
          o_mem[int'(row_i)*DV + int'(c_base) + t] <= d4_s[t];
        if (last_c) begin
          c_base <= '0;
          row_i  <= row_i + IW'(1);
        end else begin
          c_base <= c_base + CW'(LANES);
        end
      end
    end
  end

  // -- CHUNK 6: outputs --------------------------------------------------------
  assign in_ready     = (curr_state == IDLE);
  assign out_valid    = (curr_state == DONE);
  assign d4_in_valid  = (curr_state == ISSUE);
  assign d4_out_ready = (curr_state == WAIT);

  genvar g;
  generate
    for (g = 0; g < N*DV; g++) begin : gen_pack
      assign O_flat[g*DW +: DW] = o_mem[g];
    end
  endgenerate

endmodule
