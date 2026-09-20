// M8 · FlashAttention-lite: single-head attention with ONLINE softmax.
//
// Same result as attention_top (M5), fundamentally different dataflow.
//
// WHAT M5 MEASURED, AND WHY THIS EXISTS. Profiling M5 showed the softmax stage
// was 449 of 515 cycles at LANES=4 -- 87% of runtime, and completely immune to
// adding multipliers. Both of M5's problems have one root: softmax is defined
// over a whole row, so the naive design waits for the whole row to exist,
// materializes N x N matrices S and P, and then does one divide per element.
//
// THE TRICK. Carry a running max and retroactively fix up what you already
// accumulated. For each block of BLK keys:
//
//     m_new = max(m_run, max_j s_j)
//     corr  = exp(m_run - m_new)                      <= 1.0, the rebase factor
//     l_run = l_run * corr + sum_j exp(s_j - m_new)
//     acc_c = acc_c * corr + sum_j exp(s_j - m_new) * V[j][c]
//
// and only at the very end,  O[i][c] = acc_c / l_run.
//
// `corr` is the whole idea. When a later block holds a bigger score, everything
// accumulated under the old max is wrong by exactly exp(m_old - m_new) -- a
// CONSTANT. So one multiply rebases the entire history, which is why softmax,
// which looks irreducibly global, is streamable.
//
// WHAT IT BUYS (vs M5, same numbers, same LUT):
//   storage  O(N^2) for S and P   ->  O(DV) accumulator + 2 scalars per row
//   divides  N*N (one per P elem) ->  N (one reciprocal per row)
// At N=4 that is 4x fewer divides; at N=128 it is 128x, and the O(N^2) buffers
// that would never have fit on the FPGA simply do not exist.
//
// WHAT IT COSTS (measured, python/05_online_softmax_model.py, 200 seeds):
//   `corr` is applied once per BLOCK and its rounding COMPOUNDS N/BLK times
//   down the row -- the only place in the design where an error feeds back into
//   itself. Mean output error: BLK=1 0.004082, BLK=2 0.003924, BLK=4 0.003686
//   (M5 naive: 0.004071). So streaming harder is streaming less accurately.
//   Flash is a memory/divide win, NOT a free accuracy win.
//
// Spec: python/05_online_softmax_model.py. This RTL must match it bit-exactly.

module flash_top #(
  parameter int DW        = 16,
  parameter int FRAC      = 8,
  parameter int N         = 4,
  parameter int D         = 4,
  parameter int DV        = 4,
  parameter int BLK       = 2,    // keys folded per step; also the dot4 lane count
  parameter int RECIP_SH  = 24,   // reciprocal precision: r = 2^RECIP_SH / l
  parameter     EXP_FILE  = "rtl/vectors/exp_lut.hex"
)(
  input  logic clk,
  input  logic rst_n,

  // ---- input stream: Q row-major, then K row-major, then V row-major ------
  // One Q8.8 word per beat. The flat-bus interface this replaces was 131,072
  // bits wide at N=128 D=64, which is not a port that exists.
  input  logic                s_valid,
  output logic                s_ready,
  input  logic [DW-1:0]       s_data,
  input  logic                s_last,      // checked against the expected count

  // ---- output stream: O row-major ----------------------------------------
  output logic                m_valid,
  input  logic                m_ready,
  output logic [DW-1:0]       m_data,
  output logic                m_last,

  // ---- profiling taps for axil_regs --------------------------------------
  output logic                busy,
  output logic                done_pulse,
  output logic                ph_load,
  output logic                ph_compute,
  output logic                ph_store,
  output logic                in_stall,
  output logic                out_stall,
  output logic                err_overrun,   // beat arrived outside the load phase
  output logic                err_underrun   // s_last came early or late
);

  // ---- derived widths --------------------------------------------------------
  localparam int IW    = (N   > 1) ? $clog2(N)   : 1;
  localparam int CW    = (DV  > 1) ? $clog2(DV)  : 1;
  localparam int BW    = (BLK > 1) ? $clog2(BLK) : 1;
  // l_run: sum of N exps, each <= 2^FRAC. DW+IW is a comfortable bound.
  localparam int LW    = DW + IW + 1;
  // acc: sum of N products of two Q8.8 values -> Q16.16 plus log2(N) headroom.
  localparam int ACCW  = 2*DW + IW + 1;
  localparam int RNUMW = RECIP_SH + 1;                 // reciprocal numerator width
  localparam logic signed [DW-1:0] NEG_INF = {1'b1, {(DW-1){1'b0}}};

  // load/store sequencing. Two counters rather than one flat index divided by D,
  // because a divide by a non-power-of-two would cost real logic.
  localparam int LCW      = (D > DV) ? ((D  > 1) ? $clog2(D)  : 1)
                                     : ((DV > 1) ? $clog2(DV) : 1);
  localparam int OCW      = (N*DV > 1) ? $clog2(N*DV) : 1;
  logic [1:0]      ld_sel;          // 0 = Q, 1 = K, 2 = V
  logic [IW-1:0]   ld_row;
  logic [LCW-1:0]  ld_col;
  logic [OCW-1:0]  st_cnt;
  logic [LCW-1:0]  col_max;
  logic            ld_last, st_last;
  assign col_max = (ld_sel == 2'd2) ? LCW'(DV - 1) : LCW'(D - 1);
  assign ld_last = (ld_sel == 2'd2) && (ld_row == IW'(N-1)) && (ld_col == LCW'(DV-1));
  assign st_last = (st_cnt == OCW'(N*DV - 1));

  initial if (N % BLK != 0) $error("BLK=%0d must divide N=%0d", BLK, N);

  // ---- storage ---------------------------------------------------------------
  logic [D*DW-1:0]  q_mem [N];
  logic [D*DW-1:0]  k_mem [N];
  logic [DV*DW-1:0] v_mem [N];
  logic signed [DW-1:0] o_mem [N*DV];

  // THE ENTIRE RUNNING STATE. This is what replaces M5's N x N S and P buffers.
  logic signed [DW-1:0]   m_run;          // running max of scores seen so far
  logic        [LW-1:0]   l_run;          // running sum of exps, Q(FRAC)
  logic signed [ACCW-1:0] acc [DV];       // running sum of exp*V, Q(2*FRAC)

  logic signed [DW-1:0] s_blk [BLK];      // this block's scores
  logic        [DW-1:0] e_mem [BLK];      // this block's exps
  logic signed [DW-1:0] m_new_r;          // latched new max
  logic        [DW-1:0] corr_r;           // latched rebase factor, Q(FRAC)

  logic [IW-1:0] row_i, j_base;
  logic [CW-1:0] c_cnt;
  logic [BW-1:0] b_cnt;

  typedef enum logic [3:0] {
    IDLE, LOAD, SCORE_ISSUE, SCORE_WAIT, REBASE, EXPF, FOLD,
    RECIP_LOAD, RECIP_ITER, SCALE_OUT, DRAIN
  } state_t;
  state_t curr_state, next_state;

  // ---- MAC lanes: identical to qkt's, one per key in the block ---------------
  logic                 d4_in_valid, d4_out_ready;
  logic [BLK-1:0]       d4_in_ready, d4_out_valid;
  logic signed [DW-1:0] d4_s [BLK];
  logic d4_all_in_ready, d4_all_out_valid;
  assign d4_all_in_ready  = &d4_in_ready;
  assign d4_all_out_valid = &d4_out_valid;

  genvar l;
  generate
    for (l = 0; l < BLK; l++) begin : gen_lane
      dot4 #(.DW(DW), .FRAC(FRAC), .D(D), .SCALE_EN(1), .ROUND_EN(0)) u_dot (
        .clk(clk), .rst_n(rst_n),
        .in_valid(d4_in_valid), .in_ready(d4_in_ready[l]),
        .a_flat(q_mem[row_i]), .b_flat(k_mem[j_base + IW'(l)]),
        .out_valid(d4_out_valid[l]), .out_ready(d4_out_ready), .s(d4_s[l])
      );
    end
  endgenerate

  // ---- the new max, combinational over the live lane outputs ------------------
  logic signed [DW-1:0] m_new_c;
  always_comb begin
    m_new_c = m_run;
    for (int t = 0; t < BLK; t++)
      if (d4_s[t] > m_new_c) m_new_c = d4_s[t];
  end

  // ---- shared exp ROM ---------------------------------------------------------
  // ONE rom, two users, muxed by state: the rebase factor in SCORE_WAIT, the
  // score exponentials in EXPF. They never need it on the same cycle.
  logic signed [DW:0] exp_x;
  logic        [DW-1:0] exp_e;
  always_comb begin
    if (curr_state == SCORE_WAIT)
      exp_x = $signed({m_run[DW-1], m_run}) - $signed({m_new_c[DW-1], m_new_c});
    else
      exp_x = $signed({s_blk[b_cnt][DW-1], s_blk[b_cnt]})
            - $signed({m_new_r[DW-1], m_new_r});
  end
  exp_rom #(.DW(DW), .FRAC(FRAC), .EXP_FILE(EXP_FILE)) u_exp (.x(exp_x), .e(exp_e));

  // ---- signed-arithmetic helpers ---------------------------------------------
  // SystemVerilog makes an ENTIRE expression unsigned if ANY operand is
  // unsigned. corr and e are naturally unsigned (both are >= 0), but they
  // multiply signed accumulators -- so widen them into explicit signed values
  // first. Skipping this turns a negative acc into a huge positive one.
  logic signed [DW:0]   corr_s, e_s;
  assign corr_s = $signed({1'b0, corr_r});
  assign e_s    = $signed({1'b0, e_mem[b_cnt]});

  localparam signed [ACCW+DW:0] RND_F = 1 <<< (FRAC - 1);

  logic signed [ACCW+DW:0] acc_rebased;
  assign acc_rebased = ($signed(acc[c_cnt]) * corr_s + RND_F) >>> FRAC;

  logic [LW+DW:0] l_rebased;
  assign l_rebased = (({{(DW+1){1'b0}}, l_run} * {{(LW){1'b0}}, corr_r})
                      + (1 << (FRAC - 1))) >> FRAC;

  logic signed [DW-1:0] v_elem;
  logic signed [2*DW:0] ev_prod;
  assign v_elem  = $signed(v_mem[j_base + IW'(b_cnt)][int'(c_cnt)*DW +: DW]);
  assign ev_prod = e_s * $signed({v_elem[DW-1], v_elem});

  // ---- reciprocal divider (restoring, one quotient bit per cycle) -------------
  logic [RNUMW-1:0] div_num, div_q;
  logic [LW-1:0]    div_den;
  logic [LW:0]      div_rem, rem_shift;
  logic             rem_ge;
  logic [$clog2(RNUMW+1)-1:0] div_cnt;
  assign rem_shift = {div_rem[LW-1:0], div_num[RNUMW-1]};
  assign rem_ge    = (rem_shift >= {1'b0, div_den});

  // O = acc * r >> RECIP_SH, rounded and saturated back into DW bits.
  logic signed [ACCW+RNUMW:0] o_scaled;
  localparam signed [ACCW+RNUMW:0] RND_R  = 1 <<< (RECIP_SH - 1);
  localparam signed [ACCW+RNUMW:0] O_MAX  =  (1 <<< (DW-1)) - 1;
  localparam signed [ACCW+RNUMW:0] O_MIN  = -(1 <<< (DW-1));
  assign o_scaled = ($signed(acc[c_cnt]) * $signed({1'b0, div_q}) + RND_R) >>> RECIP_SH;

  logic signed [DW-1:0] o_sat;
  always_comb begin
    if      (o_scaled > O_MAX) o_sat = O_MAX[DW-1:0];
    else if (o_scaled < O_MIN) o_sat = O_MIN[DW-1:0];
    else                       o_sat = o_scaled[DW-1:0];
  end

  // ---- loop-bound wires -------------------------------------------------------
  logic last_c, last_b, last_blk, last_row;
  assign last_c   = (c_cnt  == CW'(DV - 1));
  assign last_b   = (b_cnt  == BW'(BLK - 1));
  assign last_blk = (j_base >= IW'(N - BLK));
  assign last_row = (row_i  == IW'(N - 1));

  // ---- FSM --------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) curr_state <= IDLE;
    else        curr_state <= next_state;
  end

  always_comb begin
    next_state = curr_state;
    case (curr_state)
      IDLE:        if (s_valid) next_state = LOAD;
      LOAD:        if (s_valid && s_ready && ld_last) next_state = SCORE_ISSUE;
      SCORE_ISSUE: if (d4_all_in_ready)  next_state = SCORE_WAIT;
      SCORE_WAIT:  if (d4_all_out_valid) next_state = REBASE;
      REBASE:      if (last_c) next_state = EXPF;      // DV cycles
      EXPF:        if (last_b) next_state = FOLD;      // BLK cycles
      FOLD:        if (last_b && last_c) begin         // BLK*DV cycles
                     if (last_blk) next_state = RECIP_LOAD;
                     else          next_state = SCORE_ISSUE;
                   end
      RECIP_LOAD:  next_state = RECIP_ITER;
      RECIP_ITER:  if (div_cnt == 0) next_state = SCALE_OUT;
      SCALE_OUT:   if (last_c) begin                   // DV cycles
                     if (last_row) next_state = DRAIN;
                     else          next_state = SCORE_ISSUE;
                   end
      DRAIN:       if (m_valid && m_ready && st_last) next_state = IDLE;
      default:     next_state = IDLE;
    endcase
  end

  // ---- datapath ----------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      row_i <= '0; j_base <= '0; c_cnt <= '0; b_cnt <= '0;
      ld_sel <= '0; ld_row <= '0; ld_col <= '0; st_cnt <= '0;
      m_run <= NEG_INF; l_run <= '0; m_new_r <= '0; corr_r <= '0;
      div_num <= '0; div_den <= '0; div_rem <= '0; div_q <= '0; div_cnt <= '0;
      for (int t = 0; t < N;    t++) begin q_mem[t]<='0; k_mem[t]<='0; v_mem[t]<='0; end
      for (int t = 0; t < DV;   t++) acc[t]   <= '0;
      for (int t = 0; t < BLK;  t++) begin s_blk[t]<='0; e_mem[t]<='0; end
      for (int t = 0; t < N*DV; t++) o_mem[t] <= '0;
    end else begin
      // Streaming load. One word per beat, routed by (ld_sel, ld_row, ld_col).
      // The accumulators are armed on the FIRST beat rather than on a separate
      // start pulse, so there is no window where a beat can arrive before the
      // state it will be folded into has been cleared.
      if (curr_state == IDLE && s_valid) begin
        ld_sel <= '0; ld_row <= '0; ld_col <= '0;
        row_i  <= '0; j_base <= '0; m_run <= NEG_INF; l_run <= '0;
        for (int t = 0; t < DV; t++) acc[t] <= '0;
      end

      if (curr_state == LOAD && s_valid && s_ready) begin
        case (ld_sel)
          2'd0: q_mem[ld_row][ld_col*DW +: DW] <= s_data;
          2'd1: k_mem[ld_row][ld_col*DW +: DW] <= s_data;
          default: v_mem[ld_row][ld_col*DW +: DW] <= s_data;
        endcase
        if (ld_col == col_max) begin
          ld_col <= '0;
          if (ld_row == IW'(N-1)) begin
            ld_row <= '0;
            ld_sel <= ld_sel + 2'd1;
          end else ld_row <= ld_row + IW'(1);
        end else ld_col <= ld_col + LCW'(1);
      end

      if (curr_state == SCALE_OUT && last_c && last_row) st_cnt <= '0;
      if (curr_state == DRAIN && m_valid && m_ready)     st_cnt <= st_cnt + OCW'(1);

      case (curr_state)
        SCORE_WAIT: if (d4_all_out_valid) begin
          for (int t = 0; t < BLK; t++) s_blk[t] <= d4_s[t];
          m_new_r <= m_new_c;
          corr_r  <= exp_e;                      // exp(m_run - m_new), Q(FRAC)
          c_cnt   <= '0;
        end

        // Rebase the history by corr. One accumulator element per cycle, so one
        // multiplier is reused DV times instead of DV multipliers sitting idle.
        REBASE: begin
          acc[c_cnt] <= acc_rebased[ACCW-1:0];
          if (c_cnt == '0) l_run <= l_rebased[LW-1:0];
          if (last_c) begin c_cnt <= '0; b_cnt <= '0; end
          else              c_cnt <= c_cnt + CW'(1);
        end

        // Exponentiate this block's scores and fold them into the row sum.
        EXPF: begin
          e_mem[b_cnt] <= exp_e;
          l_run        <= l_run + LW'(exp_e);
          if (last_b) begin b_cnt <= '0; c_cnt <= '0; end
          else              b_cnt <= b_cnt + BW'(1);
        end

        // acc[c] += e_j * V[j][c], one product per cycle over the BLK x DV grid.
        FOLD: begin
          acc[c_cnt] <= acc[c_cnt] + ACCW'(ev_prod);
          if (last_c) begin
            c_cnt <= '0;
            if (last_b) begin
              b_cnt <= '0;
              m_run <= m_new_r;                  // the block is now history
              if (!last_blk) j_base <= j_base + IW'(BLK);
            end else b_cnt <= b_cnt + BW'(1);
          end else c_cnt <= c_cnt + CW'(1);
        end

        // ONE divide for the whole row -- M5 did N of these per row.
        RECIP_LOAD: begin
          div_num <= RNUMW'(1 << RECIP_SH) + RNUMW'(l_run >> 1);
          div_den <= l_run;
          div_rem <= '0;
          div_q   <= '0;
          div_cnt <= ($clog2(RNUMW+1))'(RNUMW - 1);   // NOT RNUMW: see softmax.sv
        end

        RECIP_ITER: begin
          if (rem_ge) div_rem <= rem_shift - {1'b0, div_den};
          else        div_rem <= rem_shift;
          div_q   <= {div_q[RNUMW-2:0], rem_ge};
          div_num <= {div_num[RNUMW-2:0], 1'b0};
          div_cnt <= div_cnt - 1'b1;
          if (div_cnt == 0) c_cnt <= '0;
        end

        // Multiply the accumulator by the reciprocal -- no divide per element.
        SCALE_OUT: begin
          o_mem[int'(row_i)*DV + int'(c_cnt)] <= o_sat;
          if (last_c) begin
            c_cnt <= '0;
            if (!last_row) begin                 // reset the running state
              row_i  <= row_i + IW'(1);
              j_base <= '0;
              m_run  <= NEG_INF;
              l_run  <= '0;
              for (int t = 0; t < DV; t++) acc[t] <= '0;
            end
          end else c_cnt <= c_cnt + CW'(1);
        end

        default: ;
      endcase
    end
  end

  assign s_ready      = (curr_state == LOAD);
  assign m_valid      = (curr_state == DRAIN);
  assign m_data       = o_mem[st_cnt];
  assign m_last       = st_last;
  assign d4_in_valid  = (curr_state == SCORE_ISSUE);
  assign d4_out_ready = (curr_state == SCORE_WAIT);

  // ---- profiling ------------------------------------------------------------
  // ph_compute is everything that is neither load nor store nor idle, so the
  // three phases plus idle account for every cycle by construction. That is why
  // axil_regs can treat a TOTAL-versus-sum discrepancy as a real signal.
  assign busy       = (curr_state != IDLE);
  assign ph_load    = (curr_state == LOAD);
  assign ph_store   = (curr_state == DRAIN);
  assign ph_compute = busy && !ph_load && !ph_store;
  assign in_stall   = ph_load  && !s_valid;
  assign out_stall  = ph_store && !m_ready;
  assign done_pulse = (curr_state == DRAIN) && m_valid && m_ready && st_last;

  // A beat offered while we are not loading is dropped on the floor; say so
  // rather than silently losing it. Same for a framing marker in the wrong place.
  assign err_overrun  = s_valid && !s_ready && (curr_state != IDLE);
  assign err_underrun = (curr_state == LOAD) && s_valid && s_ready &&
                        (s_last ^ ld_last);

endmodule
