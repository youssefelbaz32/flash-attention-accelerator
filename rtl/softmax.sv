// M5 · module 4/5: fixed-point softmax
//
// P[i][j] = exp(S[i][j] - m[i]) / sum_j exp(S[i][j] - m[i])
//
// This is the module that does not exist in the Python model. `np.exp` and `/`
// are one character each; here they are a ROM and a restoring divider, and the
// whole error budget of the accelerator is decided by how they are sized.
//
// THREE SUB-BLOCKS, three costs:
//
//  1. SUBTRACT  x = S - m, in DW+1 bits. The difference of two DW-bit signed
//     numbers needs one extra bit (32767 - (-32768) does not fit in 16). x <= 0
//     by construction because m is the row max -- which is the entire reason
//     row_max.sv runs first.
//
//  2. EXP LUT   a ROM of EXP_N entries covering x in [-EXP_RANGE, 0].
//     index = (-x) >> EXP_SH, saturated at EXP_N-1. Because x <= 0 we only
//     store the negative half-line: half the ROM for free. Entries past
//     exp(-6.24) quantize to 0 in Q8.8, so EXP_RANGE=8 is already generous.
//     The table is NOT written here -- it is $readmemh'd from the same .hex
//     that python/04_rtl_fixed_model.py generates, so the software model and
//     the hardware cannot drift apart. One definition, two consumers.
//
//  3. DIVIDE    P = ((e << FRAC) + sum/2) / sum, one sequential restoring
//     divider reused for all N*N elements. NUMW cycles per element.
//     The +sum/2 is round-to-nearest; floor division biases all N terms the
//     same way and cost 21% mean error in a 200-seed sweep.
//
//     LATENCY/AREA ALTERNATIVE (not taken, deliberately): compute one
//     reciprocal per ROW and multiply, turning N divides into 1 divide + N
//     multiplies -- roughly 4x faster here. Rejected for M5 because it adds a
//     second rounding step that would no longer match the Python spec
//     bit-exactly, and M5's contract is exactness. Revisit at M6, where the
//     online-softmax rescaling changes the arithmetic anyway.
//
// Interface: valid/ready. One input beat takes S and m; one output beat gives P.

module softmax #(
  parameter int DW        = 16,
  parameter int FRAC      = 8,
  parameter int N         = 4,
  parameter int EXP_N     = 256,   // LUT entries
  parameter int EXP_RANGE = 8,     // LUT covers x in [-EXP_RANGE, 0], real units
  parameter     EXP_FILE  = "rtl/vectors/exp_lut.hex"
)(
  input  logic clk,
  input  logic rst_n,

  input  logic              in_valid,
  output logic              in_ready,
  input  logic [N*N*DW-1:0] S_flat,
  input  logic [N*DW-1:0]   m_flat,

  output logic              out_valid,
  input  logic              out_ready,
  output logic [N*N*DW-1:0] P_flat
);

  // -- derived geometry --------------------------------------------------------
  localparam int IW     = (N > 1) ? $clog2(N) : 1;
  // sum of N entries, each <= 2^FRAC -> needs FRAC+1+clog2(N) bits; DW is wider
  // than FRAC+1 in every sane config, so DW+clog2(N) is a safe, simple bound.
  localparam int SUMW   = DW + IW;
  // numerator = (e << FRAC) + sum/2 ; +1 bit so the add can never wrap
  localparam int NUMW   = DW + FRAC + 1;

  // -- storage -----------------------------------------------------------------
  logic signed [DW-1:0] s_mem [N*N];
  logic signed [DW-1:0] m_mem [N];
  logic        [DW-1:0] e_mem [N];        // exp values for the row in flight
  logic        [DW-1:0] p_mem [N*N];

  logic [SUMW-1:0] sum_reg;               // running sum of the row's exps
  logic [IW-1:0]   i_cnt, j_cnt;

  typedef enum logic [2:0] {
    IDLE, EXP, DIV_LOAD, DIV_ITER, DIV_STORE, DONE
  } state_t;
  state_t curr_state, next_state;

  // -- sub-block 1+2: subtract and LUT lookup ---------------------------------
  // The subtract needs DW+1 bits: the difference of two DW-bit signed numbers
  // does not fit in DW (32767 - (-32768) = 65535). The lookup itself lives in
  // exp_rom, shared with the M6 flash datapath.
  logic signed [DW:0] x;
  logic        [DW-1:0] lut_val;

  assign x = $signed({s_mem[int'(i_cnt)*N + int'(j_cnt)][DW-1],
                      s_mem[int'(i_cnt)*N + int'(j_cnt)]})
           - $signed({m_mem[i_cnt][DW-1], m_mem[i_cnt]});

  exp_rom #(.DW(DW), .FRAC(FRAC), .EXP_N(EXP_N), .EXP_RANGE(EXP_RANGE),
            .EXP_FILE(EXP_FILE)) u_exp (.x(x), .e(lut_val));

  // -- sub-block 3: restoring divider ------------------------------------------
  // Classic shift-compare-subtract: one quotient bit per cycle, MSB first.
  // All operands are non-negative here (exp >= 0, sum > 0), so no sign handling.
  logic [NUMW-1:0] div_num, div_q;
  logic [SUMW-1:0] div_den;
  logic [SUMW:0]   div_rem;               // one bit wider: holds rem<<1 | bit
  logic [$clog2(NUMW+1)-1:0] div_cnt;

  logic [SUMW:0] rem_shift;
  logic          rem_ge;
  assign rem_shift = {div_rem[SUMW-1:0], div_num[NUMW-1]};   // bring down next bit
  assign rem_ge    = (rem_shift >= {1'b0, div_den});

  logic last_j, last_elem;
  assign last_j    = (j_cnt == IW'(N-1));
  assign last_elem = last_j && (i_cnt == IW'(N-1));

  // -- FSM ---------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) curr_state <= IDLE;
    else        curr_state <= next_state;
  end

  always_comb begin
    next_state = curr_state;
    case (curr_state)
      IDLE:      if (in_valid) next_state = EXP;
      // walk the row computing exps and accumulating their sum
      EXP:       if (last_j) next_state = DIV_LOAD;
      DIV_LOAD:  next_state = DIV_ITER;
      DIV_ITER:  if (div_cnt == 0) next_state = DIV_STORE;
      DIV_STORE: begin
                   if (last_elem)   next_state = DONE;
                   else if (last_j) next_state = EXP;       // next row: new exps
                   else             next_state = DIV_LOAD;  // same row: next j
                 end
      DONE:      if (out_ready) next_state = IDLE;
      default:   next_state = IDLE;
    endcase
  end

  // -- datapath ----------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      i_cnt <= '0; j_cnt <= '0; sum_reg <= '0;
      div_num <= '0; div_den <= '0; div_rem <= '0; div_q <= '0; div_cnt <= '0;
      for (int t = 0; t < N*N; t++) begin s_mem[t] <= '0; p_mem[t] <= '0; end
      for (int t = 0; t < N;   t++) begin m_mem[t] <= '0; e_mem[t] <= '0; end
    end else begin
      if (in_valid && in_ready) begin
        for (int t = 0; t < N*N; t++) s_mem[t] <= S_flat[t*DW +: DW];
        for (int t = 0; t < N;   t++) m_mem[t] <= m_flat[t*DW +: DW];
        i_cnt <= '0; j_cnt <= '0; sum_reg <= '0;
      end

      case (curr_state)
        EXP: begin
          e_mem[j_cnt] <= lut_val;
          sum_reg      <= sum_reg + SUMW'(lut_val);
          if (last_j) j_cnt <= '0;                 // rewind to divide the row
          else        j_cnt <= j_cnt + IW'(1);
        end

        DIV_LOAD: begin
          // (e << FRAC) + sum/2  -- the round-to-nearest numerator
          div_num <= (NUMW'(e_mem[j_cnt]) << FRAC) + NUMW'(sum_reg >> 1);
          div_den <= sum_reg;
          div_rem <= '0;
          div_q   <= '0;
          // NUMW-1, not NUMW. The FSM's exit test (div_cnt == 0) is evaluated
          // while curr_state is STILL DIV_ITER, so the datapath body executes
          // on that final cycle too. Loading NUMW gives NUMW+1 iterations and
          // an exactly-2x quotient -- a power-of-two error is always a
          // shift-count bug.
          div_cnt <= ($clog2(NUMW+1))'(NUMW - 1);
        end

        DIV_ITER: begin
          // shift-compare-subtract, one quotient bit per cycle
          if (rem_ge) div_rem <= rem_shift - {1'b0, div_den};
          else        div_rem <= rem_shift;
          div_q   <= {div_q[NUMW-2:0], rem_ge};
          div_num <= {div_num[NUMW-2:0], 1'b0};
          div_cnt <= div_cnt - 1'b1;
        end

        DIV_STORE: begin
          p_mem[int'(i_cnt)*N + int'(j_cnt)] <= div_q[DW-1:0];
          if (last_j) begin
            j_cnt   <= '0;
            i_cnt   <= i_cnt + IW'(1);
            sum_reg <= '0;                          // rearm for the next row
          end else begin
            j_cnt <= j_cnt + IW'(1);
          end
        end

        default: ;
      endcase
    end
  end

  assign in_ready  = (curr_state == IDLE);
  assign out_valid = (curr_state == DONE);

  genvar g;
  generate
    for (g = 0; g < N*N; g++) begin : gen_pack
      assign P_flat[g*DW +: DW] = p_mem[g];
    end
  endgenerate

endmodule
