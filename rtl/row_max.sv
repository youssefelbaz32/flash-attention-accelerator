// M5 · module 3/5: row-max engine
//
// m[i] = max_j S[i][j]   for each of the N rows of the score matrix.
//
// WHY THIS MODULE EXISTS AT ALL: softmax(S) == softmax(S - c) for any constant
// c, because the exp() factors out of numerator and denominator and cancels.
// Choosing c = row max makes every exp argument <= 0, so exp() lands in (0,1]
// and can never overflow. In float that is a numerical nicety; in fixed point
// it is structural -- it is the reason the exp LUT only has to cover the
// negative half-line, which halves the ROM. The "stable softmax" line in the
// Python model is literally this block of hardware.
//
// Area choice: ONE comparator, scanning row-major over N*N cycles. A per-row
// comparator tree would finish in log2(N) levels x N cycles at N times the
// comparators; at these dimensions the scan is free and the tree is waste.
// If N grows and this becomes the critical path, parameterize it the way qkt
// parameterizes LANES.
//
// Interface: valid/ready. One input beat takes the whole S matrix, one output
// beat presents all N maxima.

module row_max #(
  parameter int DW = 16,   // word width, Q8.8
  parameter int N  = 4     // sequence length -> N rows, N columns
)(
  input  logic clk,
  input  logic rst_n,

  input  logic              in_valid,
  output logic              in_ready,
  input  logic [N*N*DW-1:0] S_flat,    // S[i][j] at [(i*N+j)*DW +: DW]

  output logic              out_valid,
  input  logic              out_ready,
  output logic [N*DW-1:0]   m_flat     // m[i] at [i*DW +: DW]
);

  localparam int IW = (N > 1) ? $clog2(N) : 1;

  // The identity element for signed max. Starting the running max here means
  // the first real compare always wins, so no "first element" special case.
  localparam logic signed [DW-1:0] NEG_INF = {1'b1, {(DW-1){1'b0}}};   // -32768

  typedef enum logic [1:0] { IDLE, SCAN, DONE } state_t;
  state_t curr_state, next_state;

  logic signed [DW-1:0] s_mem [N*N];
  logic signed [DW-1:0] m_mem [N];
  logic signed [DW-1:0] cur_max;        // running max of the row in flight
  logic [IW-1:0] i_cnt, j_cnt;          // row / column being examined

  logic last_j, last_elem;
  assign last_j    = (j_cnt == IW'(N-1));
  assign last_elem = last_j && (i_cnt == IW'(N-1));

  // THE comparator -- one instance, shared by every element of the matrix.
  logic signed [DW-1:0] cand, nxt_max;
  assign cand    = s_mem[int'(i_cnt)*N + int'(j_cnt)];
  assign nxt_max = (cand > cur_max) ? cand : cur_max;

  always_ff @(posedge clk) begin
    if (!rst_n) curr_state <= IDLE;
    else        curr_state <= next_state;
  end

  always_comb begin
    next_state = curr_state;
    case (curr_state)
      IDLE: if (in_valid) next_state = SCAN;
      SCAN: if (last_elem) next_state = DONE;
      DONE: if (out_ready) next_state = IDLE;
      default: next_state = IDLE;
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      i_cnt <= '0; j_cnt <= '0; cur_max <= NEG_INF;
      for (int t = 0; t < N*N; t++) s_mem[t] <= '0;
      for (int t = 0; t < N;   t++) m_mem[t] <= '0;
    end else begin
      if (in_valid && in_ready) begin                 // capture on the beat
        for (int t = 0; t < N*N; t++) s_mem[t] <= S_flat[t*DW +: DW];
        i_cnt <= '0; j_cnt <= '0; cur_max <= NEG_INF;
      end else if (curr_state == SCAN) begin
        if (last_j) begin                             // row finished
          m_mem[i_cnt] <= nxt_max;
          cur_max <= NEG_INF;                         // rearm for the next row
          j_cnt   <= '0;
          i_cnt   <= i_cnt + IW'(1);
        end else begin
          cur_max <= nxt_max;
          j_cnt   <= j_cnt + IW'(1);
        end
      end
    end
  end

  assign in_ready  = (curr_state == IDLE);
  assign out_valid = (curr_state == DONE);

  genvar g;
  generate
    for (g = 0; g < N; g++) begin : gen_pack
      assign m_flat[g*DW +: DW] = m_mem[g];
    end
  endgenerate

endmodule
