// M5 · module 2/5: QK^T engine
//
// Computes the full score matrix  S[i][j] = (Q_row_i . K_row_j) / sqrt(D)
// for all i,j in [0,N). Sequences N*N dot products over LANES parallel dot4
// instances, driving each one's valid/ready handshake from a controller FSM.
//
// The 1/sqrt(D) scaling is NOT done here -- dot4 is instantiated with
// SCALE_EN=1 so the scale folds into dot4's single final rounding shift. Doing
// it again at this level would round twice and lose a bit for free.
//
// LANES is the area/latency knob and the reason this module is parameterized:
//   LANES=1 -> 1 multiplier,  N*N*(D+2) cycles  (small, slow)
//   LANES=N -> N multipliers, N*(D+2) cycles    (N x the DSPs, N x faster)
// Sweeping it produces the LUT/DSP-vs-latency Pareto curve. LANES must divide N.
//
// Interface: valid/ready (AXI4-Stream handshake semantics). One input beat
// loads all of Q and K; one output beat presents the whole S matrix.
//
// Target: Vivado XSim + synth (Xilinx/AMD). Also simulates under `iverilog -g2012`.

// =============================================================================
// CHUNK 1: module declaration + parameters + ports
// -----------------------------------------------------------------------------
module qkt #(
  parameter int DW    = 16,   // word width, Q8.8
  parameter int FRAC  = 8,    // fractional bits
  parameter int N     = 4,    // sequence length (rows of Q, rows of K)
  parameter int D     = 4,    // head dim (features contracted by the dot product)
  parameter int LANES = 1     // dot4 instances working in parallel; must divide N
)(
  input  logic clk,
  input  logic rst_n,                       // active-LOW synchronous reset

  input  logic                  in_valid,
  output logic                  in_ready,
  input  logic [N*D*DW-1:0]     Q_flat,     // Q[i][k] at [(i*D+k)*DW +: DW]
  input  logic [N*D*DW-1:0]     K_flat,     // K[j][k] at [(j*D+k)*DW +: DW]

  output logic                  out_valid,
  input  logic                  out_ready,
  output logic [N*N*DW-1:0]     S_flat      // S[i][j] at [(i*N+j)*DW +: DW]
);

  // Index widths. $clog2(N) is 0 when N==1, and a [-1:0] range is illegal, so
  // every counter width is floored at 1.
  localparam int IW = (N > 1) ? $clog2(N) : 1;

  initial begin
    if (N % LANES != 0) $error("LANES=%0d must divide N=%0d", LANES, N);
    if (LANES > N)      $error("LANES=%0d exceeds N=%0d", LANES, N);
  end

// =============================================================================
// CHUNK 2: internal signals
// -----------------------------------------------------------------------------
  typedef enum logic [1:0] { IDLE, ISSUE, WAIT, DONE } state_t;
  state_t curr_state, next_state;

  // Row storage. Each entry is a D*DW flat vector so it can be handed to a
  // dot4 .a_flat/.b_flat port with no repacking. Deliberately UNSIGNED: this is
  // a container of D separate Q8.8 words, not one wide signed number -- dot4
  // re-interprets each DW slice as signed internally.
  logic [D*DW-1:0]      q_mem [N];
  logic [D*DW-1:0]      k_mem [N];
  logic signed [DW-1:0] s_mem [N*N];   // scores, Q8.8, signed (these ARE numbers)

  // Pair counters: which (query row, key column-block) is in flight.
  logic [IW-1:0] row_i;    // 0..N-1     query row index
  logic [IW-1:0] j_base;   // 0,LANES,.. base key index of the current block

  // dot4 fan-out/fan-in. All LANES instances are started on the same cycle with
  // identical-latency work, so they move in lockstep; the reductions below are
  // belt-and-braces (and stay correct if a lane ever gains its own FSM).
  logic                 d4_in_valid;
  logic [LANES-1:0]     d4_in_ready;
  logic                 d4_out_ready;
  logic [LANES-1:0]     d4_out_valid;
  logic signed [DW-1:0] d4_s [LANES];

  // `assign`, NOT `logic x = &y;` -- the latter is a one-shot initializer that
  // samples at time 0 and never updates again.
  logic d4_all_in_ready, d4_all_out_valid;
  assign d4_all_in_ready  = &d4_in_ready;
  assign d4_all_out_valid = &d4_out_valid;

  // Loop bounds as wires so the FSM reads as intent, not arithmetic.
  // last_j: this block covers the final LANES keys of the row. Compared against
  // the CONSTANT (N-LANES) rather than testing (j_base+LANES == N), because
  // j_base is IW bits wide and j_base+LANES wraps to 0 at exactly that point.
  logic last_j, last_pair;
  assign last_j    = (j_base >= IW'(N - LANES));
  assign last_pair = last_j && (row_i == IW'(N - 1));

// =============================================================================
// CHUNK 3: dot4 instantiation (LANES of them)
//   Every lane shares the SAME a_flat (one query row) and takes a DIFFERENT
//   b_flat (key row j_base+l). That is the reuse that makes lanes worth it:
//   Q is broadcast, K is the axis being parallelized.
// -----------------------------------------------------------------------------
  genvar l;
  generate
    for (l = 0; l < LANES; l++) begin : gen_lane
      dot4 #(.DW(DW), .FRAC(FRAC), .D(D), .SCALE_EN(1)) u_dot (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (d4_in_valid),
        .in_ready  (d4_in_ready[l]),
        .a_flat    (q_mem[row_i]),
        .b_flat    (k_mem[j_base + IW'(l)]),
        .out_valid (d4_out_valid[l]),
        .out_ready (d4_out_ready),
        .s         (d4_s[l])
      );
    end
  endgenerate

// =============================================================================
// CHUNK 4: controller FSM
//   IDLE  -- wait for the Q/K input beat
//   ISSUE -- present one (row_i, j_base) pair to the lanes, wait for accept
//   WAIT  -- wait for the lane results, capture them
//   DONE  -- hold S_flat until the consumer takes it
// -----------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) curr_state <= IDLE;
    else        curr_state <= next_state;
  end

  always_comb begin
    next_state = curr_state;
    case (curr_state)
      IDLE:  if (in_valid) next_state = ISSUE;
      ISSUE: if (d4_all_in_ready) next_state = WAIT;          // beat accepted
      // written as if/else rather than a ternary: a ?: between two enum
      // literals yields a plain vector, which strict tools reject on an
      // enum-typed assignment without an explicit cast
      WAIT:  if (d4_all_out_valid) begin
               if (last_pair) next_state = DONE;
               else           next_state = ISSUE;
             end
      DONE:  if (out_ready) next_state = IDLE;
      default: next_state = IDLE;   // 4 states fill a 2-bit encoding, but keep
    endcase                         // the recovery arm for upset-tolerance
  end

// =============================================================================
// CHUNK 5: datapath -- load Q/K, advance counters, capture scores
// -----------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      row_i  <= '0;
      j_base <= '0;
      for (int t = 0; t < N; t++) begin
        q_mem[t] <= '0;
        k_mem[t] <= '0;
      end
      for (int t = 0; t < N*N; t++) s_mem[t] <= '0;
    end else begin
      // Capture on the ACCEPT BEAT itself (in_valid && in_ready), never a cycle
      // later -- a conforming producer may drop Q_flat/K_flat immediately after.
      if (in_valid && in_ready) begin
        for (int t = 0; t < N; t++) begin
          q_mem[t] <= Q_flat[t*D*DW +: D*DW];
          k_mem[t] <= K_flat[t*D*DW +: D*DW];
        end
        row_i  <= '0;
        j_base <= '0;
      end

      // Lane results land together. RHS reads the PRE-increment counters
      // because nonblocking assignment defers the update to the clock edge.
      if (curr_state == WAIT && d4_all_out_valid) begin
        for (int t = 0; t < LANES; t++)
          s_mem[int'(row_i)*N + int'(j_base) + t] <= d4_s[t];

        if (last_j) begin
          j_base <= '0;
          row_i  <= row_i + IW'(1);   // wraps to 0 on the last pair; harmless,
        end else begin                // the FSM has already left for DONE
          j_base <= j_base + IW'(LANES);
        end
      end
    end
  end

// =============================================================================
// CHUNK 6: handshake outputs + S_flat assembly
// -----------------------------------------------------------------------------
  assign in_ready     = (curr_state == IDLE);
  assign out_valid    = (curr_state == DONE);

  assign d4_in_valid  = (curr_state == ISSUE);
  assign d4_out_ready = (curr_state == WAIT);

  // Flatten s_mem into the output bus, same row-major packing the ports document.
  genvar g;
  generate
    for (g = 0; g < N*N; g++) begin : gen_pack
      assign S_flat[g*DW +: DW] = s_mem[g];
    end
  endgenerate

endmodule
