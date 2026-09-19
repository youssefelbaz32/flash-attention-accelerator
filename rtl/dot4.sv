// M5 · module 1/5: dot-product MAC unit  (the atom inside QK^T)
//
// Computes score = sum_{k=0..D-1} a[k]*b[k]  for two length-D Q8.8 vectors.
// One TIME-MULTIPLEXED signed multiplier reused over D cycles (area-efficient),
// driven by a small FSM. Output is Q8.8.
//
// OPTIONAL 1/sqrt(D) SCALING (SCALE_EN=1): folded into the SAME final shift as the
// fixed-point normalization, so the result is rounded exactly ONCE, at full
// accumulator width. With M = log2(D):
//     1/sqrt(D) = 2^-floor(M/2) * c ,   c = 1        if M even
//                                       c = 1/sqrt2  if M odd
// An odd exponent leaves one factor of 2 under the root, and no shift can produce
// an irrational factor — hence the HALF_MUL constant multiply. qkt instantiates
// with SCALE_EN=1; SCALE_EN=0 leaves this a pure dot product (and reduces to
// exactly `acc >>> FRAC`, so the two configurations share one datapath).
//
// Fixed-point: Q8.8 (DW=16, FRAC=8). Q8.8 * Q8.8 = Q16.16, and the accumulator
// keeps that binary point while adding integer headroom for the D-term sum:
// ACCW = 2*DW + clog2(D) = 34 bits at the default config, i.e. Q18.16. Width and
// format are independent — accumulating only grows the integer side; the binary
// point moves exactly twice (at the multiply, and at the single final shift).
//
// Interface: valid/ready, following AXI4-Stream handshake semantics (valid/ready
// alone is not a full AXIS port; wrapping it as one at M6 is straightforward).
// The whole vector is accepted in one input beat (a_flat/b_flat are D elements
// packed LSB-first) and is CAPTURED ON THE ACCEPT BEAT ITSELF (valid && ready) --
// capturing a cycle later is a protocol violation, since a conforming producer
// may drop the data immediately after the transfer. Per-element streaming at M6.
//
// Target: Vivado XSim + synth (Xilinx/AMD). Also simulates locally under
// `iverilog -g2012`. Synthesizable SV subset.

// =============================================================================
// CHUNK 1: module declaration + parameters + ports
//   params: DW=16, FRAC=8, D=4, ACCW=40
//   ports :
//     clk, rst_n                         (active-LOW sync reset)
//     in_valid  (in) , in_ready  (out)   input handshake
//     a_flat, b_flat (in, [D*DW-1:0])    two packed Q8.8 vectors, elem k = [k*DW +: DW]
//     out_valid (out), out_ready (in)    output handshake
//     s         (out, signed [DW-1:0])   Q8.8 score
// -----------------------------------------------------------------------------
// TODO(you): module dot4 #(...) ( ... );

module dot4 #(
  parameter DW=16, FRAC=8, D=4,
  // Accumulator width, DERIVED so it tracks the parameter sweep instead of being
  // a magic number: 2*DW holds one Q16.16 product, +clog2(D) is the integer
  // growth from summing D of them. Overridable if you want extra headroom.
  parameter ACCW = 2*DW + $clog2(D),
  parameter bit SCALE_EN = 0,
  // ROUND_EN=1 adds half an LSB before the final shift (round-to-nearest
  // instead of truncate-toward-negative-infinity). Costs one adder. Worth it
  // in pv, NOT worth it in qkt -- see the 200-seed sweep in
  // python/04_rtl_fixed_model.py: 21% mean-error cut in pv, 0.3% in qkt.
  parameter bit ROUND_EN = 0
)(
  input  logic clk,
  input  logic rst_n,

  input  logic in_valid,
  output logic in_ready,
  input  logic [D*DW-1:0] a_flat,
  input  logic [D*DW-1:0] b_flat,

  output logic out_valid,
  input  logic out_ready,
  output logic signed [DW-1:0] s
);

    logic signed [DW-1:0] a_reg [D];
    logic signed [DW-1:0] b_reg [D];
    localparam int M = $clog2(D);
    localparam int IDXW = (D > 1) ? $clog2(D) : 1;   // never zero-width
    localparam int SQRTD_SH = SCALE_EN ? M/2: 0;
    localparam bit ODD_EXP = SCALE_EN && (M % 2 == 1);
    // Only the 1/sqrt(D) path needs D to be a power of two (the shift has to be
    // an integer). With SCALE_EN=0 this is a plain dot product and ANY D is
    // legal -- pv relies on that when it contracts over N.
    initial if (SCALE_EN && (D != (1 <<< M)))
      $error("SCALE_EN=1 needs D a power of 2; D=%0d is not", D);

    localparam signed [DW-1:0] HALF_MUL = ODD_EXP ? ((7071 * (1 <<< FRAC)) + 5000) / 10000 : (1 <<< FRAC);


// =============================================================================
// CHUNK 2: internal signals
//   state enum (IDLE/MAC/DONE), a_reg[D], b_reg[D], idx counter, acc (ACCW)
// -----------------------------------------------------------------------------
// TODO(you)

    typedef enum logic [1:0] {
        IDLE,
        MAC,
        DONE
    } state_t;

    state_t curr_state;
    state_t next_state;
    logic [IDXW-1:0] idx_counter;
    logic signed [ACCW-1:0] acc;




// =============================================================================
// CHUNK 3: FSM  (state register + next-state logic)
// -----------------------------------------------------------------------------
// TODO(you)

always_ff @(posedge clk) begin
    if (~rst_n) begin
        curr_state <= IDLE;
    end else curr_state <= next_state;
end

always_comb begin
    next_state = curr_state;
    case(curr_state)
        IDLE:
            if (in_valid) next_state = MAC;
            else next_state = IDLE;
        MAC:
            if (idx_counter == IDXW'(D-1)) next_state = DONE;
            else next_state = MAC;

        DONE:
            if (out_valid && out_ready) next_state = IDLE;
            else next_state = DONE;

        // 3 states in a 2-bit encoding leaves 0x3 reachable in silicon even
        // though it is unreachable by design — recover to a safe state.
        default: next_state = IDLE;
    endcase

end

// =============================================================================
// CHUNK 4: datapath  (load latch, MAC accumulate, >>>FRAC + saturate)
// -----------------------------------------------------------------------------
// TODO(you)

always_ff @(posedge clk) begin
    if (~rst_n) begin
        for (int i = 0; i < D; i++) begin
            a_reg[i] <= '0;
            b_reg[i] <= '0; 
        end
        idx_counter <= '0;
        acc <= '0;
    end else begin
        if (in_ready && in_valid) begin
            for (int i = 0; i < D; i++) begin
                a_reg [i] <= a_flat[i * DW +: DW]; // select from i*DW and up DW 
                b_reg [i] <= b_flat[i * DW +: DW];
            end
            idx_counter <= '0;
            acc <= '0;
        end else if (curr_state == MAC) begin
            acc <= acc + (a_reg[idx_counter] * b_reg[idx_counter]);
            idx_counter <= idx_counter + 1;
        end
    end

end


// =============================================================================
// CHUNK 5: handshake outputs  (in_ready, out_valid, s)
// -----------------------------------------------------------------------------
// TODO(you)

assign in_ready = (curr_state == IDLE);
assign out_valid = (curr_state == DONE);

//divide by 2^FRAC to get to Q8.8
logic signed [ACCW+DW-1:0] scaled;
localparam int TOT_SH = 2*FRAC + SQRTD_SH;
// Half-LSB of the FINAL shift, so it is added once at full accumulator width --
// adding it per-term would round D times and defeat the purpose.
localparam signed [ACCW+DW-1:0] RND = ROUND_EN ? (1 <<< (TOT_SH-1)) : '0;
assign scaled = (acc * HALF_MUL + RND) >>> TOT_SH;

localparam signed [ACCW+DW-1:0] sMAX =  (1 <<< (DW-1)) - 1;
localparam signed [ACCW+DW-1:0] sMIN = -(1 <<< (DW-1));

// Compares run at full ACCW+DW width (matched operand widths — a narrow bound
// against a wide value is how sMIN silently became +32768 once before). The
// truncation on assignment is intentional: the clamped value fits DW by
// construction, so take the low DW bits explicitly rather than implicitly.
always_comb begin
    if (scaled > sMAX) s = sMAX[DW-1:0];
    else if (scaled < sMIN) s = sMIN[DW-1:0];
    else s = scaled[DW-1:0];
end


endmodule