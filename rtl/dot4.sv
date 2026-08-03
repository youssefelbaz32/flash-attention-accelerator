// M5 · module 1/5: dot-product MAC unit  (the atom inside QK^T)
//
// Computes score = sum_{k=0..D-1} a[k]*b[k]  for two length-D Q8.8 vectors.
// One TIME-MULTIPLEXED signed multiplier reused over D cycles (area-efficient),
// driven by a small FSM. Output is Q8.8. (/sqrt(d) scaling is applied later, at
// the QK^T level — this unit is a pure dot product.)
//
// Fixed-point: Q8.8 (DW=16, FRAC=8). Q8.8 * Q8.8 = Q16.16 (32b). Accumulate raw
// Q16.16 in a WIDE acc (ACCW) for headroom, then >>> FRAC + saturate to DW once.
//
// Interface: Avalon-ST-style valid/ready. Whole vector accepted in one input beat
// (a_flat/b_flat are D elements packed LSB-first). Streaming per-element comes at M6.
//
// Target: ModelSim (sim) + Quartus (synth), Intel FPGA. Synthesizable SV subset.
// Write CHUNK BY CHUNK. Do not fill everything at once.

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
  parameter DW=16, FRAC=8, D=4, ACCW=40
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


// =============================================================================
// CHUNK 2: internal signals
//   state enum (IDLE/LOAD/MAC/DONE), a_reg[D], b_reg[D], idx counter, acc (ACCW)
// -----------------------------------------------------------------------------
// TODO(you)

    typedef enum logic [1:0] {
        IDLE,
        LOAD,
        MAC,
        DONE
    } state_t;

    state_t curr_state;
    state_t next_state;
    logic [$clog2(D)-1:0] idx_counter;
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
            if (in_valid && in_ready) next_state = LOAD;
            else next_state = IDLE;
        LOAD: 
            next_state = MAC;
        MAC:
            if (idx_counter == D-1) next_state = DONE;
            else next_state = MAC;

        DONE: 
            if (out_valid && out_ready) next_state = IDLE;
            else next_state = DONE;
    endcase

end

// =============================================================================
// CHUNK 4: datapath  (load latch, MAC accumulate, >>>FRAC + saturate)
// -----------------------------------------------------------------------------
// TODO(you)

always_ff @(posedge clk) begin
    if (~rst_n) begin
        a_reg <= '{default: 0};
        b_reg <= '{default: 0};
        idx_counter <= '0;
        acc <= '0;
    end else begin
        if (curr_state == LOAD) begin
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
logic signed [ACCW-1: 0] shifted;
assign shifted = acc >>> FRAC; // >>> to preserve sign bit instead of >>


localparam signed sMAX = {1'b0, {(DW-1){1'b1}}};
localparam signed sMIN = {1'b1, {(DW-1){1'b0}}};

always_comb begin
    if (shifted > sMAX) s = sMAX;
    else if (shifted < sMIN) s = sMIN;
    else s = shifted[DW-1:0];
end


endmodule