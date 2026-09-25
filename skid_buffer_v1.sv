module skid_buffer #(
    parameter DATA_WIDTH = 8
) (
    input  logic                  clk,
    input  logic                  resetn,

    input  logic                  s_valid,
    input  logic [DATA_WIDTH-1:0] s_data,
    output logic                  s_ready,

    output logic                  m_valid,
    output logic [DATA_WIDTH-1:0] m_data,
    input  logic                  m_ready
);

// combinational bypass
// we have two states, EMPTY and FULL

logic input_fire, output_fire;

logic [DATA_WIDTH - 1: 0 ] skid_slot;
logic full;
// so what needs to be combinatinal?

assign s_ready = !full;
assign m_data = full ? skid_slot : s_data;
assign m_valid = full || s_valid;

assign input_fire = s_ready && s_valid;
assign output_fire = m_ready && m_valid;

always_ff @ (posedge clk) begin
    if (!resetn) begin
        full <= 1'b0;
        skid_slot <= '0;
    end else begin
        
        if (full) begin
            if (output_fire) begin
                full <= 1'b0;
            end
        end else begin
            if (input_fire && !output_fire) begin
                full <= 1'b1;
                skid_slot <= s_data;
            end
        end

    end

end





endmodule