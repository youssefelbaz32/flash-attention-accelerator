module registered_skid_buffer #(
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

    // Your implementation

    logic input_fire, output_fire;

    logic [DATA_WIDTH - 1: 0] skid_slot;
    logic full;

    assign input_fire = s_ready && s_valid;
    assign output_fire = m_ready && m_valid;


    assign s_ready = !full;

    always_ff @(posedge clk) begin
        if (!resetn) begin
            full <= 0;
            skid_slot <= 0;
            m_data <= 0;
            m_valid <= 0;
        end else begin
            if (full) begin
                if (output_fire) begin
                    m_data <= skid_slot;
                    full <= 1'b0;
                    m_valid <= 1'b1;
                end

            end else begin
                if (m_valid && !m_ready) begin //this causes backpressure
                    if (input_fire) begin
                        skid_slot <= s_data;
                        full <= 1'b1;
                    end
                end 
                else if (input_fire) begin
                    m_data <= s_data;
                    m_valid <= 1'b1;
                end else if (output_fire) begin
                    m_valid <= 1'b0;
                end
            end

        end
    end

endmodule