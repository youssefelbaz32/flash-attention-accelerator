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

    // Your implementation

endmodule